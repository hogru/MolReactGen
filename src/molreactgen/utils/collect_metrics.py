# coding=utf-8
# src/molreactgen/utils/collect_metrics.py
"""Collect experiment metrics from a number of files and save them to a single file.

The code is specific to this project. It is not intended to be used as a general
purpose tool. By adding functionality over time it has become a bit messy.
I live with that for now.
"""

import argparse
import datetime
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Optional, Union

import pandas as pd  # type: ignore
import wandb  # type: ignore
from codetiming import Timer
from humanfriendly import format_timespan  # type: ignore
from loguru import logger
from rich.console import Console
from rich.table import Table

from molreactgen.helpers import configure_logging, determine_log_level
from molreactgen.tokenizer import (
    REGEX_PATTERN_ATOM,
    REGEX_PATTERN_CHAR,
    REGEX_PATTERN_SMARTS,
)
from molreactgen.train import WANDB_PROJECT_NAME

# Global variables, defaults
# DEFAULT_OUTPUT_FILE_NAME: Final[str] = "./metrics"
SCOPE_TOKENIZER: Final[str] = "tokenizer"
SCOPE_TRAINING: Final[str] = "training"
SCOPE_MODEL: Final[str] = "model"
SCOPE_WANDB: Final[str] = "wandb"
SCOPE_MOLS_EVAL: Final[str] = "evaluation_mols"
SCOPE_RTS_EVAL: Final[str] = "evaluation_rts"

# File names
TOKENIZER_FILE: Final[str] = "tokenizer.json"
CONFIG_FILE: Final[str] = "config.json"
CHECKPOINT_FILE: Final[str] = "checkpoint-"
MISC_TRAIN_FILE: Final[str] = "misc_training_information.json"
SCHEDULER_FILE: Final[str] = "scheduler.pt"
MODEL_FILE: Final[str] = "pytorch_model.bin"
TRAINER_STATE_FILE: Final[str] = "trainer_state.json"
ALL_RESULTS_FILE: Final[str] = "all_results.json"
README_FILE: Final[str] = "README.md"
MODEL_LINK_TO_FILE: Final[str] = "link_to_model"
GENERATION_ARGS_FILE: Final[str] = "generate_args.json"
GENERATED_SMILES_FILE: Final[str] = "generated_smiles.csv"
GENERATED_SMARTS_FILE: Final[str] = "generated_smarts.csv"
GENERATION_STATS_FILE: Final[str] = "generate_stats.json"
EVALUATED_FILE: Final[str] = "assess_stats.json"

# Keys and Values in JSON files
PRE_TOKENIZER_KEY: Final[str] = "pre_tokenizer"
PRE_TOKENIZER_TYPE_KEY: Final[str] = "type"
PRE_TOKENIZER_PATTERN_KEY: Final[str] = "pattern"
PRE_TOKENIZER_REGEX_KEY: Final[str] = "Regex"
PRE_TOKENIZER_SPLIT_KEY: Final[str] = "SPLIT"
PRE_TOKENIZER_BYTELEVEL_KEY: Final[str] = "BYTELEVEL"
PRE_TOKENIZER_CHAR_TYPE: Final[str] = "Char"
PRE_TOKENIZER_ATOM_TYPE: Final[str] = "Atom"
PRE_TOKENIZER_SMARTS_TYPE: Final[str] = "Smarts"
PRE_TOKENIZER_BYTELEVEL_TYPE: Final[str] = "ByteLevel"
TOKENIZER_MODEL_KEY: Final[str] = "model"
TOKENIZER_TYPE_KEY: Final[str] = "type"
TOKENIZER_VOCAB_KEY: Final[str] = "vocab"
TOKENIZER_ADDED_TOKENS_KEY: Final[str] = "added_tokens"
MODEL_LAYER_KEY: Final[str] = "n_layer"
MODEL_HEAD_KEY: Final[str] = "n_head"
MODEL_HIDDEN_DIM_KEY: Final[str] = "n_embd"
MODEL_SIZE_KEY: Final[str] = "model_size"
DATASET_DIR_KEY: Final[str] = "dataset_dir"
WANDB_RUN_ID_KEY: Final[str] = "wandb_run_id"
WANDB_RUN_NAME_KEY: Final[str] = "wandb_run_name"
EPOCH_KEY: Final[str] = "epoch"
TRAIN_RUNTIME_KEY: Final[str] = "train_runtime"
TRAIN_LOSS_KEY: Final[str] = "train_loss"
VAL_LOSS_KEY: Final[str] = "best_metric"
EVAL_ACC_KEY: Final[str] = "eval_accuracy"
TEST_LOSS_KEY: Final[str] = "test_loss"
TEST_ACC_KEY: Final[str] = "test_accuracy"
TEST_PPL_KEY: Final[str] = "test_perplexity"
LOG_HISTORY_KEY: Final[str] = "log_history"
GENERATION_NUM_BEAMS_KEY: Final[str] = "num_beams"
GENERATION_NUM_TEMPERATURE_KEY: Final[str] = "temperature"
GENERATION_NUM_REPETITION_PENALTY_KEY: Final[str] = "repetition_penalty"
MOLS_VALIDITY_KEY: Final[str] = "Validity"
MOLS_UNIQUENESS_KEY: Final[str] = "Uniqueness"
MOLS_NOVELTY_KEY: Final[str] = "Novelty"
MOLS_FCD_KEY: Final[str] = "FCD"
MOLS_FCD_GUACAMOL_KEY: Final[str] = "FCD GuacaMol"
RTS_COUNT_KEY: Final[str] = "counts"
RTS_FRAC_KEY: Final[str] = "relative_fractions"
RTS_VALIDITY_KEY: Final[str] = "valid"
RTS_UNIQUENESS_KEY: Final[str] = "unique"
RTS_FEASIBILITY_KEY: Final[str] = "feasible"
RTS_KNOWN_EITHER_KEY: Final[str] = "known"
RTS_KNOWN_VAL_KEY: Final[str] = "known_from_valid_set"
RTS_KNOWN_TEST_KEY: Final[str] = "known_from_test_set"

# Search strings in MD files
BATCH_SIZE_KEY: Final[str] = "train_batch_size:"
LR_KEY: Final[str] = "learning_rate:"

# Wandb related
WANDB_OUTPUT_DIR: Final[str] = "output_dir"

# Formatter strings
DEFAULT_FORMATTER_FLOAT: Final[str] = ".4f"


@dataclass
class Metric:
    column_name: str  # Not used during output
    scope: tuple[str, ...]  # might change to Enum
    dtype: type
    value: Any  # Any (default) value when instantiating the metric is not used later when evaluating it
    formatter: str = ""
    ref: str = ""
    idx: int = 0
    getter: Optional[Callable] = None
    retrieved: bool = False

    def __hash__(self) -> int:
        return hash(self.column_name)


@dataclass
class WandbMetric:
    # run_id: str
    run_name: str
    output_dir: str


class WandbMetrics:
    def __init__(
        self, entity: Optional[str] = None, project_name: Optional[str] = None
    ) -> None:
        self._api = wandb.Api()
        self._entity = self._api.default_entity if entity is None else entity
        self._project_name = (
            WANDB_PROJECT_NAME if project_name is None else project_name
        )
        self._runs: dict[str, WandbMetric] = {}
        self._metrics = self.read_metrics()

    def read_metrics(self) -> dict[str, WandbMetric]:
        self._runs = self._api.runs("/".join((self._entity, self._project_name)))
        return {
            run.id: WandbMetric(  # type: ignore
                run.name, Path(run.config.get(WANDB_OUTPUT_DIR, "")).stem  # type: ignore
            )
            for run in self._runs
        }

    def get_run_id(
        self,
        output_dir: Union[str, os.PathLike],
        raise_error_if_ambiguous: bool = False,
    ) -> Optional[str]:
        output_dir = Path(output_dir).stem
        run_ids = [
            run_id
            for run_id, metric in self._metrics.items()
            if metric.output_dir == output_dir
        ]
        if not run_ids:
            return None

        if len(run_ids) > 1:
            if raise_error_if_ambiguous:
                raise ValueError(
                    f"Found multiple runs with output_dir {output_dir}: {run_ids}"
                )
            logger.debug(f"Found multiple runs with output_dir {output_dir}: {run_ids}")
            return "Multiple"

        return run_ids[0]

    def get_run_name(
        self,
        output_dir: Union[str, os.PathLike],
        raise_error_if_ambiguous: bool = False,
    ) -> Optional[str]:
        output_dir = Path(output_dir).stem
        run_names = [
            metric.run_name
            for metric in self._metrics.values()
            if metric.output_dir == output_dir
        ]
        if not run_names:
            return None

        if len(run_names) > 1:
            if raise_error_if_ambiguous:
                raise ValueError(
                    f"Found multiple runs with output_dir {output_dir}: {run_names}"
                )
            logger.debug(
                f"Found multiple runs with output_dir {output_dir}: {run_names}"
            )
            return "Multiple"

        return run_names[0]


class Experiment:
    # _getters: dict[str, Callable[[Path], Any]] = {
    _getters: dict[str, str] = {
        "pre_tokenizer": "_get_pre_tokenizer",
        "algorithm": "_get_algorithm",
        "vocab_size": "_get_vocab_size",
        "layers": "_get_layers",
        "heads": "_get_heads",
        "hidden_dim": "_get_hidden_dim",
        "model_size": "_get_model_size",
        "dataset_dir": "_get_dataset_dir",
        "num_epochs": "_get_num_epochs",
        "batch_size": "_get_batch_size",
        "lr": "_get_lr",
        "wandb_run_id": "_get_wandb_run_id",
        "wandb_run_name": "_get_wandb_run_name",
        "train_runtime": "_get_train_runtime",
        "train_loss": "_get_train_loss",
        "val_loss": "_get_val_loss",
        "val_acc": "_get_val_acc",
        "test_loss": "_get_test_loss",
        "test_acc": "_get_test_acc",
        "test_ppl": "_get_test_perplexity",
        "num_beams": "_get_num_beams",
        "temperature": "_get_temperature",
        "repetition_penalty": "_get_repetition_penalty",
        "validity_mols": "_get_validity_mols",
        "uniqueness_mols": "_get_uniqueness_mols",
        "novelty_mols": "_get_novelty_mols",
        "fcd_mols": "_get_fcd",
        "fcd_g_mols": "_get_fcd_g",
        "validity_rts": "_get_validity_rts",
        "uniqueness_rts": "_get_uniqueness_rts",
        "feasibility_rts": "_get_feasibility",
        "known_either_rts": "_get_known_either",
        "known_val_rts": "_get_known_val",
        "known_test_rts": "_get_known_test",
    }

    _indices: dict[str, int] = {}
    for _idx, _getter in enumerate(_getters, start=1):
        _indices[_getter] = _idx

    def __init__(
        self,
        directory: Union[str, os.PathLike],
        tokenizer_scope: bool = True,
        model_scope: bool = True,
        train_scope: bool = True,
        eval_scope: bool = True,
        wandb_scope: Optional[WandbMetrics] = None,
    ) -> None:
        self.directory: Path = Path(directory).resolve()
        self._tokenizer_scope = tokenizer_scope
        self._model_scope = model_scope
        self._train_scope = train_scope
        self._eval_scope = eval_scope
        self._wandb_scope = wandb_scope
        self._metrics: dict[str, Metric] = {
            "pre_tokenizer": Metric(
                column_name="pre_tokenizer",
                scope=("tokenizer",),
                dtype=str,
                value=None,
            ),
            "algorithm": Metric(
                column_name="tokenization_algorithm",
                scope=("tokenizer",),
                dtype=str,
                value=None,
            ),
            "vocab_size": Metric(
                column_name="vocab_size",
                scope=("tokenizer",),
                dtype=int,
                value=None,
            ),
            "layers": Metric(
                column_name="num_layers",
                scope=("model",),
                dtype=int,
                value=None,
            ),
            "heads": Metric(
                column_name="num_heads",
                scope=("model",),
                dtype=int,
                value=None,
            ),
            "hidden_dim": Metric(
                column_name="hidden_dim",
                scope=("model",),
                dtype=int,
                value=None,
            ),
            "model_size": Metric(
                column_name="model_size",
                scope=("model",),
                dtype=str,
                value=None,
            ),
            "dataset_dir": Metric(
                column_name="dataset_dir",
                scope=("training",),
                dtype=str,
                value=None,
            ),
            "num_epochs": Metric(
                column_name="number_of_epochs",
                scope=("training",),
                dtype=int,
                value=None,
            ),
            "batch_size": Metric(
                column_name="batch_size",
                scope=("training",),
                dtype=int,
                value=None,
            ),
            "lr": Metric(
                column_name="learning_rate",
                scope=("training",),
                dtype=float,
                value=None,
                formatter=DEFAULT_FORMATTER_FLOAT,
            ),
            "wandb_run_id": Metric(
                column_name="wandb_run_id",
                scope=("wandb",),
                dtype=str,
                value=None,
            ),
            "wandb_run_name": Metric(
                column_name="wandb_run_name",
                scope=("wandb",),
                dtype=str,
                value=None,
            ),
            "train_runtime": Metric(
                column_name="training_runtime",
                scope=("training",),
                dtype=str,
                value=None,
            ),
            "train_loss": Metric(
                column_name="training_loss",
                scope=("training",),
                dtype=float,
                value=None,
                formatter=DEFAULT_FORMATTER_FLOAT,
            ),
            "val_loss": Metric(
                column_name="validation_loss",
                scope=("training",),
                dtype=float,
                value=None,
                formatter=DEFAULT_FORMATTER_FLOAT,
            ),
            "val_acc": Metric(
                column_name="validation_accuracy",
                scope=("training",),
                dtype=float,
                value=None,
                formatter=DEFAULT_FORMATTER_FLOAT,
            ),
            "test_loss": Metric(
                column_name="test_loss",
                scope=("training",),
                dtype=float,
                value=None,
                formatter=DEFAULT_FORMATTER_FLOAT,
            ),
            "test_acc": Metric(
                column_name="test_accuracy",
                scope=("training",),
                dtype=float,
                value=None,
                formatter=DEFAULT_FORMATTER_FLOAT,
            ),
            "test_ppl": Metric(
                column_name="test_perplexity",
                scope=("training",),
                dtype=float,
                value=None,
                formatter=DEFAULT_FORMATTER_FLOAT,
            ),
            "num_beams": Metric(
                column_name="num_beams",
                scope=("evaluation_mols", "evaluation_rts"),
                dtype=int,
                value=None,
            ),
            "temperature": Metric(
                column_name="temperature",
                scope=("evaluation_mols", "evaluation_rts"),
                dtype=float,
                value=None,
                formatter=".2f",
            ),
            "repetition_penalty": Metric(
                column_name="repetition_penalty",
                scope=("evaluation_mols", "evaluation_rts"),
                dtype=float,
                value=None,
                formatter=".2f",
            ),
            "validity_mols": Metric(
                column_name="validity",
                scope=("evaluation_mols",),
                dtype=float,
                value=None,
                formatter=DEFAULT_FORMATTER_FLOAT,
            ),
            "uniqueness_mols": Metric(
                column_name="uniqueness",
                scope=("evaluation_mols",),
                dtype=float,
                value=None,
                formatter=DEFAULT_FORMATTER_FLOAT,
            ),
            "novelty_mols": Metric(
                column_name="novelty",
                scope=("evaluation_mols",),
                dtype=float,
                value=None,
                formatter=DEFAULT_FORMATTER_FLOAT,
            ),
            "fcd_mols": Metric(
                column_name="fcd",
                scope=("evaluation_mols",),
                dtype=float,
                value=None,
                formatter=DEFAULT_FORMATTER_FLOAT,
            ),
            "fcd_g_mols": Metric(
                column_name="fcd_guacamol",
                scope=("evaluation_mols",),
                dtype=float,
                value=None,
                formatter=DEFAULT_FORMATTER_FLOAT,
            ),
            "validity_rts": Metric(
                column_name="validity",
                scope=("evaluation_rts",),
                dtype=float,
                value=None,
                formatter=DEFAULT_FORMATTER_FLOAT,
            ),
            "uniqueness_rts": Metric(
                column_name="uniqueness",
                scope=("evaluation_rts",),
                dtype=float,
                value=None,
                formatter=DEFAULT_FORMATTER_FLOAT,
            ),
            "feasibility_rts": Metric(
                column_name="feasibility",
                scope=("evaluation_rts",),
                dtype=float,
                value=None,
                formatter=DEFAULT_FORMATTER_FLOAT,
            ),
            "known_either_rts": Metric(
                column_name="known_from_validation_or_test",
                scope=("evaluation_rts",),
                dtype=int,
                value=None,
            ),
            "known_val_rts": Metric(
                column_name="known_from_validation",
                scope=("evaluation_rts",),
                dtype=int,
                value=None,
            ),
            "known_test_rts": Metric(
                column_name="known_from_test",
                scope=("evaluation_rts",),
                dtype=int,
                value=None,
            ),
        }

        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize the metrics."""

        if self._metrics.keys() != self._getters.keys():
            raise RuntimeError("The keys of metrics and getters must be equal")

        for getter in self._getters.values():
            if not callable(getattr(self, getter, False)):
                raise RuntimeError(
                    f"All getters must exist and be callable. "
                    f"Getter {getter} does not exist or is not callable."
                )

        for idx, getter in enumerate(self._getters, start=1):
            self._metrics[getter].ref = getter
            self._metrics[getter].getter = getattr(self, self._getters[getter])

    @property
    def directory(self) -> Path:
        return self._directory

    @directory.setter
    def directory(self, value: Path) -> None:
        if hasattr(self, "_directory"):
            raise AttributeError("directory is already set, can not be changed")

        if not isinstance(value, Path):
            raise TypeError("directory must be a pathlib.Path object")

        # noinspection PyAttributeOutsideInit
        self._directory = value

    def _is_valid_directory(self) -> bool:
        """Check if the directory is valid."""

        if (
            not isinstance(self.directory, Path)
            or (not self.directory.is_dir())
            or self.directory.is_symlink()
        ):
            return False

        return (
            not self.directory.stem.startswith(CHECKPOINT_FILE)
            or not (self.directory / SCHEDULER_FILE).is_file()
        )

    def _is_model_directory(self) -> bool:
        """Check if the directory is a model directory."""

        return self._is_valid_directory() and (self.directory / MODEL_FILE).is_file()

    def _is_generated_molecules_directory(self) -> bool:
        """Check if the directory holds generated molecules."""

        return (
            self._is_valid_directory()
            and (self.directory / GENERATED_SMILES_FILE).is_file()
        )

    def _is_generated_reaction_templates_directory(self) -> bool:
        """Check if the directory holds generated reaction templates."""

        return (
            self._is_valid_directory()
            and (self.directory / GENERATED_SMARTS_FILE).is_file()
        )

    def _is_evaluated_directory(self) -> bool:
        """Check if the directory has been evaluated."""

        return (
            self._is_generated_molecules_directory()
            and (self.directory / EVALUATED_FILE).is_file()
        ) or (
            self._is_generated_reaction_templates_directory()
            and (self.directory / GENERATION_STATS_FILE).is_file()
        )

    def _get_pre_tokenizer(self) -> Optional[str]:
        """Get the pre-tokenizer."""

        try:
            with open(self.model_directory / TOKENIZER_FILE) as f:  # type: ignore
                tokenizer = json.load(f)
        except (FileNotFoundError, TypeError):
            return None

        try:
            if (
                tokenizer[PRE_TOKENIZER_KEY] is not None
                and tokenizer[PRE_TOKENIZER_KEY][PRE_TOKENIZER_TYPE_KEY].upper()
                == PRE_TOKENIZER_BYTELEVEL_KEY
            ):
                return PRE_TOKENIZER_BYTELEVEL_TYPE
        except KeyError:
            return None

        try:
            if (
                tokenizer[PRE_TOKENIZER_KEY] is None
                or tokenizer[PRE_TOKENIZER_KEY][PRE_TOKENIZER_PATTERN_KEY][
                    PRE_TOKENIZER_REGEX_KEY
                ]
                == REGEX_PATTERN_CHAR
            ):
                return PRE_TOKENIZER_CHAR_TYPE
        except KeyError:
            return None

        try:
            if (
                tokenizer[PRE_TOKENIZER_KEY][PRE_TOKENIZER_TYPE_KEY].upper()
                == PRE_TOKENIZER_SPLIT_KEY
                and tokenizer[PRE_TOKENIZER_KEY][PRE_TOKENIZER_PATTERN_KEY][
                    PRE_TOKENIZER_REGEX_KEY
                ]
                == REGEX_PATTERN_ATOM
            ):
                return PRE_TOKENIZER_ATOM_TYPE
        except KeyError:
            return None

        try:
            if (
                tokenizer[PRE_TOKENIZER_KEY][PRE_TOKENIZER_TYPE_KEY].upper()
                == PRE_TOKENIZER_SPLIT_KEY
                and tokenizer[PRE_TOKENIZER_KEY][PRE_TOKENIZER_PATTERN_KEY][
                    PRE_TOKENIZER_REGEX_KEY
                ]
                == REGEX_PATTERN_SMARTS
            ):
                return PRE_TOKENIZER_SMARTS_TYPE
        except KeyError:
            return None

        return None

    def _get_algorithm(self) -> Optional[str]:
        """Get the tokenization algorithm."""

        try:
            with open(self.model_directory / TOKENIZER_FILE) as f:  # type: ignore
                return json.load(f)[TOKENIZER_MODEL_KEY][TOKENIZER_TYPE_KEY]
        except (FileNotFoundError, KeyError, TypeError):
            return None

    def _get_vocab_size(self) -> Optional[int]:
        """Get the vocabulary size (without special tokens)."""

        try:
            with open(self.model_directory / TOKENIZER_FILE) as f:  # type: ignore
                tokenizer = json.load(f)
        except (FileNotFoundError, TypeError):
            return None

        try:
            return len(tokenizer[TOKENIZER_MODEL_KEY][TOKENIZER_VOCAB_KEY]) - len(
                tokenizer[TOKENIZER_ADDED_TOKENS_KEY]
            )
        except KeyError:
            return None

    def _get_layers(self) -> Optional[int]:
        """Get the number of layers."""

        try:
            with open(self.model_directory / CONFIG_FILE) as f:  # type: ignore
                return json.load(f)[MODEL_LAYER_KEY]

        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _get_heads(self) -> Optional[int]:
        """Get the number of heads."""

        try:
            with open(self.model_directory / CONFIG_FILE) as f:  # type: ignore
                return json.load(f)[MODEL_HEAD_KEY]

        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _get_hidden_dim(self) -> Optional[int]:
        """Get the hidden dimension size."""

        try:
            with open(self.model_directory / CONFIG_FILE) as f:  # type: ignore
                return json.load(f)[MODEL_HIDDEN_DIM_KEY]

        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _get_model_size(self) -> Optional[str]:
        """Get the model's size (trainable params in millions)."""

        try:
            with open(self.model_directory / MISC_TRAIN_FILE) as f:  # type: ignore
                return json.load(f)[MODEL_SIZE_KEY]

        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _get_dataset_dir(self) -> Optional[str]:
        """Get the directory path to the training/validation/test data."""

        try:
            with open(self.model_directory / MISC_TRAIN_FILE) as f:  # type: ignore
                return json.load(f)[DATASET_DIR_KEY]

        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _get_num_epochs(self) -> Optional[int]:
        """Get the number of epochs."""

        try:
            with open(self.model_directory / ALL_RESULTS_FILE) as f:  # type: ignore
                # Hugging Face reports the number of epochs as a float
                # Sometimes the number is very close to the number of epochs, but not exactly
                # Therefore, we round the number of epochs
                return int(round(json.load(f)[EPOCH_KEY], 0))

        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _get_batch_size(self) -> Optional[int]:
        """Get the batch size."""

        try:
            content = Path(self.model_directory / README_FILE).read_text()  # type: ignore
        except (FileNotFoundError, TypeError):
            return None

        pattern = re.compile(rf"(^\s*)-(\s*{BATCH_SIZE_KEY}\s*)(\d*)", re.MULTILINE)
        match = pattern.search(content)
        return None if match is None else int(match[3])

    def _get_lr(self) -> Optional[float]:
        """Get the learning rate."""

        try:
            content = Path(self.model_directory / README_FILE).read_text()  # type: ignore
        except (FileNotFoundError, TypeError):
            return None

        pattern = re.compile(rf"(^\s*)-(\s*{LR_KEY}\s*)(\d*\.\d*)", re.MULTILINE)
        match = pattern.search(content)
        return None if match is None else float(match[3])

    def _get_wandb_run_id(self) -> Optional[str]:
        """Get the wandb run id."""

        if self._wandb_scope is None or self.model_directory is None:
            return None

        # return self._wandb_scope.get_run_id(self.model_directory)
        remote_wandb_run_id = self._wandb_scope.get_run_id(self.model_directory)

        if remote_wandb_run_id is not None:
            try:
                with open(self.model_directory / MISC_TRAIN_FILE) as f:  # type: ignore
                    local_wandb_run_id = json.load(f)[WANDB_RUN_ID_KEY]

            except (FileNotFoundError, TypeError, KeyError):
                local_wandb_run_id = None

            if local_wandb_run_id is not None:
                if remote_wandb_run_id != local_wandb_run_id:
                    logger.warning(
                        f"Remote wandb run id {remote_wandb_run_id} differs from "
                        f"local wandb run id {local_wandb_run_id}"
                    )

        return remote_wandb_run_id

    def _get_wandb_run_name(self) -> Optional[str]:
        """Get the wandb run name."""

        if self._wandb_scope is None or self.model_directory is None:
            return None

        # return self._wandb_scope.get_run_name(self.model_directory)
        remote_wandb_run_name = self._wandb_scope.get_run_name(self.model_directory)

        if remote_wandb_run_name is not None:
            try:
                with open(self.model_directory / MISC_TRAIN_FILE) as f:  # type: ignore
                    local_wandb_run_name = json.load(f)[WANDB_RUN_NAME_KEY]

            except (FileNotFoundError, TypeError, KeyError):
                local_wandb_run_name = None

            if local_wandb_run_name is not None:
                if remote_wandb_run_name != local_wandb_run_name:
                    logger.warning(
                        f"Remote wandb run name {remote_wandb_run_name} differs from "
                        f"local wandb run name {local_wandb_run_name}"
                    )

        return remote_wandb_run_name

    def _get_train_runtime(self) -> Optional[str]:
        """Get the training runtime (duration)"""

        try:
            with open(self.model_directory / ALL_RESULTS_FILE) as f:  # type: ignore
                train_runtime_in_seconds = int(json.load(f)[TRAIN_RUNTIME_KEY])
                # not sure if timedelta always works but good enough for the purpose of this script
                return str(datetime.timedelta(seconds=train_runtime_in_seconds))
        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _get_train_loss(self) -> Optional[float]:
        """Get the last training loss."""

        try:
            with open(self.model_directory / ALL_RESULTS_FILE) as f:  # type: ignore
                return json.load(f)[TRAIN_LOSS_KEY]
        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _get_val_loss(self) -> Optional[float]:
        """Get the validation loss."""

        try:
            with open(self.model_directory / TRAINER_STATE_FILE) as f:  # type: ignore
                return json.load(f)[VAL_LOSS_KEY]
        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _get_val_acc(self) -> Optional[float]:
        """Get the validation accuracy."""

        try:
            with open(self.model_directory / TRAINER_STATE_FILE) as f:  # type: ignore
                log_history = json.load(f)[LOG_HISTORY_KEY]
                return next(
                    (
                        log[EVAL_ACC_KEY]
                        for log in reversed(log_history)
                        if EVAL_ACC_KEY in log
                    )
                )
        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _get_test_loss(self) -> Optional[float]:
        """Get the test loss."""

        try:
            with open(self.model_directory / ALL_RESULTS_FILE) as f:  # type: ignore
                return json.load(f)[TEST_LOSS_KEY]
        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _get_test_acc(self) -> Optional[float]:
        """Get the test accuracy."""

        try:
            with open(self.model_directory / ALL_RESULTS_FILE) as f:  # type: ignore
                return json.load(f)[TEST_ACC_KEY]
        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _get_test_perplexity(self) -> Optional[float]:
        """Get the test perplexity."""

        try:
            with open(self.model_directory / ALL_RESULTS_FILE) as f:  # type: ignore
                return json.load(f)[TEST_PPL_KEY]
        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _read_generation_args_file(self) -> Optional[dict[str, Any]]:
        """Read the generation arguments file."""

        try:
            with open(
                self.generated_directory / GENERATION_ARGS_FILE  # type: ignore
            ) as f:
                return json.load(f)
        except (FileNotFoundError, TypeError):
            return None

    def _get_num_beams(self) -> Optional[int]:
        """Get the number of beams during generation."""

        generation_args = self._read_generation_args_file()

        try:
            return generation_args[GENERATION_NUM_BEAMS_KEY]  # type: ignore
        except (KeyError, TypeError):
            return None

    def _get_temperature(self) -> Optional[float]:
        """Get the generation temperature."""

        generation_args = self._read_generation_args_file()

        try:
            return generation_args[GENERATION_NUM_TEMPERATURE_KEY]  # type: ignore
        except (KeyError, TypeError):
            return None

    def _get_repetition_penalty(self) -> Optional[float]:
        """Get the repetition penalty during generation."""

        generation_args = self._read_generation_args_file()

        try:
            return generation_args[GENERATION_NUM_REPETITION_PENALTY_KEY]  # type: ignore
        except (KeyError, TypeError):
            return None

    def _get_validity_mols(self) -> Optional[float]:
        """Get the validity of the generated molecules."""

        try:
            with open(self.generated_molecules_directory / EVALUATED_FILE) as f:  # type: ignore
                return json.load(f)[MOLS_VALIDITY_KEY]
        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _get_uniqueness_mols(self) -> Optional[float]:
        """Get the uniqueness of the generated molecules."""

        try:
            with open(self.generated_molecules_directory / EVALUATED_FILE) as f:  # type: ignore
                return json.load(f)[MOLS_UNIQUENESS_KEY]
        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _get_novelty_mols(self) -> Optional[float]:
        """Get the novelty of the generated molecules."""

        try:
            with open(self.generated_molecules_directory / EVALUATED_FILE) as f:  # type: ignore
                return json.load(f)[MOLS_NOVELTY_KEY]
        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _get_fcd(self) -> Optional[float]:
        """Get the FCD of the generated molecules."""

        try:
            with open(self.generated_molecules_directory / EVALUATED_FILE) as f:  # type: ignore
                return json.load(f)[MOLS_FCD_KEY]
        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _get_fcd_g(self) -> Optional[float]:
        """Get the FCD (Guacamol style)  of the generated molecules."""

        try:
            with open(self.generated_molecules_directory / EVALUATED_FILE) as f:  # type: ignore
                return json.load(f)[MOLS_FCD_GUACAMOL_KEY]
        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _read_generation_stats_file(self) -> Optional[dict[str, Any]]:
        """Read the generation file."""

        try:
            with open(
                self.generated_reaction_templates_directory / GENERATION_STATS_FILE  # type: ignore
            ) as f:
                return json.load(f)
        except (FileNotFoundError, TypeError):
            return None

    def _get_validity_rts(self) -> Optional[float]:
        """Get the validity of the generated reaction templates."""

        generation_stats = self._read_generation_stats_file()

        try:
            return generation_stats[RTS_FRAC_KEY][RTS_VALIDITY_KEY]  # type: ignore
        except (KeyError, TypeError):
            return None

    def _get_uniqueness_rts(self) -> Optional[float]:
        """Get the uniqueness of the generated reaction_templates."""

        generation_stats = self._read_generation_stats_file()

        try:
            return generation_stats[RTS_FRAC_KEY][RTS_UNIQUENESS_KEY]  # type: ignore
        except (KeyError, TypeError):
            return None

    def _get_feasibility(self) -> Optional[float]:
        """Get the feasibility of the generated reaction templates."""

        generation_stats = self._read_generation_stats_file()

        try:
            return generation_stats[RTS_FRAC_KEY][RTS_FEASIBILITY_KEY]  # type: ignore
        except (KeyError, TypeError):
            return None

    def _get_known_either(self) -> Optional[int]:
        """Get the reaction templates known from either validation or test set."""

        generation_stats = self._read_generation_stats_file()

        try:
            return generation_stats[RTS_COUNT_KEY][RTS_KNOWN_EITHER_KEY]  # type: ignore
        except (KeyError, TypeError):
            return None

    def _get_known_val(self) -> Optional[int]:
        """Get the reaction templates known from the validation set."""

        generation_stats = self._read_generation_stats_file()

        try:
            return generation_stats[RTS_COUNT_KEY][RTS_KNOWN_VAL_KEY]  # type: ignore
        except (KeyError, TypeError):
            return None

    def _get_known_test(self) -> Optional[int]:
        """Get the reaction templates known from the test set."""

        generation_stats = self._read_generation_stats_file()

        try:
            return generation_stats[RTS_COUNT_KEY][RTS_KNOWN_TEST_KEY]  # type: ignore
        except (KeyError, TypeError):
            return None

    @property
    def tokenizer_metrics(self) -> dict[str, Any]:
        """Get the tokenizer metrics."""

        return {k: v for k, v in self._metrics.items() if SCOPE_TOKENIZER in v.scope}

    @property
    def model_metrics(self) -> dict[str, Any]:
        """Get the model metrics."""

        return {k: v for k, v in self._metrics.items() if SCOPE_MODEL in v.scope}

    @property
    def training_metrics(self) -> dict[str, Any]:
        """Get the training metrics."""

        return {k: v for k, v in self._metrics.items() if SCOPE_TRAINING in v.scope}

    @property
    def wandb_metrics(self) -> dict[str, Any]:
        """Get the model metrics."""

        return {k: v for k, v in self._metrics.items() if SCOPE_WANDB in v.scope}

    @property
    def evaluation_mols_metrics(self) -> dict[str, Any]:
        """Get the evaluation molecules metrics."""

        return {k: v for k, v in self._metrics.items() if SCOPE_MOLS_EVAL in v.scope}

    @property
    def evaluation_rts_metrics(self) -> dict[str, Any]:
        """Get the evaluation reaction templates metrics."""

        return {k: v for k, v in self._metrics.items() if SCOPE_RTS_EVAL in v.scope}

    @property
    def valid(self) -> bool:
        # return self._is_valid_directory()
        return (
            self._is_model_directory()
            or self._is_generated_molecules_directory()
            or self._is_generated_reaction_templates_directory()
        )

    @property
    def has_model(self) -> bool:
        return self._is_model_directory()
        # return self.has_model is not None

    @property
    def has_generated_molecules(self) -> bool:
        return self._is_generated_molecules_directory()

    @property
    def has_generated_reaction_templates(self) -> bool:
        return self._is_generated_reaction_templates_directory()

    @property
    def has_evaluation(self) -> bool:
        return self._is_evaluated_directory()

    @property
    def model_directory(self) -> Optional[Path]:
        if self.has_model:
            return self.directory
        elif (
            (self.has_generated_molecules or self.has_generated_reaction_templates)
            and (self.directory / MODEL_LINK_TO_FILE).is_symlink()
            # That would be a nice check, but I would need to create an experiment to check this
            # (or copy the code)
            # and self._is_model_directory(
            #     (self.directory / MODEL_LINK_TO_FILE).resolve()
            # )
        ):
            return (self.directory / MODEL_LINK_TO_FILE).resolve()
        else:
            return None

    @property
    def generated_molecules_directory(self) -> Optional[Path]:
        return self.directory if self.has_generated_molecules else None

    @property
    def generated_reaction_templates_directory(self) -> Optional[Path]:
        return self.directory if self.has_generated_reaction_templates else None

    @property
    def generated_directory(self) -> Optional[Path]:
        if self.has_generated_molecules and not self.has_generated_reaction_templates:
            return self.generated_molecules_directory
        elif self.has_generated_reaction_templates and not self.has_generated_molecules:
            return self.generated_reaction_templates_directory
        elif not self.has_generated_molecules:
            return None
        else:
            raise RuntimeError(
                "This experiment contains both generated molecules and reaction templates"
            )

    def get_creation_date(
        self, formatter: str = "%Y-%m-%d %H:%M"
    ) -> tuple[datetime.datetime, str]:
        """Get the creation date of an experiment."""

        # This is platform specific, but I don't care for now
        # see also https://stackoverflow.com/questions/237079/how-do-i-get-file-creation-and-modification-date-times
        creation_date = datetime.datetime.fromtimestamp(
            os.path.getctime(self.directory)
        )
        return creation_date, format(creation_date, formatter)

    @classmethod
    def get_sort_idx(cls, metric: str) -> int:
        return cls._indices.get(metric, 999999)

    @property
    def available_metrics(self) -> tuple[str, ...]:
        available_metrics: dict[str, Any] = {}  # TODO replace with sorted set
        if self.model_directory is not None:
            if self._tokenizer_scope:
                available_metrics |= self.tokenizer_metrics
            if self._model_scope:
                available_metrics |= self.model_metrics
            if self._train_scope:
                available_metrics |= self.training_metrics
            if self._wandb_scope is not None:
                available_metrics |= self.wandb_metrics

        if self.has_generated_molecules and self._eval_scope:
            available_metrics |= self.evaluation_mols_metrics

        if self.has_generated_reaction_templates and self._eval_scope:
            available_metrics |= self.evaluation_rts_metrics

        return tuple(available_metrics.keys())

    def get_metric(
        self,
        metric: str,
        raise_error_on_none: bool = False,
        raise_error_if_not_available: bool = True,
    ) -> Optional[Metric]:
        metric_ = self._metrics.get(metric, None)
        if metric_ is None:
            if raise_error_if_not_available:
                raise AttributeError(f"Metric {metric} is not available")
            else:
                return None

        getter = self._metrics[metric].getter
        if not callable(getter):
            if raise_error_if_not_available:
                raise AttributeError(
                    f"Can not determine value of metric {metric} (no getter)"
                )
            else:
                return None

        if metric_.retrieved is True:
            return metric_

        value = getter()
        if value is None and raise_error_on_none:
            raise ValueError(f"Metric {metric} is None")

        if value is not None and not isinstance(value, self._metrics[metric].dtype):
            logger.warning(
                f"Metric {metric} has the wrong type: "
                f"expected {self._metrics[metric].dtype}, got {type(value)}"
            )

        self._metrics[metric].value = value
        self._metrics[metric].retrieved = True
        return self._metrics[metric]

    def __eq__(self, other: object) -> bool:
        return (
            self.directory == other.directory
            if isinstance(other, Experiment)
            else NotImplemented  # type: ignore
        )

    def __hash__(self) -> int:
        return hash(self.directory)

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}({self.directory})"

    def __str__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name} in {self.directory}"


def collect_experiments(
    directory: Union[str, os.PathLike],
    args: argparse.Namespace,
    wandb_metrics: Optional[WandbMetrics] = None,
) -> list[Experiment]:
    """Collect experiments from a directory and return a list of Experiments in from that directory."""

    directory = Path(directory).resolve()
    dirs = [directory]
    dirs.extend(
        [d for d in sorted(directory.rglob("*")) if d.is_dir() and not d.is_symlink()]
    )
    experiments = [
        Experiment(
            d, args.tokenizer, args.model, args.train, args.evaluate, wandb_metrics
        )
        for d in dirs
    ]
    return [e for e in experiments if e.valid]


def _get_available_metrics(
    experiments: Union[Experiment, Iterable[Experiment]]
) -> tuple[str, ...]:
    """Get all available metrics from a number of experiments."""

    if isinstance(experiments, Experiment):
        experiments = [experiments]

    available_metrics: set[str] = set()
    for e in experiments:
        available_metrics |= set(e.available_metrics)

    try:
        e = next(iter(experiments))
    except StopIteration:
        return ()

    return tuple(sorted(available_metrics, key=lambda m: e.get_sort_idx(m)))


def _build_row_from_experiment(
    experiment: Experiment,
    metrics: Iterable[str],
    report_dir_stem: bool = True,
    use_metric_formatter: bool = True,
) -> tuple[Any, ...]:
    """Build a row from an experiment for printing / saving it."""

    gen_dir: Optional[Union[str, os.PathLike]]
    if experiment.generated_directory is None:
        gen_dir = None
    elif report_dir_stem:
        gen_dir = experiment.generated_directory.stem
    else:
        gen_dir = experiment.generated_directory

    # gen_dir = (
    #     None
    #     if experiment.generated_directory is None
    #     else experiment.generated_directory.stem
    # )

    model_dir: Optional[Union[str, os.PathLike]]
    if experiment.model_directory is None:
        model_dir = None
    elif report_dir_stem:
        model_dir = experiment.model_directory.stem
    else:
        model_dir = experiment.model_directory

    # model_dir = (
    #     None if experiment.model_directory is None else experiment.model_directory.stem
    # )

    row = [gen_dir, model_dir]
    for m in metrics:
        metric = experiment.get_metric(m, raise_error_if_not_available=False)
        if metric is None or metric.value is None:
            row.append(None)
        elif use_metric_formatter and isinstance(metric.formatter, str):
            row.append(format(metric.value, metric.formatter))
        else:
            row.append(format(metric.value, ""))

    return tuple(row)


def print_experiments(
    experiments: Union[Experiment, Iterable[Experiment]],
    title: str = "",
) -> None:
    """Print metrics from a number of experiments."""

    if isinstance(experiments, Experiment):
        experiments = [experiments]

    if any(not isinstance(e, Experiment) for e in experiments):
        raise TypeError("experiments must be an (Iterable of type) Experiment")

    logger.debug("Collecting available metrics...")
    available_metrics = _get_available_metrics(experiments)

    logger.debug("Building table...")
    table = Table(title=title)
    table.add_column(
        "Generated directory",
        justify="left",
        header_style="bold",
        no_wrap=False,
        overflow="fold",
    )
    table.add_column(
        "Model directory",
        justify="left",
        header_style="bold",
        no_wrap=False,
        overflow="fold",
    )
    for m in available_metrics:
        table.add_column(m, justify="right", header_style="bold", min_width=len(m))

    logger.debug("Adding rows...")
    # with Progress(
    #     SpinnerColumn(),
    #     TextColumn("[progress.description]{task.description}"),
    #     BarColumn(),
    #     MofNCompleteColumn(),
    #     TaskProgressColumn(),
    #     TimeRemainingColumn(elapsed_when_finished=True),
    #     refresh_per_second=5,
    # ) as progress:
    #     task = progress.add_task(
    #         "Collecting metrics...",
    #         total=len(experiments),
    #     )

    for e in experiments:
        if e.valid:
            row = _build_row_from_experiment(e, available_metrics)
            table.add_row(*row)
            # progress.advance(task)

    logger.debug("Printing table...")
    console = Console()
    console.print(table)


def save_experiments(
    experiments: Union[Experiment, Iterable[Experiment]],
    file_path: Union[str, os.PathLike],
    file_format: str = "ALL",
) -> None:
    def convert_to_str(value: Any) -> str:
        return str(value)

    if isinstance(experiments, Experiment):
        experiments = [experiments]

    if any(not isinstance(e, Experiment) for e in experiments):
        raise TypeError("experiments must be an (Iterable of type) Experiment")

    file_path = Path(file_path).resolve()
    logger.debug(f"Saving to {file_path}...")

    if file_format not in {"ALL", "CSV", "JSON", "MD"}:
        raise ValueError("file_format must be one of 'ALL', 'CSV', 'MD', 'JSON'")

    logger.debug("Collecting available metrics...")
    available_metrics = _get_available_metrics(experiments)

    logger.debug("Adding rows...")
    rows = []
    # with Progress(
    #     SpinnerColumn(),
    #     TextColumn("[progress.description]{task.description}"),
    #     BarColumn(),
    #     MofNCompleteColumn(),
    #     TaskProgressColumn(),
    #     TimeRemainingColumn(elapsed_when_finished=True),
    #     refresh_per_second=5,
    # ) as progress:
    #     task = progress.add_task(
    #         "Collecting metrics...",
    #         total=len(experiments),
    #     )

    for e in experiments:
        row = _build_row_from_experiment(
            e, available_metrics, report_dir_stem=False, use_metric_formatter=False
        )
        rows.append(row)
        # progress.advance(task)

    # TODO Use column_names but at this point I deal with metrics as strings only (not Metric objects)
    logger.debug("Building table...")
    df = pd.DataFrame(
        rows, columns=["Generated directory", "Model directory", *available_metrics]
    )

    logger.debug("Saving file(s)...")
    if file_format.upper() in {"CSV", "ALL"}:
        df.to_csv(file_path.with_suffix(".csv"), index=False)

    if file_format.upper() in {"JSON", "ALL"}:
        # with open(file_path.with_suffix(".json"), "w") as f:
        #     json.dump(df.values.tolist(), f, cls=PathEncoder, indent=4)
        df.to_json(
            file_path.with_suffix(".json"),
            orient="index",
            double_precision=3,
            indent=2,
            default_handler=convert_to_str,
        )

    if file_format.upper() in {"MD", "ALL"}:
        df.to_markdown(file_path.with_suffix(".md"), index=False)


@logger.catch
def main() -> None:
    """Collect metrics from a number of experiments and save them to a single file."""

    # Prepare argument parser
    parser = argparse.ArgumentParser(
        description="Collect metrics from a number of files and save them to a single file."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="the directory to collect metrics from.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=False,
        # default=DEFAULT_OUTPUT_FILE_NAME,
        # help="file path to save the metrics to, default: '%(default)s'.",
        help="file path to save the metrics to, default: files with the directory name in local directory.",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="if specified, collect all metrics.",
    )
    parser.add_argument(
        "--evaluate",
        "-e",
        action="store_true",
        help="if specified, collect evaluation metrics.",
    )
    parser.add_argument(
        "--tokenizer",
        "-k",
        action="store_true",
        help="if specified, collect tokenizer metrics.",
    )
    parser.add_argument(
        "--model",
        "-m",
        action="store_true",
        help="if specified, collect model metrics.",
    )
    parser.add_argument(
        "--train",
        "-t",
        action="store_true",
        help="if specified, collect train metrics.",
    )
    parser.add_argument(
        "--wandb",
        "-w",
        action="store_true",
        help="if specified, collect wandb metrics.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="log_level",
        action="append_const",
        const=-1,
        help="increase verbosity from default.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        dest="log_level",
        action="append_const",
        const=1,
        help="decrease verbosity from default.",
    )

    args = parser.parse_args()

    # Configure logging
    log_level: int = determine_log_level(args.log_level)
    configure_logging(log_level)

    logger.heading("Collecting experiment metrics...")  # type: ignore

    # Prepare and check (global) variables
    directory_path = Path(args.directory).resolve()
    if args.output is None:
        output_file_path = directory_path / "metrics"  # directory_path.stem
    else:
        output_file_path = Path(args.output).resolve()

    if not directory_path.is_dir():
        raise ValueError(f"Directory '{directory_path}' does not exist")

    logger.debug(f"Directory path: {directory_path}")
    logger.debug(f"Output file path: {output_file_path}")
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Start timer
    with Timer(
        name="collect_metrics",
        text=lambda secs: f"Metrics collected in {format_timespan(secs)}",
        logger=logger.info,
    ):
        if args.all:
            args.tokenizer = True
            args.model = True
            args.train = True
            args.wandb = True
            args.evaluate = True

        if args.wandb:
            logger.info("Collecting wandb metrics...")
            wandb_metrics = WandbMetrics()
        else:
            wandb_metrics = None

        logger.info(f"Collecting metrics from {directory_path}...")
        experiments = collect_experiments(directory_path, args, wandb_metrics)
        if len(experiments) == 0:
            logger.warning("No experiments found")
            return
        logger.info(f"Found {len(experiments)} experiment(s)")
        logger.info("Printing experiment results...")
        print_experiments(experiments, title="Experiment Results")

        logger.info(
            f"Saving experiments to {output_file_path} as CSV, JSON and Markdown"
        )
        save_experiments(experiments, f"{output_file_path}")


if __name__ == "__main__":
    main()
