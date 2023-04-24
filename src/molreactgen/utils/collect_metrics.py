# coding=utf-8
# src/molreactgen/utils/collect_metrics.py
"""Collect experiment metrics from a number of files and save them to a single file."""

import argparse
import datetime
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Optional, Union

# import pandas as pd  # type: ignore
from codetiming import Timer
from humanfriendly import format_timespan  # type: ignore
from loguru import logger
from rich.console import Console
from rich.table import Table

import wandb
from molreactgen.helpers import configure_logging, determine_log_level
from molreactgen.tokenizer import REGEX_PATTERN_ATOM, REGEX_PATTERN_SMARTS
from molreactgen.train import WANDB_PROJECT_NAME

# Global variables, defaults
DEFAULT_OUTPUT_FILE_NAME: Final[str] = "./metrics.csv"
# TODO add global variables for all strings here and replace them in the code
# LEARNING_RATE_STR: Final[str] = "learning_rate"


@dataclass
class Metric:
    column_name: str
    scope: str  # TODO might change to Enum
    dtype: type
    value: Any
    formatter: str = ""
    _ref: str = ""
    _idx: int = 0
    _getter: Optional[Callable] = None

    def __hash__(self) -> int:
        return hash(self.column_name)


class Experiment:
    # _getters: dict[str, Callable[[Path], Any]] = {
    _getters: dict[str, str] = {
        "pre_tokenizer": "_get_pre_tokenizer",
        "algorithm": "_get_algorithm",
        "vocab_size": "_get_vocab_size",
        "num_epochs": "_get_num_epochs",
        "batch_size": "_get_batch_size",
        "lr": "_get_lr",
        "wandb_run_id": "_get_wandb_run_id",
        "wandb_run_name": "_get_wandb_run_name",
        "train_loss": "_get_train_loss",
        "val_loss": "_get_val_loss",
        "val_acc": "_get_val_acc",
        "test_loss": "_get_test_loss",
        "test_acc": "_get_test_acc",
        "test_perplexity": "_get_test_perplexity",
        "validity_mols": "_get_validity_mols",
        "uniqueness_mols": "_get_uniqueness_mols",
        "novelty_mols": "_get_novelty",
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

    def __init__(self, directory: Union[str, os.PathLike]) -> None:
        self.directory: Path = Path(directory).resolve()
        self._metrics: dict[str, Metric] = {
            "pre_tokenizer": Metric(
                column_name="pre_tokenizer",
                scope="tokenizer",
                dtype=str,
                value=None,
            ),
            "algorithm": Metric(
                column_name="tokenization_algorithm",
                scope="tokenizer",
                dtype=str,
                value=None,
            ),
            "vocab_size": Metric(
                column_name="vocab_size",
                scope="tokenizer",
                dtype=int,
                value=None,
            ),
            "num_epochs": Metric(
                column_name="number_of_epochs",
                scope="training",
                dtype=int,
                value=None,
            ),
            "batch_size": Metric(
                column_name="batch_size",
                scope="training",
                dtype=int,
                value=None,
            ),
            "lr": Metric(
                column_name="learning_rate",
                scope="training",
                dtype=float,
                value=None,
                formatter=".4f",
            ),
            "wandb_run_id": Metric(
                column_name="wandb_run_id",
                scope="wandb",
                dtype=str,
                value=None,
            ),
            "wandb_run_name": Metric(
                column_name="wandb_run_name",
                scope="wandb",
                dtype=str,
                value=None,
            ),
            "train_loss": Metric(
                column_name="training_loss",
                scope="training",
                dtype=float,
                value=None,
                formatter=".3f",
            ),
            "val_loss": Metric(
                column_name="validation_loss",
                scope="training",
                dtype=float,
                value=None,
                formatter=".3f",
            ),
            "val_acc": Metric(
                column_name="validation_accuracy",
                scope="training",
                dtype=float,
                value=None,
                formatter=".3f",
            ),
            "test_loss": Metric(
                column_name="test_loss",
                scope="training",
                dtype=float,
                value=None,
                formatter=".3f",
            ),
            "test_acc": Metric(
                column_name="test_accuracy",
                scope="training",
                dtype=float,
                value=None,
                formatter=".3f",
            ),
            "test_perplexity": Metric(
                column_name="test_perplexity",
                scope="training",
                dtype=float,
                value=None,
                formatter=".3f",
            ),
            "validity_mols": Metric(
                column_name="validity",
                scope="evaluation_mols",
                dtype=float,
                value=None,
                formatter=".3f",
            ),
            "uniqueness_mols": Metric(
                column_name="uniqueness",
                scope="evaluation_mols",
                dtype=float,
                value=None,
                formatter=".3f",
            ),
            "novelty_mols": Metric(
                column_name="novelty",
                scope="evaluation_mols",
                dtype=float,
                value=None,
                formatter=".3f",
            ),
            "fcd_mols": Metric(
                column_name="fcd",
                scope="evaluation_mols",
                dtype=float,
                value=None,
                formatter=".3f",
            ),
            "fcd_g_mols": Metric(
                column_name="fcd_guacamol",
                scope="evaluation_mols",
                dtype=float,
                value=None,
                formatter=".3f",
            ),
            "validity_rts": Metric(
                column_name="validity",
                scope="evaluation_rts",
                dtype=float,
                value=None,
                formatter=".3f",
            ),
            "uniqueness_rts": Metric(
                column_name="uniqueness",
                scope="evaluation_rts",
                dtype=float,
                value=None,
                formatter=".3f",
            ),
            "feasibility_rts": Metric(
                column_name="feasibility",
                scope="evaluation_rts",
                dtype=float,
                value=None,
                formatter=".3f",
            ),
            "known_either_rts": Metric(
                column_name="known_from_validation_or_test",
                scope="evaluation_rts",
                dtype=int,
                value=None,
            ),
            "known_val_rts": Metric(
                column_name="known_from_validation",
                scope="evaluation_rts",
                dtype=int,
                value=None,
            ),
            "known_test_rts": Metric(
                column_name="known_from_test",
                scope="evaluation_rts",
                dtype=int,
                value=None,
            ),
        }

        self._init_metrics()

        # TODO add the following metrics
        # Step 1:
        # wandb: sweep id (if applicable) (map via output directory)
        #
        # Step 2:
        # generation reaction templates: validity, uniqueness, feasibility, known x 3
        # model: type, layers, heads, hidden_dim
        # dataset: name, ...

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
            # self._metrics[getter]._idx = idx
            self._metrics[getter]._ref = getter
            self._metrics[getter]._getter = getattr(self, self._getters[getter])

    @staticmethod
    def _is_valid_directory(path: Path) -> bool:
        """Check if a directory is a valid directory."""

        if not isinstance(path, Path) or (not path.is_dir()) or path.is_symlink():
            return False

        return (
            not path.stem.startswith("checkpoint-")
            or not (path / "scheduler.pt").is_file()
        )

    @staticmethod
    def _is_model_directory(path: Path) -> bool:
        """Check if a directory is a model directory."""

        return (
            Experiment._is_valid_directory(path)
            and (path / "pytorch_model.bin").is_file()
        )

    @staticmethod
    def _is_generated_molecules_directory(path: Path) -> bool:
        """Check if a directory holds generated molecules."""

        return (
            Experiment._is_valid_directory(path)
            and (path / "generated_smiles.csv").is_file()
        )

    @staticmethod
    def _is_generated_reaction_templates_directory(path: Path) -> bool:
        """Check if a directory holds generated reaction templates."""

        return (
            Experiment._is_valid_directory(path)
            and (path / "generated_reaction_templates.csv").is_file()
        )

    @staticmethod
    def _is_evaluated_directory(path: Path) -> bool:
        """Check if a directory has been evaluated."""

        return (
            Experiment._is_generated_molecules_directory(path)
            and (path / "evaluation.json").is_file()
        ) or (
            Experiment._is_generated_reaction_templates_directory(path)
            and (path / "generation_stats.json").is_file()
        )

    @staticmethod
    def _get_pre_tokenizer(path: Path) -> Optional[str]:
        """Get the pre-tokenizer."""

        try:
            with open(path / "tokenizer.json") as f:
                tokenizer = json.load(f)
        except FileNotFoundError:
            return None

        try:
            if tokenizer["pre_tokenizer"] is None:
                return "Char"
        except KeyError:
            return None

        try:
            if (
                tokenizer["pre_tokenizer"]["type"].upper() == "SPLIT"
                and tokenizer["pre_tokenizer"]["pattern"]["Regex"] == REGEX_PATTERN_ATOM
            ):
                return "Atom"
        except KeyError:
            return None

        try:
            if (
                tokenizer["pre_tokenizer"]["type"].upper() == "SPLIT"
                and tokenizer["pre_tokenizer"]["pattern"]["Regex"]
                == REGEX_PATTERN_SMARTS
            ):
                return "Smarts"
        except KeyError:
            return None

        return None

    @staticmethod
    def _get_algorithm(path: Path) -> Optional[str]:
        """Get the tokenization algorithm."""

        try:
            with open(path / "tokenizer.json") as f:
                return json.load(f)["model"]["type"]
        except (FileNotFoundError, KeyError):
            return None

    @staticmethod
    def _get_vocab_size(path: Path) -> Optional[int]:
        """Get the vocabulary size (without special tokens)."""

        try:
            with open(path / "tokenizer.json") as f:
                tokenizer = json.load(f)
        except FileNotFoundError:
            return None

        try:
            return len(tokenizer["model"]["vocab"]) - len(tokenizer["added_tokens"])
        except KeyError:
            return None

    @staticmethod
    def _get_num_epochs(path: Path) -> Optional[int]:
        """Get the number of epochs."""

        try:
            with open(path / "all_results.json") as f:
                return int(json.load(f)["epoch"])
        except (FileNotFoundError, KeyError):
            return None

    @staticmethod
    def _get_batch_size(path: Path) -> Optional[int]:
        """Get the batch size."""

        try:
            content = Path(path / "README.md").read_text()
        except FileNotFoundError:
            return None

        pattern = re.compile(r"(^\s*)-(\s*train_batch_size:\s*)(\d*)", re.MULTILINE)
        match = pattern.search(content)
        return None if match is None else int(match[3])

    @staticmethod
    def _get_lr(path: Path) -> Optional[int]:
        """Get the learning rate."""

        try:
            content = Path(path / "README.md").read_text()
        except FileNotFoundError:
            return None

        pattern = re.compile(r"(^\s*)-(\s*learning_rate:\s*)(\d*\.\d*)", re.MULTILINE)
        match = pattern.search(content)
        return None if match is None else float(match[3])

    @staticmethod
    def _get_wandb_run_id(path: Path) -> Optional[str]:
        """Get the wandb run id."""

        api = wandb.Api()
        entity = api.default_entity
        runs = api.runs("/".join((entity, WANDB_PROJECT_NAME)))
        return next(
            (
                run.id
                for run in runs
                if Path(run.config.get("output_dir", "")).stem == path.stem
            ),
            None,
        )

    @staticmethod
    def _get_wandb_run_name(path: Path) -> Optional[str]:
        """Get the wandb run name."""

        # TODO very inefficient, but would need to cache some information
        #  or get rid of one function == one metric
        api = wandb.Api()
        entity = api.default_entity
        runs = api.runs("/".join((entity, WANDB_PROJECT_NAME)))
        return next(
            (
                run.name
                for run in runs
                if Path(run.config.get("output_dir", "")).stem == path.stem
            ),
            None,
        )

    @staticmethod
    def _get_train_loss(path: Path) -> Optional[float]:
        """Get the last training loss."""

        try:
            with open(path / "all_results.json") as f:
                return json.load(f)["train_loss"]
        except (FileNotFoundError, KeyError):
            return None

    @staticmethod
    def _get_val_loss(path: Path) -> Optional[float]:
        """Get the validation loss."""

        try:
            with open(path / "trainer_state.json") as f:
                return json.load(f)["best_metric"]
        except (FileNotFoundError, KeyError):
            return None

    @staticmethod
    def _get_val_acc(path: Path) -> Optional[float]:
        """Get the validation accuracy."""

        try:
            with open(path / "trainer_state.json") as f:
                log_history = json.load(f)["log_history"]
                return next(
                    (
                        log["eval_accuracy"]
                        for log in reversed(log_history)
                        if "eval_accuracy" in log
                    )
                    # log_history[-1]["eval_accuracy"],
                )
        except (FileNotFoundError, KeyError):
            return None

    @staticmethod
    def _get_test_loss(path: Path) -> Optional[float]:
        """Get the test loss."""

        try:
            with open(path / "all_results.json") as f:
                return json.load(f)["test_loss"]
        except (FileNotFoundError, KeyError):
            return None

    @staticmethod
    def _get_test_acc(path: Path) -> Optional[float]:
        """Get the test accuracy."""

        try:
            with open(path / "all_results.json") as f:
                return json.load(f)["test_accuracy"]
        except (FileNotFoundError, KeyError):
            return None

    @staticmethod
    def _get_test_perplexity(path: Path) -> Optional[float]:
        """Get the test loss."""

        try:
            with open(path / "all_results.json") as f:
                return json.load(f)["perplexity"]  # TODO change key to test_perplexity
        except (FileNotFoundError, KeyError):
            return None

    @staticmethod
    def _get_validity_mols(path: Path) -> Optional[float]:
        """Get the validity of the generated items."""

        try:
            with open(path / "evaluation.json") as f:
                return json.load(f)["Validity"]
        except (FileNotFoundError, KeyError):
            return None

    @staticmethod
    def _get_uniqueness_mols(path: Path) -> Optional[float]:
        """Get the uniqueness of the generated items."""

        try:
            with open(path / "evaluation.json") as f:
                return json.load(f)["Uniqueness"]
        except (FileNotFoundError, KeyError):
            return None

    @staticmethod
    def _get_novelty(path: Path) -> Optional[float]:
        """Get the novelty of the generated items."""

        try:
            with open(path / "evaluation.json") as f:
                return json.load(f)["Novelty"]
        except (FileNotFoundError, KeyError):
            return None

    @staticmethod
    def _get_fcd(path: Path) -> Optional[float]:
        """Get the FCD of the generated items."""

        try:
            with open(path / "evaluation.json") as f:
                return json.load(f)["FCD"]
        except (FileNotFoundError, KeyError):
            return None

    @staticmethod
    def _get_fcd_g(path: Path) -> Optional[float]:
        """Get the FCD (Guacamol style)  of the generated items."""

        try:
            with open(path / "evaluation.json") as f:
                return json.load(f)["FCD GuacaMol"]
        except (FileNotFoundError, KeyError):
            return None

    @staticmethod
    def _read_generation_file(path: Path) -> Optional[dict[str, Any]]:
        """Read the generation file."""

        try:
            with open(path / "generation_stats.json") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    @staticmethod
    def _get_validity_rts(path: Path) -> Optional[float]:
        """Get the validity of the generated reaction templates."""

        getter = getattr(Experiment, "_read_generation_file", None)
        if callable(getter):
            generation_stats = getter(path)

            try:
                return generation_stats["valid"]
            except KeyError:
                return None

        return None

    @staticmethod
    def _get_uniqueness_rts(path: Path) -> Optional[float]:
        """Get the uniqueness of the generated reaction_templates."""

        getter = getattr(Experiment, "_read_generation_file", None)
        if callable(getter):
            generation_stats = getter(path)

            try:
                return generation_stats["unique"]
            except KeyError:
                return None

        return None

    @staticmethod
    def _get_feasibility(path: Path) -> Optional[float]:
        """Get the feasibility of the generated reaction templates."""

        getter = getattr(Experiment, "_read_generation_file", None)
        if callable(getter):
            generation_stats = getter(path)

            try:
                return generation_stats["feasible"]
            except KeyError:
                return None

        return None

    @staticmethod
    def _get_known_either(path: Path) -> Optional[int]:
        """Get the reaction templates known from either validation or test set."""

        getter = getattr(Experiment, "_read_generation_file", None)
        if callable(getter):
            generation_stats = getter(path)

            try:
                return generation_stats["known"]
            except KeyError:
                return None

        return None

    @staticmethod
    def _get_known_val(path: Path) -> Optional[int]:
        """Get the reaction templates known from the validation set."""
        getter = getattr(Experiment, "_read_generation_file", None)
        if callable(getter):
            generation_stats = getter(path)

            try:
                return generation_stats["known_from_valid_set"]
            except KeyError:
                return None

        return None

    @staticmethod
    def _get_known_test(path: Path) -> Optional[int]:
        """Get the reaction templates known from the test set."""
        getter = getattr(Experiment, "_read_generation_file", None)
        if callable(getter):
            generation_stats = getter(path)

            try:
                return generation_stats["known_from_test_set"]
            except KeyError:
                return None

        return None

    @property
    def tokenizer_metrics(self) -> Optional[dict[str, Any]]:
        """Get the tokenizer metrics."""

        return {k: v for k, v in self._metrics.items() if v.scope == "tokenizer"}

    @property
    def training_metrics(self) -> Optional[dict[str, Any]]:
        """Get the training metrics."""

        return {k: v for k, v in self._metrics.items() if v.scope == "training"}

    @property
    def model_metrics(self) -> Optional[dict[str, Any]]:
        """Get the model metrics."""

        return {k: v for k, v in self._metrics.items() if v.scope == "model"}

    @property
    def wandb_metrics(self) -> Optional[dict[str, Any]]:
        """Get the model metrics."""

        return {k: v for k, v in self._metrics.items() if v.scope == "wandb"}

    @property
    def evaluation_mols_metrics(self) -> Optional[dict[str, Any]]:
        """Get the evaluation molecules metrics."""

        return {k: v for k, v in self._metrics.items() if v.scope == "evaluation_mols"}

    @property
    def evaluation_rts_metrics(self) -> Optional[dict[str, Any]]:
        """Get the evaluation reaction templates metrics."""

        return {k: v for k, v in self._metrics.items() if v.scope == "evaluation_rts"}

    @property
    def valid(self) -> bool:
        return self._is_valid_directory(self.directory)

    @property
    def has_model(self) -> bool:
        return self._is_model_directory(self.directory)

    @property
    def has_generated_molecules(self) -> bool:
        return self._is_generated_molecules_directory(self.directory)

    @property
    def has_generated_reaction_templates(self) -> bool:
        return self._is_generated_reaction_templates_directory(self.directory)

    @property
    def has_evaluation(self) -> bool:
        return self._is_evaluated_directory(self.directory)

    @property
    def model_directory(self) -> Optional[Path]:
        if self.has_model:
            return self.directory
        elif (
            (self.has_generated_molecules or self.has_generated_reaction_templates)
            and (self.directory / "link_to_model").is_symlink()
            and self._is_model_directory((self.directory / "link_to_model").resolve())
        ):
            return (self.directory / "link_to_model").resolve()
        else:
            # raise AttributeError("This experiment does not contain a model")
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
        elif (
            not self.has_generated_molecules
            and not self.has_generated_reaction_templates
        ):
            return None
        else:
            raise RuntimeError(
                "This experiment contains both generated molecules and reaction templates"
            )

    @staticmethod
    def get_creation_date(
        path: Path, formatter: str = "%Y-%m-%d %H:%M"
    ) -> tuple[datetime.datetime, str]:
        """Get the creation date of an experiment."""

        # This is platform specific, but I don't care for now
        # see also https://stackoverflow.com/questions/237079/how-do-i-get-file-creation-and-modification-date-times
        path = Path(path).resolve()
        creation_date = datetime.datetime.fromtimestamp(os.path.getctime(path))
        return creation_date, format(creation_date, formatter)

    @classmethod
    def get_sort_idx(cls, metric: str) -> int:
        return cls._indices.get(metric, 999999)

    @property
    def available_metrics(self) -> tuple[str]:
        available_metrics: dict[str, Any] = {}
        if self.has_model:
            available_metrics = (
                self.tokenizer_metrics
                | self.model_metrics
                | self.training_metrics
                | self.wandb_metrics
            )

        if self.has_generated_molecules:
            available_metrics |= self.evaluation_mols_metrics

        if self.has_generated_reaction_templates:
            available_metrics |= self.evaluation_rts_metrics

        return tuple(available_metrics.keys())

    def get_metric(
        self,
        metric: str,
        raise_error_on_none: bool = False,
        raise_error_if_not_available: bool = True,
    ) -> Optional[Metric]:
        # getter = getattr(self, self._getters[metric])
        metric_ = self._metrics.get(metric, None)
        if metric_ is None:
            if raise_error_if_not_available:
                raise AttributeError(f"Metric {metric} is not available")
            else:
                return None

        getter = self._metrics[metric]._getter
        # getter = self._metrics[.get("metric].getter
        if not callable(getter):
            # if metric not in self._metrics.keys() or not callable(getter):
            if raise_error_if_not_available:
                raise AttributeError(
                    f"Can not determine value of metric {metric} (no getter)"
                )
            else:
                return None

        # try:
        if self.has_model and (
            metric in self.tokenizer_metrics
            or metric in self.model_metrics
            or metric in self.training_metrics
            or metric in self.wandb_metrics
        ):
            directory = self.model_directory
        elif self.has_generated_molecules and metric in self.evaluation_mols_metrics:
            directory = self.generated_molecules_directory
        elif (
            self.has_generated_reaction_templates
            and metric in self.evaluation_rts_metrics
        ):
            directory = self.generated_reaction_templates_directory
        else:
            raise AttributeError(
                f"Can not determine value of metric {metric} (no directory)"
            )

        value = getter(directory)
        if value is None and raise_error_on_none:
            raise ValueError(f"Metric {metric} is None")

        if value is not None and not isinstance(value, self._metrics[metric].dtype):
            logger.warning(
                f"Metric {metric} has the wrong type: "
                f"expected {self._metrics[metric].dtype}, got {type(value)}"
            )

        # except (NameError, TypeError):
        #     logger.warning(
        #         f"Could not get metric {metric} for experiment in {self.directory}"
        #     )
        #     value = None

        self._metrics[metric].value = value
        return self._metrics[metric]

    def __eq__(self, other: object) -> bool:
        return (
            self.directory == other.directory
            if isinstance(other, Experiment)
            else NotImplemented
        )

    def __hash__(self) -> int:
        return hash(self.directory)

    def __repr__(self) -> str:
        return f"Experiment({self.directory})"

    def __str__(self) -> str:
        return f"Experiment in {self.directory}"


def collect_experiments(directory: Union[str, os.PathLike]) -> list[Experiment]:
    """Collect experiments from a directory and return a list of Experiments in from that directory.

    Args:
        directory: the directory to collect experiments from.

    Returns:
        a list of experiments.
    """

    directory = Path(directory).resolve()
    dirs = [
        d for d in sorted(directory.rglob("*")) if d.is_dir() and not d.is_symlink()
    ]
    experiments = [Experiment(d) for d in dirs]
    return [e for e in experiments if e.valid]


def collect_metrics(
    experiments: Union[Experiment, Iterable[Experiment]]
) -> dict[Experiment, list[Metric]]:
    """Collect metrics from a number of experiments.

    Args:
        experiments: the experiment(s) to collect metrics from.

    Returns:
        a dict with the collected metrics.
    """

    if isinstance(experiments, Experiment):
        experiments = [experiments]

    if any(not isinstance(e, Experiment) for e in experiments):
        raise TypeError("experiments must be an (Iterable of type) Experiment")

    metrics: dict[Experiment, list[Metric]] = {}
    for exp in experiments:
        metrics[exp]: list[Metric] = []
        none_metrics: list[str] = []
        print(f"Looking at Experiment in {exp.directory}")
        if exp.has_generated_molecules:
            print(
                f"Generated molecules dir: {exp.generated_molecules_directory.stem}, "
                f"created at {exp.get_creation_date(exp.generated_molecules_directory)[1]}"
            )
        if exp.has_generated_reaction_templates:
            print(
                f"Generated reaction templates dir: {exp.generated_reaction_templates_directory.stem}, "
                f"created at {exp.get_creation_date(exp.generated_reaction_templates_directory)[1]}"
            )
        if exp.has_model:
            print(
                f"Model dir: {exp.model_directory.stem}, "
                f"created at {exp.get_creation_date(exp.model_directory)[1]}"
            )

        for m in exp.available_metrics:
            metric = exp.get_metric(m)
            if metric.value is None:
                none_metrics.append(m)
            else:
                print(f"{m}: {format(metric.value, metric.formatter)}, ", end="")
            metrics[exp].append(metric)
        print()

        if none_metrics:
            print(f"Metrics with None value: {none_metrics}")

        print("----------------------------------------")
        # TODO print number if directories with metrics

    return metrics


def print_metrics(
    experiment_metrics: dict[Experiment, list[Metric]],
    title: str = "Experiment Results",
) -> None:
    """Print metrics from a number of experiments."""

    available_metrics = {
        m._ref for e in experiment_metrics for m in experiment_metrics[e]
    }
    e = next(iter(experiment_metrics))
    available_metrics = sorted(available_metrics, key=lambda m: e.get_sort_idx(m))

    table = Table(title=title)
    table.add_column("Generated directory", justify="left", header_style="bold")
    table.add_column("Model directory", justify="left", header_style="bold")
    for m in available_metrics:
        table.add_column(m, justify="right", header_style="bold", min_width=len(m))

    for e in experiment_metrics:
        gen_dir = None if e.generated_directory is None else e.generated_directory.stem
        model_dir = None if e.model_directory is None else e.model_directory.stem
        row = [gen_dir, model_dir]
        for m in available_metrics:
            metric = e.get_metric(m, raise_error_if_not_available=False)
            row.append(None) if metric.value is None else row.append(
                format(metric.value, metric.formatter)
            )
        table.add_row(*row)

    console = Console()
    console.print(table)


@logger.catch
def main() -> None:
    """Collect metrics from a number of files and save them to a single file."""

    # TODO allow for selection of metrics / scopes (wandb is expensive)

    # Prepare argument parser
    parser = argparse.ArgumentParser(
        description="Collect metrics from a number of files and save them to a single file."
    )
    parser.add_argument(  # TODO allow for multiple directories
        "directory",
        type=Path,
        help="the directory to collect metrics from.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        default=DEFAULT_OUTPUT_FILE_NAME,
        help="file path to save the metrics to, default: '%(default)s'.",
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

    logger.heading("Collecting metrics...")  # type: ignore

    # Prepare and check (global) variables
    directory_path = Path(args.directory).resolve()
    output_file_path = Path(args.output).resolve()

    logger.debug(f"Directory path: {directory_path}")
    logger.debug(f"Output file path: {output_file_path}")
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Start timer
    with Timer(
        name="collect_metrics",
        text=lambda secs: f"Metrics collected in {format_timespan(secs)}",
        logger=logger.info,
    ):
        experiments = collect_experiments(directory_path)
        metrics = collect_metrics(experiments)
        print_metrics(metrics)

        # TODO output metrics, with rich table? (R&D)

        logger.info(f"Writing metrics to {output_file_path}")
        # TODO write metrics to csv file


if __name__ == "__main__":
    main()
