# coding=utf-8
# train.py
"""
Auto-Regressive Molecule and Reaction Template Generator
Causal language modeling (CLM) with a transformer decoder model
Author: Stephan Holzgruber
Student ID: K08608294
"""
import argparse

# Parts of this file are based on the following huggingface example:
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
import json
import math
import os
import sys
import tempfile
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from random import randint
from typing import Any, Final, Optional, Union

# Most of Hugging Face has poor type hints, trying to avoid mypy errors
import datasets
import evaluate  # type: ignore
import torch
import transformers  # type: ignore
import wandb
from datasets import Dataset, DatasetDict, Features, Value, load_dataset  # type: ignore
from loguru import logger
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    GPT2TokenizerFast,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)

# from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint  # type: ignore
from transformers.utils import check_min_version  # type: ignore
from transformers.utils.versions import require_version  # type: ignore

from molreactgen.generate import create_and_save_generation_config
from molreactgen.helpers import configure_logging, guess_project_root_dir
from molreactgen.tokenizer import (
    DATASET_COLUMN_NAME,
    enclose_function,
    get_merges,
    get_modified_vocab,
    get_tokenizer,
    tokenize_function,
)

CONFIG_OVERWRITE_ARGS_FLAG: Final[str] = "--config_file"
PROJECT_ROOT_DIR: Final = guess_project_root_dir()
DEFAULT_CONFIG_DIR: Final[Path] = PROJECT_ROOT_DIR / "src/molreactgen/conf"
WANDB_PROJECT_NAME: Final[str] = "MolReactGen"


###############################################################################
# Initial checks and setup                                                    #
###############################################################################

# Check Hugging Face versions (only)
hint = "To fix: Install package requirements with poetry install"
check_min_version("4.24.0")
require_version("datasets>=1.8.0", hint)
require_version("evaluate>=0.3.0", hint)

# The HF datacollator seems to first call tokenizer.encode() and then tokenizer.pad()
# It would be faster to call tokenizer.__call__() which does both in one go
# This environment variable suppresses this warning
# Based on
# https://discuss.huggingface.co/t/get-using-the-call-method-is-faster-warning-with-datacollatorwithpadding/23924
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

# Configure logging to wandb
# os.environ["WANDB_DISABLED"] = "false"
# os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME
# os.environ["WANDB_LOG_MODEL"] = "end"
# os.environ["WANDB_WATCH"] = "gradients"


###############################################################################
# Prepare config                                                              #
###############################################################################

# Model related
MODEL_CONFIG_CLASSES: Final = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES: Final = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# TODO Add default values to metadata
# Dataclasses for command line arguments
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    # Theoretically we can pass any model type for causal language modeling
    # However, this is only tested with GPT2
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_positions=128,n_embd=132,n_layer=12,n_head=12"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    def __post_init__(self) -> None:
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory with the dataset file(s)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    validation_split_percentage: Optional[int] = field(
        default=10,
        metadata={
            "help": "The percentage of the data set used as a validation set (random split)."
        },
    )
    test_split_percentage: Optional[int] = field(
        default=10,
        metadata={
            "help": "The percentage of the data set used as a test set (random split)."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pre_tokenizer: Optional[str] = field(
        default="ATOM",
        metadata={
            "help": "The pre-tokenizer  to use; one of 'char', 'atom', 'smarts'."
        },
    )
    algorithm: Optional[str] = field(
        default="WORDLEVEL",
        metadata={
            "help": "The tokenizer algorithm to use; one of "
            "'wordlevel', 'bpe', 'wordpiece', 'unigram'."
        },
    )
    vocab_size: Optional[int] = field(
        default=0,
        metadata={
            "help": "The vocabulary size for 'bpe', 'wordpiece' and 'unigram' tokenization algorithms "
            "(not relevant for 'wordlevel')."
        },
    )
    vocab_min_frequency: Optional[int] = field(
        default=1,
        metadata={
            "help": "The minimum frequency a pair should have in order to be merged. "
            "Only relevant for the 'bpe' and 'wordpiece' algorithms."
        },
    )
    map_tokenizers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to train a new tokenizer and map the vocabulary into a pre-trained tokenizer."
        },
    )
    # TODO implement map_strategy, currently only linear is supported
    map_strategy: Optional[str] = field(
        default="linear",
        metadata={
            "help": "The strategy to use when mapping the vocabulary; one of 'linear', 'random'. "
            "Only relevant when 'map_tokenizers' is set to True. Currently defaults to 'linear'."
        },
    )

    def __post_init__(self) -> None:
        if self.dataset_name is not None:
            raise NotImplementedError("Datasets from the HF hub are not supported yet.")

        if self.dataset_name is None and self.dataset_dir is None:
            raise ValueError("Need either a dataset name or a dataset directory.")
        elif self.dataset_dir is not None:
            self.dataset_dir = (
                (Path(PROJECT_ROOT_DIR) / self.dataset_dir).resolve().as_posix()
            )
            for file in Path(self.dataset_dir).glob("*"):
                if (
                    file.suffix.lower() not in (".csv", ".json", ".txt")
                    and file.name != ".DS_Store"  # macOS specific
                ):
                    raise ValueError(
                        f"Dataset directory {self.dataset_dir} contains file {file.name} "
                        f"with unsupported extension {file.suffix}"
                    )
        if self.map_tokenizers:
            if (
                self.pre_tokenizer is None
                or self.algorithm is None
                or str(self.pre_tokenizer.upper()) != "CHAR"
                or str(self.algorithm.upper()) != "BPE"
            ):
                warnings.warn(
                    "Mapping tokenizers is currently only supported for character-level tokenization with BPE. "
                    "Set the pre-tokenizer and algorithm to 'char' and 'bpe' respectively."
                )
            self.pre_tokenizer = "CHAR"
            self.algorithm = "BPE"
            if self.map_strategy is not None and self.map_strategy.upper() != "LINEAR":
                warnings.warn(
                    "Only the 'linear' mapping strategy is currently supported. Set it to 'linear'."
                )
                self.map_strategy = "linear"


@dataclass
class AdditionalArguments:
    """
    Additional miscellaneous training arguments which can not be passed to the Trainer directly.
    """

    random_seed: bool = field(
        default=False,
        metadata={
            "help": "Whether a random seed should be configured; overwrites a fixed seed value. Defaults to False."
        },
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of evaluation calls with worsened metrics after which the training stops."
        },
    )
    early_stopping_threshold: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "The threshold for the metric to be satisfy an early stopping condition."
        },
    )
    unique_output_subdir: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to create a unique sub-directory in the output directory or not."
        },
    )


###############################################################################
# Helper to parse arguments (command line or .yaml                            #
###############################################################################


def get_training_config() -> tuple[argparse.Namespace, ...]:
    # Extract the arguments which are not overwritten via command line flags/.args files
    config_file_parser = argparse.ArgumentParser()
    config_file_parser.add_argument(
        CONFIG_OVERWRITE_ARGS_FLAG, type=str, action="append"
    )
    _, remaining_args = config_file_parser.parse_known_args()

    # Determine .args and .yaml files in arguments
    args_file_names = {
        arg
        for arg in remaining_args
        if arg.lower().endswith(".args") and not arg.startswith("--")
    }
    yaml_file_names = {
        arg
        for arg in remaining_args
        if arg.lower().endswith(".yaml") and not arg.startswith("--")
    }

    # Check if arguments make sense
    if len(args_file_names) and len(yaml_file_names):
        raise ValueError(
            "Both .args and .yaml config files specified, can only use one configuration file (format)"
        )

    if len(args_file_names) > 1 or len(yaml_file_names) > 1:
        raise ValueError(
            "Can only create configuration from a single configuration file"
        )

    if len(yaml_file_names) == 1 and len(sys.argv) > 2:
        raise ValueError(
            "Configuration with a .yaml file does not allow for additional command line arguments"
        )

    # Parse the arguments, taking care of the different config scenarios
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            AdditionalArguments,
        ),
        description="Train a model on a dataset",
    )

    # we have a .yaml file as a single argument
    if len(sys.argv) == 2 and sys.argv[1].lower().endswith(".yaml"):
        # print(f"Config via .yaml file {sys.argv[1]}")
        yaml_file_path = Path(sys.argv[1]).resolve()
        if not yaml_file_path.is_file():
            raise FileNotFoundError(f"Config file {yaml_file_path} not found")
        model_args, data_args, training_args, additional_args = parser.parse_yaml_file(
            yaml_file_path.as_posix(),
        )

    # we have a (main) .args file, plus potentially overwrites
    elif len(args_file_names) == 1 and list(args_file_names)[0].lower().endswith(
        ".args"
    ):
        main_args_file_name = list(args_file_names)[0]
        main_args_file_path = Path(main_args_file_name).resolve()
        # print(f"Config via .args file {main_args_file_path}")
        if not main_args_file_path.is_file():
            raise FileNotFoundError(f"Config file {main_args_file_path} not found")
        (
            model_args,
            data_args,
            training_args,
            additional_args,
            unused_args,
        ) = parser.parse_args_into_dataclasses(
            args_filename=main_args_file_path.as_posix(),
            return_remaining_strings=True,
            args_file_flag=CONFIG_OVERWRITE_ARGS_FLAG,
        )
        unused_args.remove(main_args_file_name)  # must use original string
        if unused_args:
            raise ValueError(f"Unknown configuration arguments: {unused_args}")

    # no config given, look for default config
    elif len(args_file_names) == 0:
        main_args_file_path = (
            Path(DEFAULT_CONFIG_DIR) / Path(sys.argv[0]).with_suffix(".args").name
        )
        # print(f"Config via default .args file {main_args_file_path}")
        if not main_args_file_path.is_file():
            raise FileNotFoundError(f"Config file {main_args_file_path} not found")
        (
            model_args,
            data_args,
            training_args,
            additional_args,
            unused_args,
        ) = parser.parse_args_into_dataclasses(
            args_filename=main_args_file_path,
            return_remaining_strings=True,
            args_file_flag=CONFIG_OVERWRITE_ARGS_FLAG,
        )
        if unused_args:
            raise ValueError(f"Unknown configuration arguments: {unused_args}")

    # Something we have not considered (hopefully not)
    else:
        raise RuntimeError(
            "Can not determine valid configuration option (should not happen, check code)"
        )

    return model_args, data_args, training_args, additional_args


###############################################################################
# Functions for data manipulation                                             #
###############################################################################


# TODO implement this function
def load_raw_dataset_from_hub(  # type: ignore  # noqa  # Remove once implemented
    dataset_name: str,  # noqa
    *,
    dataset_config_name: Optional[str] = None,  # noqa
    cache_dir: Optional[str] = None,  # noqa
    use_auth_token: bool = False,  # noqa
) -> DatasetDict:
    ...


def load_raw_dataset_from_dir(
    data_dir: str,
    *,
    validation_split_percentage: int = 10,
    test_split_percentage: int = 10,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> DatasetDict:
    features = Features({"0": Value(dtype="string")})  # type: ignore
    # mypy complains about the type of dataset, but it's correct
    # see also https://discuss.huggingface.co/t/...
    # ...mypy-and-datasetdict-error-incompatible-return-value-type-got-union-datasetdict-dataset-...
    # ...iterabledatasetdict-iterabledataset-expected-datasetdict/17177
    dataset: DatasetDict = load_dataset(
        data_dir, features=features, cache_dir=cache_dir, header=None
    )  # type: ignore
    if "validation" not in dataset.keys():
        if "train" not in dataset.keys():
            raise RuntimeError("Can't load dataset, no train split found")
        # Create random split
        dataset_shuffled: Dataset = dataset["train"].shuffle(seed=seed)
        val_len: int = int(len(dataset_shuffled) * validation_split_percentage / 100.0)
        test_len: int = int(len(dataset_shuffled) * test_split_percentage / 100.0)
        train_len: int = len(dataset_shuffled) - val_len - test_len
        logger.debug(
            f"Create random split with lengths {train_len}, {val_len}, {test_len}"
        )
        assert len(dataset_shuffled) == train_len + val_len + test_len
        dataset["train"] = dataset_shuffled.select(range(train_len))
        dataset["validation"] = dataset_shuffled.select(
            range(train_len, train_len + val_len)
        )
        dataset["test"] = dataset_shuffled.select(
            range(train_len + val_len, train_len + val_len + test_len)
        )

    dataset = dataset.rename_column("0", DATASET_COLUMN_NAME)
    return dataset


###############################################################################
# Main training loop                                                          #
###############################################################################


# @logger.catch
def main() -> None:
    # -----------------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------------

    # Parse arguments from command line or yaml configuration file
    model_args, data_args, training_args, additional_args = get_training_config()

    # Configure logging
    log_level = (
        training_args.get_process_log_level()
    )  # This returns a log level depending on main process yes/no etc.
    # TODO make address configurable
    configure_logging(log_level, address=("logs3.papertrailapp.com", 32501))
    datasets.utils.logging.set_verbosity(log_level)
    datasets.utils.logging.disable_progress_bar()  # type: ignore
    evaluate.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log a small summary on each process
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set seed in random, numpy, torch (not only for training, but also for dataset creation, if applicable)
    if additional_args.random_seed:
        training_args.seed = randint(0, 2**32 - 1)
    set_seed(training_args.seed)

    logger.debug(f"Data arguments\n{data_args}")
    logger.debug(f"Model arguments\n{model_args}")
    logger.debug(f"Training arguments\n{training_args}")
    logger.debug(f"Additional arguments\n{additional_args}")

    # -----------------------------------------------------------------------------
    # Load datasets
    # -----------------------------------------------------------------------------

    raw_datasets: DatasetDict
    if data_args.dataset_name is not None:
        logger.info(f"Loading dataset {data_args.dataset_name} from hub...")
        raw_datasets = load_raw_dataset_from_hub(
            data_args.dataset_name,
            dataset_config_name=data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=model_args.use_auth_token,
        )
    else:
        logger.info(f"Loading dataset from directory {data_args.dataset_dir}...")
        raw_datasets = load_raw_dataset_from_dir(
            data_args.dataset_dir,
            validation_split_percentage=data_args.validation_split_percentage,
            test_split_percentage=data_args.test_split_percentage,
            cache_dir=model_args.cache_dir,
            seed=training_args.seed,
        )
    logger.debug(raw_datasets)

    # -----------------------------------------------------------------------------
    # Configure model - Step 1 (before tokenizer config)
    # -----------------------------------------------------------------------------

    # Define base model config
    logger.info("Configuring model...")
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.model_name_or_path:
        logger.info(f"Loading configuration from {model_args.model_name_or_path}")
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )

    elif model_args.config_name:
        logger.info(f"Loading configuration from {model_args.config_name}")
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)

    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.info(f"Training model type {model_args.model_type} from scratch")
        # logger.debug(f"Old config {config}")
        if model_args.config_overrides is not None:
            logger.info("Overriding model config...")
            logger.debug(f"{model_args.config_overrides}")
            config_overrides = model_args.config_overrides.replace(" ", "")
            config.update_from_string(config_overrides)
            logger.debug(f"New config:\n{config}")

    # -----------------------------------------------------------------------------
    # Configure / Train tokenizer
    # -----------------------------------------------------------------------------

    logger.info("Configuring tokenizer(s)...")
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": True,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    need_tokenizer_from_scratch: bool
    if data_args.map_tokenizers:
        need_tokenizer_from_scratch = True
        logger.info("Loading pre-trained tokenizer...")
        tokenizer_pretrained: PreTrainedTokenizerFast
        if model_args.tokenizer_name:
            logger.info(f"Loading tokenizer from {model_args.tokenizer_name}")
            tokenizer_pretrained = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name, **tokenizer_kwargs
            )
        elif model_args.model_name_or_path:
            logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
            tokenizer_pretrained = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path, **tokenizer_kwargs
            )
        else:
            raise ValueError(
                "Mapping tokenizers requires a pre-trained model or tokenizer"
            )

    else:
        tokenizer: PreTrainedTokenizerFast
        if model_args.tokenizer_name:
            need_tokenizer_from_scratch = False
            logger.info(f"Loading tokenizer from {model_args.tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name, **tokenizer_kwargs
            )
        elif model_args.model_name_or_path:
            need_tokenizer_from_scratch = False
            logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path, **tokenizer_kwargs
            )
        else:
            need_tokenizer_from_scratch = True
            pass

    if need_tokenizer_from_scratch:
        logger.info("Building & training tokenizer from scratch...")
        logger.debug(
            f"Pre-tokenizer: {data_args.pre_tokenizer}, "
            f"algorithm: {data_args.algorithm}, "
            f"vocabulary size: {data_args.vocab_size}, "
            f"min frequency: {data_args.vocab_min_frequency}, "
            f"max length: {config.n_positions}"
        )
        data_iterator = raw_datasets["train"][DATASET_COLUMN_NAME]
        tokenizer_from_scratch: PreTrainedTokenizerFast = get_tokenizer(
            pre_tokenizer=data_args.pre_tokenizer,
            algorithm=data_args.algorithm,
            train_source=data_iterator,
            vocab_size=data_args.vocab_size,
            min_frequency=data_args.vocab_min_frequency,
            model_max_length=config.n_positions,
        )

    if data_args.map_tokenizers:
        assert need_tokenizer_from_scratch
        # Get the token frequencies by tokenizing the training data
        all_tokens: list[str] = []
        for item in data_iterator:
            tokens = tokenizer_from_scratch.tokenize(item)
            all_tokens.extend(tokens)
        token_counter = Counter(all_tokens)

        # Add tokens that did not make into the token_counter, i.e. don't occur for whatever reason
        counter_set = {t[0] for t in token_counter.most_common(None)}
        vocab_set = set(tokenizer_from_scratch.get_vocab().keys())
        delta_set = (
            vocab_set - counter_set - set(tokenizer_from_scratch.all_special_tokens)
        )
        delta_counter = {k: 0 for k in delta_set}
        token_counter.update(delta_counter)

        # Make "room" for the tokens with zero frequency and the special tokens of the pre-trained tokenizer
        end_idx = (
            len(tokenizer_pretrained)
            - len(tokenizer_pretrained.all_special_tokens)
            - len(delta_counter)
        )

        # Add BOS and EOS frequencies
        item_count = len(data_iterator)
        token_counter.update(
            {
                tokenizer_from_scratch.bos_token: item_count,
                tokenizer_from_scratch.eos_token: item_count,
            }
        )

        # Map vocab from the new tokenizer to the pre-trained tokenizer
        # Assuming byte-level encoding here
        # Start at index 256, i.e. 0..255 used for byte-level encoding
        vocab_modified = get_modified_vocab(
            tokenizer_pretrained,
            token_counter,
            mapping_strategy=data_args.map_strategy,
            start_idx=256,
            end_idx=end_idx,
        )

        # Build a new gpt2 tokenizer with the modified vocabulary
        with tempfile.TemporaryDirectory(), tempfile.NamedTemporaryFile(
            mode="w", suffix=".json"
        ) as f_vocab_modified, tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt"
        ) as f_merge_modified:
            # Write vocabulary file
            json.dump(vocab_modified, f_vocab_modified, ensure_ascii=False)

            # Create and write merges file
            merges = get_merges(tokenizer_from_scratch)
            f_merge_modified.writelines("\n".join(merges))
            # f_merge_modified.write("#This is empty")

            # Flush files before reading them; otherwise an error is raised
            f_vocab_modified.flush()
            f_merge_modified.flush()

            # Load a new gpt2 tokenizer with those two files
            tokenizer = GPT2TokenizerFast(
                vocab_file=f_vocab_modified.name,
                merges_file=f_merge_modified.name,
                model_max_length=config.n_positions,
                # padding_side="right",
                # truncation_side="left",
                pad_token=tokenizer_pretrained.eos_token,
            )

    elif not data_args.map_tokenizers and need_tokenizer_from_scratch:
        tokenizer = tokenizer_from_scratch

    elif not data_args.map_tokenizers and not need_tokenizer_from_scratch:
        pass

    else:
        raise RuntimeError("This should not happen, check code.")

    logger.info("Tokenizing datasets...")
    with training_args.main_process_first(desc="Tokenize dataset (map)"):
        if data_args.map_tokenizers:
            enclosed_datasets = raw_datasets.map(
                partial(
                    enclose_function,
                    start_token=tokenizer_from_scratch.bos_token,
                    end_token=tokenizer_from_scratch.eos_token,
                ),
                batched=True,
                num_proc=4,
                load_from_cache_file=True,
                desc="Enclose datasets",
            )
            raw_datasets = enclosed_datasets

        tokenized_datasets: DatasetDict = raw_datasets.map(
            partial(tokenize_function, tokenizer=tokenizer),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=DATASET_COLUMN_NAME,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenize datasets",
        )

    # -----------------------------------------------------------------------------
    # Configure model - Step 2 (after tokenizer config)
    # -----------------------------------------------------------------------------

    # Create causal language model
    if model_args.model_name_or_path:
        logger.info(
            f"Loading pre-trained model from {model_args.model_name_or_path}..."
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Update model with information from tokenizer
        logger.info("Updating model configuration with tokenizer information...")
        logger.debug(
            f"Vocabulary size: {tokenizer.vocab_size}, "
            f"BOS token id: {tokenizer.bos_token_id}, "
            f"EOS token id: {tokenizer.eos_token_id}, "
            f"PAD token: {tokenizer.pad_token}"
        )
        model_overrides: dict[str, Any] = {
            "vocab_size": tokenizer.vocab_size,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
        }
        config.update(model_overrides)
        logger.info("Creating model...")
        model = AutoModelForCausalLM.from_config(config)
    n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    logger.info(f"Model size: {n_params/2**20:.2f}M params")

    # Reshape the embeddings if necessary
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        logger.error("Reshaping of model's embedding necessary (should NOT happen)")
        logger.error(
            f"Tokenizer length: {len(tokenizer)}, embedding size: {embedding_size}"
        )
        # model.resize_token_embeddings(len(tokenizer))
        raise RuntimeError("Length of tokenizer is greater than embedding size")

    # -----------------------------------------------------------------------------
    # Load last checkpoint
    # -----------------------------------------------------------------------------

    logger.info("Configuring training...")

    # Detect last checkpoint
    last_checkpoint: Optional[str] = None
    training_args.output_dir = (
        Path(PROJECT_ROOT_DIR) / training_args.output_dir
    ).as_posix()
    if additional_args.unique_output_subdir:
        training_args.output_dir = (
            Path(training_args.output_dir)
            / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_experiment"
        ).as_posix()

    if (
        Path(training_args.output_dir).is_dir()
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        # get_last_checkpoint() not documented, see code here:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if (  # no checkpoint found but other files in directory
            last_checkpoint is None
            and len(list(Path(training_args.output_dir).iterdir())) > 0
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (  # checkpoint found but training not configured to resume from this checkpoint
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # -----------------------------------------------------------------------------
    # Configure training data
    # -----------------------------------------------------------------------------

    # Define training/validation sets and limit to max length if necessary
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset: Dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples: int = min(
                len(train_dataset), data_args.max_train_samples
            )
            logger.info(f"Limit training samples to {max_train_samples}")
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset: Dataset = tokenized_datasets["validation"]
        if data_args.max_val_samples is not None:
            max_eval_samples: int = min(len(eval_dataset), data_args.max_val_samples)
            logger.info(f"Limit validation samples to {max_eval_samples}")
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Configure data collator
    # Pad to multiples of 8 for NVIDIA tensor cores, see
    # https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, pad_to_multiple_of=8, mlm=False
    )

    # -----------------------------------------------------------------------------
    # Define helper functions for metric computation
    # -----------------------------------------------------------------------------

    logger.info("Setting up metric computation...")
    metric = evaluate.load("accuracy")

    def preprocess_logits_for_metrics(
        logits: Union[tuple[torch.Tensor, ...], torch.Tensor],
        labels: torch.Tensor,  # noqa
    ) -> torch.Tensor:
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]

        return logits.argmax(dim=-1)

    def compute_metrics(
        eval_preds: tuple[torch.Tensor, torch.Tensor]
    ) -> Optional[dict[str, float]]:
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics, but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)

        # metric.compute() returns Optional[dict[str, float]] but is not annotated as such
        return metric.compute(predictions=preds, references=labels)  # type: ignore

    # -----------------------------------------------------------------------------
    # Do final configuration steps and initialize trainer
    # -----------------------------------------------------------------------------

    callbacks: list[TrainerCallback] = []

    # Configure early stopping
    if additional_args.early_stopping_patience is not None:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=additional_args.early_stopping_patience,
            early_stopping_threshold=additional_args.early_stopping_threshold,
        )
        callbacks.append(early_stopping_callback)

    # Configure lr scheduler (done via argument)
    # We could build a more advanced lr scheduler "by hand" instead of passing an argument, see
    # https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/trainer#transformers.Trainer.create_scheduler
    # Cosine with restarts **has** to be built manually, due to a hugging face issue, see
    # https://github.com/huggingface/transformers/issues/20552

    # Configure wandb
    if training_args.report_to is not None and "wandb" in training_args.report_to:
        os.environ["WANDB_DISABLED"] = "false"
        os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME
        os.environ["WANDB_LOG_MODEL"] = "end"
        os.environ["WANDB_WATCH"] = "gradients"
        # os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = "true"

        _ = wandb.init(
            project=WANDB_PROJECT_NAME,
            job_type="train",
            anonymous="allow",
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # TODO configure optimizer manually, HF deprecated the support to configure it via arguments
    # Will be an issues with transformers version 5

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
        callbacks=callbacks,
    )

    # -----------------------------------------------------------------------------
    # Train / Fine-tune the model
    # -----------------------------------------------------------------------------

    # Training (train/validation set)
    if training_args.do_train:
        checkpoint: Optional[Union[bool, str]] = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        if checkpoint is not None and isinstance(checkpoint, bool):
            logger.info(f"Resuming training from checkpoint {checkpoint}")
        elif checkpoint is not None and isinstance(checkpoint, str):
            logger.info(f"Resuming training from checkpoint {training_args.output_dir}")
        logger.heading("Start training...")  # type: ignore
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        logger.info("Saving model...")
        trainer.save_model()

        # Save training metrics
        logger.info("Saving training metrics...")
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation (test set)
    if training_args.do_predict:
        if "test" not in tokenized_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = tokenized_datasets["test"]

        # There is also Trainer.predict() but it returns more than needed here
        logger.info("Evaluating trained model on test set...")
        metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")

        # Save test metrics
        logger.info("Saving evaluation metrics...")
        try:
            perplexity = math.exp(metrics["test_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    if wandb.run is not None:
        wandb.finish()

    # -----------------------------------------------------------------------------
    # (Prepare to) push the model/dataset to the hub
    # -----------------------------------------------------------------------------

    # Create default generation config
    _ = create_and_save_generation_config(
        Path(trainer.args.output_dir), split_into_chunks=False
    )

    # Prepare metadata for the model/dataset card
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-generation",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        logger.info("Pushing model/dataset to the hub...")
        trainer.push_to_hub(**kwargs)
    else:
        logger.info("Creating model/dataset card template...")
        trainer.create_model_card(**kwargs)


# For xla_spawn (TPUs)
def _mp_fn(index: int) -> None:  # noqa
    main()


if __name__ == "__main__":
    main()
