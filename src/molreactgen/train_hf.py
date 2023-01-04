# coding=utf-8
# Parts of this file are based on the following huggingface example:
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""
Auto-Regressive Molecule and Reaction Template Generator
Causal language modeling (CLM) with a transformer decoder model
Author: Stephan Holzgruber
Student ID: K08608294
"""

import logging
import math
import os
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Optional

import datasets
import evaluate
import transformers
from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from tokenizers import (  # SentencePieceBPETokenizer,; SentencePieceUnigramTokenizer,
    Regex,
    Tokenizer,
    decoders,
)
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece

# from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import (
    BpeTrainer,
    UnigramTrainer,
    WordLevelTrainer,
    WordPieceTrainer,
)
from transformers import (  # default_data_collator,
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)

# from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# import wandb

###############################################################################
# Initial checks and setup                                                    #
###############################################################################

# Check versions
# The HF example requires 4.25.0.dev, but the official release as of 2022-11-29 is 4.24.0
# Will try to use the official release first, if it fails, use dev version
# check_min_version("4.25.0.dev0")
check_min_version("4.24.0")
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
require_version(
    "datasets>=1.8.0",
    "To fix: see https://github.com/huggingface/transformers/blob/main/examples/"
    "pytorch/language-modeling/requirements.txt",
)

# The HF datacollator seems to first call tokenizer.encode() and then tokenizer.pad()
# It would be faster to call tokenizer.__call__() which does both in one go
# This environment variable suppresses the warning
# Based on
# https://discuss.huggingface.co/t/get-using-the-call-method-is-faster-warning-with-datacollatorwithpadding/23924
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

# Configure logging to wandb
os.environ["WANDB_DISABLED"] = "false"
os.environ["WANDB_PROJECT"] = "MolGenHF"
os.environ["WANDB_LOG_MODEL"] = "true"

logger = logging.getLogger(__name__)

###############################################################################
# Prepare config                                                              #
###############################################################################

# Data related
DATASET_COLUMN_NAME = "items"
HUB_MODEL_ID = "hogru/molgen-hf"

# Tokenizer related
BOS_TOKEN: str = "^"
EOS_TOKEN: str = "_"
PAD_TOKEN: str = " "
UNK_TOKEN: str = "§"
ADD_TOKEN: str = "°"  # Not sure if needed at all; might be used to map special model tokens to like CLS, SEP etc.

# SMILES_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
# SMARTS_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|Au?|Fe?|Zn?|Mg?|Li?|Ga?|As?|Ru?|Eu?|Ta?|Ga?|Yb?|Dy?|N|O|S|P|F|I|H|K|U|W|V|Y|b|c|n|o|s|i|p|D\d+|[a-z]|\(|\)|\.|=|\#|-|\+|\;|\%|\\|\/|:|~|@|\?|>>?|\*|\$|\d+)"""
# REGEX_PATTERN = SMILES_REGEX_PATTERN

REGEX_INPUT = {
    # everything within brackets becomes a single token; might play with that, alternative below
    "bracket": r"\[[^\]]+]",
    # "bracket": r"\[|\]",
    "atom_a": r"A[cglmrstu]?",
    "atom_b": r"B[aeihkr]?",
    "atom_c": r"C[adeflmorsu]?",
    "atom_d": r"D[bsy]?",
    "atom_e": r"E[rsu]?",
    "atom_f": r"F[emr]?",
    "atom_g": r"G[ade]?",
    "atom_h": r"H[efgos]?",
    "atom_i": r"I[nr]?",
    "atom_k": r"K[r]?",
    "atom_l": r"L[aru]?",
    "atom_m": r"M[dgnot]?",
    "atom_n": r"N[abdeiop]?",
    "atom_o": r"Os?",
    "atom_p": r"P[abdmortu]?",
    "atom_r": r"R[aefghnu]?",
    "atom_s": r"S[bcdegimnr]?",
    "atom_t": r"T[abcehilm]?",
    "atom_u": r"U",
    "atom_v": r"V",
    "atom_w": r"W",
    "atom_x": r"Xe",
    "atom_y": r"Yb?",
    "atom_z": r"Z[nr]?",
    "aromatic": r"as|b|c|n|o|p|se?",
    "parenthesis": r"\(|\)",
    # "@" is also a bond type (any ring), order of RegEx matters for "@@" token
    "chiral": r"@@?",
    # charge "-" is also a (single aliphatic) bond, order of RegEx matters
    "charge": r"\+\d+|\++|-\d+|-+",
    # "#" (triple bond) amended with "\d{1}" to allow for SMARTS syntax
    # "$" (quadruple bond) amended with "\(?" to allow for recursive SMARTS syntax
    # ":" (aromatic bond) with digit also used for atom mapping
    "bond": r"\.|-|=|#\d{1}|\$\(?|:[0-9]*|~|@|\/\??|\\\??",
    "ring": r"\%[0-9]{2}",
    "digit": r"[0-9]",
    "additional": r"\*|>>?|D\d{1}|H\d{1}|h\d{1}|R\d{1}|r\d{1}|v\d{1}|X\d{1}|x\d{1}|#\d{1}",
    "logical": r"!|&|,|;",
    # "recursive": r"\$\(",  # redundant
}

# the replace() is just a safety net against double "|" in the RegEx
REGEX_PATTERN_SMARTS = "|".join(REGEX_INPUT.values()).replace("||", "|")
REGEX_PATTERN_ATOM = REGEX_PATTERN_SMARTS.replace(r"\[[^\]]+]", r"\[|\]")
MIN_VOCAB_SIZE_UNIGRAM = 100

# Model related
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    # use_fast_tokenizer: bool = field(
    #     default=True,
    #     metadata={
    #         "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
    #     },
    # )
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

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    # train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    # validation_file: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    # )
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
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    # block_size: Optional[int] = field(
    #     default=None,
    #     metadata={
    #         "help": (
    #             "Optional input sequence length after tokenization. "
    #             "The training dataset will be truncated in block of this size for training. "
    #             "Default to the model max input length for single sentence inputs (take into account special tokens)."
    #         )
    #     },
    # )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=10,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    test_split_percentage: Optional[int] = field(
        default=10,
        metadata={
            "help": "The percentage of the train set used as test set in case there's no test split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    # keep_linebreaks: bool = field(
    #     default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    # )
    pre_tokenizer: Optional[str] = field(
        default="SMARTS",
        metadata={
            "help": "The pre-tokenizer  to use; one of " "char, atom, smarts."
        },
    )

    algorithm: Optional[str] = field(
        default="WORDLEVEL",
        metadata={
            "help": "The tokenizer algorithm to use; one of "
            "wordlevel, bpe, wordpiece, unigram."
        },
    )
    vocab_min_frequency: Optional[int] = field(
        default=1,
        metadata={
            "help": "The minimum frequency a pair should have in order to be merged."
        },
    )
    vocab_size: Optional[int] = field(
        default=0,
        metadata={
            "help": "The vocabulary size for BPE and UNIGRAM tokenization algorithms, "
            "irrelevant for CHAR, WORDLEVEL and SENTENCEPIECE_BPE."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.dataset_dir is None:
            raise ValueError(
                "Need either a dataset name or a dataset directory with training (and validation) file(s)."
            )
        else:
            for file in Path(self.dataset_dir).glob("*"):
                if (
                    file.suffix.lower() not in (".csv", ".json", ".txt")
                    and file.name != ".DS_Store"
                ):
                    raise ValueError(
                        f"Dataset directory {self.dataset_dir} contains file {file.name} "
                        f"with unsupported extension {file.suffix}"
                    )


###############################################################################
# Functions for data manipulation                                             #
###############################################################################


# TODO implement and type check
def load_raw_dataset_from_hub(
    dataset_name, dataset_config_name, cache_dir, use_auth_token
):
    ...


def load_raw_dataset_from_dir(
    data_dir: str,
    *,
    validation_split_percentage: int = 10,
    test_split_percentage: int = 10,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> DatasetDict:
    features = Features({"0": Value(dtype="string")})
    dataset: DatasetDict = load_dataset(
        data_dir, features=features, cache_dir=cache_dir, header=None
    )
    if "validation" not in dataset.keys():
        if "train" not in dataset.keys():
            raise RuntimeError("Can't load dataset, no train split found")
        dataset_shuffled: Dataset = dataset["train"].shuffle(seed=seed)
        val_len = int(
            len(dataset_shuffled) * validation_split_percentage / 100.0
        )
        test_len = int(len(dataset_shuffled) * test_split_percentage / 100.0)
        train_len = len(dataset_shuffled) - val_len - test_len
        assert len(dataset_shuffled) == train_len + val_len + test_len
        dataset["train"] = dataset_shuffled.select(range(train_len))
        dataset["validation"] = dataset_shuffled.select(
            range(train_len, train_len + val_len)
        )
        dataset["test"] = dataset_shuffled.select(
            range(train_len + val_len, train_len + val_len + test_len)
        )
    #     dataset["validation"] = load_dataset(
    #         data_dir,
    #         features=features,
    #         split=f"train[:{validation_split_percentage}%]",
    #         cache_dir=cache_dir,
    #         header=None,
    #     )
    #     dataset["train"] = load_dataset(
    #         data_dir,
    #         features=features,
    #         split=f"train[{validation_split_percentage}%:]",
    #         cache_dir=cache_dir,
    #         header=None,
    #     )

    dataset = dataset.rename_column("0", DATASET_COLUMN_NAME)
    return dataset


###############################################################################
# Tokenization-related functions                                              #
###############################################################################


def token_in_regex(token: str, regex: str):
    token = str(token)
    regex = str(regex)
    regex_pattern = re.compile(regex)
    found = regex_pattern.findall(token)
    if len(found) > 0:
        return True
    else:
        return False


def get_tokenizer(
    pre_tokenizer: str,
    algorithm: str,
    train_source: Sequence[str],
    *,
    vocab_size: int = 0,
    model_max_length: int = 1024,
    min_frequency: int = 1,
    bos_token: str = BOS_TOKEN,
    eos_token: str = EOS_TOKEN,
    pad_token: str = PAD_TOKEN,
    unk_token: str = UNK_TOKEN,
    regex_pattern: Optional[str] = None,
    byte_level: bool = False,
    # save_path: str = "./tokenizers/",
) -> PreTrainedTokenizerFast:

    pre_tokenizer = str(pre_tokenizer).upper()
    algorithm = str(algorithm).upper()
    vocab_size = max(
        0, int(vocab_size)
    )  # Zero works with all algorithms except Unigram (runs "infinitely")
    model_max_length = int(model_max_length)
    min_frequency = max(1, int(min_frequency))
    byte_level = bool(byte_level)
    try:
        add_token = str(ADD_TOKEN)
    except (NameError, ValueError):
        add_token = "°"

    special_tokens = [bos_token, eos_token, pad_token, unk_token, add_token]
    vocab_size = vocab_size + len(special_tokens)

    # if pre_tokenizer in ("ATOM", "SMARTS") and algorithm in (
    #     "BPE",
    #     "WORDPIECE",
    #     "UNIGRAM",
    # ):
    #     logger.warning(
    #         f"Combination of pre-tokenizer {pre_tokenizer} and algorithm {algorithm} not supported. "
    #         f"Using pre-tokenizer CHAR instead."
    #     )
    #     pre_tokenizer = "CHAR"

    if regex_pattern is None:
        if pre_tokenizer == "CHAR":
            regex_pattern = ""
        elif pre_tokenizer == "ATOM":
            regex_pattern = REGEX_PATTERN_ATOM
        elif pre_tokenizer == "SMARTS":
            regex_pattern = REGEX_PATTERN_SMARTS
        else:
            raise ValueError(
                f"Pre-tokenizer {pre_tokenizer} not supported. "
                f"Choose from CHAR, ATOM, SMARTS."
            )

    if pre_tokenizer in ("ATOM", "SMARTS"):
        for token in special_tokens:
            if token_in_regex(token, regex_pattern):
                raise ValueError(
                    f"Special token '{token}' invalid, can be parsed by '{pre_tokenizer}' regular expression"
                )

    regex_pattern = Regex(regex_pattern)

    # elif tokenizer == "SMARTS":
    #     for token in special_tokens:
    #         if token_in_regex(token, smarts_regex_pattern):
    #             raise ValueError(
    #                 f"Special token '{token}' invalid, can be parsed by '{tokenizer}' regular expression"
    #             )
    #     regex_pattern = Regex(smarts_regex_pattern)
    #
    # else:
    #     raise ValueError(f"Unknown tokenizer '{tokenizer}'")

    tokenizer: Tokenizer

    if algorithm == "WORDLEVEL":
        if min_frequency > 1:
            logger.warning(
                f"Min frequency {min_frequency} is not supported for {algorithm} "
                f"tokenizer algorithm, setting min frequency to 1."
            )
            min_frequency = 1
        tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
        tokenizer.pre_tokenizer = Split(
            pattern=regex_pattern, behavior="isolated", invert=False
        )
        trainer = WordLevelTrainer(
            special_tokens=special_tokens,
            min_frequency=min_frequency,
            show_progress=True,
        )
        tokenizer.train_from_iterator(train_source, trainer=trainer)

    elif algorithm == "BPE":
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        # tokenizer.pre_tokenizer = Split(
        #     pattern=regex_pattern, behavior="contiguous", invert=False
        # )
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=min_frequency,
            show_progress=True,
        )
        tokenizer.train_from_iterator(train_source, trainer=trainer)

    elif algorithm == "WORDPIECE":
        tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=min_frequency,
            show_progress=True,
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##", cleanup=False)
        tokenizer.train_from_iterator(train_source, trainer=trainer)

    elif algorithm == "UNIGRAM":
        old_vocab_size = vocab_size
        vocab_size = max(MIN_VOCAB_SIZE_UNIGRAM, vocab_size)
        if vocab_size > old_vocab_size:
            logger.warning(
                f"Vocab size deemed too small for Unigram, increasing from {old_vocab_size} to {vocab_size}"
            )
        tokenizer = Tokenizer(Unigram())
        # tokenizer.pre_tokenizer = Split(
        #     pattern=regex_pattern, behavior="isolated", invert=False
        # )
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            unk_token=unk_token,
            show_progress=True,
        )
        tokenizer.train_from_iterator(train_source, trainer=trainer)

    # SentencePiece is poorly documented on HuggingFace
    # see code at https://github.com/huggingface/tokenizers/tree/main/bindings/python/py_src/tokenizers/implementations
    # elif algorithm == "SENTENCEPIECE_BPE":
    #     tokenizer = SentencePieceBPETokenizer(
    #         unk_token=unk_token, add_prefix_space=False
    #     )
    #     # SentencePieceBPETokenizer sets the normalizer to NFKC by default
    #     # Not needed, but doesn't hurt; can't set to None
    #     # tokenizer.normalizer = None
    #     # tokenizer.pre_tokenizer = Split(
    #     #     pattern=regex_pattern, behavior="isolated", invert=False
    #     # )
    #     tokenizer.train_from_iterator(
    #         train_source,
    #         vocab_size=vocab_size,
    #         min_frequency=min_frequency,
    #         show_progress=True,
    #         # limit_alphabet=1000,  # TODO R&D, make configurable, default=1000
    #         special_tokens=special_tokens,
    #     )
    #
    # elif algorithm == "SENTENCEPIECE_UNIGRAM":
    #     old_vocab_size = vocab_size
    #     vocab_size = max(MIN_VOCAB_SIZE_UNIGRAM, vocab_size)
    #     if vocab_size > old_vocab_size:
    #         logger.warning(
    #             f"Vocab size deemed too small for unigram tokenizer, increasing from {old_vocab_size} to {vocab_size}"
    #         )
    #     tokenizer = SentencePieceUnigramTokenizer(add_prefix_space=False)
    #     # SentencePieceUnigramTokenizers sets the normalizer to [Nmt, NFKC, Replace(Regex(" {2,}"), " ")] by default
    #     # TODO R&D what this does
    #     # tokenizer.normalizer = None
    #     # tokenizer.pre_tokenizer = Split(
    #     #     pattern=regex_pattern, behavior="isolated", invert=False
    #     # )
    #     tokenizer.train_from_iterator(
    #         train_source,
    #         vocab_size=vocab_size,
    #         show_progress=True,
    #         special_tokens=special_tokens,
    #         unk_token=unk_token,
    #     )

    else:
        raise ValueError(f"Unknown tokenization algorithm: {algorithm}")

    tokenizer.post_processor = TemplateProcessing(
        single=bos_token + " $A " + eos_token,
        special_tokens=[
            (bos_token, tokenizer.token_to_id(bos_token)),
            (eos_token, tokenizer.token_to_id(eos_token)),
        ],
    )

    # file_path = Path(save_path) / ((algorithm + ".json").lower())
    # file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        tokenizer.save(f.name)
        tokenizer_pretrained = PreTrainedTokenizerFast(
            tokenizer_file=f.name,  # file_path.as_posix(),
            model_max_length=model_max_length,
            padding_side="right",
            truncation_side="left",
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
        )

    return tokenizer_pretrained


def tokenize_function(
    batch: dict[str, list],
    tokenizer: transformers.PreTrainedTokenizerFast,
) -> BatchEncoding:
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    # tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    # with CaptureLogger(tok_logger) as cl:

    outputs = tokenizer(
        batch[DATASET_COLUMN_NAME],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        # Recommended by Hugging Face for best performance
        # https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/data_collator#
        # transformers.DataCollatorForLanguageModeling
        # But not recognized by GPT2LMHeadModel.forward
        return_special_tokens_mask=True,
    )
    # clm input could be much much longer than block_size
    # if "Token indices sequence length is longer than the" in cl.out:
    #     tok_logger.warning(
    #             "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
    #             " before being passed to the model."
    #     )

    return outputs


###############################################################################
# Main training loop                                                          #
###############################################################################


def main():
    # Parse arguments from command line or yaml configuration file
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=Path(sys.argv[1]).resolve().as_posix()
        )  # (os.path.abspath(sys.argv[1])
    else:
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log a small summary on each process
    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    #     + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    # )

    logger.info(f"Training/evaluation parameters \n{training_args}")

    # Set seed in random, numpy, torch (not only for training, but also for dataset creation, if applicable)
    set_seed(training_args.seed)

    # -----------------------------------------------------------------------------
    # Load datasets
    # -----------------------------------------------------------------------------

    if data_args.dataset_name is not None:
        raw_datasets = load_raw_dataset_from_hub(
            data_args.dataset_name,
            data_args.dataset_config_name,
            model_args.cache_dir,
            model_args.use_auth_token,
        )
    else:
        raw_datasets = load_raw_dataset_from_dir(
            data_args.dataset_dir,
            validation_split_percentage=data_args.validation_split_percentage,
            test_split_percentage=data_args.test_split_percentage,
            cache_dir=model_args.cache_dir,
            seed=training_args.seed,
        )
    print(raw_datasets)

    # -----------------------------------------------------------------------------
    # Configure model - Step 1 (before tokenizer config)
    # -----------------------------------------------------------------------------

    # Define base model config
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    # TODO check first 2 IFs (currently not used)
    if model_args.config_name:
        logger.info(f"Loading configuration from {model_args.config_name}")
        config = AutoConfig.from_pretrained(
            model_args.config_name, **config_kwargs
        )
    elif model_args.model_name_or_path:
        logger.info(
            f"Loading configuration from {model_args.model_name_or_path}"
        )
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.info(f"Training model type {model_args.model_type} from scratch")
        logger.info(f"Old config {config}")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config_overrides = model_args.config_overrides.replace(" ", "")
            config.update_from_string(config_overrides)
            logger.info(f"New config: {config}")

    # -----------------------------------------------------------------------------
    # Configure / Train tokenizer
    # -----------------------------------------------------------------------------

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": True,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        data_iterator = raw_datasets["train"][DATASET_COLUMN_NAME]
        tokenizer = get_tokenizer(
            pre_tokenizer=data_args.pre_tokenizer,
            algorithm=data_args.algorithm,
            train_source=data_iterator,
            vocab_size=data_args.vocab_size,
            min_frequency=data_args.vocab_min_frequency,
            model_max_length=config.n_positions,
        )

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
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

    # Update model with information from tokenizer
    model_overrides: dict[str, Any] = {
        "vocab_size": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
    }
    config.update(model_overrides)

    # Create causal language model
    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(
            dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()
        )
        logger.info(f"Model size: {n_params/2**20:.2f}M params")

    # Reshape the embeddings if necessary
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # -----------------------------------------------------------------------------
    # Configure training
    # -----------------------------------------------------------------------------

    # Detect last checkpoint
    last_checkpoint = None
    if (
        # os.path.isdir(training_args.output_dir)
        Path(training_args.output_dir).is_dir()
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if (
            last_checkpoint is None
            # and len(os.listdir(training_args.output_dir)) > 0
            and len(list(Path(training_args.output_dir).iterdir())) > 0
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None
            and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Define training/validation sets and limit to max length if necessary
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples
            )
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples
            )
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        # labels = labels[:, :].reshape(-1)
        # preds = preds[:, :].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    # Configure early stopping
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)

    # TODO: might build the lr scheduler "by hand" due to this hugging face bug:
    # https://github.com/huggingface/transformers/issues/20552

    # Configure data collator
    # Pad to multiples of 8 for NVIDIA tensor cores
    # see: https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, pad_to_multiple_of=8, mlm=False
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        # data_collator=default_data_collator,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
        callbacks=[early_stopping_callback],
    )

    # -----------------------------------------------------------------------------
    # Train / Fine-tune the model
    # -----------------------------------------------------------------------------

    # Training (train/validation set)
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

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
        logger.info("*** Evaluate on test set ***")
        if "test" not in tokenized_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = tokenized_datasets["test"]

        # There is also a Trainer.predict() but it returns more than needed here
        metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")

        try:
            perplexity = math.exp(metrics["test_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

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
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


# def _mp_fn(index):
#     # For xla_spawn (TPUs)
#     main()


if __name__ == "__main__":
    main()
