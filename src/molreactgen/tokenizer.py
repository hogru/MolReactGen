# coding=utf-8
"""
Auto-Regressive Molecule and Reaction Template Generator
Causal language modeling (CLM) with a transformer decoder model
Author: Stephan Holzgruber
Student ID: K08608294
"""
import json
import logging
import re
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Optional, Sequence, Union

# Most of Hugging Face has poor type hints, trying to avoid mypy errors
import transformers  # type: ignore
from tokenizers import Regex, Tokenizer, decoders  # type: ignore
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece  # type: ignore
from tokenizers.pre_tokenizers import Split  # type: ignore
from tokenizers.processors import TemplateProcessing  # type: ignore
from tokenizers.trainers import (  # type: ignore
    BpeTrainer,
    UnigramTrainer,
    WordLevelTrainer,
    WordPieceTrainer,
)
from transformers import BatchEncoding, PreTrainedTokenizerFast

# Tokenizer related
DATASET_COLUMN_NAME: Final = "items"
DELETED_TOKEN_PREFIX: Final = "§§§_"
BOS_TOKEN: Final = "^"
EOS_TOKEN: Final = "_"
PAD_TOKEN: Final = " "
UNK_TOKEN: Final = "§"
ADD_TOKEN: Final = "°"  # Not sure if needed at all; might be used to map special model tokens to like CLS, SEP etc.
MODEL_MAX_LENGTH: Final = 1024

REGEX_INPUT: Final = {
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

# replace() is just a safety net against double "|" in the RegEx
REGEX_PATTERN_CHAR: Final[str] = ""
REGEX_PATTERN_SMARTS: Final = "|".join(REGEX_INPUT.values()).replace("||", "|")
REGEX_PATTERN_ATOM: Final = REGEX_PATTERN_SMARTS.replace(r"\[[^\]]+]", r"\[|\]")
MIN_VOCAB_SIZE_UNIGRAM: Final = 44  # for comparison with SMARTS + WORDLEVEL tokenizer, which has 44 non-special tokens


logger = logging.getLogger(__name__)


def _filter_invalid_tokenizer_combos(
    pre_tokenizer: str, algorithm: str, vocab_size: int
) -> None:
    if pre_tokenizer in {"ATOM", "SMARTS"} and algorithm != "WORDLEVEL":
        raise ValueError(
            f"Pre-tokenizer {pre_tokenizer} must be used with WORDLEVEL algorithm"
        )

    if algorithm == "WORDLEVEL" and vocab_size != 0:
        raise ValueError(f"Algorithm {algorithm} must be used with a vocab size of 0")

    # if algorithm == "UNIGRAM" and vocab_size < MIN_VOCAB_SIZE_UNIGRAM:
    #     raise ValueError(
    #         f"Algorithm {algorithm} must be used with a vocab size of at least {MIN_VOCAB_SIZE_UNIGRAM}"
    #     )


def token_in_regex(token: str, regex: str) -> bool:
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
    model_max_length: int = MODEL_MAX_LENGTH,
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

    _filter_invalid_tokenizer_combos(pre_tokenizer, algorithm, vocab_size)

    try:
        add_token = str(ADD_TOKEN)
    except (NameError, ValueError):
        add_token = "°"

    special_tokens = [bos_token, eos_token, pad_token, unk_token, add_token]
    vocab_size += len(special_tokens)

    if regex_pattern is None:
        if pre_tokenizer == "CHAR":
            regex_pattern = REGEX_PATTERN_CHAR
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
    #         # limit_alphabet=1000,  # might make configurable, default=1000
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


# Can't use a post processor in the tokenizer, since the post-processor is already set and
# there is no Sequence of post processors
# Add BOS and EOS to the data instead
def enclose_function(
    batch: Mapping[str, list[str]],  # list[Union[str, Sequence[str]]]],
    start_token: str,
    end_token: str,
) -> dict[str, list[str]]:
    enclosed = [start_token + line + end_token for line in batch[DATASET_COLUMN_NAME]]
    return {DATASET_COLUMN_NAME: enclosed}


def tokenize_function(
    batch: Mapping[str, list[Union[str, Sequence[str]]]],
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
    # clm input could be much longer than block_size
    # if "Token indices sequence length is longer than the" in cl.out:
    #     tok_logger.warning(
    #             "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
    #             " before being passed to the model."
    #     )

    return outputs


def get_modified_vocab(
    tokenizer_original: PreTrainedTokenizerFast,
    modified_freq: Counter[str],
    *,
    mapping_strategy: str = "linear",
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> dict[str, int]:
    # Check arguments
    if not isinstance(tokenizer_original, PreTrainedTokenizerFast):
        raise TypeError(
            f"tokenizer_original must of type PreTrainedTokenizerFast, but is {type(tokenizer_original)}"
        )

    if not isinstance(modified_freq, Counter):
        raise TypeError(
            f"modified_freq must of type Tally, but is {type(modified_freq)}"
        )

    vocab_original: dict[str, int] = tokenizer_original.get_vocab()
    if len(modified_freq) > len(vocab_original):
        raise ValueError(
            f"The length of modified_freq ({len(modified_freq)}) is larger than "
            f"the length of tokenizer_original ({len(vocab_original)})"
        )

    if int(start_idx) < 0:
        raise ValueError("start_idx must be a positive number")

    end_idx = len(vocab_original) if end_idx is None else int(end_idx)

    if not start_idx + len(modified_freq) < end_idx < len(vocab_original):
        raise ValueError(
            f"end_idx must be a positive number, be larger than start_idx plus the length "
            f"of the new vocabulary and smaller than "
            f"{len(vocab_original)}, the length of tokenizer_original"
        )

    if mapping_strategy.upper() == "LINEAR":
        # Build linear mapping from modified_freq to vocab_original
        # The most frequent element in modified_freq gets the start_idx
        # A (hypothetical) element with frequency zero becomes the end_idx
        vocab_modified = vocab_original.copy()
        mapping_d: int = end_idx
        mapping_k: float = (end_idx - start_idx) / modified_freq.most_common(1)[0][1]
        print(f"Linear mapping: y = -{mapping_k}x + {mapping_d}")

        # These sets are used to
        # (a) check that we do not double assign indices
        # (b) all tokens have a single entry
        # (c) assign dummy tokens to token ids to make sure that the vocab is contiguous
        indices_used: set[int] = set()
        indices_deleted: set[int] = set()

        # Helper function that uses the linear function to map from
        # the token token_frequency to a token in the (to be built) modified gpt2 tokenizer
        def map_from_to(from_: int) -> int:
            to = int(-mapping_k * from_ + mapping_d) - 1
            while (to := to + 1) in indices_used:
                pass
            if not (start_idx <= to < len(vocab_original)):
                raise RuntimeError("Cannot fit modified_freq into tokenizer_original")

            indices_used.add(to)
            return to

        idx_original: Optional[int]
        # Go over all tokens in modified_freq and do the following
        for token_modified, count in modified_freq.most_common(None):
            # Delete the original entry from the vocab (if it exists) and remember its index
            idx_original = vocab_modified.pop(token_modified, None)
            if idx_original is not None:
                indices_deleted.add(idx_original)
                # print(f"Change token {token_modified} with idx {idx_original}")
                # vocab_modified[token_modified + TOKEN_SUFFIX] = idx_original

            # Map its count / frequency to the gpt2 tokenizer vocab
            idx_new = map_from_to(count)

            print(f"Token {token_modified} gets an ID of {idx_new}")
            # Replace the original token at this index with the new token
            token_original = tokenizer_original.convert_ids_to_tokens(idx_new)
            vocab_modified.pop(token_original, None)
            vocab_modified[token_modified] = idx_new

        # Replace all deleted tokens with a dummy token to ensure a contiguous vocabulary
        indices_deleted -= indices_used
        for idx in sorted(indices_deleted):
            print(
                f"Overwrite original token {tokenizer_original.convert_ids_to_tokens(idx)} "
                f"at ID {idx} with {DELETED_TOKEN_PREFIX+str(idx)}"
            )
            vocab_modified[DELETED_TOKEN_PREFIX + str(idx)] = idx

        # Return the sorted modified vocab
        vocab_modified = dict(sorted(vocab_modified.items(), key=lambda item: item[1]))

    else:
        raise ValueError(f"Unknown mapping strategy {mapping_strategy}")

    return vocab_modified


def get_merges(tokenizer: PreTrainedTokenizerFast) -> list[str]:
    # Save the tokenizer and read the merges from file
    with tempfile.TemporaryDirectory() as d_tokenizer:
        tokenizer.save_pretrained(d_tokenizer)
        tokenizer_file = Path(d_tokenizer) / "tokenizer.json"
        with open(tokenizer_file, "r") as f:
            items: dict[Any, Any] = json.load(f)
            model: dict[str, Any] = items.get("model", {})
            merges: list[str] = model.get("merges", [])

    return merges
