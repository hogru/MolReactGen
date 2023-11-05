# coding=utf-8
# src/molreactgen/tokenizer.py
"""Helper functions for tokenizing the dataset.

Functions:
    get_tokenizer:
        Returns a trained Hugging Face fast tokenizer for the given pre-tokenizer, tokenization algorithm and dataset.
    tokenize_function:
        Tokenizes a string using the given tokenizer.
    enclose_function:
        Encloses a string in 'non-special' (i.e. the tokens are part of the normal vocabulary) BOS and EOS tokens.
        Used for fine-tuning the pre-trained model only.
    get_modified_vocab:
        Maps the tokens of one tokenizer to the tokens of another tokenizer.
        Used for fine-tuning the pre-trained model only.
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

from tokenizers import Regex, Tokenizer, decoders  # type: ignore

# Tokenizer related
DATASET_COLUMN_NAME: Final[str] = "items"
DELETED_TOKEN_PREFIX: Final[str] = "§§§_"
BOS_TOKEN: Final[str] = "^"
EOS_TOKEN: Final[str] = "_"
PAD_TOKEN: Final[str] = " "
UNK_TOKEN: Final[str] = "§"
ADD_TOKEN: Final[
    str
] = "°"  # Not sure if needed at all; might be used to map special model tokens to like CLS, SEP etc.
MODEL_MAX_LENGTH: Final[int] = 1024
WORDPIECE_MAX_INPUT_CHARS_PER_WORD: Final[int] = MODEL_MAX_LENGTH

# noinspection SpellCheckingInspection
REGEX_INPUT: Final[dict[str, str]] = {
    # everything within brackets becomes a single token; might play with that, alternative below
    "bracket": r"\[[^\]]+]",
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
REGEX_PATTERN_SMARTS: Final[str] = "|".join(REGEX_INPUT.values()).replace("||", "|")
REGEX_PATTERN_ATOM: Final[str] = REGEX_PATTERN_SMARTS.replace(r"\[[^\]]+]", r"\[|\]")
MIN_VOCAB_SIZE_UNIGRAM: Final[
    int
] = 44  # for comparison with SMARTS + WORDLEVEL tokenizer, which has 44 non-special tokens


logger = logging.getLogger(__name__)


def _filter_invalid_tokenizer_combos(
    pre_tokenizer: str, algorithm: str, vocab_size: int
) -> None:
    """Check if the given tokenizer combination is valid.

    This is mainly used to handle the wandb sweeps, which do not allow for conditional parameters.
    Therefore, we iterate over all combinations, check if the given combination is valid and
    raise an error if not. The wandb sweep will then skip this combination and continue with the next one.

    Args:
        pre_tokenizer: The pre-tokenizer to use.
        algorithm: The tokenization algorithm to use.
        vocab_size: The vocabulary size to use.

    Raises:
        ValueError: If the given combination is invalid.
    """

    # This is a safety check for a Hugging Face issue described here:
    # https://github.com/huggingface/tokenizers/issues/1369
    # Once this is resolved we can use the Split pre-tokenizer with all tokenization algorithms
    if pre_tokenizer in {"ATOM", "SMARTS"} and algorithm != "WORDLEVEL":
        raise ValueError(
            f"Pre-tokenizer {pre_tokenizer} must be used with WORDLEVEL algorithm"
        )

    if algorithm == "WORDLEVEL" and vocab_size != 0:
        raise ValueError(f"Algorithm {algorithm} must be used with a vocab size of 0")

    if algorithm in {"BPE", "WORDPIECE", "UNIGRAM"} and vocab_size == 0:
        raise ValueError(
            f"Algorithm {algorithm} should be used with a vocab size larger than 0."
            f"Use WORDLEVEL algorithm for a vocab size of 0."
        )


def _token_in_regex(token: str, regex: str) -> bool:
    """Checks whether a token matches a given regex.

    Used to check whether a special token like BOS is part of the regex pattern.
    If so, it should not be used as a special token.

    Args:
        token: the (special) token to check
        regex: the regex pattern to check against

    Returns:
        True if the token is part of the regex pattern, False otherwise.
    """

    regex_pattern = re.compile(regex)
    found = regex_pattern.findall(token)
    return len(found) > 0


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
) -> PreTrainedTokenizerFast:
    """Get a Hugging Face fast tokenizer with the given parameters.

    Args:
        pre_tokenizer: The pre-tokenizer to use. One of 'CHAR', 'ATOM', 'SMARTS'.
        algorithm: The tokenization algorithm to use. One of 'WORDLEVEL', 'BPE', 'WORDPIECE', 'UNIGRAM'.
        train_source: The list of strings to train the tokenizer on.
        vocab_size: The vocabulary size to use. Defaults to 0, which means a minimum vocab size is used.
            Can not be used with 'UNIGRAM' algorithm.
        model_max_length: The maximum sequence length of the model. Defaults to MODEL_MAX_LENGTH.
        min_frequency: The minimum frequency a pair should have in order to be merged. Defaults to 1.
        bos_token: The beginning of sequence token. Defaults to BOS_TOKEN.
        eos_token: The end of sequence token. Defaults to EOS_TOKEN.
        pad_token: The padding token. Defaults to PAD_TOKEN.
        unk_token: The unknown token. Defaults to UNK_TOKEN.
        regex_pattern: The regex pattern to use. Defaults to None, which means a default pattern is used.

    Returns:
        The trained Hugging Face fast tokenizer.

    Raises:
        ValueError: If the given argument combination is invalid.
    """

    pre_tokenizer = pre_tokenizer.upper()
    algorithm = algorithm.upper()
    vocab_size = max(
        0, vocab_size
    )  # Zero works with all algorithms except Unigram (runs "infinitely")
    min_frequency = max(1, min_frequency)

    _filter_invalid_tokenizer_combos(pre_tokenizer, algorithm, vocab_size)

    # Safety check for special 'additional' token (not used at the moment)
    try:
        add_token = str(ADD_TOKEN)
    except (NameError, ValueError):
        add_token = "°"

    special_tokens = [bos_token, eos_token, pad_token, unk_token, add_token]
    vocab_size += len(special_tokens)

    # Determine regex pattern for pre-tokenizer
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

    # Check if special tokens are part of the regex pattern
    if pre_tokenizer in ("ATOM", "SMARTS"):
        for token in special_tokens:
            if _token_in_regex(token, regex_pattern):
                raise ValueError(
                    f"Special token '{token}' invalid, can be parsed by '{pre_tokenizer}' regular expression"
                )

    regex_pattern = Regex(regex_pattern)

    # Lots of disabling of PyCharm warnings here, because the Hugging Face Tokenizer API seems to be
    # a bit inconsistent with the documentation
    tokenizer: Tokenizer

    # Train the tokenizer
    if algorithm == "WORDLEVEL":
        if min_frequency > 1:
            logger.warning(
                f"Min frequency {min_frequency} is not supported for {algorithm} "
                f"tokenizer algorithm, setting min frequency to 1."
            )
            min_frequency = 1
        # noinspection PyArgumentList
        tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
        # noinspection PyPropertyAccess
        tokenizer.pre_tokenizer = Split(
            pattern=regex_pattern, behavior="isolated", invert=False
        )
        # noinspection PyArgumentList
        trainer = WordLevelTrainer(
            special_tokens=special_tokens,
            min_frequency=min_frequency,
            show_progress=True,
        )
        tokenizer.train_from_iterator(train_source, trainer=trainer)

    elif algorithm == "BPE":
        # noinspection PyArgumentList
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        # The BPE algorithm seemingly does not work properly with the Split pre-tokenizer
        # see Hugging Face issue https://github.com/huggingface/tokenizers/issues/1369
        # Therefore, we do not use not a pre-tokenizer
        # noinspection PyPropertyAccess
        # tokenizer.pre_tokenizer = Split(
        #     pattern=regex_pattern, behavior="isolated", invert=False
        # )
        # noinspection PyArgumentList
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=min_frequency,
            show_progress=True,
        )
        tokenizer.train_from_iterator(train_source, trainer=trainer)

    elif algorithm == "WORDPIECE":
        # noinspection PyArgumentList
        tokenizer = Tokenizer(
            WordPiece(
                unk_token=unk_token,
                max_input_chars_per_word=WORDPIECE_MAX_INPUT_CHARS_PER_WORD,
            )
        )
        # The WordPiece algorithm seemingly does not work properly with the Split pre-tokenizer
        # see Hugging Face issue https://github.com/huggingface/tokenizers/issues/1369
        # Therefore, we do not use not a pre-tokenizer
        # noinspection PyPropertyAccess
        # tokenizer.pre_tokenizer = Split(
        #     pattern=regex_pattern, behavior="isolated", invert=False
        # )
        # noinspection PyArgumentList
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=min_frequency,
            show_progress=True,
        )
        # noinspection PyPropertyAccess
        tokenizer.decoder = decoders.WordPiece(prefix="##", cleanup=False)
        tokenizer.train_from_iterator(train_source, trainer=trainer)

    elif algorithm == "UNIGRAM":
        # Set a minimum vocab size for Unigram
        old_vocab_size = vocab_size
        vocab_size = max(MIN_VOCAB_SIZE_UNIGRAM, vocab_size)
        if vocab_size > old_vocab_size:
            logger.warning(
                f"Vocab size deemed too small for Unigram, increasing from {old_vocab_size} to {vocab_size}"
            )
        # noinspection PyArgumentList
        tokenizer = Tokenizer(Unigram())
        # The Unigram algorithm seemingly does not work properly with the Split pre-tokenizer
        # see Hugging Face issue https://github.com/huggingface/tokenizers/issues/1369
        # Therefore, we do not use not a pre-tokenizer
        # noinspection PyPropertyAccess
        # tokenizer.pre_tokenizer = Split(
        #     pattern=regex_pattern, behavior="isolated", invert=False
        # )
        # noinspection PyArgumentList
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            unk_token=unk_token,
            show_progress=True,
        )
        tokenizer.train_from_iterator(train_source, trainer=trainer)

    else:
        raise ValueError(f"Unknown tokenization algorithm: {algorithm}")

    # Enclose tokens in special tokens BOS and EOS
    # noinspection PyPropertyAccess
    # noinspection PyArgumentList
    tokenizer.post_processor = TemplateProcessing(
        single=bos_token + " $A " + eos_token,
        special_tokens=[
            (bos_token, tokenizer.token_to_id(bos_token)),
            (eos_token, tokenizer.token_to_id(eos_token)),
        ],
    )

    # Save the tokenizer and load it again as a fast tokenizer
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
    """Enclose each string in the batch with the start and end token.

    Intended to be used with Hugging Face datasets' map function.
    For our custom tokenizers enclosing is done via the tokenizer´s post processor.
    For the pre-trained tokenizers this is done via this function. The pre-trained tokenizers might already
    implement a post processor, and we can't add another one (no sequence of post processors).
    In this case we add the BOS and EOS tokens to the data instead.
    The BOS and EOS tokens are non-special tokens from the tokenizer´s "perspective".

    Args:
        batch: A batch of data
        start_token: The start token
        end_token: The end token

    Returns:
        The batch with the start and end token added to each string
    """

    enclosed = [start_token + line + end_token for line in batch[DATASET_COLUMN_NAME]]
    return {DATASET_COLUMN_NAME: enclosed}


def tokenize_function(
    batch: Mapping[str, list[Union[str, Sequence[str]]]],
    tokenizer: transformers.PreTrainedTokenizerFast,
) -> BatchEncoding:
    """Tokenize a batch of data.

    Intended to be used with Hugging Face datasets' map function.

    Args:
        batch: A batch of data
        tokenizer: The tokenizer to use

    Returns:
        The batch encoded by the tokenizer
    """

    return tokenizer(
        batch[DATASET_COLUMN_NAME],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        # Recommended by Hugging Face for best performance
        # https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/data_collator#
        # transformers.DataCollatorForLanguageModeling
        # But not recognized by GPT2LMHeadModel.forward
        # Raises a warning which can be ignored
        return_special_tokens_mask=True,
    )


def get_modified_vocab(
    tokenizer_original: PreTrainedTokenizerFast,
    modified_freq: Counter[str],
    *,
    mapping_strategy: str = "linear",
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> dict[str, int]:
    """Maps a tokenizer´s vocabulary into a new pre-trained tokenizer.

    The mapping is based on the frequency of the tokens in the original tokenizer´s vocabulary.
    The mapping applied linear from the original (small) vocab into to the new (large) vocab.

    Args:
        tokenizer_original: The original, trained tokenizer.
        modified_freq: The frequency of the tokens in the original tokenizer´s vocabulary.
        mapping_strategy: The strategy to use for mapping the tokens. Currently only "linear" is supported.
        start_idx: The index to start the mapping at. For GPT-2 byte-level BPE tokenizer this is 256.
        end_idx: The index to end the mapping at. If None, the end index is the length of the original vocab.

    Returns:
        A pre-trained tokenizer with the modified vocabulary.
    """

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

    if start_idx < 0:
        raise ValueError("start_idx must be a positive number")

    end_idx = len(vocab_original) if end_idx is None else int(end_idx)

    if not start_idx + len(modified_freq) < end_idx < len(vocab_original):
        raise ValueError(
            f"end_idx must be a positive number, be larger than start_idx plus the length "
            f"of the new vocabulary and smaller than "
            f"{len(vocab_original)}, the length of tokenizer_original"
        )

    # TODO implement additional mapping strategies, currently only 'linear' is supported
    if mapping_strategy.upper() == "LINEAR":
        # Build linear mapping from modified_freq to vocab_original
        # The most frequent element in modified_freq gets the start_idx
        # A (hypothetical) element with frequency zero becomes the end_idx
        vocab_modified = vocab_original.copy()
        mapping_d: int = end_idx
        mapping_k: float = (end_idx - start_idx) / modified_freq.most_common(1)[0][1]
        logger.debug(f"Linear mapping: y = -{mapping_k}x + {mapping_d}")

        # These sets are used to
        # (a) check that we do not double assign indices
        # (b) all tokens have a single entry
        # (c) assign dummy tokens to token ids to make sure that the vocab is contiguous
        indices_used: set[int] = set()
        indices_deleted: set[int] = set()

        # Helper function that uses the linear function to map from
        # the token token_frequency to a token in the (to be built) modified gpt2 tokenizer
        def map_from_to(from_: int) -> int:
            """Maps a token index from the original tokenizer to a token index in the modified tokenizer.

            Args:
                from_: The token index in the original tokenizer

            Returns:
                The token index in the modified tokenizer
            """

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

            # Map its count / frequency to the gpt2 tokenizer vocab
            idx_new = map_from_to(count)

            logger.debug(f"Token {token_modified} gets an ID of {idx_new}")
            # Replace the original token at this index with the new token
            token_original = tokenizer_original.convert_ids_to_tokens(idx_new)
            vocab_modified.pop(token_original, None)
            vocab_modified[token_modified] = idx_new

        # Replace all deleted tokens with a dummy token to ensure a contiguous vocabulary
        indices_deleted -= indices_used
        for idx in sorted(indices_deleted):
            logger.debug(
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
    """Get the merge information from a tokenizer´s algorithm.

    Args:
        tokenizer: The tokenizer to get the merges from.

    Returns:
        A list of merges.
    """

    # Save the tokenizer and read the merges from its configuration file
    with tempfile.TemporaryDirectory() as d_tokenizer:
        tokenizer.save_pretrained(d_tokenizer)
        tokenizer_file = Path(d_tokenizer) / "tokenizer.json"
        with open(tokenizer_file, "r") as f:
            items: dict[Any, Any] = json.load(f)
            model: dict[str, Any] = items.get("model", {})
            merges: list[str] = model.get("merges", [])

    return merges
