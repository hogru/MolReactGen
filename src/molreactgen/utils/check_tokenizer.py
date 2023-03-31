# coding=utf-8
# src/molreactgen/utils/check_tokenizer.py
"""Check whether a given tokenizer can successfully encode and decode a given dataset.

Functions:
    main:
        The main function of the script.
"""

import argparse
from pathlib import Path
from typing import Final

import pandas as pd  # type: ignore
import transformers  # type: ignore

from molreactgen.tokenizer import get_tokenizer

VALID_PRE_TOKENIZERS: Final[tuple[str, ...]] = (
    "char",
    "atom",
    "smarts",
)

VALID_ALGORITHMS: Final[tuple[str, ...]] = (
    "wordlevel",
    "bpe",
    "wordpiece",
    "unigram",
)


def main() -> None:
    """Check whether a given tokenizer can successfully encode and decode a given dataset."""

    parser = argparse.ArgumentParser(
        description="Check a tokenizer against a dataset",
    )
    parser.add_argument(
        "-p",
        "--pre-tokenizer",
        type=str.lower,
        required=True,
        choices=VALID_PRE_TOKENIZERS,
        help="the pre-tokenizer",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str.lower,
        required=True,
        choices=VALID_ALGORITHMS,
        help="the tokenization algorithm",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        required=True,
        help="the dataset file to check",
    )

    args = parser.parse_args()

    file = Path(args.file).resolve()
    if not file.is_file():
        raise FileNotFoundError(f"Could not find file {file}")

    print(f"Reading data from {file}...")
    df = pd.read_csv(file, usecols=[0], header=None)  # assuming no header, first column
    original: list[str] = df[0].tolist()

    print("Getting tokenizer...")
    tokenizer: transformers.PreTrainedTokenizerFast = get_tokenizer(
        pre_tokenizer=args.pretokenizer,
        algorithm=args.algorithm,
        train_source=original,
    )

    print("Encoding...")
    encoded: transformers.BatchEncoding = tokenizer(original)

    print("Decoding...")
    decoded: list[str] = tokenizer.batch_decode(
        encoded["input_ids"], skip_special_tokens=True
    )

    print("Comparing...")
    for i, (ori, enc, dec) in enumerate(zip(original, encoded, decoded)):
        dec = dec.replace(" ", "")
        if ori != dec:
            print(f"Problem found in row {i+1}:")
            print(f"Original: {ori}")
            print(f"Encoded: {enc}")
            print(f"Decoded: {dec}")
            print("Stopping...")
            break

    print("Success!")


if __name__ == "__main__":
    main()
