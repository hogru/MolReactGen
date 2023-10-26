# coding=utf-8
# src/molreactgen/utils/train_tokenizers.py
"""Train the tokenizers for a given dataset."""

import argparse
from pathlib import Path
from random import randint
from typing import Final

import pandas as pd
from loguru import logger

from molreactgen.helpers import configure_logging, guess_project_root_dir
from molreactgen.tokenizer import _filter_invalid_tokenizer_combos, get_tokenizer

# Global variables, defaults
PROJECT_ROOT_DIR: Final[Path] = guess_project_root_dir()
TOKENIZERS_DIR: Final[Path] = PROJECT_ROOT_DIR / "tokenizers"
PRE_TOKENIZERS: Final[tuple[str, ...]] = ("char", "atom", "smarts")
ALGORITHMS: Final[tuple[str, ...]] = ("wordlevel", "bpe", "wordpiece", "unigram")
MIN_FREQUENCY: int = 1
VOCAB_SIZES: Final[tuple[int, ...]] = (0, 44, 88, 176)
# TEST_TOKENIZER: Final[bool] = True


def train_tokenizers(dataset_file_path: Path, tokenizers_dir: Path) -> None:
    dataset_file_path = Path(dataset_file_path).resolve()
    tokenizers_dir = Path(tokenizers_dir).resolve()
    dataset_name = Path(dataset_file_path).parent.parent.stem
    logger.info(f"Training tokenizers for dataset {dataset_name}...")

    # Load data
    df = pd.read_csv(dataset_file_path, header=None)
    train_source = df.iloc[:, 0].values.tolist()

    # Train each tokenizer combo
    for pre_tokenizer in PRE_TOKENIZERS:
        for algorithm in ALGORITHMS:
            for vocab_size in VOCAB_SIZES:
                pre_tokenizer = pre_tokenizer.upper()
                algorithm = algorithm.upper()

                # Omit invalid tokenizer combos
                try:
                    _filter_invalid_tokenizer_combos(
                        pre_tokenizer, algorithm, vocab_size
                    )
                except ValueError:
                    continue

                logger.info(
                    f"Pre-Tokenizer: {pre_tokenizer}, algorithm: {algorithm}, vocab size: {vocab_size}"
                )
                # Get tokenizer
                tokenizer = get_tokenizer(
                    pre_tokenizer=pre_tokenizer,
                    algorithm=algorithm,
                    min_frequency=MIN_FREQUENCY,
                    vocab_size=vocab_size,
                    train_source=train_source,
                )

                # Save tokenizer
                # -1 for the UNK token, not part of special ids/tokens
                resulting_vocab_size = (
                    tokenizer.vocab_size - len(tokenizer.all_special_ids) - 1
                )
                logger.info(
                    f"Vocab size (without / with special tokens): "
                    f"{resulting_vocab_size} / {tokenizer.vocab_size}"
                )

                # Test tokenizer
                # if TEST_TOKENIZER:
                item = train_source[randint(0, len(train_source) - 1)]
                encoded = tokenizer(
                    item,
                    add_special_tokens=True,
                    padding=True,
                    return_attention_mask=True,
                    return_overflowing_tokens=False,
                    return_tensors="pt",
                )
                input_ids = encoded.input_ids[0]
                decoded = tokenizer.decode(input_ids, skip_special_tokens=True).replace(
                    " ", ""
                )
                if decoded != item:
                    logger.error(
                        f"Decoded item does not match original item:\n"
                        f"Original item: {item}\n"
                        f"Decoded item: {decoded}\n"
                        f"Do NOT save tokenizer: {pre_tokenizer.lower()}_"
                        f"{algorithm.lower()}_{resulting_vocab_size}"
                    )
                    # raise RunTimeError("Decoded item does not match original item.")
                else:
                    tokenizer.save_pretrained(
                        f"{tokenizers_dir}/{dataset_name}/{pre_tokenizer.lower()}_"
                        f"{algorithm.lower()}_{resulting_vocab_size}"
                    )


@logger.catch
def main() -> None:
    """Main training wrapper function.

    Reads the command line arguments and calls tokenizer training function.
    """

    # Prepare argument parser
    parser = argparse.ArgumentParser(
        description="Train the tokenizers on a given dataset."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="file path to the dataset",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"directory path for the trained tokenizers, default: "
        f"{TOKENIZERS_DIR}",
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging()

    # Check arguments
    dataset_file_path = Path(args.dataset).resolve()
    if not dataset_file_path.is_file():
        raise FileNotFoundError(f"File '{dataset_file_path}' does not exist.")
    logger.info(f"Dataset file path: {dataset_file_path}")

    tokenizers_dir = Path(args.output).resolve() if args.output else TOKENIZERS_DIR
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Tokenizers directory: {tokenizers_dir}")

    # Train tokenizers
    train_tokenizers(dataset_file_path, tokenizers_dir)


if __name__ == "__main__":
    main()
