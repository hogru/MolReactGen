# coding=utf-8
import argparse
from importlib import reload
from pathlib import Path

import transformers  # type: ignore

import molgen.huggingface.train_hf

reload(molgen.huggingface.train_hf)
import pandas as pd  # type: ignore

from molgen.huggingface.train_hf import get_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check basic (pre-) tokenizer against dataset file",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="the dataset file to check",
    )

    args = parser.parse_args()
    file: Path = Path(args.file)
    if not file.is_file():
        raise ValueError(f"Invalid file: {file}")
    print(f"Reading data from {file}...")
    df: pd.DataFrame = pd.read_csv(file, usecols=[0], header=None)
    original: list[str] = df[0].tolist()
    print("Getting tokenizer...")
    tokenizer: transformers.PreTrainedTokenizerFast = get_tokenizer(
        algorithm="WORDLEVEL", train_source=original
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
            print(f"Problem found with row {i}:")
            print(f"Original: {ori}")
            print(f"Encoded: {enc}")
            print(f"Decoded: {dec}")
            break
    print("Done!")


if __name__ == "__main__":
    main()
