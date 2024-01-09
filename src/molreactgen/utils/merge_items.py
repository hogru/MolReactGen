# coding=utf-8
# src/molreactgen/utils/merge_items.py
"""Merge generated items (molecules or reaction templates) into a single file.
"""

import argparse
from os import PathLike
from pathlib import Path
from typing import Any, Final, Optional, Sequence, Union

import pandas as pd  # type: ignore

DEFAULT_FILE_EXTENSION: Final[str] = "*.csv"
DEFAULT_TARGET_FILE_SUFFIX: Final[str] = "_merged.csv"
DEFAULT_CHECK_HEADER: Final[bool] = True
DEFAULT_DROP_DUPLICATES: Final[bool] = True
DEFAULT_COLUMN_FOR_DUPLICATE_CHECK: Final[Optional[Union[int, str]]] = 0


def get_args() -> argparse.Namespace:
    """Read command line arguments"""

    # Prepare argument parser
    parser = argparse.ArgumentParser(
        description="Merge items in CSV files into a single CSV."
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="the directory with the files to be merged.",
    )
    parser.add_argument(
        "--target_file",
        "-t",
        type=Path,
        help=f"the file path to merge the source files into, default: "
        f"a file named after the source-dir with the suffix {DEFAULT_TARGET_FILE_SUFFIX}",
    )
    parser.add_argument(
        "--file_extension",
        "-x",
        type=str.lower,
        default=DEFAULT_FILE_EXTENSION,
        help="the file extension of the files to be merged, default: '%(default)s'.",
    )
    parser.add_argument(
        "--column",
        "-c",
        default=DEFAULT_COLUMN_FOR_DUPLICATE_CHECK,
        help="the column to use for duplicate check; "
        "either 'none' or column number or column name, default: '%(default)s'.",
    )
    parser.add_argument(
        "--skip_header_check",
        "-s",
        default=False,
        action="store_true",
        help="do not check for consistent headers in the input files.",
    )
    parser.add_argument(
        "--allow_duplicates",
        "-a",
        default=False,
        action="store_true",
        help="allow for duplicate entries in the target file.",
    )

    return parser.parse_args()


def get_file_list(
    directory: PathLike, search_pattern: str = DEFAULT_FILE_EXTENSION
) -> list[Path]:
    """Get a list of the source files"""
    directory = Path(directory).resolve()
    return [
        d
        for d in sorted(directory.rglob(search_pattern))
        if d.is_file() and not d.is_symlink()
    ]


def merge_csv_files(
    file_list: Sequence[PathLike],
    check_header: bool = DEFAULT_CHECK_HEADER,
    drop_duplicates: bool = DEFAULT_DROP_DUPLICATES,
    column_for_duplicate_check: Optional[
        Union[int, str]
    ] = DEFAULT_COLUMN_FOR_DUPLICATE_CHECK,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge the source files into a single pandas dataframe"""

    if not file_list:
        raise ValueError("The file list is empty.")

    # Header check must be turned on if duplicates should be dropped based on column name
    if (
        not check_header
        and drop_duplicates
        and isinstance(column_for_duplicate_check, str)
    ):
        raise ValueError(
            f"Can not use column name {column_for_duplicate_check} to drop duplicates without header check."
        )

    # Replace column name "none" with python None
    if (
        isinstance(column_for_duplicate_check, str)
        and column_for_duplicate_check.lower() == "none"
    ):
        column_for_duplicate_check = None

    stats: dict[str, Any] = {}

    # Check whether all files have the same header line
    if check_header:
        print("Checking headers for consistency ...")
        first_file_headings = pd.read_csv(file_list[0], nrows=0).columns.tolist()
        for file_name in file_list[1:]:
            current_file_headings = pd.read_csv(file_name, nrows=0).columns.tolist()
            if first_file_headings != current_file_headings:
                raise ValueError(
                    f"The headings in {file_name} do not match the headings in {file_list[0]}."
                )

    # Merge the source files into a single dataframe
    merged_df = pd.read_csv(file_list[0])
    for file_name in file_list[1:]:
        df = pd.read_csv(file_name)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    stats["items_total"] = len(merged_df)

    # Drop duplicates
    if drop_duplicates:
        if check_header and column_for_duplicate_check is not None:
            if (
                isinstance(column_for_duplicate_check, int)
                and len(first_file_headings) > column_for_duplicate_check
            ):
                subset = first_file_headings[column_for_duplicate_check]
            elif (
                isinstance(column_for_duplicate_check, str)
                and column_for_duplicate_check in first_file_headings
            ):
                subset = column_for_duplicate_check
            else:
                subset = None
        else:
            subset = None
        print(f"Dropping duplicates based on column {subset} ...")
        merged_df.drop_duplicates(subset=subset, inplace=True, ignore_index=True)
        stats["items_deduplicated"] = len(merged_df)

    return merged_df, stats


def save_df(df: pd.DataFrame, file_path: PathLike):
    """Save a dataframe to disk"""
    df.to_csv(file_path, index=False)


def main() -> None:
    args = get_args()

    # Generate target file name
    if args.target_file is None:
        target_file_path = args.source_dir / Path(
            args.source_dir.name + DEFAULT_TARGET_FILE_SUFFIX
        )
    else:
        target_file_path = Path(args.target_file)

    print(
        f"Getting source files with extension {args.file_extension} from {args.source_dir} ..."
    )
    file_list = get_file_list(args.source_dir, search_pattern=args.file_extension)

    print(f"Merging {len(file_list)} source files ...")
    df, stats = merge_csv_files(
        file_list,
        check_header=not args.skip_header_check,
        drop_duplicates=not args.allow_duplicates,
        column_for_duplicate_check=args.column,
    )
    print("Source files merged with the following counts:")
    print(stats)

    print(f"Saving merged file to {target_file_path} ...")
    save_df(df, target_file_path)
    print("Done!")


if __name__ == "__main__":
    main()
