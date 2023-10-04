# coding=utf-8
# src/molreactgen/utils/rename_files.py
"""A helper file to rename files for consistency.
"""

import argparse
from pathlib import Path
from typing import Final, Union

FILES_TO_RENAME: Final[dict[str, str]] = {
    "generate_cl_args.json": "generate_args.json",
    "evaluate_cl_args.json": "assess_args.json",
    "generation_stats.json": "generate_stats.json",
    "evaluation.json": "assess_stats.json",
    "generated_reaction_templates.csv": "generated_smarts.csv",
    "generated_reaction_templates_redundant.csv": "generated_smarts_feasible_only.csv",
}


def rename_file(
    old_file: Union[str, Path], new_file: Union[str, Path], overwrite: bool = False
) -> None:
    old_file = Path(old_file)
    new_file = Path(new_file)

    if not old_file.is_file():
        raise FileNotFoundError(f"File to rename {old_file} does not exist")

    if new_file.exists() and not overwrite:
        raise FileExistsError(
            f"New file {new_file} already exists (set overwrite=True) to enforce overwrite"
        )

    print(f"    Rename from {old_file.name} to {new_file.name}")
    old_file.rename(new_file)


def rename_files_in_directory(
    directory: Union[str, Path], overwrite: bool = False
) -> None:
    directory = Path(directory)

    if not directory.is_dir():
        raise ValueError(f"Directory '{directory}' does not exist")

    for file in sorted(directory.rglob("*")):
        if file.is_dir() and not file.is_symlink():
            print(f"Scanning directory {file}...")
        if file.is_file() and not file.is_symlink() and file.name in FILES_TO_RENAME:
            rename_file(file, file.parent / FILES_TO_RENAME[file.name], overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename files in directory (change source to change affected files)."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="the directory (including its sub-directories) to rename the files in.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrites existing files.",
    )

    args = parser.parse_args()
    rename_files_in_directory(args.directory, args.overwrite)


if __name__ == "__main__":
    main()
