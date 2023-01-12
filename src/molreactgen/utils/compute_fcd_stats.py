# coding=utf-8
"""
Auto-Regressive Molecule and Reaction Template Generator
Causal language modeling (CLM) with a transformer decoder model
Author: Stephan Holzgruber
Student ID: K08608294
"""

import argparse
import warnings
from pathlib import Path
from typing import Any

from codetiming import Timer
from humanfriendly import format_timespan  # type: ignore

from molreactgen.evaluate_fcd import (
    get_stats_from_molecules,
    read_molecules_from_file,
    save_stats_to_file,
)
from molreactgen.molecule import canonicalize_molecules

DEFAULT_FCD_STATS_FILE: str = "./fcd_stats.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pre-compute FCD metrics (activations' mean, sigma) for a dataset",
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="dataset (csv with SMILES strings) to pre-compute FCD metrics for",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_FCD_STATS_FILE,
        type=Path,
        help="output file to write the FCD metrics to, default '%(default)s'",
    )

    args = parser.parse_args()

    # Start timer
    with Timer(
        name="compute_fcd_metrics",
        text=lambda secs: f"Computed FCD metrics in {format_timespan(secs)}",
    ):

        dataset_file_path = Path(args.dataset).resolve()
        if not dataset_file_path.is_file():
            raise ValueError(f"Dataset file {dataset_file_path} does not exist")

        output_file_path = Path(args.output).resolve()
        # Add .pkl extension if none present
        if output_file_path.suffix == "":
            output_file_path = output_file_path.with_suffix(".pkl")
            print(
                f"Added file extension .pkl to output file: {output_file_path.name}"
            )
        # Raise warning if output file already exists
        if output_file_path.is_file():
            warnings.warn(
                f"Output file {output_file_path.name} already exists, overwriting it"
            )
            output_file_path.unlink()

        print(f"Reading molecules from {dataset_file_path.name}...")
        molecules = read_molecules_from_file(dataset_file_path)

        print(f"Canonicalizing {len(molecules):,} molecules...")
        canon_molecules = canonicalize_molecules(molecules, double_check=True)
        assert all(canon_molecules), "Not all molecules could be canonicalized"

        print(
            f"Computing FCD metrics for {len(canon_molecules):,} canonicalized molecules..."
        )
        stats: dict[str, Any] = get_stats_from_molecules(canon_molecules, canonicalize=False)  # type: ignore

        print(f"Saving FCD metrics to {output_file_path.name}...")
        save_stats_to_file(output_file_path, stats)


if __name__ == "__main__":
    main()
