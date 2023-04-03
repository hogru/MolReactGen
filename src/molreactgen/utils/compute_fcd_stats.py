# coding=utf-8
# src/molreactgen/utils/compute_fcd_stats.py
"""Precompute and save the Fréchet ChemNet Distance stats (model activations) for a given dataset.

The precomputed FCD stats can be used to compute the FCD metrics for assessing a model's performance.

Functions:
    main:
        The main function of the script.
"""

import argparse
from pathlib import Path
from typing import Any, Final

from codetiming import Timer
from humanfriendly import format_timespan  # type: ignore
from loguru import logger

from molreactgen.evaluate import (
    get_stats_from_molecules,
    read_molecules_from_file,
    save_stats_to_file,
)
from molreactgen.helpers import configure_logging, determine_log_level
from molreactgen.molecule import canonicalize_molecules

DEFAULT_FCD_STATS_FILE: Final[str] = "./fcd_stats.pkl"


def main() -> None:
    """Precompute and save the model activations for a given dataset."""

    parser = argparse.ArgumentParser(
        description="pre-compute FCD stats (model activations) for a given dataset",
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="dataset (csv with SMILES strings) to pre-compute the FCD stats for",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_FCD_STATS_FILE,
        type=Path,
        help="output file to write the FCD stats to, default '%(default)s'",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="log_level",
        action="append_const",
        const=-1,
        help="increase verbosity from default.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="log_level",
        action="append_const",
        const=1,
        help="decrease verbosity from default.",
    )

    args = parser.parse_args()

    # Configure logging
    log_level: int = determine_log_level(args.log_level)
    configure_logging(log_level)

    # Start timer
    with Timer(
        name="compute_fcd_stats",
        text=lambda secs: f"Computed FCD stats in {format_timespan(secs)}",
    ):
        logger.heading("Computing FCD stats...")  # type: ignore
        dataset_file_path = Path(args.dataset).resolve()
        if not dataset_file_path.is_file():
            raise ValueError(f"Dataset file {dataset_file_path} does not exist")

        output_file_path = Path(args.output).resolve()
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Add .pkl extension if none present
        if output_file_path.suffix == "":
            output_file_path = output_file_path.with_suffix(".pkl")
            logger.debug(
                f"Added file extension .pkl to output file: {output_file_path.name}"
            )
        # Give warning if output file already exists
        if output_file_path.is_file():
            logger.warning(
                f"Output file {output_file_path.name} already exists, overwriting it"
            )
            output_file_path.unlink()

        logger.info(f"Reading molecules from {dataset_file_path.name}...")
        molecules = read_molecules_from_file(dataset_file_path)

        logger.info(f"Canonicalizing {len(molecules):,} molecules...")
        canon_molecules = canonicalize_molecules(molecules, double_check=True)
        assert all(canon_molecules), "Not all molecules could be canonicalized"

        logger.info(
            f"Computing FCD stats for {len(canon_molecules):,} canonicalized molecules..."
        )
        stats: dict[str, Any] = get_stats_from_molecules(canon_molecules, canonicalize=False)  # type: ignore

        logger.info(f"Saving FCD stats to {output_file_path}...")
        save_stats_to_file(output_file_path, stats)


if __name__ == "__main__":
    main()
