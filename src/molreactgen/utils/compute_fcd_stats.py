import argparse
import warnings
from pathlib import Path

from codetiming import Timer  # type: ignore
from humanfriendly import format_timespan  # type: ignore

from molgen.evaluate import (
    get_stats_from_molecules,
    read_molecules_from_file,
    save_stats_to_file,
)
from molgen.molecule import canonicalize_molecules

DEFAULT_FCD_STATS_FILE = "./fcd_stats.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute FCD metrics (activations' mean, sigma) for a dataset",
        allow_abbrev=True,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        type=str,
        help="Dataset (csv with SMILES strings) to pre-compute FCD metrics for",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_FCD_STATS_FILE,
        type=str,
        help=f"Output file to write the FCD metrics to, default '{DEFAULT_FCD_STATS_FILE}'",
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

        print(
            f"Computing FCD metrics for {len(canon_molecules):,} canonicalized molecules..."
        )
        stats = get_stats_from_molecules(canon_molecules, canonicalize=False)

        print(f"Saving FCD metrics to {output_file_path.name}...")
        save_stats_to_file(output_file_path, stats)


if __name__ == "__main__":
    main()
