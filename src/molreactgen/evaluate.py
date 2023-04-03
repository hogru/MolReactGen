# coding=utf-8
# src/molreactgen/evaluate_fcd.py
"""Evaluate the generated molecules using different metrics.

Functions:
    read_molecules_from_file:
        Read molecules from a CSV file.
    get_stats_from_molecules:
        Calculate the model activations for a list of molecules.
    get_stats_from_file:
        Read precomputed model activations from a file.
    get_basic_stats:
        Calculate basic metrics (validity, uniqueness, novelty) for the generated molecules.
    get_reference_stats:
        Compute the model activations for the reference molecules.
    get_fcd:
        Calculate the Fréchet ChemNet Distance between the generated and reference molecules, or,
        between the generated molecules and precomputed model activations.
    evaluate_molecules:
        Evaluate the generated molecules using different metrics.
    main:
        The main function of the script.
"""

import argparse
import json
import pickle
from collections.abc import Mapping, Sequence
from math import exp
from pathlib import Path
from typing import Final, Optional, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore
from codetiming import Timer
from humanfriendly import format_timespan  # type: ignore
from loguru import logger

from fcd_torch.fcd_torch.fcd import FCD
from molreactgen.helpers import (
    configure_logging,
    determine_log_level,
    get_device,
    get_num_workers,
    guess_project_root_dir,
    save_commandline_arguments,
)
from molreactgen.molecule import canonicalize_molecules

T = TypeVar("T", bound=npt.NBitBase)

# Global variables, defaults
PROJECT_ROOT_DIR: Final[Path] = guess_project_root_dir()
DEFAULT_OUTPUT_FILE_NAME: Final[str] = "evaluation.json"
ARGUMENTS_FILE_NAME: Final[str] = "evaluate_cl_args.json"

VALID_EVALUATION_MODES: Final[tuple[str, ...]] = (
    "reference",
    "stats",
)


def read_molecules_from_file(
    file: Path, num_molecules: Optional[int] = None, valid_only: bool = False
) -> list[str]:
    """Read molecules from a CSV file.
    Args:
        file: the path to the CSV file.
        num_molecules: the first num_molecules of molecules to provide. Defaults to None, i.e. all molecules.

    Returns:
        A list of molecules as strings.

    Raises:
        ValueError: if num_molecules is greater than the number of molecules in the file.
    """

    file = Path(file).resolve()
    # Cope with different column names, allows for several columns names or no header
    molecules_df: pd.DataFrame
    try:
        molecules_df = pd.read_csv(file, usecols=["smiles", "valid"])
        if valid_only:
            molecules_df = molecules_df[molecules_df["valid"]]["smiles"]
        else:
            molecules_df = molecules_df["smiles"]

    except ValueError:
        try:
            molecules_df = pd.read_csv(file, usecols=["smiles"])
        except ValueError:
            molecules_df = pd.read_csv(file, header=None, usecols=[0])
        if valid_only:
            raise ValueError(
                "The 'valid_only' option is not supported for this file, check file format"
            )

    # Delete rows with NaN values
    molecules: list[str] = molecules_df.dropna().values.squeeze()

    # If num_molecules is provided, only return the first num_molecules
    if num_molecules is not None:
        if num_molecules > len(molecules):
            adjective_str = "valid" if valid_only else "generated"
            raise ValueError(
                f"The number of molecules to be evaluated ({num_molecules:,d}) is greater than "
                f"the number of {adjective_str} molecules ({len(molecules):,d})"
            )
        molecules = molecules[:num_molecules]

    return molecules


def get_stats_from_molecules(
    molecules: Sequence[str],
    canonicalize: bool = True,
) -> dict[str, "np.floating[T]"]:
    """Calculate the model activations for a list of molecules.

    Args:
        molecules: the molecules for which to calculate the model activations.
        canonicalize: whether to canonicalize the molecules before calculating the model activations.

    Returns:
        a dictionary containing the model activations.
    """

    fcd_fn = _get_fcd_instance(canonicalize=canonicalize)
    stats: dict[str, "np.floating[T]"] = fcd_fn.precalc(molecules)
    return stats


def get_stats_from_file(file: Path) -> dict[str, "np.floating[T]"]:
    """Read precomputed model activations from a file.
    Args:
        file: the path to the file containing the precomputed model activations.

    Returns:
        a dictionary containing the model activations.

    Raises:
        ValueError: if the file does not contain a known format.
    """

    with open(file, "rb") as f:
        stats = pickle.load(f)

    if isinstance(stats, tuple) and len(stats) == 2:  # assume fcd style stats
        mu, sigma = stats
        stats_dict = {  # convert to format that fcd_torch expects
            "mu": mu,
            "sigma": sigma,
        }
    elif isinstance(stats, Mapping):  # assume fcd_torch style stats
        stats_dict = dict(stats)
    else:
        raise ValueError(f"Unknown stats format, stats: {stats}")

    return stats_dict


def get_basic_stats(
    mols_generated: Sequence[str], mols_reference: Sequence[str]
) -> tuple[float, float, float]:
    """Calculate validity, uniqueness, novelty for the generated molecules.

    Args:
        mols_generated: the generated molecules to calculate the metrics for.
        mols_reference: the reference molecules to calculate some metrics against.

    Returns:
        a tuple containing the validity, uniqueness, novelty.
    """

    mols_canonical = canonicalize_molecules(
        mols_generated, strict=False, double_check=True
    )
    mols_valid = [mol for mol in mols_canonical if mol is not None]
    mols_unique = set(mols_valid)
    mols_novel = mols_unique - set(mols_reference)
    len_mols_generated = len(mols_generated)
    len_mols_valid = len(mols_valid)
    len_mols_unique = len(mols_unique)
    len_mols_novel = len(mols_novel)

    # The stats / percentages are calculated hierarchically (as in GuacaMol)
    validity = len_mols_valid / len_mols_generated if len_mols_generated > 0 else 0.0
    uniqueness = len_mols_unique / len_mols_valid if len_mols_valid > 0 else 0.0
    novelty = len_mols_novel / len_mols_unique if len_mols_unique > 0 else 0.0

    return validity, uniqueness, novelty


def get_reference_stats(
    molecule_file_path: Optional[Path] = None,
    stats_file_path: Optional[Path] = None,
) -> dict[str, "np.floating[T]"]:
    """Read precomputed model activations from a file or calculate them from a list of molecules.

    Args:
        molecule_file_path: the path to the file containing the molecules for which to calculate the model activations.
        stats_file_path: the path to the file containing the precomputed model activations.

    Returns:
        a dictionary containing the model activations.

    Raises:
        ValueError: if neither molecule_file_path nor stats_file_path is provided or if both are provided.
    """

    stats: dict[str, "np.floating[T]"]

    if molecule_file_path is not None and stats_file_path is None:
        molecules = read_molecules_from_file(molecule_file_path)
        stats = get_stats_from_molecules(molecules)

    elif molecule_file_path is None and stats_file_path is not None:
        stats = get_stats_from_file(stats_file_path)

    else:
        raise ValueError(
            "Exactly one of molecule_file_path and stats_file_path must be provided"
        )

    return stats


def _get_fcd_instance(canonicalize: bool = True) -> "FCD":
    num_workers = min(get_num_workers(), 4)
    device = get_device()
    return FCD(canonize=canonicalize, n_jobs=num_workers, device=device, pbar=True)


def _get_fcd_from_molecules(
    mols_generated: Sequence[str], mols_reference: Sequence[str]
) -> float:
    fcd_fn: FCD = _get_fcd_instance()
    return fcd_fn(gen=mols_generated, ref=mols_reference)


def _get_fcd_from_stats(
    molecules: Sequence[str], stats: Mapping[str, "np.floating[T]"]
) -> float:
    fcd_fn: FCD = _get_fcd_instance()
    return fcd_fn(gen=molecules, pref=stats)


def get_fcd(
    mols_generated: Sequence[str],
    *,
    mols_reference: Optional[Sequence[str]] = None,
    reference_stats: Optional[Mapping[str, "np.floating[T]"]] = None,
) -> float:
    """Compute the FCD between the generated molecules and the reference molecules / stats.

    Provide either mols_reference or reference_stats, but not both.

    Args:
        mols_generated: the generated molecules to calculate the FCD for.
        mols_reference: the reference molecules to calculate the FCD against.
        reference_stats: the precomputed model activations to calculate the FCD against.

    Returns:
        the FCD value.

    Raises:
        ValueError: if neither mols_reference nor reference_stats is provided or if both are provided.
    """

    if mols_reference is not None and reference_stats is None:
        fcd_value = _get_fcd_from_molecules(mols_generated, mols_reference)
    elif reference_stats is not None:
        fcd_value = _get_fcd_from_stats(mols_generated, reference_stats)
    else:
        raise ValueError("Either mols_reference or reference_stats must be provided")

    return fcd_value


def save_stats_to_file(file: Path, stats: Mapping[str, "np.floating[T]"]) -> None:
    """Save precomputed model activations to a file.

    Args:
        file: the path to the file to save the model activations to.
        stats: the model activations to save.
    """

    with open(file, "wb") as f:
        pickle.dump(stats, f)


def evaluate_molecules(
    mode: str,
    *,
    generated_file_path: Path,
    reference_file_path: Optional[Path] = None,
    stats_file_path: Optional[Path] = None,
    num_molecules: Optional[int] = None,
) -> dict[str, float]:
    """Evaluate the generated molecules.

    Mode "stats" requires a file containing the precomputed model activations.
    Mode "reference" requires a file containing the reference molecules.
    For mode "stats" a "reference_file_path" can still be provided; it is used for basic statistics.

    Args:
        mode: the evaluation mode. Either "stats" or "reference".
        generated_file_path: the path to the file containing the generated molecules.
        reference_file_path: the path to the file containing the reference molecules.
        stats_file_path: the path to the file containing the precomputed model activations.
        num_molecules: the number of molecules to evaluate. If None, all molecules are evaluated.

    Returns:
        a dictionary containing the evaluation results.
    """

    if mode not in VALID_EVALUATION_MODES:
        raise ValueError(f"Invalid evaluation mode: {mode}")

    if reference_file_path is None and stats_file_path is None:
        raise ValueError(
            "At least one of 'reference_file_path' and 'stats_file_path' must be passed"
        )

    # Load generated molecules
    if num_molecules is None:
        num_molecules_str = "all"
    else:
        num_molecules_str = f"first {num_molecules:,d}"

    logger.info(f"Loading {num_molecules_str} *generated* molecules...")
    mols_generated = read_molecules_from_file(
        generated_file_path, num_molecules, valid_only=False
    )

    # Calculate basic stats; compare generated and reference molecules (if available)
    if reference_file_path is not None:
        logger.info("Loading reference molecules...")
        mols_reference = read_molecules_from_file(reference_file_path)
        logger.info("Calculating basic stats...")
        validity, uniqueness, novelty = get_basic_stats(mols_generated, mols_reference)
        basic_stats = {
            "Validity": validity,
            "Uniqueness": uniqueness,
            "Novelty": novelty,
        }
    else:
        mols_reference = None
        basic_stats = {}

    # Get reference stats
    # mypy complains "Type variable "molreactgen.evaluate.T" is unbound  [valid-type]"
    # Might R&D why this is the case, but for now, we ignore it
    reference_stats: dict[str, "np.floating[T]"]  # type: ignore

    if mode == "reference":
        logger.info("Calculating reference stats from reference molecules...")
        reference_stats = get_reference_stats(reference_file_path, None)
    elif mode == "stats":
        logger.info("Loading pre-computed reference stats...")
        reference_stats = get_reference_stats(None, stats_file_path)
    else:
        raise ValueError(f"Invalid evaluation mode {mode}")

    # Calculate FCD value
    logger.info("Calculating FCD value...")
    logger.info(f"Loading {num_molecules_str} *valid* molecules...")
    mols_generated = read_molecules_from_file(
        generated_file_path, num_molecules, valid_only=True
    )
    fcd_value = get_fcd(
        mols_generated=mols_generated,
        mols_reference=mols_reference,
        reference_stats=reference_stats,
    )

    return {
        "FCD": fcd_value,
        "FCD GuacaMol": exp(-0.2 * fcd_value),
    } | basic_stats


@logger.catch
def main() -> None:
    """Main function of the evaluation script.

    Reads the command line arguments and calls evaluate_molecules().
    Saves the evaluation results to a file.
    """

    # Prepare argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate Fréchet ChemNet Distance (FCD) between generated SMILES molecules "
        "and a reference set."
    )
    parser.add_argument(
        "mode",
        type=str.lower,
        choices=VALID_EVALUATION_MODES,
        help="the evaluation mode.",
    )
    parser.add_argument(
        "-g",
        "--generated",
        type=Path,
        required=True,
        help="file path to the generated molecules.",
    )
    parser.add_argument(
        "-n",
        "--num_molecules",
        type=int,
        required=False,
        default=None,
        help="number of molecules to evaluate from generated molecules, default: all molecules.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        default=None,
        help=f"file path for the evaluation statistic, "
        f"default: {DEFAULT_OUTPUT_FILE_NAME} in the directory of the generated molecules.",
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=Path,
        required=False,
        default=None,
        help="file path to the reference molecules.",
    )
    parser.add_argument(
        "-s",
        "--stats",
        type=Path,
        required=False,
        default=None,
        help="file path to pre-calculated FCD statistics of reference molecules.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="log_level",
        action="append_const",
        const=-1,
        help="increase verbosity from default.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        dest="log_level",
        action="append_const",
        const=1,
        help="decrease verbosity from default.",
    )

    args = parser.parse_args()

    # Configure logging
    log_level: int = determine_log_level(args.log_level)
    configure_logging(log_level)
    save_commandline_arguments(
        args, Path(args.generated).parent / ARGUMENTS_FILE_NAME, ("log_level",)
    )

    logger.heading("Evaluating molecules...")  # type: ignore

    # Prepare and check (global) variables
    generated_file_path = Path(args.generated).resolve()
    logger.debug(f"Generated molecules file path: {generated_file_path}")
    if not generated_file_path.is_file():
        raise FileNotFoundError(
            f"Generated molecules file '{generated_file_path}' not found"
        )

    reference_file_path: Optional[Path]
    stats_file_path: Optional[Path]

    if args.mode == "reference" and args.reference is not None:
        stats_file_path = None
        reference_file_path = Path(args.reference).resolve()
        logger.debug(f"Reference molecules file path: {reference_file_path}")
        if not reference_file_path.is_file():
            raise FileNotFoundError(
                f"Reference molecules file '{reference_file_path}' not found"
            )

    elif args.mode == "stats":
        if args.stats is None:
            raise ValueError(
                "Reference statistics file path must be provided in 'stats' mode"
            )
        stats_file_path = Path(args.stats).resolve()
        logger.debug(f"Reference statistics file path: {stats_file_path}")

        if not stats_file_path.is_file():
            raise FileNotFoundError(
                f"Reference statistics file '{stats_file_path}' not found"
            )

        if args.reference is not None:
            reference_file_path = Path(args.reference).resolve()
            logger.debug(f"Reference molecules file path: {reference_file_path}")
            if not reference_file_path.is_file():
                raise FileNotFoundError(
                    f"Reference molecules file '{reference_file_path}' not found"
                )

        else:
            reference_file_path = None

    else:
        raise ValueError(
            "Reference molecules file path must be provided in 'reference' mode"
        )

    if args.num_molecules is not None and args.num_molecules <= 0:
        raise ValueError(
            f"Number of molecules to evaluate must be greater than 0, "
            f"got {args.num_molecules}"
        )

    if args.output is None:
        output_file_path = generated_file_path.parent / DEFAULT_OUTPUT_FILE_NAME
    else:
        output_file_path = Path(args.output).resolve()
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Output file path: '{output_file_path}'")

    # Start timer
    with Timer(
        name="evaluate_molecules",
        text=lambda secs: f"Molecules evaluated in {format_timespan(secs)}",
        logger=logger.info,
    ):
        # Evaluate molecules
        stats = evaluate_molecules(
            args.mode,
            generated_file_path=generated_file_path,
            reference_file_path=reference_file_path,
            stats_file_path=stats_file_path,
            num_molecules=args.num_molecules,
        )

        # Display and save evaluation stats to file
        logger.info("Evaluation statistics:")
        for k, v in stats.items():
            logger.info(f"{k}: {v:.4f}")

        logger.info(f"Saving evaluation statistics to {output_file_path}")
        with open(output_file_path, "w") as f:
            json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()
