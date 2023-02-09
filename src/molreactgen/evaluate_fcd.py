# coding=utf-8
"""
Auto-Regressive Molecule and Reaction Template Generator
Causal language modeling (CLM) with a transformer decoder model
Author: Stephan Holzgruber
Student ID: K08608294
"""
import argparse
import pickle
from collections.abc import Mapping, Sequence

# from datetime import datetime
# from importlib import reload
from math import exp
from pathlib import Path
from typing import Optional, TypeVar

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

# import molreactgen.config
# import molreactgen.helpers
# import molreactgen.molecule
#
# reload(molreactgen.config)
# reload(molreactgen.helpers)
# reload(molreactgen.molecule)


# try:
#     # noinspection PyUnresolvedReferences
#     import fcd
#     FCD_AVAILABLE = True
# except ImportError:
#     FCD_AVAILABLE = False
#     from molgen.fcd_torch.fcd_torch import FCD  # type: ignore


T = TypeVar("T", bound=npt.NBitBase)


# Global variables, defaults
PROJECT_ROOT_DIR: Path = guess_project_root_dir()
DEFAULT_OUTPUT_FILE_SUFFIX = "_evaluation.csv"
ARGUMENTS_FILE_NAME = "evaluate_cl_args.json"
# DEFAULT_OUTPUT_FILE_PATH = (
#     f"../../data/generated/{datetime.now():%Y-%m-%d_%H-%M}_evaluation.csv"
# )

VALID_EVALUATION_MODES = [
    "reference",
    "stats",
]


def read_molecules_from_file(file: Path) -> list[str]:
    molecules_df: pd.DataFrame
    try:
        molecules_df = pd.read_csv(file, usecols=["canonical_smiles"])
    except ValueError:
        try:
            molecules_df = pd.read_csv(file, usecols=["smiles"])
        except ValueError:
            molecules_df = pd.read_csv(file, header=None, usecols=[0])

    molecules: list[str] = molecules_df.dropna().values.squeeze()

    return molecules


def get_stats_from_molecules(
    molecules: Sequence[str],
    canonicalize: bool = True,
) -> dict[str, "np.floating[T]"]:
    # if FCD_AVAILABLE:
    #     if canonicalize:
    #         molecules = canonicalize_molecules(molecules)
    #     model = fcd.load_ref_model()
    #     activations = fcd.get_predictions(model, molecules)
    #     mu = np.mean(activations, axis=0)
    #     sigma = np.cov(activations.T)
    #     stats = {"mu": mu, "sigma": sigma}
    #
    # else:
    #     fcd_fn = _get_fcd_instance(canonicalize=canonicalize)
    #     stats = fcd_fn.precalc(molecules)

    fcd_fn = _get_fcd_instance(canonicalize=canonicalize)
    stats: dict[str, "np.floating[T]"] = fcd_fn.precalc(molecules)

    return stats


def get_stats_from_file(file: Path) -> dict[str, "np.floating[T]"]:
    with open(file, "rb") as f:
        stats = pickle.load(f)

    if isinstance(stats, tuple) and len(stats) == 2:  # assume fcd style stats
        mu, sigma = stats
        stats_dict = {
            "mu": mu,
            "sigma": sigma,
        }  # = format that fcd_torch expects
    elif isinstance(stats, Mapping):  # assume fcd_torch style stats
        stats_dict = dict(stats)
    else:
        raise ValueError(f"Unknown stats format, stats: {stats}")

    return stats_dict


def get_basic_stats(
    mols_generated: Sequence[str], mols_reference: Sequence[str]
) -> tuple[float, float, float]:

    mols_canonical = canonicalize_molecules(
        mols_generated, strict=False, double_check=True
    )
    mols_valid = [mol for mol in mols_canonical if mol is not None]
    mols_unique = set(mols_valid)
    mols_novel = mols_unique - set(mols_reference)
    len_mols_generated = len(mols_generated)
    validity = len(mols_valid) / len_mols_generated
    uniqueness = len(mols_unique) / len_mols_generated
    novelty = len(mols_novel) / len_mols_generated

    return validity, uniqueness, novelty  # TODO might replace with dict


def get_reference_stats(
    molecule_file_path: Optional[Path] = None,
    stats_file_path: Optional[Path] = None,
) -> dict[str, "np.floating[T]"]:

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
    fcd_fn = FCD(
        canonize=canonicalize, n_jobs=num_workers, device=device, pbar=True
    )

    return fcd_fn


def _get_fcd_from_molecules(
    mols_generated: Sequence[str], mols_reference: Sequence[str]
) -> float:
    # if FCD_AVAILABLE:
    #     model = fcd.load_ref_model()
    #     fcd_value = fcd.get_fcd(model, mols_generated, mols_reference)
    #
    # else:
    #     fcd_fn: FCD = _get_fcd_instance()
    #     fcd_value: float = fcd_fn(gen=mols_generated, ref=mols_reference)

    fcd_fn: FCD = _get_fcd_instance()
    fcd_value: float = fcd_fn(gen=mols_generated, ref=mols_reference)

    return fcd_value


def _get_fcd_from_stats(
    molecules: Sequence[str], stats: Mapping[str, "np.floating[T]"]
) -> float:
    # if FCD_AVAILABLE:
    #     model = fcd.load_ref_model()
    #     activations = fcd.get_predictions(model, molecules)
    #     mu = np.mean(activations, axis=0)
    #     sigma = np.cov(activations.T)
    #
    #     fcd_value = fcd.calculate_frechet_distance(
    #         mu1=mu,
    #         mu2=stats["mu"],
    #         sigma1=sigma,
    #         sigma2=stats["sigma"],
    #     )
    #
    # else:
    #     fcd_fn: FCD = _get_fcd_instance()
    #     fcd_value: float = fcd_fn(gen=molecules, pref=stats)

    fcd_fn: FCD = _get_fcd_instance()
    fcd_value: float = fcd_fn(gen=molecules, pref=stats)

    return fcd_value


def get_fcd(
    mols_generated: Sequence[str],
    *,
    mols_reference: Optional[Sequence[str]] = None,
    reference_stats: Optional[Mapping[str, "np.floating[T]"]] = None,
) -> float:
    if mols_reference is not None and reference_stats is None:
        fcd_value = _get_fcd_from_molecules(mols_generated, mols_reference)
    elif reference_stats is not None:
        fcd_value = _get_fcd_from_stats(mols_generated, reference_stats)
    else:
        raise ValueError(
            "Either mols_reference or reference_stats must be provided"
        )

    return fcd_value


def save_stats_to_file(
    file: Path, stats: Mapping[str, "np.floating[T]"]
) -> None:
    with open(file, "wb") as f:
        pickle.dump(stats, f)


def evaluate_molecules(
    mode: str,
    *,
    generated_file_path: Path,
    reference_file_path: Optional[Path] = None,
    stats_file_path: Optional[Path] = None,
) -> dict[str, float]:

    if mode not in VALID_EVALUATION_MODES:
        raise ValueError(f"Invalid evaluation mode: {mode}")

    if reference_file_path is None and stats_file_path is None:
        raise ValueError(
            "At least one of 'reference_file_path' and 'stats_file_path' must be passed"
        )
    else:
        pass

    # Load generated molecules
    logger.info("Loading generated molecules...")
    mols_generated = read_molecules_from_file(generated_file_path)

    # Calculate basic stats; compare generated and reference molecules (if available)
    if reference_file_path is not None:
        logger.info("Loading reference molecules...")
        mols_reference = read_molecules_from_file(reference_file_path)
        logger.info("Calculating basic stats...")
        validity, uniqueness, novelty = get_basic_stats(
            mols_generated, mols_reference
        )
        basic_stats = {
            "Validity": validity,
            "Uniqueness": uniqueness,
            "Novelty": novelty,
        }
    else:
        mols_reference = None
        basic_stats = {}

    # Get reference stats
    # TODO mypy complains "Missing type parameters for generic type "floating"
    reference_stats: dict[str, "np.floating"]
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
        "-o",
        "--output",
        type=Path,
        required=False,
        default=None,
        help=f"file path for the evaluation statistic, "
        f"default: the file name of the generated molecules with suffix {DEFAULT_OUTPUT_FILE_SUFFIX}.",
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
        if args.stats is not None:
            stats_file_path = Path(args.stats).resolve()
            logger.debug(f"Reference statistics file path: {stats_file_path}")
            if not stats_file_path.is_file():
                raise FileNotFoundError(
                    f"Reference statistics file '{stats_file_path}' not found"
                )
        else:
            raise ValueError(
                "Reference statistics file path must be provided in 'stats' mode"
            )
        if args.reference is not None:
            reference_file_path = Path(args.reference).resolve()
            logger.debug(
                f"Reference molecules file path: {reference_file_path}"
            )
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

    if args.output is None:
        # TODO determine output file path from generated file path
        pass
    else:
        output_file_path = Path(args.output).resolve()
        logger.debug(f"Output file path: '{output_file_path}'")
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

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
        )

        # Display and save stats to file
        for k, v in stats.items():
            logger.info(f"{k}: {v:.4f}")

        # TODO Save output to file


if __name__ == "__main__":
    main()
