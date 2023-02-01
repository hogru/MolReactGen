# coding=utf-8
import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd  # type: ignore
import pooch  # type: ignore
from loguru import logger
from tdc.generation import MolGen  # type: ignore

from molreactgen.helpers import (
    configure_logging,
    determine_log_level,
    guess_project_root_dir,
)

# Determining if tqdm is installed and if so, use it for the download progressbar
# Might experiment with a custom progressbar later
# Sample code: https://www.fatiando.org/pooch/latest/progressbars.html#custom-progressbar
progressbar: bool
try:
    from tqdm import tqdm  # noqa: F401 # type: ignore

    progressbar = True
except ImportError:
    progressbar = False

PROJECT_ROOT_DIR: Path = guess_project_root_dir()
RAW_DATA_DIR: Path = PROJECT_ROOT_DIR / "data" / "raw"
PREP_DATA_DIR: Path = PROJECT_ROOT_DIR / "data" / "prep"
GZIP_FILE_EXTENSIONS = (".xz", ".gz", ".bz2")

VALID_DATASETS: tuple[str, ...] = (
    "all",
    "debug",
    "guacamol",
    "uspto50k",
    "zinc",
)

RAW_DIRS: dict[str, Path] = {
    "debug": RAW_DATA_DIR / "debug",
    "guacamol": RAW_DATA_DIR / "guacamol",
    "uspto50k": RAW_DATA_DIR / "uspto50k",
    "usptofull": RAW_DATA_DIR / "usptofull",
    "zinc": RAW_DATA_DIR / "zinc",
}

PREP_DIRS: dict[str, Path] = {
    "debug": PREP_DATA_DIR / "debug" / "csv",
    "guacamol": PREP_DATA_DIR / "guacamol" / "csv",
    "uspto50k": PREP_DATA_DIR / "uspto50k" / "csv",
    "usptofull": PREP_DATA_DIR / "usptofull" / "csv",
    "zinc": PREP_DATA_DIR / "zinc" / "csv",
}

# TODO Change debug github link once merged into main
POOCHES: dict[str, pooch.Pooch] = {
    "debug": pooch.create(
        path=RAW_DIRS["debug"].as_posix(),
        # base_url="https://github.com/hogru/MolReactGen/tree/feature/download_data/data/raw/debug/",
        base_url="https://github.com/hogru/MolReactGen/raw/313d5981eb2a339d5baae42917ab55ba03082b7d/data/raw/debug/",
        registry={
            "debug_train.csv": None,  # We can change the debug data as we like, so no hash code
            "debug_val.csv": None,
            "debug_test.csv": None,
            "debug_all.csv": None,
        },
    ),
    "guacamol": pooch.create(
        path=RAW_DIRS["guacamol"].as_posix(),
        # base_url="https://figshare.com/ndownloader/files/",
        base_url="doi:10.6084/",
        registry={
            # "13612760": "3c67ee945f351dbbdc02d9016da22efaffc32a39d882021b6f213d5cd60b6a80",
            # "13612766": "124c4e76062bebf3a9bba9812e76fea958a108f25e114a98ddf49c394c4773bf",
            # "13612757": "0b7e1e88e7bd07ee7fe5d2ef668e8904c763635c93654af094fa5446ff363015",
            # "13612745": "ef19489c265c8f5672c6dc8895de0ebe20eeeb086957bd49421afd7bdf429bef",
            "m9.figshare.7322228.v2/guacamol_v1_train.smiles": "3c67ee945f351dbbdc02d9016da22efaffc32a39d882021b6f213d5cd60b6a80",
            "m9.figshare.7322243.v3/guacamol_v1_valid.smiles": "124c4e76062bebf3a9bba9812e76fea958a108f25e114a98ddf49c394c4773bf",
            "m9.figshare.7322246.v2/guacamol_v1_test.smiles": "0b7e1e88e7bd07ee7fe5d2ef668e8904c763635c93654af094fa5446ff363015",
            "m9.figshare.7322252.v2/guacamol_v1_all.smiles": "ef19489c265c8f5672c6dc8895de0ebe20eeeb086957bd49421afd7bdf429bef",
        },
    ),
    "uspto50k": pooch.create(
        path=RAW_DIRS["uspto50k"].as_posix(),
        # base_url="https://github.com/ml-jku/mhn-react/blob/main/data/",
        base_url="https://github.com/ml-jku/mhn-react/raw/de0fda32f76f866835aa65a6ff857964302b2178/data/",
        registry={
            "USPTO_50k_MHN_prepro.csv.gz": "be7a15e61a8c22cd2bf48be8ce710bbe98843a60a8cf7b3d2d1b667768253c23",
        },
    ),
}


FILE_NAME_TRANSLATIONS: dict[str, str] = {
    # "13612760": "guacamol_v1_train.csv",
    # "13612766": "guacamol_v1_valid.csv",
    # "13612757": "guacamol_v1_test.csv",
    # "13612745": "guacamol_v1_all.csv",
    "m9.figshare.7322228.v2/guacamol_v1_train.smiles": "guacamol_v1_train.csv",
    "m9.figshare.7322243.v3/guacamol_v1_valid.smiles": "guacamol_v1_valid.csv",
    "m9.figshare.7322246.v2/guacamol_v1_test.smiles": "guacamol_v1_test.csv",
    "m9.figshare.7322252.v2/guacamol_v1_all.smiles": "guacamol_v1_all.csv",
    # "USPTO_50k_MHN_prepro.csv.gz": "USPTO_50k_MHN_prepro.csv",
    "USPTO_50k_MHN_prepro.csv.gz.decomp": "USPTO_50k_MHN_prepro.csv",
}

"""
def flatten_dir(file_name: str, action: str, pup: pooch.Pooch) -> str:
    file_path = Path(file_name)
    action = str(action).lower()

    cache_dir = Path(pup.path)
    download_dir = file_path.parent
    if cache_dir != download_dir:
        # Move file to cache dir
        shutil.move(file_path, cache_dir)
        file_path = cache_dir / file_path.name
        download_dir.rmdir()

    return file_path.as_posix()
"""


def _cleanse_and_copy_data(
    input_file_path: Path, output_file_path: Path
) -> None:
    df = pd.read_csv(input_file_path, usecols=[0], header=None)
    first_row = df[0][0]
    if first_row.upper().startswith(
        ("SMILES", "REACTION", "SMARTS", "SMIRKS", "SELFIES", "MOL")
    ):
        df = df.iloc[1:].reset_index(drop=True)
    df.to_csv(output_file_path, header=False, index=False)


def _download_pooched_dataset(
    dataset: str, raw_dir: Path, enforce_download: bool
) -> None:
    assert raw_dir.samefile(POOCHES[dataset].path)
    if enforce_download:
        for file in POOCHES[dataset].registry:
            if (raw_dir / file).exists():
                logger.info(f"Deleting file {file}...")
                (raw_dir / file).unlink()

    processor: pooch.processors
    for file in POOCHES[dataset].registry:
        processor = (
            pooch.Decompress() if file.endswith(GZIP_FILE_EXTENSIONS) else None
        )
        file_name = Path(
            POOCHES[dataset].fetch(
                file,
                processor=processor,
                progressbar=progressbar,
            )
        )
        logger.info(f"Loaded file {file_name.name}")


def _prepare_pooched_dataset(
    dataset: str, raw_dir: Path, prep_dir: Path
) -> None:
    assert raw_dir.samefile(POOCHES[dataset].path)
    prep_dir.mkdir(parents=True, exist_ok=True)
    for file in POOCHES[dataset].registry:
        # sub_dir = Path(file).parent
        file_renamed = FILE_NAME_TRANSLATIONS.get(file, file)
        _cleanse_and_copy_data(raw_dir / file, prep_dir / file_renamed)


def _download_debug_dataset(raw_dir: Path, enforce_download: bool) -> None:
    _download_pooched_dataset("debug", raw_dir, enforce_download)


def _prepare_debug_dataset(raw_dir: Path, prep_dir: Path) -> None:
    _prepare_pooched_dataset("debug", raw_dir, prep_dir)


def _download_guacamol_dataset(raw_dir: Path, enforce_download: bool) -> None:
    _download_pooched_dataset("guacamol", raw_dir, enforce_download)


def _prepare_guacamol_dataset(raw_dir: Path, prep_dir: Path) -> None:
    _prepare_pooched_dataset("guacamol", raw_dir, prep_dir)


def _download_uspto_50k_dataset(raw_dir: Path, enforce_download: bool) -> None:
    _download_pooched_dataset("uspto50k", raw_dir, enforce_download)


def _prepare_uspto_50k_dataset(raw_dir: Path, prep_dir: Path) -> None:
    # assert raw_dir.samefile(POOCHES["uspto50k"].path)
    # prep_dir.mkdir(parents=True, exist_ok=True)
    # for file in POOCHES["uspto50k"].registry:
    #     file_renamed = FILE_NAME_TRANSLATIONS.get(file, file)
    #     print(raw_dir / file)
    #     with gzip.open(raw_dir / file, "rb") as f_in:
    #         with open(raw_dir / file_renamed, "wb") as f_out:
    #             print("do I get here?")
    #             shutil.copyfileobj(f_in, f_out)

    # Setup file, column, split names
    raw_file = raw_dir / "USPTO_50k_MHN_prepro.csv.gz.decomp"
    if not Path(raw_file).is_file():
        raise FileNotFoundError(
            f"File {raw_file} not found. The raw file name seems to have changed."
        )
    files: dict[str, str] = {
        "known": "USPTO_50k_known.csv",  # "known" means in either in validation or test set (but not train set)
        "train": "USPTO_50k_train.csv",
        "valid": "USPTO_50k_val.csv",
        "test": "USPTO_50k_test.csv",
    }
    splits: tuple[str, ...] = ("train", "valid", "test")
    # Historically there were two different reaction template sets, one with atom mapping and one without
    # We will use the one with atom mapping, but keep the code for the other one for reference / just in case
    # dirs = ["with_mapping", "without_mapping"]
    # columns: list[str] = [
    # "reaction_smarts_with_atom_mapping",
    # "reaction_smarts_without_atom_mapping",
    # ]
    column: str = "reaction_smarts_with_atom_mapping"

    # Read raw data, rename/append columns
    df: dict[str, pd.DataFrame] = dict()
    df["raw"] = pd.read_csv(raw_file, header=0)
    df["all"] = df["raw"][["id", "split", "prod_smiles", "reaction_smarts"]]
    df["all"].rename(
        columns={
            "id": "USPTO-50k_id",
            "prod_smiles": "product_smiles",
            "reaction_smarts": column,
        },
        inplace=True,
    )
    # df["all"]["reaction_smarts_without_atom_mapping"] = remove_atom_mapping(
    #     df["all"]["reaction_smarts_with_atom_mapping"]
    # )

    # Prepare "known" reactions from validation and test set (can include duplicates)
    df["known"] = df["all"][df["all"]["split"].isin(["valid", "test"])].copy()

    # Prepare train, validation, and test set and save in corresponding files
    # for sub_dir, column in zip(dirs, columns):
    # (prep_dir / sub_dir).mkdir(parents=True, exist_ok=True)
    # file_name = prep_dir / sub_dir / files["known"]
    file_name = prep_dir / files["known"]
    df["known"].to_csv(file_name, header=True, index=False)
    logger.debug(
        # f"Save {len(df['known'])} known reactions (including duplicates) to {sub_dir}/{file_name.name}..."
        f"Save {len(df['known'])} known reactions (including duplicates) to {file_name.name}..."
    )
    for split in splits:
        # file_name = prep_dir / sub_dir / files[split]
        file_name = prep_dir / files[split]
        # df[split] = df["all"][df["all"]["split"] == split][[column]].copy()
        df[split] = df["all"][df["all"]["split"] == split][[column]].copy()
        # Remove duplicates
        len_before = len(df[split])
        df[split].drop_duplicates(inplace=True)
        len_after = len(df[split])
        logger.debug(
            f"Remove {len_before - len_after} duplicates from {split} set, {len_after} reactions left..."
        )
        if split == "train":
            # Remove entries that are also in the "known" set
            len_before = len(df[split])
            df[split] = df[split][
                # ~df[split][column].isin(df["known"][column])
                ~df[split][column].isin(df["known"][column])
            ].copy()
            len_after = len(df[split])
            logger.debug(
                f"Remove {len_before - len_after} reactions from train set that are also in the "
                f"validation and/or test set, {len_after} rows left..."
            )
        # logger.debug(f"Save {split} reactions to {sub_dir}/{file_name.name}...")
        logger.debug(f"Save {split} reactions to {file_name.name} ...")
        df[split].to_csv(file_name, header=False, index=False)

        # Make sure that train set is not included in known set
        train_known_intersection = set(df["known"][column].values) & set(
            df["train"].iloc[:, 0].values
        )
        assert (
            len(train_known_intersection) == 0
        ), "Train set includes reactions from validation and/or test set!"


def _download_zinc_dataset(raw_dir: Path, enforce_download: bool) -> None:
    file = "zinc.tab"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if enforce_download:
        if (raw_dir / file).exists():
            logger.info(f"Deleting file {file}...")
            (raw_dir / file).unlink()

    if (raw_dir / file).is_file():
        logger.info(f"File {file} already exists, skipping download...")
    else:
        logger.info(f"Cacheing file {file} to {raw_dir}...")
        _ = MolGen(
            name="ZINC",
            path=raw_dir,
            print_stats=False,
        )


def _prepare_zinc_dataset(raw_dir: Path, prep_dir: Path) -> None:
    file = "zinc.tab"
    prep_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / file
    prep_file = (prep_dir / raw_file.name).with_suffix(".csv")
    _cleanse_and_copy_data(raw_file, prep_file)


DOWNLOAD_FNS: dict[str, Callable[[Path, bool], None]] = {
    "debug": _download_debug_dataset,
    "guacamol": _download_guacamol_dataset,
    "uspto50k": _download_uspto_50k_dataset,
    "zinc": _download_zinc_dataset,
}


def download_dataset(
    dataset: str, raw_dir: Path, enforce_download: bool = False
) -> None:
    download_fn = DOWNLOAD_FNS.get(dataset, None)
    if download_fn is None:
        raise ValueError(f"Invalid dataset: {dataset}")
    else:
        enforce_str = (
            " (force deleting cached files)" if enforce_download else ""
        )
        logger.info(
            f"Downloading dataset {dataset} to {raw_dir}{enforce_str}..."
        )
        return download_fn(raw_dir, enforce_download)


PREPARE_FNS: dict[str, Callable[[Path, Path], None]] = {
    "debug": _prepare_debug_dataset,
    "guacamol": _prepare_guacamol_dataset,
    "uspto50k": _prepare_uspto_50k_dataset,
    "zinc": _prepare_zinc_dataset,
}


def prepare_dataset(dataset: str, raw_dir: Path, prep_dir: Path) -> None:
    prepare_fn = PREPARE_FNS.get(dataset, None)
    if prepare_fn is None:
        raise ValueError(f"Invalid dataset: {dataset}")
    else:
        logger.info(
            f"Preparing dataset {dataset} from {raw_dir} into {prep_dir}..."
        )
        return prepare_fn(raw_dir, prep_dir)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Prepare data for training of the Hugging Face model."
    )
    parser.add_argument(
        "dataset",
        type=str.lower,
        nargs="?",
        default="all",
        choices=VALID_DATASETS,
        help="the dataset to prepare, default: '%(default)s' for all datasets.",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=Path,
        default=f"{PREP_DATA_DIR}",
        help="root of the data directory, default: '%(default)s'.",
    )
    parser.add_argument(
        "-e",
        "--enforce_download",
        default=False,
        action="store_true",
        help="enforce downloading the dataset(s), default: '%(default)s'.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="log_level",
        action="append_const",
        const=-1,
    )
    parser.add_argument(
        "--quiet",
        "-q",
        dest="log_level",
        action="append_const",
        const=1,
    )

    args = parser.parse_args()
    log_level: int = determine_log_level(args.log_level)
    configure_logging(log_level)

    if "all" in args.dataset:
        datasets = sorted(set(VALID_DATASETS) - {"all"})
    else:
        datasets = [args.dataset]

    # TODO what if data_dir is set?
    # data_dir = args.data_dir.resolve()

    for dataset in datasets:
        logger.heading(f"Preparing dataset {dataset}...")  # type: ignore
        raw_dir = RAW_DIRS[dataset]
        download_dataset(dataset, raw_dir, args.enforce_download)
        prep_dir = PREP_DIRS[dataset]
        prepare_dataset(dataset, raw_dir, prep_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
