# coding=utf-8
import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd  # type: ignore
from tdc.generation import MolGen  # type: ignore

# This creates a huge list of mypy issues (without it, no issues)
# TODO re-check once molgen.molecule is properly type checked
from molgen.molecule import remove_atom_mapping  # type: ignore

# from shutil import copy2


VALID_DATASETS: tuple[str, ...] = (
    "all",
    "debug",
    "guacamol",
    "uspto50k",
    "zinc",
)

'''
class DatasetAction(argparse.Action):
    """An option to handle multiple datasets without "all" being a valid option"""
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print(values)
        print(option_string)
        if values is None or len(values) == 0:
            values = VALID_DATASETS
        elif isinstance(values, str):
            values = [values]
        elif isinstance(values, Iterable):
            pass
        else:
            raise TypeError(f"Invalid type for argument {option_string}: {type(values)}")

        # values = [v.lower() for v in values]

        for v in values:
            if v not in VALID_DATASETS:
                raise ValueError(f"Invalid value for argument {option_string}: {v}")

        setattr(namespace, self.dest, values)
'''


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


def _download_debug_dataset(raw_dir: Path, enforce_download: bool) -> bool:
    print(
        "Download not intended, skipping (just checking if files exist locally) ..."
    )
    files = (
        "debug_all.csv",
        "debug_train.csv",
        "debug_val.csv",
        "debug_test.csv",
    )
    for file in files:
        if not (raw_dir / file).is_file():
            raise FileNotFoundError(f"File {file} not found in {raw_dir}")
    return False


def _prepare_debug_dataset(raw_dir: Path, prep_dir: Path) -> None:
    files = (
        "debug_all.csv",
        "debug_train.csv",
        "debug_val.csv",
        "debug_test.csv",
    )
    prep_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        # copy2(raw_dir / file, prep_dir / file)
        _cleanse_and_copy_data(raw_dir / file, prep_dir / file)


def _download_guacamol_dataset(raw_dir: Path, enforce_download: bool) -> bool:
    # TODO implement download from source (wget)
    print("Download not yet implemented, checking if files exist locally ...")
    files = (
        "guacamol_v1_all.smiles",
        "guacamol_v1_train.smiles",
        "guacamol_v1_valid.smiles",
        "guacamol_v1_test.smiles",
    )
    for file in files:
        if not (raw_dir / file).is_file():
            raise FileNotFoundError(f"File {file} not found in {raw_dir}")
    return False


def _prepare_guacamol_dataset(raw_dir: Path, prep_dir: Path) -> None:
    files = (
        "guacamol_v1_all.smiles",
        "guacamol_v1_train.smiles",
        "guacamol_v1_valid.smiles",
        "guacamol_v1_test.smiles",
    )
    prep_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        raw_file = raw_dir / file
        prep_file = (prep_dir / raw_file.name).with_suffix(".csv")
        # copy2(raw_file, prep_file)
        _cleanse_and_copy_data(raw_file, prep_file)


def _download_uspto_50k_dataset(raw_dir: Path, enforce_download: bool) -> bool:
    # TODO implement download from source (wget)
    # raw_url = "https://github.com/ml-jku/mhn-react/blob/main/data/USPTO_50k_MHN_prepro.csv.gz"
    print("Download not yet implemented, checking if file exist locally ...")
    file = "USPTO_50k_MHN_prepro.csv"
    if not (raw_dir / file).is_file():
        raise FileNotFoundError(f"File {file} not found in {raw_dir}")
    return False


def _prepare_uspto_50k_dataset(raw_dir: Path, prep_dir: Path) -> None:
    # raw_url = "https://github.com/ml-jku/mhn-react/blob/main/data/USPTO_50k_MHN_prepro.csv.gz"

    # Setup file, column, split names
    raw_file = raw_dir / "USPTO_50k_MHN_prepro.csv"
    files: dict[str, str] = {
        "known": "USPTO_50k_known.csv",  # "known" means in either in validation or test set (but not train set)
        "train": "USPTO_50k_train.csv",
        "valid": "USPTO_50k_val.csv",
        "test": "USPTO_50k_test.csv",
    }
    splits: tuple[str, ...] = ("train", "valid", "test")
    dirs = ["with_mapping", "without_mapping"]
    columns: list[str] = [
        "reaction_smarts_with_atom_mapping",
        "reaction_smarts_without_atom_mapping",
    ]

    # Read raw data, rename/append columns
    df: dict[str, pd.DataFrame] = dict()
    df["raw"] = pd.read_csv(raw_file, header=0)
    df["all"] = df["raw"][["id", "split", "prod_smiles", "reaction_smarts"]]
    df["all"].rename(
        columns={
            "id": "USPTO-50k_id",
            "prod_smiles": "product_smiles",
            "reaction_smarts": "reaction_smarts_with_atom_mapping",
        },
        inplace=True,
    )
    df["all"]["reaction_smarts_without_atom_mapping"] = remove_atom_mapping(
        df["all"]["reaction_smarts_with_atom_mapping"]
    )

    # Prepare "known" reactions from validation and test set (can include duplicates)
    df["known"] = df["all"][df["all"]["split"].isin(["valid", "test"])].copy()

    # Prepare train, validation, and test set and save in corresponding files
    for sub_dir, column in zip(dirs, columns):
        (prep_dir / sub_dir).mkdir(parents=True, exist_ok=True)
        file_name = prep_dir / sub_dir / files["known"]
        df["known"].to_csv(file_name, header=True, index=False)
        print(
            f"Save {len(df['known'])} known reactions (including duplicates) to {sub_dir}/{file_name.name} ..."
        )
        for split in splits:
            file_name = prep_dir / sub_dir / files[split]
            df[split] = df["all"][df["all"]["split"] == split][[column]].copy()
            # Remove duplicates
            len_before = len(df[split])
            df[split].drop_duplicates(inplace=True)
            len_after = len(df[split])
            print(
                f"Remove {len_before - len_after} duplicates from {split} set, {len_after} reactions left ..."
            )
            if split == "train":
                # Remove entries that are also in the "known" set
                len_before = len(df[split])
                df[split] = df[split][
                    ~df[split][column].isin(df["known"][column])
                ].copy()
                len_after = len(df[split])
                print(
                    f"Remove {len_before - len_after} reactions from train set that are also in the "
                    f"validation and/or test set, {len_after} rows left ..."
                )
            print(f"Save {split} reactions to {sub_dir}/{file_name.name} ...")
            df[split].to_csv(file_name, header=False, index=False)

        # Make sure that train set is not included in known set
        train_known_intersection = set(df["known"][column].values) & set(
            df["train"].iloc[:, 0].values
        )
        # print(len(train_known_intersection))
        assert (
            len(train_known_intersection) == 0
        ), "Train set includes reactions from validation and/or test set!"


def _download_zinc_dataset(raw_dir: Path, enforce_download: bool) -> bool:
    file = "zinc.tab"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if enforce_download:
        print(f"Enforcing download, deleting file {file} ...")
        (raw_dir / file).unlink()

    if (raw_dir / file).is_file():
        print(f"File {file} already exists in {raw_dir}")
        return False
    else:
        print(f"Downloading file {file} to {raw_dir} ...")
        _ = MolGen(
            name="ZINC",
            path=raw_dir,
            print_stats=False,
        )
        return True


def _prepare_zinc_dataset(raw_dir: Path, prep_dir: Path) -> None:
    file = "zinc.tab"
    prep_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / file
    prep_file = (prep_dir / raw_file.name).with_suffix(".csv")
    # copy2(raw_file, prep_file)
    _cleanse_and_copy_data(raw_file, prep_file)


def download_dataset(
    dataset: str, raw_dir: Path, enforce_download: bool = False
) -> bool:
    download_fns: dict[str, Callable[[Path, bool], bool]] = {
        "debug": _download_debug_dataset,
        "guacamol": _download_guacamol_dataset,
        "uspto50k": _download_uspto_50k_dataset,
        "zinc": _download_zinc_dataset,
    }

    download_fn = download_fns.get(dataset, None)
    if download_fn is None:
        raise ValueError(f"Invalid dataset: {dataset}")
    else:
        print(f"Downloading dataset {dataset} to {raw_dir} ...")
        return download_fn(raw_dir, enforce_download)


def prepare_dataset(dataset: str, raw_dir: Path, prep_dir: Path) -> None:
    prepare_fns: dict[str, Callable[[Path, Path], None]] = {
        "debug": _prepare_debug_dataset,
        "guacamol": _prepare_guacamol_dataset,
        "uspto50k": _prepare_uspto_50k_dataset,
        "zinc": _prepare_zinc_dataset,
    }

    prepare_fn = prepare_fns.get(dataset, None)
    if prepare_fn is None:
        raise ValueError(f"Invalid dataset: {dataset}")
    else:
        print(f"Preparing dataset {dataset} into {prep_dir} ...")
        return prepare_fn(raw_dir, prep_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare data for training of the huggingface model"
    )
    parser.add_argument(
        "dataset",
        type=str.lower,
        # action=DatasetAction,
        nargs="?",
        # const="all",
        default="all",
        choices=VALID_DATASETS,
        help="the dataset to prepare, default: '%(default)s' for all datasets",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=Path,
        default="../../../data",
        help="root of the data directory, default: '%(default)s'",
    )
    parser.add_argument(
        "-e",
        "--enforce_download",
        default=False,
        action="store_true",
        help="enforce downloading the dataset(s), default: '%(default)s'",
    )
    # parser.add_argument('-p', '--preprocess', default=True, action='store_true', help="preprocess the dataset(s)")

    args = parser.parse_args()

    if "all" in args.dataset:
        datasets = sorted(set(VALID_DATASETS) - {"all"})
    else:
        datasets = [args.dataset]

    data_dir = args.data_dir.resolve()

    for dataset in datasets:
        raw_dir = data_dir / "raw" / dataset
        download_dataset(dataset, raw_dir, args.enforce_download)
        prep_dir = data_dir / "prep" / dataset / "csv"
        prepare_dataset(dataset, raw_dir, prep_dir)

    print("Done!")


if __name__ == "__main__":
    main()
