# coding=utf-8
"""
Auto-Regressive Molecule and Reaction Template Generator
Causal language modeling (CLM) with a transformer decoder model
Author: Stephan Holzgruber
Student ID: K08608294
"""


# import operator
# import pickle
import re

# import reprlib
from collections.abc import (  # Callable,; Iterator,; MutableMapping,
    Iterable,
    Sequence,
)
from functools import cached_property, partial

# from importlib import reload
from multiprocessing import Pool

# from pathlib import Path
# from statistics import mean, median
from typing import Any, Optional, Union, overload

# from loguru import logger
from rdkit import Chem, rdBase  # type: ignore

# reload(molgen.config)
# reload(molgen.helpers)
# reload(molgen.tokenizer)
# from molgen.config import PathLike
from molreactgen.helpers import get_num_workers

# import molgen.config
# import molgen.helpers
# import molgen.tokenizer


# from molgen.tokenizer import Tokenizer, __tokenizer_version__


def _canonicalize_molecule(
    molecule: str, is_smarts: bool, remove_atom_mapping: bool, strict: bool
) -> Optional[str]:
    if is_smarts:
        mol = Chem.MolFromSmarts(molecule)
    else:
        mol = Chem.MolFromSmiles(molecule)

    if mol is None:
        if strict:
            raise ValueError(
                f"{molecule} is not a valid {'SMARTS' if is_smarts else 'SMILES'} string"
            )
        else:
            canonical_molecule = None
    else:
        if remove_atom_mapping:
            for atom in mol.GetAtoms():
                if atom.HasProp("molAtomMapNumber"):
                    atom.ClearProp("molAtomMapNumber")
        # The try/except block is here to catch an issue with RDKit (which might be my issue)
        try:
            canonical_molecule = Chem.MolToSmiles(mol)
        except RuntimeError as e:
            if not strict and str(e).startswith("Pre-condition Violation"):
                canonical_molecule = None
            else:
                raise RuntimeError(e) from e
    return canonical_molecule


def canonicalize_smiles(  # TODO refactor into canonicalize_molecule()
    molecule: str,
    strict: bool = True,
    double_check: bool = False,
) -> Optional[str]:

    return canonicalize_molecule(
        molecule,
        is_smarts=False,
        remove_atom_mapping=False,
        strict=strict,
        double_check=double_check,
    )


def canonicalize_molecule(
    molecule: str,
    *,
    is_smarts: bool = False,
    remove_atom_mapping: bool = False,
    strict: bool = True,
    double_check: bool = False,
) -> Optional[str]:

    molecule = str(molecule)
    is_smarts = bool(is_smarts)
    remove_atom_mapping = bool(remove_atom_mapping)
    strict = bool(strict)
    double_check = bool(double_check)

    rdBase.DisableLog("rdApp.error")
    canonical_molecule = _canonicalize_molecule(
        molecule,
        is_smarts=is_smarts,
        remove_atom_mapping=remove_atom_mapping,
        strict=strict,
    )
    if double_check and canonical_molecule is not None:
        canonical_molecule = _canonicalize_molecule(
            canonical_molecule,
            is_smarts=is_smarts,
            remove_atom_mapping=remove_atom_mapping,
            strict=strict,
        )
    rdBase.EnableLog("rdApp.error")

    return canonical_molecule


def canonicalize_molecules(
    molecules: Union[str, Sequence[str]],
    num_workers: Optional[int] = None,
    strict: bool = False,
    double_check: bool = False,
) -> list[Optional[str]]:

    # double_check parameter is required due to a bug in RDKit
    # see https://github.com/rdkit/rdkit/issues/5455

    if not isinstance(molecules, Iterable):
        molecules = [molecules]

    if not all([isinstance(molecule, str) for molecule in molecules]):
        raise TypeError("All molecules must be of type str")

    if num_workers is None:
        num_workers = get_num_workers()
    else:
        num_workers = int(num_workers)

    chunk_size = max(
        len(molecules) // (num_workers * 4), 1
    )  # heuristics; make chunk size smaller than necessary
    _canonicalize_fn = partial(
        canonicalize_smiles, strict=strict, double_check=double_check
    )

    pool = Pool(num_workers)
    canonical_smiles_generator = pool.imap(
        _canonicalize_fn, molecules, chunksize=chunk_size
    )
    pool.close()
    pool.join()
    canonical_smiles: list[Optional[str]] = list(canonical_smiles_generator)

    return canonical_smiles


# Taken from https://github.com/ml-jku/mhn-react/blob/main/mhnreact/molutils.py
# Modified to work within this project


# This let's mypy complain about
# "Overloaded function signatures 1 and 2 overlap with incompatible return types"
# see also https://github.com/python/mypy/issues/11001
@overload
def remove_atom_mapping(smarts: str) -> str:  # type: ignore
    ...


@overload
def remove_atom_mapping(smarts: Sequence[str]) -> list[str]:
    ...


def remove_atom_mapping(
    smarts: Union[str, Sequence[str]]
) -> Union[str, list[str]]:
    """Removes a number after a ':'"""
    if isinstance(smarts, str) or not isinstance(smarts, Iterable):
        smarts = [smarts]

    smarts = [re.sub(r":\d+", "", str(s)) for s in smarts]

    if len(smarts) == 1:
        return smarts[0]
    else:
        return smarts


def canonicalize_template(
    smarts: str, strict: bool = True, double_check: bool = False
) -> Optional[str]:
    smarts = str(smarts)
    # order the list of smiles + canonicalize it
    results = []
    for reaction_parts in smarts.split(">>"):
        parts: list[str] = reaction_parts.split(".")
        parts_canon: list[Optional[str]] = [
            canonicalize_molecule(
                part,
                is_smarts=True,
                remove_atom_mapping=True,
                strict=strict,
                double_check=double_check,
            )
            for part in parts
        ]
        if not all(parts_canon):
            return None
        parts_canon.sort()
        # mypy complains about the following line, namely that parts_canon can be None
        # but this is not the case, because we checked that all parts_canon are not None
        results.append(".".join(parts_canon))  # type: ignore

    return ">>".join(results)


class Reaction:
    def __init__(
        self,
        reaction_smarts: str,
        *,
        id_: Optional[str] = None,
        feasible: bool = False,
        product: Optional[str] = None,
    ) -> None:

        self.reaction_smarts = str(reaction_smarts)
        self.id = id_
        self.feasible = bool(feasible)
        self.product = product
        self.works_with: Optional[str] = None
        # self.reactants: Optional[  # TODO improve type hints once I know what this looks like
        #     str
        # ] = None

    @property
    def reaction_smarts(self) -> str:
        return self._reaction_smarts

    @reaction_smarts.setter
    def reaction_smarts(self, value: str) -> None:
        if hasattr(self, "_reaction_smarts"):
            raise AttributeError(
                "reaction_smarts is already set, can not be changed"
            )

        # noinspection PyAttributeOutsideInit
        self._reaction_smarts = str(value)

    @cached_property
    def reaction_smarts_without_atom_mapping(
        self,
    ) -> str:
        return remove_atom_mapping(self.reaction_smarts)

    @cached_property
    def reaction_smarts_canonicalized(self) -> Optional[str]:
        return canonicalize_template(
            self.reaction_smarts, strict=False, double_check=True
        )

    @property
    def valid(self) -> bool:
        return self.reaction_smarts_canonicalized is not None

    @property
    def invalid(self) -> bool:
        return not self.valid

    def is_similar_to(
        self, other: "Reaction", criterion: str = "atom_mapping"
    ) -> bool:
        criterion = str(criterion).lower()
        if isinstance(other, Reaction):
            if criterion == "atom_mapping":
                return (
                    self.reaction_smarts_without_atom_mapping
                    == other.reaction_smarts_without_atom_mapping
                )
            elif criterion == "canonical":
                return (
                    self.reaction_smarts_canonicalized
                    == other.reaction_smarts_canonicalized
                )
            else:
                raise ValueError(f"Unknown criterion: {criterion}")
        else:
            return False

    # TODO Think about what equality means for Reaction
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Reaction):
            return self.reaction_smarts == other.reaction_smarts
            # return (
            #     self.reaction_smarts_without_atom_mapping
            #     == other.reaction_smarts_without_atom_mapping
            # )
        else:
            return NotImplemented

    def __hash__(self) -> int:
        return hash(self.reaction_smarts)

    def __len__(self) -> int:
        return len(str(self))

    def __str__(self) -> str:
        return self.reaction_smarts

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return (
            f"{class_name}"
            f"(reaction_smarts={self.reaction_smarts.__repr__()}, "
            f"id={self.id})"
        )


class Molecule:
    def __init__(
        self,
        smiles: str,
        *,
        id_: Optional[str] = None,
        notation: str = "SMILES",
    ) -> None:

        self.smiles: str = str(smiles)
        self.id = id_
        # Currently only SMILES notation is supported
        # For other notations, e.g. SELFIES, the code needs to be amended
        if str(notation).upper() == "SMILES":
            self.notation: str = "SMILES"
        else:
            raise ValueError(f"{notation} is not a valid molecule notation")

    @property
    def smiles(self) -> str:
        return self._smiles

    @smiles.setter
    def smiles(self, value: str) -> None:
        if hasattr(self, "_smiles"):
            raise AttributeError("smiles is already set, can not be changed")

        # noinspection PyAttributeOutsideInit
        self._smiles = str(value)

    @cached_property
    def canonical_smiles(
        self,
    ) -> Optional[str]:
        return canonicalize_smiles(self.smiles, strict=False)

    @property
    def valid(self) -> bool:
        return self.canonical_smiles is not None

    @property
    def invalid(self) -> bool:
        return not self.valid

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Molecule):
            self_smiles = self.canonical_smiles if self.valid else self.smiles
            other_smiles = (
                other.canonical_smiles if other.valid else other.smiles
            )
            return self_smiles == other_smiles
        else:
            return NotImplemented

    def __hash__(self) -> int:
        return hash(self.canonical_smiles if self.valid else self.smiles)

    def __len__(self) -> int:
        return len(str(self))

    def __str__(self) -> str:
        return (
            self.canonical_smiles
            if self.canonical_smiles
            is not None  # same as self.valid but mypy complains about return type
            else self.smiles
        )

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return (
            f"{class_name}"
            f"(smiles={self.smiles.__repr__()}, "
            f"id={self.id.__repr__()}, "
            f"notation={self.notation.__repr__()})"
        )


# class MoleculeStore:
#     def __init__(
#         self,
#         *,
#         name: str,
#         split: str = "",
#         description: str = "",
#         source: str = "",
#         id_format: str = "07d",
#         molecules: Optional[Sequence[Molecule]] = None,
#         csv_file_path: Optional[PathLike] = None,
#     ) -> None:
#
#         logger.debug(f"Initialize MoleculeStore '{name}', split '{split}'")
#         self.name = str(name)
#         self.split = str(split)
#         self.description = str(description)
#         self.source = str(source)
#         self.id_format = str(id_format)
#         self.molecules: list[Molecule] = []
#
#         if (molecules is not None) and (csv_file_path is not None):
#             raise ValueError(
#                 "Either molecules or csv_file_path can be given, but not both"
#             )
#         if molecules is not None:
#             self.molecules = list(molecules)
#         elif csv_file_path is not None:
#             self.molecules = self._load_molecules_from_csv(csv_file_path)
#         else:
#             pass
#
#         self._molecules_added: int = 0
#         self.molecule_version: str = __molecule_version__
#         self.tokenizer_version: str = __tokenizer_version__
#         self._columns: list[str] = [
#             "id",
#             "source_id",
#             "smiles",
#             "canonical_smiles",
#         ]
#         self._tokenizer: Optional[Tokenizer] = None
#
#     @staticmethod
#     def _load_molecules_from_csv(csv_file_path: PathLike) -> list[Molecule]:
#         logger.debug(f"Loading molecules from '{csv_file_path}'")
#         df = pd.read_csv(csv_file_path, names=["smiles"], header=None)
#         first_row = df["smiles"][0]
#         if (
#             first_row.upper() == "SMILES"
#             or canonicalize_smiles(first_row, strict=False, double_check=False)
#             is None
#         ):
#             df = df.iloc[1:].reset_index()
#         molecules: list[Molecule] = [Molecule(smiles=s) for s in df["smiles"]]
#
#         return molecules
#
#     def add_molecules(
#         self,
#         molecules: Union[Molecule, Sequence[Molecule]],
#         *,
#         check_duplicate: bool = False,
#         raise_exception: bool = False,
#     ) -> int:
#
#         if not isinstance(molecules, Iterable):
#             molecules = [molecules]
#
#         # TODO This instance checks fails, probably due to the reloading of molecule.py
#         # Once the code is stable, remove the reloads and de-comment the check
#         # Without this check, a unit test fails
#         # if not all([isinstance(molecule, Molecule) for molecule in molecules]):
#         #     raise TypeError("All molecules must be of type Molecule")
#
#         if len(molecules) > 1:
#             logger.debug(
#                 f"Add {len(molecules):,d} molecules to MoleculeStore '{self.name}', split '{self.split}'"
#             )
#
#         num_current_molecules = len(self.molecules)
#         self._molecules_added += len(molecules)
#         mol_id = num_current_molecules
#
#         # TODO The duplicate check is very slow, do it differently? in parallel? at least with a progress bar?
#         # OR DO a quick check first, i.e. len(list) == len(set(list))
#
#         if check_duplicate:
#             logger.debug(
#                 "Checking for duplicate molecules (this might take a while)"
#             )
#
#         for mol in molecules:
#             if check_duplicate:
#                 if mol in self.molecules:
#                     if raise_exception:
#                         raise ValueError(f"{mol} is a duplicate molecule")
#                     continue
#
#             mol.id = self._id_formatter(mol_id)
#             mol_id += 1
#             self.molecules.append(mol)
#
#         return len(self.molecules) - num_current_molecules
#
#     @property
#     def all_smiles(self) -> list[str]:
#         return [
#             m.canonical_smiles if m.canonical_smiles is not None else m.smiles
#             for m in self.molecules
#         ]
#
#     @property
#     def all_smiles_encoded(self) -> Optional[list[Any]]:
#         self._check_tokenizer()
#         assert self._tokenizer is not None  # make mypy happy
#         # noinspection PyProtectedMember
#         return [m._encoding[self._tokenizer.name] for m in self.molecules]
#
#     @property
#     def tokenizer(self) -> Optional[Tokenizer]:
#         return self._tokenizer
#
#     @tokenizer.setter
#     def tokenizer(self, value: Tokenizer) -> None:
#         if not isinstance(value, molgen.tokenizer.Tokenizer):
#             raise TypeError("tokenizer must be of type Tokenizer")
#         if not (hasattr(value, "name") and isinstance(value.name, str)):
#             raise TypeError(
#                 "tokenizer must have a 'name' property of type 'str'"
#             )
#         # The mypy checks against Callable fail, see mypy issues #6680 and #6864
#         if not (hasattr(value, "encode") and isinstance(value.encode, Callable)):  # type: ignore
#             raise TypeError("tokenizer must have an 'encode()' Callable")
#         if not (hasattr(value, "decode") and isinstance(value.decode, Callable)):  # type: ignore
#             raise TypeError("tokenizer must have a 'decode()' Callable")
#         if not (
#             hasattr(value, "encoder_vocabulary")
#             and isinstance(value.encoder_vocabulary, MutableMapping)
#         ):
#             raise TypeError(
#                 "tokenizer must have an 'encoder_vocabulary' property of type 'MutableMapping'"
#             )
#         if not (
#             hasattr(value, "decoder_vocabulary")
#             and isinstance(value.decoder_vocabulary, MutableMapping)
#         ):
#             raise TypeError(
#                 "tokenizer must have a 'decoder_vocabulary' property of type 'MutableMapping'"
#             )
#
#         self._tokenizer = value
#
#     def _check_tokenizer(self) -> None:
#         if self._tokenizer is None:
#             raise RuntimeError("No tokenizer set")
#
#     def encode_molecules(self) -> None:
#         self._check_tokenizer()
#         assert self._tokenizer is not None  # make mypy happy
#         smiles = self.all_smiles
#         smiles_encoded = self._tokenizer.encode(smiles)
#         for mol, smi_enc in zip(self.molecules, smiles_encoded):
#             # noinspection PyProtectedMember
#             mol._encoding[self._tokenizer.name] = smi_enc
#
#         # This is meant as a safety check, but it's not really necessary
#         # TODO delete this after debugging
#         smiles_decoded = self._tokenizer.decode(smiles_encoded)
#         if smiles != smiles_decoded:
#             raise RuntimeError(
#                 "Encoding and decoding of smiles failed (are different)"
#             )
#         # for i in range(len(smiles)):
#         #     if smiles[i] != smiles_decoded[i]:
#         #         print(f"i: {i}, {smiles[i]} != {smiles_decoded[i]}")
#
#     @property
#     def vocabulary(self) -> dict[str, int]:
#         return self.encoder_vocabulary
#
#     @property
#     def encoder_vocabulary(self) -> dict[str, int]:
#         self._check_tokenizer()
#         assert self._tokenizer is not None  # make mypy happy
#         return self._tokenizer.encoder_vocabulary
#
#     @property
#     def decoder_vocabulary(self) -> dict[int, str]:
#         self._check_tokenizer()
#         assert self._tokenizer is not None  # make mypy happy
#         return self._tokenizer.decoder_vocabulary
#
#     @property
#     def vocab_size(self) -> int:
#         self._check_tokenizer()
#         assert self._tokenizer is not None  # make mypy happy
#         return self._tokenizer.vocab_size
#
#     def _id_formatter(self, n: int) -> str:
#         n = int(n)
#         return f"{self.name}{n:{self.id_format}}"
#
#     def save(
#         self,
#         *,
#         pickle_file_path: Optional[PathLike] = None,
#         csv_file_path: Optional[PathLike] = None,
#         # vocab_file_path_template: Optional[PathLike] = None,
#     ) -> None:
#
#         if pickle_file_path is not None:
#             # Save the encoded MoleculeStore to a pickle file
#             pickle_file_path = Path(pickle_file_path).resolve()
#             pickle_file_path.parent.mkdir(parents=True, exist_ok=True)
#             logger.debug(
#                 f"Save MoleculeStore (encoded) '{self.name}' with {len(self.molecules):,d} entries "
#                 f"to '{pickle_file_path}'"
#             )
#             with open(pickle_file_path, "wb") as f:
#                 pickle.dump(self, f, protocol=5)
#
#         if csv_file_path is not None:
#             # Save the list of molecules to a csv file
#             csv_file_path = Path(csv_file_path).resolve()
#             csv_file_path.parent.mkdir(parents=True, exist_ok=True)
#             logger.debug(
#                 f"Save MoleculeStore (molecule list) '{self.name}' with {len(self.molecules):,d} entries "
#                 f"to '{csv_file_path}'"
#             )
#
#             mols = [
#                 [
#                     m.id,
#                     m.source_id,
#                     m.smiles,
#                     m.canonical_smiles,
#                 ]
#                 for m in self
#             ]
#             df = pd.DataFrame(mols, columns=self._columns)
#             df.to_csv(csv_file_path, index=False)
#
#     @classmethod
#     def from_file(cls, pickle_file_path: PathLike):
#         pickle_file_path = Path(pickle_file_path).resolve()
#         logger.debug(f"Load MoleculeStore from file '{pickle_file_path}'")
#         if not pickle_file_path.is_file():
#             raise FileNotFoundError(f"File {pickle_file_path} does not exist")
#
#         with open(pickle_file_path, "rb") as f:
#             molstore = pickle.load(f)
#
#         logger.debug(
#             f"Loaded MoleculeStore '{molstore.name}', split '{molstore.split}' "
#             f"with {len(molstore.molecules):,d} molecules"
#         )
#
#         return molstore
#
#     @property
#     def unique_rate(self) -> float:
#         if self._molecules_added == 0.0:
#             return 0.0
#         else:
#             return len(self.molecules) / self._molecules_added
#
#     @property
#     def min_len(self) -> int:
#         return int(min(len(m) for m in self.molecules))
#
#     @property
#     def mean_len(self) -> float:
#         return float(mean(len(m) for m in self.molecules))
#
#     @property
#     def max_len(self) -> int:
#         return int(max(len(m) for m in self.molecules))
#
#     @property
#     def median_len(self) -> float:
#         return float(median(len(m) for m in self.molecules))
#
#     def __contains__(self, item: Any) -> bool:
#         if isinstance(item, Molecule):
#             return item in self.molecules
#         else:
#             return False
#
#     @overload
#     def __getitem__(self, key: int) -> Molecule:
#         ...
#
#     @overload
#     def __getitem__(self, key: slice) -> "MoleculeStore":
#         ...
#
#     def __getitem__(
#         self, key: Union[int, slice]
#     ) -> Union[Molecule, "MoleculeStore"]:
#         if isinstance(key, slice):
#             cls = type(self)
#             new_molstore = cls(
#                 name=self.name,
#                 split=self.split,
#                 description=self.description,
#                 source=self.source,
#                 id_format=self.id_format,
#                 molecules=self.molecules[key],
#             )
#             if self._tokenizer is not None:
#                 new_molstore.tokenizer = self._tokenizer
#                 new_molstore.encode_molecules()
#             return new_molstore
#         else:
#             index = operator.index(key)
#             return self.molecules[index]
#
#     def __iter__(self) -> Iterator[Molecule]:
#         return iter(self.molecules)
#
#     def __len__(self) -> int:
#         return len(self.molecules)
#
#     def __repr__(self) -> str:
#         # name = self.name.__repr__() if isinstance(self.name, str) else self.name
#         # description = (
#         #     self.description.__repr__()
#         #     if isinstance(self.description, str)
#         #     else self.description
#         # )
#         # source = (
#         #     self.source.__repr__()
#         #     if isinstance(self.source, str)
#         #     else self.source
#         # )
#         # id_format = (
#         #     self.id_format.__repr__()
#         #     if isinstance(self.id_format, str)
#         #     else self.id_format
#         # )
#         # version = (
#         #     self.version.__repr__()
#         #     if isinstance(self.version, str)
#         #     else self.version
#         # )
#         class_name = type(self).__name__
#         molecules = reprlib.repr(self.molecules)
#         return (
#             f"{class_name}"
#             f"(name={self.name.__repr__()}, "
#             f"split={self.split.__repr__()}, "
#             f"description={self.description.__repr__()}, "
#             f"source={self.source.__repr__()}, "
#             f"id_format={self.id_format.__repr__()}, "
#             f"molecules={molecules!r})"
#         )
