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
from typing import Any, Literal, Optional, Union, overload

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

# TODO This could/should be refactored in the following ways:
# - allow canonicalize_template() to take a list of templates and call canonicalize_molecules() internally
# But, worth it? only used in one place, the Reaction class, which is a single template by definition
# - make canonicalize_molecules() aware of the smarts parameter
# - replace calls to canonicalize_smiles with canonicalize_molecule()


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


def canonicalize_smiles(
    molecule: str,
    strict: bool = False,
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
    strict: bool = False,
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
    molecules: Sequence[str],
    num_workers: Optional[int] = None,
    strict: bool = False,
    double_check: bool = False,
) -> list[Optional[str]]:

    # double_check parameter is required due to a bug in RDKit
    # see https://github.com/rdkit/rdkit/issues/5455

    if (
        isinstance(molecules, str)
        or not isinstance(molecules, Iterable)
        or not all([isinstance(molecule, str) for molecule in molecules])
    ):
        raise TypeError("molecules must be an iterable of strings")

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

    if not all([isinstance(s, str) for s in smarts]):
        raise TypeError("All smarts must be of type str")

    smarts = [re.sub(r":\d+", "", str(s)) for s in smarts]

    if len(smarts) == 1:
        return smarts[0]
    else:
        return smarts


def canonicalize_template(
    smarts: str, strict: bool = False, double_check: bool = False
) -> Optional[str]:
    # this is faster than remove_atom_mapping via rdkit, so don't use it below
    smarts = remove_atom_mapping(str(smarts))
    # order the list of smiles + canonicalize it
    results = []
    for reaction_parts in smarts.split(">>"):
        parts: list[str] = reaction_parts.split(".")
        parts_canon: list[Optional[str]] = [
            canonicalize_molecule(
                part,
                is_smarts=True,
                remove_atom_mapping=False,
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
        split: Optional[Literal["valid", "test"]] = None,
        id_: Optional[str] = None,
        product: Optional[str] = None,
        feasible: bool = False,
    ) -> None:

        self.reaction_smarts = str(reaction_smarts)
        self.split = str(split).lower() if split is not None else None
        self.id = str(id_)
        self.product = str(product)
        self.feasible = bool(feasible)
        self.works_with: Optional[str] = None
        self.num_works_with: int = 0
        self.in_val_set: bool = False
        self.in_test_set: bool = False
        # self.reactants: Optional[str] = None

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
            f"split={self.split.__repr__()}, "
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
        return canonicalize_smiles(self.smiles, strict=False, double_check=True)

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
