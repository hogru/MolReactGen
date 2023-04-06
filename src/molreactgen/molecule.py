# coding=utf-8
# src/molreactgen/molecule.py
"""Helper classes and functions for working with molecules and reaction templates.

Classes:
    Molecule:
        A class for working with molecules.
    Reaction:
        A class for working with reaction templates.

Functions:
    canonicalize_molecule:
        Canonicalize a single molecule.
    canonicalize_smiles:
        Canonicalize a single molecule. For compatibility reasons, this is a wrapper around canonicalize_molecule().
    canonicalize_molecules:
        Canonicalize a list of molecules with multiprocessing.
    canonicalize_template:
        Canonicalize a reaction template.
    remove_atom_mapping:
        Remove atom mapping from a reaction template.
"""

import logging
import re
from collections.abc import Iterable, Sequence  # Callable,; Iterator,; MutableMapping,
from functools import cached_property, partial
from multiprocessing import Pool
from typing import Any, Literal, Optional, Union, overload

from rdkit import Chem, rdBase  # type: ignore

from molreactgen.helpers import get_num_workers

logger = logging.getLogger(__name__)


# This could/should be refactored in the following ways:
# - allow canonicalize_template() to take a list of templates and call canonicalize_molecules() internally
# But, worth it? only used in one place, the Reaction class, which is a single template by definition
# - make canonicalize_molecules() aware of the smarts parameter
# - replace calls to canonicalize_smiles with canonicalize_molecule()


def _canonicalize_molecule(
    molecule: str, is_smarts: bool, remove_atom_mapping: bool, strict: bool
) -> Optional[str]:
    """Canonicalize a single molecule.

    Args:
        molecule: A SMILES or SMARTS string.
        is_smarts: If True, the molecule is a SMARTS string. If False, the molecule is a SMILES string.
        remove_atom_mapping: If True, remove atom mapping from the molecule.
        strict: If True, raise an error if the molecule is not valid. If False, return None.

    Returns:
        The canonical SMILES string for the molecule. If the molecule is not valid, return None.

    Raises:
        ValueError: If the molecule is not valid and strict is True.
    """

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
    *,
    strict: bool = False,
    double_check: bool = False,
) -> Optional[str]:
    """Canonicalize a single molecule. For compatibility reasons, this is a wrapper around canonicalize_molecule().

    Args:
        molecule: A SMILES string.
        strict: If True, raise an error if the molecule is not valid. If False, return None.
        double_check: If True, canonicalize the molecule twice and check that the results are the same.

    Returns:
        The canonical SMILES string for the molecule. If the molecule is not valid, return None.

    Raises:
        ValueError: If the molecule is not valid and strict is True.
    """

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
    """Canonicalize a single molecule.

    Args:
        molecule: A SMILES or SMARTS string.
        is_smarts: If True, the molecule is a SMARTS string. If False, the molecule is a SMILES string.
        remove_atom_mapping: If True, remove atom mapping from the molecule.
        strict: If True, raise an error if the molecule is not valid. If False, return None.
        double_check: If True, canonicalize the molecule twice and check that the results are the same.

    Returns:
        The canonical SMILES string for the molecule. If the molecule is not valid, return None.

    Raises:
        ValueError: If the molecule is not valid and strict is True.
    """

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
    *,
    num_workers: Optional[int] = None,
    strict: bool = False,
    double_check: bool = False,
) -> list[Optional[str]]:
    """Canonicalize a list of molecules.

    Args:
        molecules: A list of SMILES strings.
        num_workers: The number of workers to use. If None, determine a "reasonable" number of workers.
        strict: If True, raise an error if the molecule is not valid. If False, return None.
        double_check: If True, canonicalize the molecule twice and check that the results are the same.

    Returns:
        A list of canonical SMILES strings for the molecules. If a molecule is not valid, return None.

    Raises:
        ValueError: If the molecule is not valid and strict is True.
    """

    # double_check parameter is required due to a bug in RDKit
    # see https://github.com/rdkit/rdkit/issues/5455

    if (
        isinstance(molecules, str)
        or not isinstance(molecules, Iterable)
        or not all(isinstance(molecule, str) for molecule in molecules)
    ):
        raise TypeError("molecules must be an iterable of strings")

    num_workers = get_num_workers() if num_workers is None else int(num_workers)

    # Canonicalize molecules in parallel
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
# noinspection PyMissingOrEmptyDocstring
@overload
def remove_atom_mapping(smarts: str) -> str:  # type: ignore
    ...


# noinspection PyMissingOrEmptyDocstring
@overload
def remove_atom_mapping(smarts: Sequence[str]) -> list[str]:
    ...


def remove_atom_mapping(smarts: Union[str, Sequence[str]]) -> Union[str, list[str]]:
    """Remove atom mapping (a number after a ':') from a SMARTS string or list of SMARTS strings.

    Args:
        smarts: A SMARTS string or list of SMARTS strings.

    Returns:
        A SMARTS string or list of SMARTS strings with atom mapping removed.
    """

    if isinstance(smarts, str) or not isinstance(smarts, Iterable):
        smarts = [smarts]

    if not all(isinstance(s, str) for s in smarts):
        raise TypeError("All smarts must be of type str")

    smarts = [re.sub(r":\d+", "", str(s)) for s in smarts]

    return smarts[0] if len(smarts) == 1 else smarts


def canonicalize_template(
    smarts: str, *, strict: bool = False, double_check: bool = False
) -> Optional[str]:
    """Canonicalize a template.

    Remove atom mappings, canonicalize reactants and products, and sort them.

    Args:
        smarts: A SMARTS string.
        strict: If True, raise an error if the template is not valid. If False, return None.
        double_check: If True, canonicalize the template parts twice and check that the results are the same.

    Returns:
        The canonical SMARTS string for the template. If the template is not valid, return None.

    Raises:
        ValueError: If the template is not valid and strict is True.
    """

    # this is faster than remove_atom_mapping via rdkit, so don't use it below
    smarts = remove_atom_mapping(smarts)
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
    """A reaction template.

    Attributes:
        reaction_smarts: The reaction template SMARTS string.
        split: Split the reaction template originates from.
        id: The reaction template ID.
        product: The product of the reaction template.
        feasible: Whether the reaction template is feasible.

    Methods:
        reaction_smarts_without_atom_mapping: The reaction template SMARTS string without atom mapping.
        reaction_smarts_canonicalized: The canonical reaction template SMARTS string.
        valid: Whether the reaction template is valid.
        invalid: Whether the reaction template is invalid.
        is_similar_to: Check if the reaction template is similar to another reaction template.
    """

    def __init__(
        self,
        reaction_smarts: str,
        *,
        split: Optional[Literal["valid", "test"]] = None,
        id_: Optional[str] = None,
        product: Optional[str] = None,
        feasible: bool = False,
    ) -> None:
        """Initialize the Reaction object.

        Args:
            reaction_smarts: The reaction template SMARTS string.
            split: Split the reaction template originates from. Defaults to None.
            id_: The reaction template ID. Defaults to None.
            product: The product of the reaction template. Defaults to None.
            feasible: Whether the reaction template is feasible. Defaults to False.
        """

        self.reaction_smarts = reaction_smarts
        self.split = str(split).lower() if split is not None else None
        self.id = id_
        self.product = product
        self.feasible = feasible
        self.works_with: Optional[str] = None
        self.num_works_with: int = 0
        self.in_val_set: bool = False
        self.in_test_set: bool = False

    # noinspection PyMissingOrEmptyDocstring
    @property
    def reaction_smarts(self) -> str:
        return self._reaction_smarts

    @reaction_smarts.setter
    def reaction_smarts(self, value: str) -> None:
        if hasattr(self, "_reaction_smarts"):
            raise AttributeError("reaction_smarts is already set, can not be changed")

        # noinspection PyAttributeOutsideInit
        self._reaction_smarts = value

    # noinspection PyMissingOrEmptyDocstring
    @cached_property
    def reaction_smarts_without_atom_mapping(
        self,
    ) -> str:
        return remove_atom_mapping(self.reaction_smarts)

    # noinspection PyMissingOrEmptyDocstring
    @cached_property
    def reaction_smarts_canonicalized(self) -> Optional[str]:
        return canonicalize_template(
            self.reaction_smarts, strict=False, double_check=True
        )

    # noinspection PyMissingOrEmptyDocstring
    @property
    def valid(self) -> bool:
        return self.reaction_smarts_canonicalized is not None

    # noinspection PyMissingOrEmptyDocstring
    @property
    def invalid(self) -> bool:
        return not self.valid

    def is_similar_to(self, other: "Reaction", criterion: str = "atom_mapping") -> bool:
        """Whether the reaction template is similar to another reaction template.

        Similar in this context means that the reaction smarts either have
        - the same form without atom mapping (criterion="atom_mapping")
        - the same canonical form (criterion="canonical"); stricter than the above

        Args:
            other: The other reaction template.
            criterion: The criterion to use for similarity. Defaults to "atom_mapping".

        Returns:
            Whether the reaction template is similar to another reaction template.

        Raises:
            ValueError: If the criterion is unknown.
        """

        criterion = criterion.lower()
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

    # might further think about what equality means for Reaction
    # That would be less strict
    # return (
    #     self.reaction_smarts_without_atom_mapping
    #     == other.reaction_smarts_without_atom_mapping
    # )
    def __eq__(self, other: Any) -> bool:
        return (
            self.reaction_smarts == other.reaction_smarts
            if isinstance(other, Reaction)
            else NotImplemented  # type: ignore
        )

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
            f"id={self.id.__repr__()})"
        )


class Molecule:
    """A molecule.

    During generation/evaluation of molecules, the following attributes are set:
    novel, unique (depend on other generated molecules).

    Attributes:
        smiles: The molecule SMILES string.
        id: The molecule ID.
        notation: The molecule notation format.
        novel: Whether the molecule is novel.
        unique: Whether the molecule is unique.

    Methods:
        canonical_smiles: The canonical SMILES string.
        valid: Whether the molecule is valid.
        invalid: Whether the molecule is invalid.

    """

    def __init__(
        self,
        smiles: str,
        *,
        id_: Optional[str] = None,
        notation: str = "SMILES",
    ) -> None:
        """Initialize the Molecule object.

        Args:
            smiles: The molecule SMILES string.
            id_: The molecule ID. Defaults to None.
            notation: The molecule notation format. Defaults to 'SMILES'.

        Raises:
            ValueError: If the notation is unknown (at the moment only 'SMILES' is supported).
        """

        self.smiles = smiles
        self.id = id_
        # Currently only SMILES notation is supported
        # For other notations, e.g. SELFIES, the code needs to be amended
        if notation.upper() == "SMILES":
            self.notation: str = "SMILES"
        else:
            raise ValueError(f"{notation} is not a valid molecule notation")

        self.unique: bool = False
        self.novel: bool = False

    # noinspection PyMissingOrEmptyDocstring
    @property
    def smiles(self) -> str:
        return self._smiles

    @smiles.setter
    def smiles(self, value: str) -> None:
        if hasattr(self, "_smiles"):
            raise AttributeError("smiles is already set, can not be changed")

        # noinspection PyAttributeOutsideInit
        self._smiles = value

    # noinspection PyMissingOrEmptyDocstring
    @cached_property
    def canonical_smiles(
        self,
    ) -> Optional[str]:
        return canonicalize_smiles(self.smiles, strict=False, double_check=True)

    # noinspection PyMissingOrEmptyDocstring
    @property
    def valid(self) -> bool:
        return self.canonical_smiles is not None

    # noinspection PyMissingOrEmptyDocstring
    @property
    def invalid(self) -> bool:
        return not self.valid

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Molecule):
            return NotImplemented

        self_smiles = self.canonical_smiles if self.valid else self.smiles
        other_smiles = other.canonical_smiles if other.valid else other.smiles
        return self_smiles == other_smiles

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
