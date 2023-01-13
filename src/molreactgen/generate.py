# coding=utf-8
"""
Auto-Regressive Molecule and Reaction Template Generator
Causal language modeling (CLM) with a transformer decoder model
Author: Stephan Holzgruber
Student ID: K08608294
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd  # type: ignore
from loguru import logger
from rdchiral.main import (  # type: ignore
    rdchiralReactants,
    rdchiralReaction,
    rdchiralRun,
)
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    pipeline,
)

from molreactgen.helpers import Counter, configure_logging
from molreactgen.molecule import Molecule, Reaction

# Global variables, defaults
VALID_GENERATION_MODES = [
    "smiles",
    "smarts",
]  # TODO Type hinting with Literal?!
DEFAULT_OUTPUT_FILE_PATH = (
    f"../../data/generated/{datetime.now():%Y-%m-%d_%H-%M}_generated.csv"
)
DEFAULT_NUM_TO_GENERATE: int = 1000
MIN_NUM_TO_GENERATE: int = 20
# TODO make dependant on generation mode
# DEFAULT_SMILES_MAX_LENGTH: int = 128
# DEFAULT_SMARTS_MAX_LENGTH: int = 896
DEFAULT_TOP_P: float = 0.95  # not used yet
DEFAULT_NUM_BEAMS: int = 5  # not used yet
DEFAULT_EARLY_STOPPING: bool = True  # not used yet


def load_existing_molecules(
    file_path: Path,
) -> list[Molecule]:

    df: pd.DataFrame = pd.read_csv(file_path, header=None)
    molecules: list[Molecule] = [Molecule(row) for row in df[0]]

    return molecules


def load_existing_reaction_templates(
    file_path: Path,
) -> list[Reaction]:
    # df_all: pd.DataFrame = pd.DataFrame()
    # for file_path in existing_smarts_file_paths:
    #     df: pd.DataFrame = pd.read_csv(file_path, header=None)
    #     df_all = pd.concat([df_all, df], ignore_index=True)
    # df_all.columns = [
    #     "existing_reaction_smarts"
    # ]

    df: pd.DataFrame = pd.read_csv(file_path, header=0)
    # reaction_templates: list[str] = df[
    #     "reaction_smarts_with_atom_mapping"
    # ].tolist()
    # reactions: set[Reaction] = {Reaction(s) for s in reaction_templates}

    reactions: list[Reaction] = [
        Reaction(
            reaction_smarts=row["reaction_smarts_with_atom_mapping"],
            id_=row["USPTO-50k_id"],
            product=row["product_smiles"],
        )
        for (_, row) in df.iterrows()
    ]
    return reactions


def generate_smiles(
    model_file_path: Path,
    existing_file_path: Path,
    num_to_generate: int,
    max_length: Optional[int] = None,
) -> tuple[Counter, pd.DataFrame]:

    # Validate arguments
    model_file_path = Path(model_file_path).resolve()
    existing_file_path = Path(existing_file_path).resolve()
    assert int(num_to_generate) > 0
    if max_length is not None:
        assert int(max_length) > 0

    # TODO add those arguments to the function definition
    # assert 0.0 <= float(top_p) <= 1.0
    # assert int(num_beams) > 0
    # early_stopping = bool(early_stopping)

    # Setup variables
    smiles: dict[str, set[Molecule]] = {
        "all_existing": set(),
        "all_valid": set(),
        "all_novel": set(),
        "pl_generated": set(),
        "pl_valid": set(),
        "pl_unique": set(),
        "pl_novel": set(),
    }
    counter = Counter(["generated", "valid", "unique", "novel"])

    # Load existing molecules
    logger.info("Loading known molecules...")
    existing_molecules: list[Molecule] = load_existing_molecules(
        existing_file_path
    )
    smiles["all_existing"] = set(existing_molecules)
    assert all(bool(s.canonical_smiles) for s in smiles["all_existing"])

    # Load model including tokenizer
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_file_path)
    logger.info("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_file_path)

    # Create generation pipeline
    logger.info("Preparing generation...")
    if max_length is None:
        max_length = model.config.n_positions
        logger.info(f"Using max_length={max_length} from model config")
    else:
        logger.info(f"Using max_length={max_length} from command line")
        if max_length > model.config.n_positions:
            max_length = model.config.n_positions
            logger.warning(
                f"max_length={max_length} is larger than model config allows, setting it to {model.config.n_positions}"
            )

    num_tries: int = 0
    max_num_tries: int = max(num_to_generate // 100, 10)
    num_to_generate_in_pipeline: int = min(
        max(MIN_NUM_TO_GENERATE, num_to_generate // 100),
        MIN_NUM_TO_GENERATE * 10,
    )
    logger.info(f"Generating {num_to_generate_in_pipeline} molecules at a time")
    prompt = tokenizer.bos_token
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer  # , device=device
    )

    # Generate molecules
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        "valid: {task.fields[valid]:>4.0%}",
        "unique: {task.fields[unique]:>4.0%}",
        "novel: {task.fields[novel]:>4.0%}",
        TimeRemainingColumn(elapsed_when_finished=True),
        refresh_per_second=2,
    ) as progress:

        task = progress.add_task(
            "Generating molecules...",
            total=num_to_generate,
            valid=0.0,
            unique=0.0,
            novel=0.0,
        )

        while len(smiles["all_novel"]) < num_to_generate:
            generated = pipe(
                prompt,
                num_return_sequences=num_to_generate_in_pipeline,
                max_new_tokens=max_length,
                return_full_text=True,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                # top_p=top_p,
                # num_beams=num_beams,
                # early_stopping=early_stopping,
                # skip_special_tokens=True
            )

            generated = {
                s["generated_text"].replace(prompt, "").replace(" ", "")
                for s in generated
            }
            smiles["pl_generated"] = {Molecule(s) for s in generated}
            smiles["pl_valid"] = {s for s in smiles["pl_generated"] if s.valid}

            smiles["pl_unique"] = smiles["pl_valid"] - smiles["all_valid"]
            smiles["all_valid"] |= smiles["pl_unique"]

            smiles["pl_novel"] = smiles["pl_unique"] - smiles["all_existing"]
            smiles["all_novel"] |= smiles["pl_novel"]

            if len(smiles["pl_novel"]) == 0:
                num_tries += 1
                if num_tries >= max_num_tries:
                    raise RuntimeError(
                        f"Aborting... no novel molecules generated for {num_tries} times"
                    )

            generated_counter = len(smiles["pl_generated"])
            valid_counter = len(smiles["pl_valid"])
            unique_counter = len(smiles["pl_unique"])
            novel_counter = len(smiles["pl_novel"])
            counter.increment("generated", generated_counter)
            counter.increment("valid", valid_counter)
            counter.increment("unique", unique_counter)
            counter.increment("novel", novel_counter)

            progress.update(
                task,
                advance=novel_counter,
                valid=valid_counter / num_to_generate_in_pipeline,
                unique=unique_counter / valid_counter,
                novel=novel_counter / unique_counter,
            )

    # Some final checks
    logger.info("Perform final plausibility checks...")
    assert len(smiles["all_valid"]) >= len(smiles["all_novel"])
    assert len(smiles["all_novel"]) >= num_to_generate

    # Generate output
    logger.info("Generating output...")
    column_smiles = [s.canonical_smiles for s in smiles["all_novel"]]
    df = pd.DataFrame(
        {
            "smiles": column_smiles,
        }
    )

    return counter, df


def generate_smarts(
    model_file_path: Path,
    existing_file_path: Path,
    num_to_generate: int,
    max_length: Optional[int] = None,
) -> tuple[Counter, pd.DataFrame]:
    def is_feasible(_reaction: Reaction) -> list[Reaction]:
        _similar_reactions = [
            r
            for r in existing_reactions
            if _reaction.is_similar_to(r, "canonical")
        ]
        if len(_similar_reactions) > 0:
            try:
                _rxn = rdchiralReaction(_reaction.reaction_smarts)
            except (TypeError, ValueError):
                return []

            _feasible_reactions: list[Reaction] = []
            for _similar_reaction in _similar_reactions:
                _product = rdchiralReactants(_similar_reaction.product)
                # broad exception clause due to RDKit raising non-Python exceptions
                # noinspection PyBroadException
                try:
                    # TODO check rdChiralRun options/arguments
                    _outcomes = rdchiralRun(
                        _rxn,
                        _product,
                        keep_mapnums=True,
                        combine_enantiomers=True,
                    )
                    if len(_outcomes) > 0:
                        _feasible_reactions.append(_similar_reaction)
                except:  # noqa: E722
                    pass

            return _feasible_reactions

        else:
            return []

    # Validate arguments
    model_file_path = Path(model_file_path).resolve()
    existing_file_path = Path(existing_file_path).resolve()
    assert int(num_to_generate) > 0
    if max_length is not None:
        assert int(max_length) > 0

    # TODO add those arguments to the function definition
    # assert 0.0 <= float(top_p) <= 1.0
    # assert int(num_beams) > 0
    # early_stopping = bool(early_stopping)

    # Setup variables
    smarts: dict[str, set[Reaction]] = {
        "all_existing": set(),
        "all_valid": set(),
        "all_feasible": set(),
        "all_known": set(),
        "all_new": set(),
        "pl_generated": set(),
        "pl_valid": set(),
        "pl_unique": set(),
    }
    counter = Counter(["generated", "valid", "unique", "feasible", "known"])

    # Load existing reaction templates
    logger.info("Loading known reaction templates...")
    # existing_smarts: set[str] = load_existing_smarts(existing_file_path)
    existing_reactions: list[Reaction] = load_existing_reaction_templates(
        existing_file_path
    )
    smarts["all_existing"] = set(existing_reactions)
    assert all(bool(s.reaction_smarts) for s in smarts["all_existing"])

    # Load model including tokenizer
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_file_path)
    logger.info("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_file_path)

    # Create generation pipeline
    logger.info("Preparing generation...")
    if max_length is None:
        max_length = model.config.n_positions
        logger.info(f"Using max_length={max_length} from model config")
    else:
        logger.info(f"Using max_length={max_length} from command line")
        if max_length > model.config.n_positions:
            max_length = model.config.n_positions
            logger.warning(
                f"max_length={max_length} is larger than model config allows, setting it to {model.config.n_positions}"
            )

    num_tries: int = 0
    max_num_tries: int = max(num_to_generate // 100, 10)
    num_to_generate_in_pipeline: int = min(
        max(MIN_NUM_TO_GENERATE, num_to_generate // 100),
        MIN_NUM_TO_GENERATE * 10,
    )
    logger.info(
        f"Generating {num_to_generate_in_pipeline} reaction templates at a time"
    )
    prompt = tokenizer.bos_token
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer  # , device=device
    )

    # Generate reaction templates
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        "valid: {task.fields[valid]:>4.0%}",
        "unique: {task.fields[unique]:>4.0%}",
        TimeRemainingColumn(elapsed_when_finished=True),
        refresh_per_second=2,
    ) as progress:

        task = progress.add_task(
            "Generating reaction templates...",
            total=num_to_generate,
            valid=0.0,
            unique=0.0,
        )

        while len(smarts["all_valid"]) < num_to_generate:
            generated = pipe(
                prompt,
                num_return_sequences=num_to_generate_in_pipeline,
                max_new_tokens=max_length,
                return_full_text=True,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                # top_p=top_p,
                # num_beams=num_beams,
                # early_stopping=early_stopping,
                # skip_special_tokens=True
            )

            generated = {
                s["generated_text"].replace(prompt, "").replace(" ", "")
                for s in generated
            }
            smarts["pl_generated"] = {Reaction(s) for s in generated}
            smarts["pl_valid"] = {s for s in smarts["pl_generated"] if s.valid}

            if len(smarts["pl_valid"]) == 0:
                num_tries += 1
                if num_tries >= max_num_tries:
                    raise RuntimeError(
                        f"Aborting... no valid reaction templates generated for {num_tries} times"
                    )

            smarts["pl_unique"] = smarts["pl_valid"] - smarts["all_valid"]
            smarts["all_valid"] |= smarts["pl_unique"]

            generated_counter = len(smarts["pl_generated"])
            valid_counter = len(smarts["pl_valid"])
            unique_counter = len(smarts["pl_unique"])
            counter.increment("generated", generated_counter)
            counter.increment("valid", valid_counter)
            counter.increment("unique", unique_counter)

            progress.update(
                task,
                advance=unique_counter,
                valid=valid_counter / num_to_generate_in_pipeline,
                unique=unique_counter / valid_counter,
            )

    # Check for chemical feasibility
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(elapsed_when_finished=True),
        refresh_per_second=2,
    ) as progress:

        task = progress.add_task(
            "Checking chemical feasibility...", total=len(smarts["all_valid"])
        )

        # Check valid reactions for chemical feasibility
        # Feasible reactions are those for which we can find a product that produces reactants with the reaction
        # Finding a product is done by looking for products of reactions that are similar to the reaction
        # Similarity is defined by the canonical templates of the reactions being the same
        for reaction in smarts["all_valid"]:
            feasible_reactions = is_feasible(reaction)
            if len(feasible_reactions) > 0:
                reaction.feasible = True
                smarts["all_feasible"].add(reaction)
                feasible_ids = [
                    s.id for s in feasible_reactions if s.id is not None
                ]
                reaction.works_with = " | ".join(feasible_ids)

            progress.update(task, advance=1)

    counter.increment("feasible", len(smarts["all_feasible"]))

    # Check for exact match with existing reaction templates
    logger.info("Checking for exact match with existing reaction templates...")
    smarts["all_known"] = smarts["all_feasible"] & smarts["all_existing"]
    smarts["all_new"] = smarts["all_valid"] - smarts["all_existing"]
    counter.increment("known", len(smarts["all_known"]))

    # Some final checks
    logger.info("Perform final plausibility checks...")
    assert len(smarts["all_valid"]) == len(smarts["all_known"]) + len(
        smarts["all_new"]
    )
    should_be_empty = smarts["all_known"] - smarts["all_existing"]
    if len(should_be_empty) > 0:
        logger.warning(
            f"Found {len(should_be_empty)} known reaction templates that are not in the existing file"
        )
        logger.warning(
            "This probably indicates a bug in the code, saving output nonetheless for inspection"
        )
    assert len(smarts["all_feasible"]) <= len(smarts["all_valid"])
    assert len(smarts["all_known"]) <= len(smarts["all_feasible"])

    # Generate output
    # TODO Amend output to include ... TBD
    logger.info("Generating output...")
    column_smarts = [s.reaction_smarts for s in smarts["all_feasible"]]
    column_known = [s in smarts["all_known"] for s in smarts["all_feasible"]]
    column_works_with = [s.works_with for s in smarts["all_feasible"]]
    # output = [s.reaction_smarts for s in smarts["all_known"]]
    # output.extend([s.reaction_smarts for s in smarts["all_new"]])
    # existing_flag = [True] * len(smarts["all_known"]) + [False] * len(
    #     smarts["all_new"]
    # )
    df = pd.DataFrame(
        {
            "feasible_reaction_smarts": column_smarts,
            "exact_match": column_known,
            "works_with": column_works_with,
        }
    )

    return counter, df


@logger.catch
def main() -> None:
    # Prepare argument parser
    parser = argparse.ArgumentParser(
        description="Generate SMILES molecules or SMARTS reaction templates."
    )
    parser.add_argument(
        "mode",
        type=str.lower,
        choices=VALID_GENERATION_MODES,
        help="the generation mode.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        required=True,
        help="directory path to the trained model.",
    )
    parser.add_argument(
        "-k",
        "--known",
        type=Path,
        required=True,
        help="file path to the known molecules or reaction templates (must be prepared by prepare_data.py).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_FILE_PATH),
        help="file path for the generated molecules or reaction templates, default: '%(default)s'.",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=DEFAULT_NUM_TO_GENERATE,
        help="number of molecules or reaction templates to generate, default: '%(default)s'.",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        help="maximum length of generated molecules or reaction templates, default: the model's sequence length.",
    )

    # TODO add more arguments for beam search, e.g.
    # TOP_P: float = 0.95  # not used yet
    # NUM_BEAMS: int = 5  # not used yet
    # EARLY_STOPPING: bool = True  # not used yet

    args = parser.parse_args()
    configure_logging()

    # Prepare and check (global) variables
    if args.mode == "smiles":
        items_name = "molecules"
        logger.log("HEADING", "Generating SMILES molecules...")
    elif args.mode == "smarts":
        items_name = "reaction templates"
        logger.log("HEADING", "Generating SMARTS reaction templates...")
    else:
        raise ValueError(f"Invalid generation mode: {args.mode}")

    model_file_path = Path(args.model).resolve()
    logger.debug(f"Model file path: '{model_file_path}'")
    if not model_file_path.is_dir():
        raise FileNotFoundError(
            f"Model in {model_file_path.as_posix()} not found"
        )

    known_file_path = Path(args.known).resolve()
    logger.debug(f"Known file path: '{known_file_path}'")
    if not known_file_path.is_file():
        raise FileNotFoundError(
            f"Known items in {known_file_path.as_posix()} not found"
        )

    output_file_path = Path(args.output).resolve()
    logger.debug(f"Output file path: '{output_file_path}'")
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    num_to_generate = args.num
    logger.debug(f"Number of {items_name} to generate: {num_to_generate}")
    max_length = args.length
    logger.debug(f"Maximum length of generated {items_name}: {max_length}")

    if args.mode == "smiles":
        counter, df = generate_smiles(
            model_file_path,
            known_file_path,
            num_to_generate,
            max_length,
        )
    elif args.mode == "smarts":
        counter, df = generate_smarts(
            model_file_path,
            known_file_path,
            num_to_generate,
            max_length,
        )
    else:
        raise ValueError(f"Invalid generation mode: {args.mode}")

    logger.info("Generation statistics")
    logger.info(f"Absolute numbers:   {counter.get_count()}")
    logger.info(f"Absolute fractions: {counter.get_absolute_fraction()}")
    logger.info(f"Relative fractions: {counter.get_relative_fraction()}")

    # Save generated items
    logger.info(f"Saving generated {items_name} to {output_file_path.name}")
    df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    main()
