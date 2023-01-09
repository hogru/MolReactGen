# coding=utf-8
"""
Auto-Regressive Molecule and Reaction Template Generator
Causal language modeling (CLM) with a transformer decoder model
Author: Stephan Holzgruber
Student ID: K08608294
"""

import argparse
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
from molreactgen.molecule import Reaction

# Global variables, defaults
VALID_GENERATION_MODES = ["smiles", "smarts"]
# TODO make dependant on generation mode
DEFAULT_OUTPUT_FILE_PATH = "../../data/generated/generated_reaction_templates.csv"
DEFAULT_NUM_TO_GENERATE: int = 1000
MIN_NUM_TO_GENERATE: int = 20
# TODO make dependant on generation mode
DEFAULT_MAX_LENGTH: int = 896
DEFAULT_TOP_P: float = 0.95  # not used yet
DEFAULT_NUM_BEAMS: int = 5  # not used yet
DEFAULT_EARLY_STOPPING: bool = True  # not used yet


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
                        # print(f"\nFeasible: {_reaction}")
                        # print(f"Product: {_similar_reaction.product}")
                        _feasible_reactions.append(_similar_reaction)
                except:  # noqa: E722
                    pass

            # _feasible: bool = len(_feasible_reactions) > 0
            # exact_match: bool = any([reaction == r for r in similar_reactions])

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
    # if not ATOM_MAPPING:  # TODO do this even WITH atom mapping?
    #     existing_smarts = {
    #         canonicalize_template(s, strict=False, double_check=True)
    #         for s in existing_smarts
    #     }

    # Load model including tokenizer
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_file_path)
    logger.info("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_file_path)

    # Create generation pipeline
    logger.info("Preparing generation...")
    if max_length is None:
        try:
            max_length = model.config.n_positions
            logger.info(f"Using max_length={max_length} from model config")
        except AttributeError:
            max_length = DEFAULT_MAX_LENGTH
            logger.warning(
                f"Could not determine max_length from model config, setting to {DEFAULT_MAX_LENGTH}"
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
            # canonical_smarts = {
            #     canonicalize_template(s, strict=False, double_check=True)
            #     for s in generated_smarts
            # }
            smarts["pl_valid"] = {s for s in smarts["pl_generated"] if s.valid}
            # if ATOM_MAPPING:
            #     valid_smarts = {
            #         gen
            #         for gen, can in zip(generated_smarts, canonical_smarts)
            #         if can is not None
            #     }
            # else:
            #     valid_smarts = {s for s in canonical_smarts if s is not None}

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

            # This results in an empty set, since after canonicalization, all invalid SMARTS are None
            # smarts["generated_invalid"] |= canonical_smarts - {None} - valid_smarts
            # This is non-empty, but not comparable since not those are not canonicalized
            # smarts["generated_invalid"] |= generated_smarts - {""} - valid_smarts

            # progress.console.print(
            #     f"valid: {valid_counter/num_to_generate_in_pipeline:.0%}, "
            #     f"unique: {unique_counter/valid_counter:.0%}"
            # )
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

        for reaction in smarts["all_valid"]:
            feasible_reactions = is_feasible(reaction)
            if len(feasible_reactions) > 0:
                # progress.console.print(f"Feasible reaction: {reaction.reaction_smarts}")
                reaction.feasible = True
                feasible_ids = [
                    s.id for s in feasible_reactions if s.id is not None
                ]
                reaction.works_with = " | ".join(feasible_ids)
                smarts["all_feasible"].add(reaction)

            # progress.console.print(
            #     f"feasible: {len(smarts['all_feasible'])}",
            # )
            progress.update(task, advance=1)

    counter.increment("feasible", len(smarts["all_feasible"]))

    # Check for exact match with existing reaction templates
    logger.info("Checking for exact match with existing reaction templates...")
    smarts["all_known"] = smarts["all_feasible"] & smarts["all_existing"]
    smarts["all_new"] = smarts["all_valid"] - smarts["all_existing"]
    counter.increment("known", len(smarts["all_known"]))

    # Some final checks
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
    # TODO SMILES not implemented yet
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
        # TODO enable code, remove exception once implemented
        # logger.log("HEADING", "Generating SMARTS reaction templates...")
        # items_name = "molecules"
        raise NotImplementedError(
            "SMILES molecule generation not implemented yet"
        )
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

    if args.mode == "smarts":
        counter, df = generate_smarts(
            model_file_path,
            known_file_path,
            num_to_generate,
            max_length,
        )
    elif args.mode == "smiles":
        raise NotImplementedError(
            "SMILES molecule generation not implemented yet"
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
