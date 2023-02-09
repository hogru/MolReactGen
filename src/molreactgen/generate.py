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
from typing import Optional, Union

import pandas as pd  # type: ignore
from loguru import logger
from rdchiral.main import (  # type: ignore
    rdchiralReactants,
    rdchiralReaction,
    rdchiralRun,
)
from rdkit import rdBase  # type: ignore
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
    AutoTokenizer,
    GenerationConfig,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    pipeline,
)

from molreactgen.helpers import (
    Tally,
    configure_logging,
    determine_log_level,
    guess_project_root_dir,
    save_commandline_arguments,
)
from molreactgen.molecule import Molecule, Reaction
from molreactgen.tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN

# Global variables, defaults
PROJECT_ROOT_DIR: Path = guess_project_root_dir()
GENERATED_DATA_DIR: Path = (
    PROJECT_ROOT_DIR / "data" / "generated" / f"{datetime.now():%Y-%m-%d_%H-%M}"
)
GENERATED_DATA_DIR.mkdir(exist_ok=False, parents=True)
ARGUMENTS_FILE_PATH = GENERATED_DATA_DIR / "generate_cl_args.json"
DEFAULT_OUTPUT_FILE_PATH = GENERATED_DATA_DIR / "generated.csv"
DEFAULT_GENERATION_CONFIG_FILE_NAME = "generation_config.json"
CSV_ID_SPLITTER = " | "

VALID_GENERATION_MODES = (
    "smiles",
    "smarts",
)
DEFAULT_NUM_TO_GENERATE: int = 1000
MIN_NUM_TO_GENERATE: int = 20
DEFAULT_NUM_BEAMS: int = 1
DEFAULT_TEMPERATURE: float = 1.0


def load_existing_molecules(
    file_path: Path,
) -> list[Molecule]:
    df: pd.DataFrame = pd.read_csv(file_path, header=None)
    molecules: list[Molecule] = [Molecule(row) for row in df[0]]
    return molecules


def load_existing_reaction_templates(
    file_path: Path,
) -> list[Reaction]:
    df: pd.DataFrame = pd.read_csv(file_path, header=0)
    reactions: list[Reaction] = [
        Reaction(
            reaction_smarts=row["reaction_smarts_with_atom_mapping"],
            split=row["split"],
            id_=row["USPTO-50k_id"],
            product=row["product_smiles"],
        )
        for (_, row) in df.iterrows()
    ]
    return reactions


def _load_model(
    model_file_path: Path,
) -> AutoModelForCausalLM:
    model_file_path = Path(model_file_path).resolve()
    logger.debug(f"Loading model from {model_file_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_file_path)
    return model


def _load_tokenizer(
    tokenizer_file_path: Path,
) -> PreTrainedTokenizerFast:
    tokenizer_file_path = Path(tokenizer_file_path).resolve()
    logger.debug(f"Loading tokenizer from {tokenizer_file_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path)
    return tokenizer


def _is_finetuned_model(
    tokenizer: PreTrainedTokenizerFast,
    *,
    from_scratch_bos_token: str = BOS_TOKEN,
    from_scratch_eos_token: str = EOS_TOKEN,
) -> bool:
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError(
            f"tokenizer must be a PreTrainedTokenizerFast, but is {type(tokenizer)}"
        )

    if (
        tokenizer.bos_token == from_scratch_bos_token
        and tokenizer.eos_token == from_scratch_eos_token
        and tokenizer.bos_token_id < 256
        and tokenizer.eos_token_id < 256
    ):
        return False

    elif (
        tokenizer.bos_token != from_scratch_bos_token
        and tokenizer.eos_token != from_scratch_eos_token
        and tokenizer.bos_token_id > 255
        and tokenizer.eos_token_id > 255
    ):
        return True

    else:
        raise RuntimeError(
            f"Cannot determine if model is fine-tuned or not. "
            f"Tokenizer BOS token (id): {tokenizer.bos_token} ({tokenizer.bos_token_id}), "
            f"Tokenizer EOS token (id): {tokenizer.eos_token} ({tokenizer.eos_token_id}), "
            f"From-scratch BOS token: {from_scratch_bos_token}, "
            f"From-scratch EOS token: {from_scratch_eos_token}"
        )


def _determine_max_length(
    model: AutoModelForCausalLM, max_length: Optional[int] = None
) -> int:
    if not isinstance(
        model, PreTrainedModel
    ):  # would like to check for AutoModelForCausalLM, but that doesn't work
        raise TypeError(
            f"model must be an AutoModelForCausalLM, but is {type(model)}"
        )

    try:
        model_max_length = int(model.config.n_positions)
    except (AttributeError, TypeError):
        model_max_length = None
        logger.warning(
            "Cannot determine the model's maximum input sequence length"
        )
        if max_length is None:
            raise ValueError(
                "Cannot determine maximum input sequence length, "
                "since max_length is also None"
            )

    if max_length is None and model_max_length is not None:
        max_length = model_max_length
        logger.debug(f"Using max_length={max_length} from model config")
    elif max_length is not None:
        logger.debug(
            f"Using max_length={max_length} from command line argument"
        )
        if model_max_length is not None and max_length > model_max_length:
            max_length = model_max_length
            logger.warning(
                f"max_length={max_length} is larger than model config allows, "
                f"setting it to {model_max_length}"
            )
    else:
        raise RuntimeError("Unexpected error")

    return max_length


def _determine_stopping_criteria(
    tokenizer: PreTrainedTokenizerFast, fine_tuned: bool
) -> Union[int, list[int]]:
    stopping_criteria: Union[int, list[int]]
    if fine_tuned:
        # This would be correct but does not work due to a bug in transformers, see:
        # https://github.com/huggingface/transformers/pull/21461
        # TODO Change once the HF bug is fixed
        # stopping_criteria = [
        #     tokenizer.convert_tokens_to_ids(EOS_TOKEN),
        #     tokenizer.eos_token_id,
        # ]
        stopping_criteria = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
    else:
        stopping_criteria = tokenizer.eos_token_id

    return stopping_criteria


def _determine_num_to_generate_in_pipeline(
    num_to_generate: int, item_name: str = "items"
) -> int:
    num_to_generate_in_pipeline: int = min(
        max(MIN_NUM_TO_GENERATE, num_to_generate // 100),
        MIN_NUM_TO_GENERATE * 10,
    )
    logger.debug(
        f"Generating {num_to_generate_in_pipeline} {item_name} at a time"
    )
    return num_to_generate_in_pipeline


def _create_generation_pipeline(
    model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast
) -> Pipeline:
    if not isinstance(
        model, PreTrainedModel
    ):  # would like to check for AutoModelForCausalLM, but that doesn't work
        raise TypeError(
            f"model must be an AutoModelForCausalLM, but is {type(model)}"
        )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError(
            f"tokenizer must be a PreTrainedTokenizerFast, but is {type(tokenizer)}"
        )

    pipe: Pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer  # ,device=device
    )
    return pipe


def _determine_max_num_tries(num_to_generate: int) -> int:
    return max(int(num_to_generate) // 100, 10)


def _determine_prompt(
    tokenizer: PreTrainedTokenizerFast, fine_tuned: bool
) -> str:
    if fine_tuned:
        prompt = BOS_TOKEN
    else:
        prompt = tokenizer.bos_token

    return prompt


def create_and_save_generation_config(
    model_file_path: Path,
    *,
    num_to_generate: int = DEFAULT_NUM_TO_GENERATE,
    max_length: Optional[int] = None,
    num_beams: int = 1,
    temperature: float = 1.0,
) -> GenerationConfig:
    model = _load_model(model_file_path)
    tokenizer = _load_tokenizer(model_file_path)
    fine_tuned = _is_finetuned_model(tokenizer)
    fine_tuned_str = "fine-tuned" if fine_tuned else "trained from scratch"
    logger.debug(f"Model assumed to be {fine_tuned_str}")
    max_length = _determine_max_length(model, max_length)
    stopping_criteria = _determine_stopping_criteria(tokenizer, fine_tuned)
    num_to_generate_in_pipeline = _determine_num_to_generate_in_pipeline(
        num_to_generate
    )
    if num_beams > 1:
        early_stopping = True
    else:
        early_stopping = False
    do_sample = True

    if (model_file_path / DEFAULT_GENERATION_CONFIG_FILE_NAME).is_file():
        generation_config, unused_kwargs = GenerationConfig.from_pretrained(
            model_file_path,
            do_sample=do_sample,
            num_return_sequences=num_to_generate_in_pipeline,
            max_new_tokens=max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stopping_criteria,
            num_beams=num_beams,
            early_stopping=early_stopping,
            temperature=temperature,
            length_penalty=0.0,  # does neither promote nor penalize long sequences
            return_unused_kwargs=True,
        )
        if len(unused_kwargs) > 0:
            logger.warning(
                f"Unused kwargs for GenerationConfig: {list(unused_kwargs)}"
            )

    else:
        generation_config = GenerationConfig(
            do_sample=do_sample,
            num_return_sequences=num_to_generate_in_pipeline,
            max_new_tokens=max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stopping_criteria,
            num_beams=num_beams,
            early_stopping=early_stopping,
            temperature=temperature,
            length_penalty=0.0,  # does neither promote nor penalize long sequences
        )

    generation_config.save_pretrained(model_file_path)
    return generation_config


def generate_smiles(
    *,
    config: GenerationConfig,
    pipe: Pipeline,
    prompt: str,
    existing_file_path: Path,
    num_to_generate: int,
    max_num_tries: int,
) -> tuple[Tally, pd.DataFrame]:

    # Validate arguments
    if not isinstance(config, GenerationConfig):
        raise TypeError(
            f"config must be a GenerationConfig, but is {type(config)}"
        )
    if not isinstance(pipe, Pipeline):
        raise TypeError(
            f"pipe must be a transformers.Pipeline, but is {type(pipe)}"
        )
    prompt = str(prompt)
    existing_file_path = Path(existing_file_path).resolve()
    assert int(num_to_generate) > 0
    assert int(max_num_tries) > 0

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
    counter = Tally(["generated", "valid", "unique", "novel"])

    # Load existing molecules
    logger.info("Loading known molecules...")
    existing_molecules: list[Molecule] = load_existing_molecules(
        existing_file_path
    )
    smiles["all_existing"] = set(existing_molecules)
    assert all(bool(s.canonical_smiles) for s in smiles["all_existing"])

    # Generate molecules
    logger.info(
        f"Starting generation... ({config.num_return_sequences} molecules at a time, "
        f"with a maximum sequence length of {config.max_new_tokens})"
    )
    num_tries = 0
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
        # Currently no batching, might add it later, but check this first:
        # https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching
        while len(smiles["all_novel"]) < num_to_generate:
            generated = pipe(
                prompt,
                return_full_text=True,
            )

            generated = {
                s["generated_text"]
                .replace(prompt, "")
                .replace(PAD_TOKEN, "")
                .replace(EOS_TOKEN, "")
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
                valid=valid_counter / generated_counter
                if generated_counter > 0
                else 0.0,
                unique=unique_counter / valid_counter
                if valid_counter > 0
                else 0.0,
                novel=novel_counter / unique_counter
                if unique_counter > 0
                else 0.0,
            )

    # Some final checks
    logger.debug("Performing final plausibility checks...")
    assert len(smiles["all_valid"]) >= len(smiles["all_novel"])
    assert len(smiles["all_novel"]) >= num_to_generate

    # Generate output
    logger.debug("Generating output...")
    column_smiles = [s.canonical_smiles for s in smiles["all_novel"]]
    df = pd.DataFrame(
        {
            "smiles": column_smiles,
        }
    )
    return counter, df


def generate_smarts(
    *,
    config: GenerationConfig,
    pipe: Pipeline,
    prompt: str,
    existing_file_path: Path,
    num_to_generate: int,
    max_num_tries: int,
) -> tuple[Tally, pd.DataFrame]:
    def get_reactions_with_feasible_products(
        _reaction: Reaction,
    ) -> list[Reaction]:
        # Determine similar reactions; similar means that the reaction smarts have the same canonical form
        _similar_reactions = [
            r
            for r in existing_reactions
            if _reaction.is_similar_to(r, "canonical")
        ]

        # If there are similar reactions, see if the reaction smarts itself is syntactically valid
        if len(_similar_reactions) > 0:
            rdBase.DisableLog("rdApp.error")
            # broad exception clause due to RDKit raising non-Python exceptions
            # noinspection PyBroadException
            try:
                _rxn = rdchiralReaction(_reaction.reaction_smarts)
            except:  # noqa: E722
                return []
            # If the reaction smarts is valid, iterate over all similar reactions and determine their products
            _feasible_reactions: list[
                Reaction
            ] = []  # technically it's a reaction, but we only need the products
            for _similar_reaction in _similar_reactions:
                _product = rdchiralReactants(_similar_reaction.product)
                # See if we can run the reaction smarts with each product
                # If we can, add the similar reaction to the list of feasible reactions
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

            rdBase.EnableLog("rdApp.error")
            return _feasible_reactions

        else:
            return []

    # Validate arguments
    if not isinstance(config, GenerationConfig):
        raise TypeError(
            f"config must be a GenerationConfig, but is {type(config)}"
        )
    if not isinstance(pipe, Pipeline):
        raise TypeError(f"pipe must be a Pipeline, but is {type(pipe)}")
    prompt = str(prompt)
    existing_file_path = Path(existing_file_path).resolve()
    assert int(num_to_generate) > 0
    assert int(max_num_tries) > 0

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
    counter = Tally(
        [
            "generated",
            "valid",
            "unique",
            "feasible",
            "known",
            "known_from_valid_set",
            "known_from_test_set",
        ]
    )

    # Load existing reaction templates
    logger.info("Loading known reaction templates...")
    existing_reactions: list[Reaction] = load_existing_reaction_templates(
        existing_file_path
    )
    smarts["all_existing"] = set(existing_reactions)
    assert all(bool(s.reaction_smarts) for s in smarts["all_existing"])

    # Generate reaction templates
    logger.info(
        f"Starting generation... ({config.num_return_sequences} reaction templates at a time, "
        f"with a maximum sequence length of {config.max_new_tokens})"
    )
    num_tries = 0
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
                return_full_text=True,
            )

            generated = {
                s["generated_text"]
                .replace(prompt, "")
                .replace(PAD_TOKEN, "")
                .replace(EOS_TOKEN, "")
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
                valid=valid_counter / generated_counter
                if generated_counter > 0
                else 0.0,
                unique=unique_counter / valid_counter
                if valid_counter > 0
                else 0.0,
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
            feasible_reactions = get_reactions_with_feasible_products(reaction)
            # If we can find a reaction with a product that works with the generated reaction,
            # then the reaction is feasible = these products/reactions "work with" the generated reaction
            if len(feasible_reactions) > 0:
                reaction.feasible = True
                smarts["all_feasible"].add(reaction)
                # Gather the IDs of the reactions that work with the generated reaction
                feasible_ids = [
                    s.id for s in feasible_reactions if s.id is not None
                ]
                reaction.works_with = CSV_ID_SPLITTER.join(feasible_ids)
                reaction.num_works_with = len(feasible_ids)
                # Gather information about the data split of the reaction that work with the generated reaction
                reaction.in_val_set = any(
                    [
                        s.split == "valid"
                        for s in existing_reactions
                        if s == reaction
                    ]
                )
                reaction.in_test_set = any(
                    [
                        s.split == "test"
                        for s in existing_reactions
                        if s == reaction
                    ]
                )
            progress.update(task, advance=1)

    counter.increment("feasible", len(smarts["all_feasible"]))

    # Check for exact match with existing reaction templates and gather sets and statistics
    logger.info("Checking for exact match with existing reaction templates...")
    smarts["all_known"] = smarts["all_feasible"] & smarts["all_existing"]
    smarts["all_new"] = smarts["all_valid"] - smarts["all_existing"]
    smarts["all_known_from_valid_set"] = {
        s for s in smarts["all_known"] if s.in_val_set
    }
    smarts["all_known_from_test_set"] = {
        s for s in smarts["all_known"] if s.in_test_set
    }
    counter.increment("known", len(smarts["all_known"]))
    counter.increment(
        "known_from_valid_set", len(smarts["all_known_from_valid_set"])
    )
    counter.increment(
        "known_from_test_set", len(smarts["all_known_from_test_set"])
    )

    # Some final checks
    logger.info("Performing final plausibility checks...")
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
    logger.debug("Generating output...")
    column_smarts = [s.reaction_smarts for s in smarts["all_feasible"]]
    column_known = [
        (s.in_val_set or s.in_test_set) for s in smarts["all_feasible"]
    ]
    column_valid_set = [s.in_val_set for s in smarts["all_feasible"]]
    column_test_set = [s.in_test_set for s in smarts["all_feasible"]]
    column_num_works_with = [s.num_works_with for s in smarts["all_feasible"]]
    column_example = [
        s.works_with.split(CSV_ID_SPLITTER, maxsplit=1)[0]
        if s.works_with is not None
        else "ERROR!"
        for s in smarts["all_feasible"]
    ]
    # column_works_with = [s.works_with for s in smarts["all_feasible"]]
    # output = [s.reaction_smarts for s in smarts["all_known"]]
    # output.extend([s.reaction_smarts for s in smarts["all_new"]])
    # existing_flag = [True] * len(smarts["all_known"]) + [False] * len(smarts["all_new"])
    df = pd.DataFrame(
        {
            "feasible_reaction_smarts": column_smarts,
            "not_trained_on_but_known": column_known,
            "known_from_valid_set": column_valid_set,
            "known_from_test_set": column_test_set,
            "num_products_works_with": column_num_works_with,
            "example_works_with_reaction_id": column_example,
            # "works_with": column_works_with,
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
        "-b",
        "--num_beams",
        type=int,
        default=DEFAULT_NUM_BEAMS,
        help="the number of beams for beam search generation, default: '%(default)s' (no beam search).",
    )
    parser.add_argument(
        "-k",
        "--known",
        type=Path,
        required=True,
        help="file path to the known molecules or reaction templates (must be prepared by prepare_data.py).",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        help="maximum length of generated molecules or reaction templates, default: the model's sequence length.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        required=True,
        help="directory path to the trained model.",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=DEFAULT_NUM_TO_GENERATE,
        help="number of molecules or reaction templates to generate, default: '%(default)s'.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_FILE_PATH),
        help="file path for the generated molecules or reaction templates, default (for this instance): '%(default)s'.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="the value used to change the token probabilities, default: '%(default)s'.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="log_level",
        action="append_const",
        const=-1,
        help="increase verbosity from default.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="log_level",
        action="append_const",
        const=1,
        help="decrease verbosity from default.",
    )

    args = parser.parse_args()

    # Configure logging
    log_level: int = determine_log_level(args.log_level)
    configure_logging(log_level)
    save_commandline_arguments(args, ARGUMENTS_FILE_PATH, ("log_level",))

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
    logger.debug(f"Model file path: {model_file_path}")
    if not model_file_path.is_dir():
        raise FileNotFoundError(
            f"Model in {model_file_path.as_posix()} not found"
        )

    known_file_path = Path(args.known).resolve()
    logger.debug(f"Known file path: {known_file_path}")
    if not known_file_path.is_file():
        raise FileNotFoundError(
            f"Known items in {known_file_path.as_posix()} not found"
        )

    output_file_path = Path(args.output).resolve()
    logger.debug(f"Output file path: {output_file_path}")
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    num_to_generate = args.num
    logger.debug(f"Number of {items_name} to generate: {num_to_generate}")
    max_length = args.length
    logger.debug(f"Maximum length of generated {items_name}: {max_length}")

    num_beams = args.num_beams
    logger.debug(f"Number of beams: {num_beams}")
    if num_beams < 1:
        raise ValueError(
            f"Number of beams must be greater than 0, not {num_beams}"
        )

    temperature = args.temperature
    logger.debug(f"Temperature: {temperature}")
    if temperature <= 0:
        raise ValueError(
            f"Temperature must be greater than 0, not {temperature}"
        )

    # Create text generation pipeline
    logger.info("Setting up generation configuration...")
    generation_config = create_and_save_generation_config(
        model_file_path,
        num_to_generate=num_to_generate,
        max_length=max_length,
        num_beams=num_beams,
        temperature=temperature,
    )

    logger.info("Loading model (with updated generation config)...")
    model = _load_model(model_file_path)

    logger.info("Loading tokenizer...")
    tokenizer = _load_tokenizer(model_file_path)
    fine_tuned = _is_finetuned_model(tokenizer)

    logger.info("Creating text generation pipeline...")
    pipe = _create_generation_pipeline(model, tokenizer)

    prompt = _determine_prompt(tokenizer, fine_tuned)
    max_num_tries = _determine_max_num_tries(num_to_generate)

    if args.mode == "smiles":
        counter, df = generate_smiles(
            config=generation_config,
            pipe=pipe,
            prompt=prompt,
            existing_file_path=known_file_path,
            num_to_generate=num_to_generate,
            max_num_tries=max_num_tries,
        )

    elif args.mode == "smarts":
        counter, df = generate_smarts(
            config=generation_config,
            pipe=pipe,
            prompt=prompt,
            existing_file_path=known_file_path,
            num_to_generate=num_to_generate,
            max_num_tries=max_num_tries,
        )

    else:
        raise ValueError(f"Invalid generation mode: {args.mode}")

    logger.info("Generation statistics")
    logger.info(f"Absolute numbers:   {counter.get_count()}")
    logger.info(f"Absolute fractions: {counter.get_absolute_fraction()}")
    logger.info(f"Relative fractions: {counter.get_relative_fraction()}")

    # TODO save statistics to file
    # plus the file path to the model
    # plus the file path to the known items
    # plus the hash codes

    # Save generated items
    logger.info(f"Saving generated {items_name} to {output_file_path}")
    df.to_csv(output_file_path, index=False)

    # TODO save a symlink to the model in the output directory


if __name__ == "__main__":
    main()
