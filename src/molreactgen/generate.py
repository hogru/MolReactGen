# coding=utf-8
# src/molreactgen/generate.py
"""Generate molecules and reaction templates resp.

Functions:
    load_existing_molecules:
        Load molecules from a CSV file.
    load_existing_reaction_templates:
        Load reaction templates from a CSV file.
    create_and_save_generation_config:
        Create and save a generation config from command line arguments.
    generate_smiles:
        Generate molecules from a trained model.
    generate_smarts:
        Generate reaction templates from a trained model.
    main:
        Entry point for the generate command line interface.
"""

import argparse
import contextlib
from datetime import datetime
from pathlib import Path
from random import randint
from typing import Final, Optional, Union

import pandas as pd  # type: ignore
import torch
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
    create_file_link,
    determine_log_level,
    guess_project_root_dir,
    save_commandline_arguments,
)
from molreactgen.molecule import Molecule, Reaction
from molreactgen.tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN

# Global variables, defaults
PROJECT_ROOT_DIR: Final[Path] = guess_project_root_dir()
GENERATED_DATA_DIR: Final[Path] = (
    PROJECT_ROOT_DIR / "data" / "generated" / f"{datetime.now():%Y-%m-%d_%H-%M}"
)
ARGUMENTS_FILE_PATH: Final[Path] = GENERATED_DATA_DIR / "generate_args.json"
DEFAULT_SMILES_OUTPUT_FILE_PATH: Final[Path] = (
    GENERATED_DATA_DIR / "generated_smiles.csv"
)
DEFAULT_SMILES_OUTPUT_FILE_PATH_NOVEL: Final[Path] = (
    GENERATED_DATA_DIR / "generated_smiles_novel_only.csv"
)
DEFAULT_SMARTS_OUTPUT_FILE_PATH: Final[Path] = (
    GENERATED_DATA_DIR / "generated_smarts.csv"
)
DEFAULT_SMARTS_OUTPUT_FILE_PATH_FEASIBLE: Final[Path] = (
    GENERATED_DATA_DIR / "generated_smarts_feasible_only.csv"
)

DEFAULT_GENERATION_CONFIG_FILE_NAME: Final[str] = "generation_config.json"
# CSV_STATS_FILE_NAME: Final[Path] = GENERATED_DATA_DIR / "generate_stats.csv"  # not used
JSON_STATS_FILE_NAME: Final[Path] = GENERATED_DATA_DIR / "generate_stats.json"
MODEL_LINK_DIR_NAME: Final[Path] = GENERATED_DATA_DIR / "link_to_model"
KNOWN_LINK_FILE_NAME: Final[Path] = GENERATED_DATA_DIR / "link_to_known.csv"
LATEST_LINK_FILE_NAME: Final[Path] = (
    GENERATED_DATA_DIR.parent / "link_to_latest_generated.csv"
)
CSV_ID_SPLITTER: Final[str] = " | "

VALID_TARGET_MODES: Final[tuple[str, ...]] = (
    "smiles",
    "smarts",
)
VALID_TARGET_MODES_HELP_STR: Final[str] = "{" + "|".join(VALID_TARGET_MODES) + "}"

DEFAULT_NUM_TO_GENERATE: Final[int] = 10_000
MIN_NUM_TO_GENERATE: Final[int] = 20
DEFAULT_NUM_BEAMS: Final[int] = 1
DEFAULT_REPETITION_PENALTY: Final[float] = 1.2
DEFAULT_SEED: Final[int] = 42
DEFAULT_TEMPERATURE: Final[float] = 1.0


def _load_existing_molecules(
    file_path: Path,
) -> list[Molecule]:
    """Load molecules from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A list of Molecule objects.
    """

    df: pd.DataFrame = pd.read_csv(file_path, header=None)
    return [Molecule(row) for row in df[0]]


def _load_existing_reaction_templates(
    file_path: Path,
) -> list[Reaction]:
    """Load reaction templates from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A list of Reaction objects.
    """

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
    """Load a model from a file path.
    Args:
        model_file_path: Path to the model file.

    Returns:
        The loaded AutoModelForCausalLM model.
    """

    model_file_path = Path(model_file_path).resolve()
    logger.debug(f"Loading model from {model_file_path}...")
    return AutoModelForCausalLM.from_pretrained(model_file_path)


def _load_tokenizer(
    tokenizer_file_path: Path,
) -> PreTrainedTokenizerFast:
    """Load a tokenizer from a file path.

    Args:
        tokenizer_file_path: Path to the tokenizer file.

    Returns:
        The loaded PreTrainedTokenizerFast tokenizer.
    """

    tokenizer_file_path = Path(tokenizer_file_path).resolve()
    logger.debug(f"Loading tokenizer from {tokenizer_file_path}...")
    return AutoTokenizer.from_pretrained(tokenizer_file_path)


def _is_finetuned_model(
    tokenizer: PreTrainedTokenizerFast,
    *,
    from_scratch_bos_token: str = BOS_TOKEN,
    from_scratch_eos_token: str = EOS_TOKEN,
) -> bool:
    """Determine if a model is fine-tuned or from scratch.

    Since we can't tell from the model itself, we have to rely on the tokenizer.
    In this context, a fine-tuned model has different BOS and EOS tokens compared to a model trained from scratch.

    Args:
        tokenizer: the tokenizer corresponding to the model.
        from_scratch_bos_token: the BOS token used for models trained from scratch.
        from_scratch_eos_token: the EOS token used for models trained from scratch.

    Returns:
        True if the model is fine-tuned, False if it is trained from scratch.

    Raises:
        RunTimeError: if the tokenizer BOS and EOS tokens are ambiguous.
    """

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
    """Determine the maximum input sequence length for a model.

    Args:
        model: the model to determine the maximum input sequence length for.
        max_length: the maximum input sequence length to use, if specified,
            can not be larger than the model's maximum input sequence length.

    Returns:
        the maximum input sequence length to use.

    Raises:
        TypeError: if both the model does not have length information and max_length is None.
    """

    if not isinstance(
        model, PreTrainedModel
    ):  # would like to check for AutoModelForCausalLM, but that doesn't work
        raise TypeError(f"model must be an AutoModelForCausalLM, but is {type(model)}")

    try:
        model_max_length = int(model.config.n_positions)
    except (AttributeError, TypeError) as e:
        model_max_length = None
        logger.warning("Cannot determine the model's maximum input sequence length")
        if max_length is None:
            raise ValueError(
                "Cannot determine maximum input sequence length, "
                "since max_length is also None"
            ) from e

    if max_length is None and model_max_length is not None:
        max_length = model_max_length
        logger.debug(f"Using max_length={max_length} from model config")
    elif max_length is not None:
        logger.debug(f"Using max_length={max_length} from command line argument")
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
    """Determine the stopping criteria for the generation pipeline.

    Args:
        tokenizer: the tokenizer corresponding to the model.
        fine_tuned: whether the model is fine-tuned or not.

    Returns:
        the stopping criteria for the generation pipeline.
    """

    # This is the "correct" code as long as this HF bug is not fixed
    # https://github.com/huggingface/transformers/pull/21461

    # return (  # type: ignore
    #     tokenizer.convert_tokens_to_ids(EOS_TOKEN)
    #     if fine_tuned
    #     else tokenizer.eos_token_id
    # )

    # Let's hope this works now
    # TODO if it does work, remove the above code and the comments hinting at the bug
    return (  # type: ignore
        [tokenizer.convert_tokens_to_ids(EOS_TOKEN), tokenizer.eos_token_id]
        if fine_tuned
        else tokenizer.eos_token_id
    )


def _determine_num_to_generate_in_pipeline(
    num_to_generate: int, item_name: str = "items"
) -> int:
    """Determine the number of items to generate during a batch in the generation pipeline.

    Args:
        num_to_generate: the total number of items to generate.
        item_name: the name of the items to generate, used for logging only.

    Returns:
        the number of items to generate during a batch in the generation pipeline.
    """

    # Empirical number to generate in a batch
    num_to_generate_in_pipeline: int = min(
        max(MIN_NUM_TO_GENERATE, num_to_generate // 100),
        MIN_NUM_TO_GENERATE * 10,
    )
    logger.debug(f"Generating {num_to_generate_in_pipeline} {item_name} at a time")
    return num_to_generate_in_pipeline


def _create_generation_pipeline(
    model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast
) -> Pipeline:
    """Create a generation pipeline.

    Args:
        model: the model to use.
        tokenizer: the tokenizer to use.

    Returns:
        the generation pipeline.
    """

    if not isinstance(
        model, PreTrainedModel
    ):  # would like to check for AutoModelForCausalLM, but that doesn't work
        raise TypeError(f"model must be an AutoModelForCausalLM, but is {type(model)}")
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError(
            f"tokenizer must be a PreTrainedTokenizerFast, but is {type(tokenizer)}"
        )

    return pipeline(
        "text-generation", model=model, tokenizer=tokenizer  # ,device=device
    )


def _determine_max_num_tries(num_to_generate: int) -> int:
    """Determine the maximum number of tries to generate a given number of items.

    Args:
        num_to_generate: the number of items to generate.

    Returns:
        the maximum number of tries to generate a given number of items.
    """

    # Empirical number of tries to generate a given number of items
    return max(num_to_generate // 100, 10)


def _determine_prompt(tokenizer: PreTrainedTokenizerFast, fine_tuned: bool) -> str:
    """Determine the prompt to use for the generation pipeline.

    Args:
        tokenizer: the tokenizer corresponding to the model.
        fine_tuned: weather the model is fine-tuned or not.

    Returns:
        the prompt to use for the generation pipeline.
    """

    return BOS_TOKEN if fine_tuned else tokenizer.bos_token  # type: ignore


def create_and_save_generation_config(
    model_file_path: Path,
    *,
    num_to_generate: int = DEFAULT_NUM_TO_GENERATE,
    split_into_chunks: bool = True,
    max_length: Optional[int] = None,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    temperature: float = 1.0,
    overwrite_pretrained_config: bool = False,
) -> GenerationConfig:
    """Create and save a generation config from command line arguments.

    Args:
        model_file_path: the path to the model file.
        num_to_generate: the number of items to generate. Defaults to DEFAULT_NUM_TO_GENERATE.
        split_into_chunks: whether to split the number of items to generate into chunks/batches. Defaults to True.
        max_length:the maximum length of the generated items. Defaults to None, i.e. the model's maximum length.
        num_beams: the number of beams to use for beam search. Defaults to 1, i.e. multinomial search.
        repetition_penalty: the repetition penalty during generation.
        temperature: the value used to modulate the next token probabilities (change logits before softmax).
            Defaults to 1.0, i.e. no modulation.
        overwrite_pretrained_config: whether to overwrite the generation config if it already exists. Defaults to False.

    Returns:
        the generation config.
    """

    model = _load_model(model_file_path)
    tokenizer = _load_tokenizer(model_file_path)
    fine_tuned = _is_finetuned_model(tokenizer)
    fine_tuned_str = "fine-tuned" if fine_tuned else "trained from scratch"
    logger.debug(f"Model assumed to be {fine_tuned_str}")

    # Generate at least one token plus the EOS token(s)
    min_length = 2 if fine_tuned else 1
    max_length = _determine_max_length(model, max_length)

    # Make "room" for the custom EOS token if model is fine-tuned
    max_length = max_length - 1 if fine_tuned else max_length
    stopping_criteria = _determine_stopping_criteria(tokenizer, fine_tuned)
    if split_into_chunks:
        num_to_generate_in_pipeline = _determine_num_to_generate_in_pipeline(
            num_to_generate
        )
    else:
        num_to_generate_in_pipeline = num_to_generate

    early_stopping = num_beams > 1
    do_sample = True

    # TODO Include again once I resolve the following issue:
    # The pre-trained GPT-2 model has an eos_token_id of 50256, but the fine-tuned model
    # needs different stopping criteria. The problem is, that HF does not like overwriting it
    # Therefore I always create a new GenerationConfig (instead of reading and changing it)

    # if (model_file_path / DEFAULT_GENERATION_CONFIG_FILE_NAME).is_file():
    #     generation_config, unused_kwargs = GenerationConfig.from_pretrained(
    #         model_file_path,
    #         do_sample=do_sample,
    #         num_return_sequences=num_to_generate_in_pipeline,
    #         min_new_tokens=min_length,
    #         max_new_tokens=max_length,
    #         pad_token_id=tokenizer.pad_token_id,
    #         # eos_token_id=stopping_criteria,
    #         num_beams=num_beams,
    #         early_stopping=early_stopping,
    #         temperature=temperature,
    #         length_penalty=0.0,  # does neither promote nor penalize long sequences
    #         return_unused_kwargs=True,
    #     )
    #     if len(unused_kwargs) > 0:
    #         logger.warning(f"Unused kwargs for GenerationConfig: {list(unused_kwargs)}")
    #
    # else:
    generation_config = GenerationConfig(
        do_sample=do_sample,
        num_return_sequences=num_to_generate_in_pipeline,
        min_new_tokens=min_length,
        max_new_tokens=max_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stopping_criteria,
        num_beams=num_beams,
        early_stopping=early_stopping,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        # length_penalty only allowed for num_beams > 1
        # length_penalty=0.0,  # does neither promote nor penalize long sequences
    )

    if overwrite_pretrained_config:
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
) -> tuple[Tally, pd.DataFrame, pd.DataFrame]:
    """Generate SMILES strings, i.e. molecules.

    Args:
        config: the generation config.
        pipe: the generation pipeline.
        prompt: the prompt to use for the generation pipeline.
        existing_file_path: the path to the existing molecules, i.e. the molecules the model was trained on.
        num_to_generate: the number of molecules to generate.
        max_num_tries: the maximum number of tries to generate the molecules before raising an exception.

    Returns:
        a tuple containing the tally, the dataframe of all valid molecules,
        and the dataframe of all generated molecules.

    Raises:
        RunTimeError: if the number of tries to generate molecules exceeds max_num_tries.
    """

    # Validate arguments
    if not isinstance(config, GenerationConfig):
        raise TypeError(f"config must be a GenerationConfig, but is {type(config)}")
    if not isinstance(pipe, Pipeline):
        raise TypeError(f"pipe must be a transformers.Pipeline, but is {type(pipe)}")
    existing_file_path = Path(existing_file_path).resolve()
    assert num_to_generate > 0
    assert max_num_tries > 0

    # Setup variables
    all_smiles: list[Molecule] = []  # need a list instead of a set for master list
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
    existing_molecules: list[Molecule] = _load_existing_molecules(existing_file_path)
    # Building the set is painfully slow (about 10min), but I don't know how to speed it up
    # The bulk of the time is spent in calculating the hashes (Molecule.__hash__)
    smiles["all_existing"] = set(existing_molecules)
    assert all(bool(s.canonical_smiles) for s in smiles["all_existing"])

    # Generate molecules
    logger.info(
        f"Starting generation... ({config.num_return_sequences:,} molecules at a time, "
        f"with a maximum sequence length of {config.max_new_tokens:,})"
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
                generation_config=config,
                return_full_text=True,
            )

            generated = {
                s["generated_text"]
                .replace(prompt, "")
                .replace(PAD_TOKEN, "")
                .replace(EOS_TOKEN, "")
                for s in generated
            }
            # I ask for at least one new token in the generation config, but
            # the pipeline still returns empty strings sometimes; therefore the if len(s) > 0
            smiles["pl_generated"] = {Molecule(s) for s in generated if s}
            all_smiles.extend(smiles["pl_generated"])
            smiles["pl_valid"] = {s for s in smiles["pl_generated"] if s.valid}

            smiles["pl_unique"] = smiles["pl_valid"] - smiles["all_valid"]
            smiles["all_valid"] |= smiles["pl_unique"]

            smiles["pl_novel"] = smiles["pl_unique"] - smiles["all_existing"]
            smiles["all_novel"] |= smiles["pl_novel"]

            # Check if we have to abort because we cannot generate valid molecules
            if len(smiles["pl_novel"]) == 0:
                num_tries += 1
                if num_tries >= max_num_tries:
                    raise RuntimeError(
                        f"Aborting... no novel molecules generated for {num_tries} times"
                    )

            # Update the sets of molecules and the corresponding counters
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
                unique=unique_counter / valid_counter if valid_counter > 0 else 0.0,
                novel=novel_counter / unique_counter if unique_counter > 0 else 0.0,
            )

    # Some final checks
    logger.debug("Performing final plausibility checks...")
    assert len(all_smiles) >= len(smiles["all_valid"])
    assert len(smiles["all_valid"]) >= len(smiles["all_novel"])
    assert len(smiles["all_novel"]) >= num_to_generate

    # Generate output
    logger.debug("Generating output...")
    # That is a bit redundant but came later and I did not want to change the generation code too much
    for s in all_smiles:
        s.unique = all_smiles.count(s) == 1
        s.novel = s not in smiles["all_existing"]

    # Generate the "simple" output
    # This is a smaller, filtered file for convenience; not necessary
    column_smiles = [s.canonical_smiles for s in smiles["all_novel"]]
    df_small = pd.DataFrame(
        {
            "smiles": column_smiles,
        }
    )

    # Generate the full output
    column_smiles = [s.smiles for s in all_smiles]
    column_canonical_smiles = [s.canonical_smiles for s in all_smiles]
    column_valid = [s.valid for s in all_smiles]
    column_unique = [s.unique for s in all_smiles]
    column_novel = [s.novel for s in all_smiles]
    df_full = pd.DataFrame(
        {
            "smiles": column_smiles,
            "canonical_smiles": column_canonical_smiles,
            "valid": column_valid,
            "unique": column_unique,
            "novel": column_novel,
        }
    )

    return counter, df_small, df_full


def generate_smarts(
    *,
    config: GenerationConfig,
    pipe: Pipeline,
    prompt: str,
    existing_file_path: Path,
    num_to_generate: int,
    max_num_tries: int,
) -> tuple[Tally, pd.DataFrame, pd.DataFrame]:
    """Generate SMARTS strings, i.e. reaction templates.

    Args:
        config: the generation config.
        pipe: the generation pipeline.
        prompt: the prompt to use for the generation pipeline.
        existing_file_path: the path to the existing reaction templates,
            i.e. the reaction templates the model was trained on.
        num_to_generate: the number of reaction templates to generate.
        max_num_tries: the maximum number of tries to generate the reaction templates before raising an exception.

    Returns:
        a tuple containing the tally, the dataframe of all feasible molecules, and a copy of that dataframe
            (the copy is work in progress and might change later).

    Raises:
        RunTimeError: if the number of tries to generate reaction templates exceeds max_num_tries.
    """

    def get_reactions_with_feasible_products(_reaction: Reaction) -> list[Reaction]:
        """Determine similar reactions to the given reaction and check whether applying the Reaction to the
        corresponding products is chemically feasible.

        Args:
            _reaction: the reaction to check.

        Returns:
            a list of reactions with products that are chemically feasible with the given reaction.
        """

        # Determine similar reactions; similar means that the reaction smarts have the same canonical form
        _similar_reactions = [
            r for r in existing_reactions if _reaction.is_similar_to(r, "canonical")
        ]

        # If there are similar reactions, see if the reaction smarts itself is syntactically valid
        if _similar_reactions:
            rdBase.DisableLog("rdApp.error")
            # broad exception clause due to RDKit raising non-Python exceptions
            # noinspection PyBroadException
            try:
                _rxn = rdchiralReaction(_reaction.reaction_smarts)
            except Exception:  # noqa: E722
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
                with contextlib.suppress(Exception):
                    # might play with rdChiralRun options/arguments
                    _outcomes = rdchiralRun(
                        _rxn,
                        _product,
                        keep_mapnums=True,
                        combine_enantiomers=True,
                    )
                    if _outcomes:
                        _feasible_reactions.append(_similar_reaction)
            rdBase.EnableLog("rdApp.error")
            return _feasible_reactions

        else:
            return []

    # Validate arguments
    if not isinstance(config, GenerationConfig):
        raise TypeError(f"config must be a GenerationConfig, but is {type(config)}")
    if not isinstance(pipe, Pipeline):
        raise TypeError(f"pipe must be a Pipeline, but is {type(pipe)}")
    existing_file_path = Path(existing_file_path).resolve()
    assert num_to_generate > 0
    assert max_num_tries > 0

    # Setup variables
    all_smarts: list[Reaction] = []  # need a list instead of a set for master list
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
    existing_reactions: list[Reaction] = _load_existing_reaction_templates(
        existing_file_path
    )
    smarts["all_existing"] = set(existing_reactions)
    assert all(bool(s.reaction_smarts) for s in smarts["all_existing"])

    # Generate reaction templates
    logger.info(
        f"Starting generation... ({config.num_return_sequences:,} reaction templates at a time, "
        f"with a maximum sequence length of {config.max_new_tokens:,})"
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
            all_smarts.extend(smarts["pl_generated"])
            smarts["pl_valid"] = {s for s in smarts["pl_generated"] if s.valid}

            # Check if we have to abort because we cannot generate valid reaction templates
            if len(smarts["pl_valid"]) == 0:
                num_tries += 1
                if num_tries >= max_num_tries:
                    raise RuntimeError(
                        f"Aborting... no valid reaction templates generated for {num_tries} times"
                    )

            # Update the reaction template sets and the corresponding counters
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
                unique=unique_counter / valid_counter if valid_counter > 0 else 0.0,
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
            if feasible_reactions:
                reaction.feasible = True
                smarts["all_feasible"].add(reaction)
                # Gather the IDs of the reactions that work with the generated reaction
                feasible_ids = [s.id for s in feasible_reactions if s.id is not None]
                reaction.works_with = CSV_ID_SPLITTER.join(feasible_ids)
                reaction.num_works_with = len(feasible_ids)
                # Gather information about the data split of the reaction that work with the generated reaction
                reaction.in_val_set = any(
                    s.split == "valid" for s in existing_reactions if s == reaction
                )
                reaction.in_test_set = any(
                    s.split == "test" for s in existing_reactions if s == reaction
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
    counter.increment("known_from_valid_set", len(smarts["all_known_from_valid_set"]))
    counter.increment("known_from_test_set", len(smarts["all_known_from_test_set"]))

    # Some final checks
    logger.info("Performing final plausibility checks...")
    assert len(smarts["all_valid"]) == len(smarts["all_known"]) + len(smarts["all_new"])
    should_be_empty = smarts["all_known"] - smarts["all_existing"]
    if should_be_empty:
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
    # That is a bit redundant but came later and I did not want to change the generation code too much
    for s in all_smarts:
        s.unique = all_smarts.count(s) == 1

    # Generate the "simple" output
    # This is a smaller, filtered file for convenience; not necessary
    column_smarts = [s.reaction_smarts for s in smarts["all_feasible"]]
    df_small = pd.DataFrame(
        {
            "smarts": column_smarts,
        }
    )

    # column_smarts = [s.reaction_smarts for s in smarts["all_feasible"]]
    # column_known = [(s.in_val_set or s.in_test_set) for s in smarts["all_feasible"]]
    # column_valid_set = [s.in_val_set for s in smarts["all_feasible"]]
    # column_test_set = [s.in_test_set for s in smarts["all_feasible"]]
    # column_num_works_with = [s.num_works_with for s in smarts["all_feasible"]]
    # column_example = [
    #     s.works_with.split(CSV_ID_SPLITTER, maxsplit=1)[0]
    #     if s.works_with is not None
    #     else "ERROR!"
    #     for s in smarts["all_feasible"]
    # ]

    # df_full = pd.DataFrame(
    #     {
    #         "feasible_reaction_smarts": column_smarts,
    #         "not_trained_on_but_known": column_known,
    #         "known_from_valid_set": column_valid_set,
    #         "known_from_test_set": column_test_set,
    #         "num_products_works_with": column_num_works_with,
    #         "example_works_with_reaction_id": column_example,
    #     }
    # )

    column_smarts = [s.reaction_smarts for s in all_smarts]
    column_valid = [s.valid for s in all_smarts]
    column_unique = [s.unique for s in all_smarts]
    column_feasible = [s in smarts["all_feasible"] for s in all_smarts]
    column_known = [(s.in_val_set or s.in_test_set) for s in all_smarts]
    column_valid_set = [s.in_val_set for s in all_smarts]
    column_test_set = [s.in_test_set for s in all_smarts]
    column_num_works_with = [s.num_works_with for s in all_smarts]
    column_example = [
        s.works_with.split(CSV_ID_SPLITTER, maxsplit=1)[0]
        if s.works_with is not None
        else None
        for s in all_smarts
    ]

    df_full = pd.DataFrame(
        {
            "smarts": column_smarts,
            "valid": column_valid,
            "unique": column_unique,
            "feasible": column_feasible,
            "not_trained_on_but_known": column_known,
            "known_from_valid_set": column_valid_set,
            "known_from_test_set": column_test_set,
            "num_products_works_with": column_num_works_with,
            "example_works_with_reaction_id": column_example,
        }
    )

    return counter, df_small, df_full


@logger.catch
def main() -> None:
    """Main generation wrapper function.

    Reads the command line arguments and calls the appropriate functions.
    Saves the generation output to a file.
    """

    # Prepare argument parser
    parser = argparse.ArgumentParser(
        description="Generate SMILES molecules or SMARTS reaction templates."
    )
    parser.add_argument(
        "target",
        type=str.lower,
        choices=VALID_TARGET_MODES,
        help="the target format to generate.",
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
        help="minimum number of molecules or reaction templates to generate, default: '%(default)s'.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"file path for the generated molecules or reaction templates, default: "
        f"'{GENERATED_DATA_DIR}/generated_{VALID_TARGET_MODES_HELP_STR}.csv'.",
    )
    parser.add_argument(
        "-p",
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
        help="the repetition penalty, default: '%(default)s'.",
    )
    parser.add_argument(
        "-r",
        "--random_seed",
        action="store_true",
        default=False,
        help="whether a random seed should be configured; overwrites a fixed seed value, default: '%(default)s'.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="the value used as to seed the the random number generator, default: '%(default)s'.",
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
    if args.target == "smiles":
        items_name = "molecules"
        logger.heading("Generating SMILES molecules...")  # type: ignore
    elif args.target == "smarts":
        items_name = "reaction templates"
        logger.heading("Generating SMARTS reaction templates...")  # type: ignore
    else:
        raise ValueError(f"Invalid generation target: {args.target}")

    model_file_path = Path(args.model).resolve()
    logger.debug(f"Model file path: {model_file_path}")
    if not model_file_path.is_dir():
        raise FileNotFoundError(f"Model in {model_file_path.as_posix()} not found")

    known_file_path = Path(args.known).resolve()
    logger.debug(f"Known file path: {known_file_path}")
    if not known_file_path.is_file():
        raise FileNotFoundError(
            f"Known items in {known_file_path.as_posix()} not found"
        )

    if args.output is None:
        if args.target == "smiles":
            output_file_path_short = DEFAULT_SMILES_OUTPUT_FILE_PATH_NOVEL
            output_file_path_full = DEFAULT_SMILES_OUTPUT_FILE_PATH
        else:
            output_file_path_short = DEFAULT_SMARTS_OUTPUT_FILE_PATH_FEASIBLE
            output_file_path_full = DEFAULT_SMARTS_OUTPUT_FILE_PATH
    else:
        output_file_path_full = Path(args.output).resolve()
        if args.target == "smiles":
            output_file_path_short = output_file_path_full.with_stem(
                output_file_path_full.stem + "_novel_only"
            )
        else:
            output_file_path_short = output_file_path_full.with_stem(
                output_file_path_full.stem + "_feasible_only"
            )

    logger.debug(f"Output file path: {output_file_path_full}")
    logger.debug(f"Secondary output file path: {output_file_path_short}")

    GENERATED_DATA_DIR.mkdir(exist_ok=True, parents=True)
    output_file_path_short.parent.mkdir(parents=True, exist_ok=True)
    output_file_path_full.parent.mkdir(parents=True, exist_ok=True)

    num_to_generate = args.num
    max_length = args.length
    if max_length is None:
        logger.info(
            f"Generate ≥ {num_to_generate:,} {items_name}; "
            f"the maximum length will be determined by the model's sequence length"
        )
    else:
        logger.info(
            f"Generate ≥ {num_to_generate:,} {items_name} with a maximum length ≤ {max_length:,}"
        )

    num_beams = args.num_beams
    logger.debug(f"Number of beams: {num_beams}")
    if num_beams < 1:
        raise ValueError(f"Number of beams must be greater than 0, not {num_beams}")

    repetition_penalty = args.repetition_penalty
    logger.debug(f"Repetition penalty: {repetition_penalty}")
    if repetition_penalty <= 0:
        raise ValueError(
            f"Repetition penalty must be greater than 0, not {repetition_penalty}"
        )

    temperature = args.temperature
    logger.debug(f"Temperature: {temperature}")
    if temperature <= 0:
        raise ValueError(f"Temperature must be greater than 0, not {temperature}")

    # Set seed in torch
    if args.random_seed:
        args.seed = randint(0, 2**32 - 1)
    logger.debug(f"Using random seed {args.seed}")
    torch.manual_seed(args.seed)

    # Create text generation pipeline
    logger.info("Setting up generation configuration...")
    generation_config = create_and_save_generation_config(
        model_file_path,
        num_to_generate=num_to_generate,
        max_length=max_length,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        overwrite_pretrained_config=True,
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

    # Start generation
    if args.target == "smiles":
        counter, df_short, df_full = generate_smiles(
            config=generation_config,
            pipe=pipe,
            prompt=prompt,
            existing_file_path=known_file_path,
            num_to_generate=num_to_generate,
            max_num_tries=max_num_tries,
        )

    elif args.target == "smarts":
        counter, df_short, df_full = generate_smarts(
            config=generation_config,
            pipe=pipe,
            prompt=prompt,
            existing_file_path=known_file_path,
            num_to_generate=num_to_generate,
            max_num_tries=max_num_tries,
        )

    else:
        raise ValueError(f"Invalid generation target: {args.target}")

    # Display and save statistics
    logger.info("Generation statistics")
    logger.info(f"Absolute numbers:   {counter.get_count(format_specifier=',')}")
    logger.info(
        f"Absolute fractions: {counter.get_absolute_fraction(format_specifier='.3f')}"
    )
    logger.info(
        f"Relative fractions: {counter.get_relative_fraction(format_specifier='.3f')}"
    )

    counter.save_to_file(JSON_STATS_FILE_NAME, format_="json")

    # Save generated items
    logger.info(f"Saving generated {items_name} to {output_file_path_short}...")
    df_short.to_csv(output_file_path_short, index=False)
    logger.info(f"Saving generated {items_name} to {output_file_path_full}...")
    df_full.to_csv(output_file_path_full, index=False)

    # Create symlinks for better traceability and convenience
    create_file_link(MODEL_LINK_DIR_NAME, model_file_path)
    create_file_link(KNOWN_LINK_FILE_NAME, known_file_path)
    create_file_link(LATEST_LINK_FILE_NAME, output_file_path_full)


if __name__ == "__main__":
    main()
