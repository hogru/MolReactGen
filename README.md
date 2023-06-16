# MolReactGen

An auto-regressive causal language model for molecule (SMILES) and reaction template (SMARTS) generation. Based on the [Hugging Face implementation](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2#openai-gpt2) of [OpenAI’s GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) [transformer decoder](https://arxiv.org/abs/1706.03762v5) model.

## Research questions:

- How well does this model perform for molecule generation, using the [GuacaMol paper](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839) as a benchmark?
- What is the effect of different tokenization approaches (different RegEx expressions as pre-tokenizers, tokenization algorithms such as BPE, WordPiece)?
- Can we use a model pre-trained on _natural_ language as a basis for fine-tuning a “_molecule_ language” model?
- Can we use this approach/model to generate reaction templates?

## Installation

Disclaimer: This is currently under development. A local (editable package) installation requires `python` ≥ 3.9, [`poetry`](https://python-poetry.org)  ≥ 1.0.8 and `pip` ≥ 22.3. Experiment results are logged to [`weights and biases`](https://wandb.ai).

```
git clone --recurse-submodules https://github.com/hogru/molreactgen
cd molreactgen
python -m pip install -e .
```

## Main files

### `src/molreactgen` directory

- `prepare_data.py` downloads and prepares the datasets
- `train.py` trains the model on a given dataset, configured via (optionally multiple) `.args` file(s) or a single `.yaml` file in the `conf` directory (see example files)
- `generate.py` generates molecules (SMILES) or reaction templates (SMARTS)
- `assess.py` (for molecules only) calculates the Fréchet ChemNet Distance (FCD) between the generated molecules and a reference set of molecules (e.g. the GuacaMol dataset) along with some basic evaluation criteria
- `molecule.py` covers helpers for the chemical space of the task
- `tokenizer.py` provides the various tokenizers
- `helpers.py` is a set of misc helpers/utils (logging etc.)

### `src/molreactgen/utils` directory

- `compute_fcd_stats.py` computes the model activations that are needed to calculate the FCD. This is a separate script because it is computationally expensive and can be reused for model comparison.
- `check_tokenizer.py` is used if a tokenizer can successfully encode and decode a dataset
- `collect_metrics.py` collects metrics from various files and `wandb` and provides them in several formats; used during experiments
- `create_plots.ipynb` is a Jupyter notebook that creates plots from the datasets; used for presentation purposes
- `*.sh` are "quick and dirty" scripts for running experiments/sweeps for hyper parameter search

### `data/raw` directory

- the (default) directory `prepare_data.py` downloads the datasets to
- a sub-directory is created for each dataset, containing the raw data files

### `data/prep` directory

- the (default) directory `prepare_data.py` prepares the datasets in
- a sub-directory is created for each dataset, containing the prepared data files

## Example Usage

### Pre-conditions

- Local repository installation (see above)
- Python 3.9 - it should work with 3.10 as well, but I haven't tested it
- `poetry` installed (see [here](https://python-poetry.org/docs/#installation))
- `poetry shell` to activate the virtual environment
- Optional: `wandb` account and API key (see [here](https://docs.wandb.ai/quickstart)); should work with an anonymous account, but I haven't tested it

cd into the `molreactgen/src/molreactgen` directory and run the following commands:

### Molecules (SMILES)

```bash
# Download and prepare dataset
python prepare_data.py guacamol

# Train the model
# add --fp16 false if your GPU does not support fp16 or you run it on a CPU (not recommended)
python train.py --args conf/guacamol.args  # this also reads the default train.args file

# Generate ≥ 10000 molecules
python generate.py smiles \
--model "../../checkpoints/<your_model>" \
--known "../../data/prep/guacamol/csv/guacamol_v1_train.csv"
--num 10000

# Calculate the stats of GuacaMol training set (needed for FCD calculation)
# This is computationally expensive and can be reused for model comparison
python utils/compute_fcd_stats.py \
"../../data/prep/guacamol/csv/guacamol_v1_train.csv" \
--output "../../data/prep/guacamol/fcd_stats/guacamol_train.pkl"

# Evaluate the generated molecules
python assess.py stats \
--generated "../../data/generated/<generation directory>/generated_smiles.csv" \
--reference "../../data/prep/guacamol/csv/guacamol_v1_train.csv" \
--stats "../../data/prep/guacamol/fcd_stats/guacamol_train.pkl" \
--num_molecules 10000
```

### Reaction Templates (SMARTS)

```bash
# Download and prepare dataset
python prepare_data.py uspto50k

# Train the model
# add --fp16 false if your GPU does not support fp16 or you run it on a CPU (not recommended)
python train.py --args conf/uspto50k.args  # this also reads the default train.args file

# Generate ≥ 10000 reaction templates
# In this case the evaluation is done during generation
python generate.py smarts \
--model "../../checkpoints/<your_model>" \
--known "../../data/prep/uspto50k/csv/USPTO_50k_known.csv"
--num 10000
```

## Known issues and limitations

- Ran only on a local GPU, not configured/tested for distributed training
- Not tested with pytorch ≥ v2.0
- Starting with transformers v5 (not out as of this writing), the optimizer must be instantiated manually
