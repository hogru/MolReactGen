![Python Version](https://img.shields.io/badge/python-3.9-blue?color=3975A5&logo=python&link=https%3A%2F%2Fwww.python.org)
![Poetry Version](https://img.shields.io/badge/poetry-1.6-blue?color=1E293B&logo=poetry&link=https%3A%2F%2Fpython-poetry.org)
![Pytorch Version](https://img.shields.io/badge/pytorch-1.13-blue?color=EE4C2C&logo=pytorch&link=https%3A%2F%2Fpytorch.org)
![Transformers Version](https://img.shields.io/badge/hf%20transformers-4.33-blue?color=FFD21E&link=https%3A%2F%2Fhuggingface.co%2Fdocs%2Ftransformers%2Findex)
[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)


![Code Style](https://img.shields.io/badge/code%20style-black-black?link=https%3A%2F%2Fpypi.org%2Fproject%2Fblack%2F)
![Imports](https://img.shields.io/badge/imports-isort-blue?color=EF8336&link=https%3A%2F%2Fpypi.org%2Fproject%2Fisort%2F)

![License](https://img.shields.io/badge/license-MIT-blue?color=7A0014&link=https%3A%2F%2Fopensource.org%2Flicense%2Fmit%2F)

# MolReactGen

> An auto-regressive causal language model for molecule (SMILES) and reaction template (SMARTS) generation. Based on the [Hugging Face implementation](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2#openai-gpt2)
of [OpenAI’s GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) [transformer decoder](https://arxiv.org/abs/1706.03762v5) model.

## Table of Contents

- [Abstract](#Abstract)
- [Research questions](#Research-questions)
- [Installation](#Installation)
- [Main files](#Main-files)
- [Usage example](#Usage-example)
- [Release History](#Release-History)
- [Known Issues](#Known-Issues)
- [Meta](#Meta)

## Abstract

This work focuses on the world of chemistry, with the goal of supporting the discovery of drugs to cure diseases or
sustainable materials for cleaner energy. The research explores the potential of a transformer decoder model in
generating chemically feasible molecules and reaction templates. We begin with contrasting the performance
of [GuacaMol](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839) for molecule generation with a transformer decoder
architecture, assessing the influence of various tokenizers on performance. The study also involves fine-tuning a
pre-trained language model and comparing its outcomes with a model trained from scratch. It utilizes multiple metrics,
including the [Fréchet ChemNet Distance](https://pubmed.ncbi.nlm.nih.gov/30118593/), to evaluate the model's ability to
generate new, valid molecules similar to the training data. The research indicates that the transformer decoder model
outperforms the GuacaMol model in terms of this metric, and is also successful in generating known reaction templates.

## Research questions

- How well does this model perform for molecule generation, using
  the [GuacaMol paper](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839) as a benchmark?
- What is the effect of different tokenization approaches (different RegEx expressions as pre-tokenizers, tokenization
  algorithms such as BPE, WordPiece)?
- Can we use a model pre-trained on _natural_ language as a basis for fine-tuning a “_molecule_ language” model?
- Can we use this approach/model to generate reaction templates?

## Installation

Disclaimer: This is currently under development. A local (editable package) installation requires `python` ≥
3.9, [`poetry`](https://python-poetry.org)  ≥ 1.0.8 and `pip` ≥ 22.3. Experiment results are logged
to [`weights and biases`](https://wandb.ai).

```
git clone --recurse-submodules https://github.com/hogru/molreactgen
cd molreactgen
python -m pip install -e .
```

## Main files

### `src/molreactgen` directory

- `prepare_data.py` downloads and prepares the datasets
- `train.py` trains the model on a given dataset, configured via (optionally multiple) `.args` file(s) or a
  single `.yaml` file in the `conf` directory (see example files)
- `generate.py` generates molecules (SMILES) or reaction templates (SMARTS)
- `assess.py` (for molecules only) calculates the Fréchet ChemNet Distance (FCD) between the generated molecules and a
  reference set of molecules (e.g. the GuacaMol dataset) along with some basic evaluation criteria
- `molecule.py` covers helpers for the chemical space of the task
- `tokenizer.py` provides the various tokenizers
- `helpers.py` is a set of misc helpers/utils (logging etc.)

### `src/molreactgen/utils` directory

- `compute_fcd_stats.py` computes the model activations that are needed to calculate the FCD. This is a separate script
  because it is computationally expensive and can be reused for model comparison.
- `check_tokenizer.py` is used if a tokenizer can successfully encode and decode a dataset
- `collect_metrics.py` collects metrics from various files and `wandb` and provides them in several formats; used during
  experiments
- `create_plots.ipynb` is a Jupyter notebook that creates plots from the datasets; used for presentation purposes
- `*.sh` are "quick and dirty" scripts for running experiments/sweeps for hyper parameter search

### `data/raw` directory

- the (default) directory `prepare_data.py` downloads the datasets to
- a sub-directory is created for each dataset, containing the raw data files

### `data/prep` directory

- the (default) directory `prepare_data.py` prepares the datasets in
- a sub-directory is created for each dataset, containing the prepared data files

## Usage example

### Pre-conditions

- Local repository installation (see above)
- Python 3.9 - it should work with ≥ 3.10 as well, but I haven't tested it
- `poetry` installed (see [here](https://python-poetry.org/docs/#installation))
- `poetry shell` in directory `molreactgen/src/molreactgen` to activate the virtual environment
- Optional: `wandb` account and API key (see [here](https://docs.wandb.ai/quickstart)); should work with an anonymous
  account, but I haven't tested it

> Note: the Hugging Face `trainer` uses its own [`accelerate`](https://huggingface.co/docs/accelerate/index) library
> under the hood. This library is supposed to support a number of distributed training backends. It should work with its
> default values for a simple setup, but you might want /need to change the `accelerate` parameters. You can do this by
> issuing the `accelerate config` command. This is my current setup:
>
> ```yaml
> compute_environment: LOCAL_MACHINE
> distributed_type: 'NO'
> downcast_bf16: 'no'
> machine_rank: 0
> main_training_function: main
> mixed_precision: fp16
> num_machines: 1
> num_processes: 1
> rdzv_backend: static
> same_network: true
> tpu_env: []
> tpu_use_cluster: false
> tpu_use_sudo: false
> use_cpu: false
> ```

### Pipeline

- `cd` into the `molreactgen/src/molreactgen` directory and run the following commands:

#### Molecules (SMILES)

```bash
# Download and prepare dataset
python prepare_data.py guacamol

# Train the model
# add --fp16 false if your GPU does not support fp16 or you run it on a CPU (not recommended)
python train.py --config_file conf/guacamol.args  # this also reads the default train.args file

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
python assess.py smiles \
--mode stats \
--generated "../../data/generated/<generation directory>/generated_smiles.csv" \
--reference "../../data/prep/guacamol/csv/guacamol_v1_train.csv" \
--stats "../../data/prep/guacamol/fcd_stats/guacamol_train.pkl" \
--num_molecules 10000
```

#### Reaction Templates (SMARTS)

```bash
# Download and prepare dataset
python prepare_data.py uspto50k

# Train the model
# add --fp16 false if your GPU does not support fp16 or you run it on a CPU (not recommended)
python train.py --config_file conf/uspto50k.args  # this also reads the default train.args file

# Generate ≥ 10000 reaction templates
# In this case the evaluation is done during generation
python generate.py smarts \
--model "../../checkpoints/<your_model>" \
--known "../../data/prep/uspto50k/csv/USPTO_50k_known.csv"
--num 10000

# Evaluate the generated reaction templates
# At the moment, the assessment is fully done during the generation already
```

### Hugging Face Models

Pre-trained models are available on [Hugging Face](https://huggingface.co), both
for [molecules](https://huggingface.co/hogru/MolReactGen-GuacaMol-Molecules) (SMILES)
and [reaction templates](https://huggingface.co/hogru/MolReactGen-USPTO50K-Reaction-Templates) (SMARTS).

## Release History

- None yet - Work in progress

## Known issues

- Ran only on a local GPU, not configured/tested for distributed training
- Not tested with pytorch ≥ v2.0
- Starting with transformers v5 (not out as of this writing), the optimizer must be instantiated manually; this requires
  a code change in `train.py`
- Does not detect Apple devices automatically; you can use command line argument `--use_mps_device true` to take advantage of Apple Silicon (assuming `pytorch` is configured correctly)
- The current `pyproject.toml` does not update to the following versions due to required testing and, in some cases, their potential breaking changes:
  - python ≥ 3.10 (should work up to 3.11 when also upgrading to pytorch ≥ 2.0)
  - pytorch ≥ 2.0 (not tested, major version)
  - transformers ≥ 4.33 (not tested, tokenizer breaking changes with ≥ 4.34)
  - tokenizers ≥ 0.14 (breaking changes)
  - pandas ≥ 2.0 (not tested, major version)

## Meta

- Stephan Holzgruber - stephan.holzgruber@gmail.com
- Distributed under the MIT license. See `LICENSE` for more information.
- [https://github.com/hogru/MolReactGen](https://github.com/hogru/MolReactGen)
