# MolReactGen

An auto-regressive causal language model for molecule (SMILES) and reaction template (SMARTS) generation. Based on the [Hugging Face implementation](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2#openai-gpt2) of [OpenAI’s GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) [transformer decoder](https://arxiv.org/abs/1706.03762v5) model.

## Main research question:

- How well does this model perform for molecule generation, using the [GuacaMol paper](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839) as a benchmark?
- What is the effect of different SMILES tokenization approaches (different RegEx expressions as pre-tokenizers, tokenization algorithms such as BPE, WordPiece)?
- Can we use this model to generate reaction templates (harder since less training data, what is the evaluation criterion?)
- Can we use a model pre-trained on _natural_ language as a basis for fine-tuning a “_molecule_ language” (SMILES, SMARTS)?

## Installation

This is currently under development and should not be considered for any serious use case besides academic curiosity. A local (editable package) installation requires `python` ≥ 3.9, [`poetry`](https://python-poetry.org)  ≥ 1.0.8 and `pip` ≥ 22.3. Experiment results are logged to [`weights and biases`](https://wandb.ai).

```
git clone https://github.com/hogru/molreactgen
cd molreactgen
python -m pip install -e .
```

## Main files

- `prepare_data.py` downloads and massages the datasets (download not implemented yet)
- `train.py` trains the model on a given dataset, configured via `yaml` files in the `conf` directory
- `generate.py` generates SMILES or SMARTS
- `evaluate_fcd.py` calculates the Frèchet ChemNet Distance (FCD) between the generated molecules and a reference set of molecules (e.g. the GuacaMol dataset)
- `molecule.py` covers helpers for the chemical space of the task
- `helpers.py` is a set of misc helpers/utils
