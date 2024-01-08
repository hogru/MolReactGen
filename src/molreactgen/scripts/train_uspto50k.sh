#!/bin/bash
# Sample training script for uspto-50k
# Current directory should contain the python training script
# i.e. src/molreactgen/

TRAIN_FILE="train.py"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Could not find $TRAIN_FILE, exit."
    exit
fi

echo "== Experiment 1/1, pre-tokenizer smarts, algorithm wordlevel, vocab size 947"
python train.py \
  --config_file conf/uspto50k.args \
  --tokenizer_name ../../tokenizers/uspto50k/smarts_wordlevel_947
