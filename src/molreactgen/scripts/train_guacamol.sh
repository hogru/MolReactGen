#!/bin/bash
# Sample training script for guacamol
# Current directory should contain the python training script
# i.e. src/molreactgen/

TRAIN_FILE="train.py"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Could not find $TRAIN_FILE, exit."
    exit
fi

echo "== Experiment 1/1, pre-tokenizer char, algorithm wordpiece, vocab size 176"
python train.py \
  --config_file conf/guacamol.args \
  --tokenizer_name ../../tokenizers/guacamol/char_wordpiece_176
