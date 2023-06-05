#!/bin/bash
# Helper script to start a number of pre-configured experiments
# Just a sequence of commands
# Current directory should contain the python training script

TRAIN_FILE="train.py"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Could not find $TRAIN_FILE, exit."
    exit
fi

echo "== Starting a sequence of experiments"

echo "== Experiment 1/1, Finetuning GPT-2"
python train.py \
  --config_file conf/guacamol_finetuning.args
