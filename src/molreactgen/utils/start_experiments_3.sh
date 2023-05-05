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

echo "== Experiment 1/3, 100 epochs, fp16, batch size 64, lr 0.0005, hidden dim 144"
python train.py \
  --config_file conf/guacamol.args \
  --tokenizer_name ../../tokenizers/char_unigram_88 \
  --config_overrides n_positions=128,n_embd=144,n_layer=12,n_head=12 \
  --num_train_epochs 100 \
  --fp16 true \
  --learning_rate 0.0005 \
  --per_device_train_batch_size 64

echo "== Experiment 2/3, 50 epochs, fp32, batch size 64, lr 0.0005, hidden dim 144"
python train.py \
  --config_file conf/guacamol.args \
  --tokenizer_name ../../tokenizers/char_unigram_88 \
  --config_overrides n_positions=128,n_embd=144,n_layer=12,n_head=12 \
  --num_train_epochs 50 \
  --fp16 false \
  --learning_rate 0.0005 \
  --per_device_train_batch_size 64

echo "== Experiment 3/3, 50 epochs, fp16, batch size 32, lr 0.0005, hidden dim 768"
python train.py \
  --config_file conf/guacamol.args \
  --tokenizer_name ../../tokenizers/char_unigram_88 \
  --config_overrides n_positions=128,n_embd=768,n_layer=12,n_head=12 \
  --num_train_epochs 50 \
  --fp16 true \
  --learning_rate 0.0005 \
  --per_device_train_batch_size 32
