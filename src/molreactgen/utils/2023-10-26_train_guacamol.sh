#!/bin/bash
# Helper script to start a number of pre-configured experiments
# Just a sequence of commands
# Goal: find the (initial) best tokenizer for the Guacamol dataset
# Current directory should contain the python training script

TRAIN_FILE="train.py"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Could not find $TRAIN_FILE, exit."
    exit
fi


echo "== Starting a sequence of experiments"

echo "== Experiment 1/11, pre-tokenizer char, algorithm wordlevel, vocab size 38"
python train.py \
  --config_file conf/guacamol.args \
  --tokenizer_name ../../tokenizers/guacamol/char_wordlevel_38

echo "== Experiment 2/11, pre-tokenizer char, algorithm bpe, vocab size 44"
python train.py \
  --config_file conf/guacamol.args \
  --tokenizer_name ../../tokenizers/guacamol/char_bpe_44

echo "== Experiment 3/11, pre-tokenizer char, algorithm bpe, vocab size 88"
python train.py \
  --config_file conf/guacamol.args \
  --tokenizer_name ../../tokenizers/guacamol/char_bpe_88

echo "== Experiment 4/11, pre-tokenizer char, algorithm bpe, vocab size 176"
python train.py \
  --config_file conf/guacamol.args \
  --tokenizer_name ../../tokenizers/guacamol/char_bpe_176

echo "== Experiment 5/11, pre-tokenizer char, algorithm wordpiece, vocab size 88"
python train.py \
  --config_file conf/guacamol.args \
  --tokenizer_name ../../tokenizers/guacamol/char_wordpiece_88

echo "== Experiment 6/11, pre-tokenizer char, algorithm wordpiece, vocab size 176"
python train.py \
  --config_file conf/guacamol.args \
  --tokenizer_name ../../tokenizers/guacamol/char_wordpiece_176

echo "== Experiment 7/11, pre-tokenizer char, algorithm unigram, vocab size 44"
python train.py \
  --config_file conf/guacamol.args \
  --tokenizer_name ../../tokenizers/guacamol/char_unigram_44

echo "== Experiment 8/11, pre-tokenizer char, algorithm unigram, vocab size 88"
python train.py \
  --config_file conf/guacamol.args \
  --tokenizer_name ../../tokenizers/guacamol/char_unigram_88

echo "== Experiment 9/11, pre-tokenizer char, algorithm unigram, vocab size 176"
python train.py \
  --config_file conf/guacamol.args \
  --tokenizer_name ../../tokenizers/guacamol/char_unigram_176

echo "== Experiment 10/11, pre-tokenizer atom, algorithm wordlevel, vocab size 50"
python train.py \
  --config_file conf/guacamol.args \
  --tokenizer_name ../../tokenizers/guacamol/atom_wordlevel_50

echo "== Experiment 11/11, pre-tokenizer smarts, algorithm wordlevel, vocab size 106"
python train.py \
  --config_file conf/guacamol.args \
  --tokenizer_name ../../tokenizers/guacamol/smarts_wordlevel_106
