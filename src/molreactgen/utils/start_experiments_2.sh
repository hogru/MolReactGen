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

# python train.py \
#   --config_file conf/guacamol.args \
#   --tokenizer_name ../../tokenizers/char_unigram_88 \
#   --learning_rate 0.0025 \
#   --per_device_train_batch_size 64

echo "== Experiment 1/1, lr 0.0025, batch size 32"
python train.py \
  --config_file conf/guacamol.args \
  --tokenizer_name ../../tokenizers/char_unigram_88 \
  --learning_rate 0.0025 \
  --per_device_train_batch_size 32

# python train.py \
#   --config_file conf/guacamol.args \
#   --tokenizer_name ../../tokenizers/char_unigram_88 \
#   --learning_rate 0.0025 \
#   --per_device_train_batch_size 256

#echo "== Experiment 2/6, lr 0.0005, batch size 64"
#python train.py \
#  --config_file conf/guacamol.args \
#  --tokenizer_name ../../tokenizers/char_unigram_88 \
#  --learning_rate 0.0005 \
#  --per_device_train_batch_size 64
#
# python train.py \
#   --config_file conf/guacamol.args \
#   --tokenizer_name ../../tokenizers/char_unigram_88 \
#   --learning_rate 0.0005 \
#   --per_device_train_batch_size 128
#
#echo "== Experiment 3/6, lr 0.0005, batch size 256"
#python train.py \
#  --config_file conf/guacamol.args \
#  --tokenizer_name ../../tokenizers/char_unigram_88 \
#  --learning_rate 0.0005 \
#  --per_device_train_batch_size 256
#
#echo "== Experiment 4/6, lr 0.0001, batch size 64"
#python train.py \
#  --config_file conf/guacamol.args \
#  --tokenizer_name ../../tokenizers/char_unigram_88 \
#  --learning_rate 0.0001 \
#  --per_device_train_batch_size 64
#
#echo "== Experiment 5/6, lr 0.0001, batch size 128"
#python train.py \
#  --config_file conf/guacamol.args \
#  --tokenizer_name ../../tokenizers/char_unigram_88 \
#  --learning_rate 0.0001 \
#  --per_device_train_batch_size 128
#
#echo "== Experiment 6/6, lr 0.0001, batch size 256"
#python train.py \
#  --config_file conf/guacamol.args \
#  --tokenizer_name ../../tokenizers/char_unigram_88 \
#  --learning_rate 0.0001 \
#  --per_device_train_batch_size 256
