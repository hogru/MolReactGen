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

#echo "== Experiment 1/9, char wordlevel"  # DONE
#python train.py \
#  --config_file conf/uspto50k.args \
#  --pre_tokenizer char \
#  --algorithm wordlevel \
#  --vocab_size 0

echo "== Experiment 2/9, char bpe 88"
python train.py \
  --config_file conf/uspto50k.args \
  --pre_tokenizer char \
  --algorithm bpe \
  --vocab_size 88

echo "== Experiment 3/9, char bpe 176"
python train.py \
  --config_file conf/uspto50k.args \
  --pre_tokenizer char \
  --algorithm bpe \
  --vocab_size 176

echo "== Experiment 4/9, char wordpiece 88"
python train.py \
  --config_file conf/uspto50k.args \
  --pre_tokenizer char \
  --algorithm wordpiece \
  --vocab_size 88

echo "== Experiment 5/9, char wordpiece 176"
python train.py \
  --config_file conf/uspto50k.args \
  --pre_tokenizer char \
  --algorithm wordpiece \
  --vocab_size 176

echo "== Experiment 6/9, char unigram 88"
python train.py \
  --config_file conf/uspto50k.args \
  --pre_tokenizer char \
  --algorithm unigram \
  --vocab_size 88

echo "== Experiment 7/9, char unigram 176"
python train.py \
  --config_file conf/uspto50k.args \
  --pre_tokenizer char \
  --algorithm unigram \
  --vocab_size 176

#echo "== Experiment 8/9, atom wordlevel"  # DONE
#python train.py \
#  --config_file conf/uspto50k.args \
#  --pre_tokenizer atom \
#  --algorithm wordlevel \
#  --vocab_size 0
#
#echo "== Experiment 9/9, smarts wordlevel "  # DONE
#python train.py \
#  --config_file conf/uspto50k.args \
#  --pre_tokenizer smarts \
#  --algorithm wordlevel \
#  --vocab_size 0
