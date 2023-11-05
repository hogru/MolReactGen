#!/bin/bash
# Helper script to train a model n times

TRAIN_FILE="train.py"
BREAK_TIME=15

if [ -z "$1" ]; then
  echo "Number of runs not supplied, exit."
  exit
fi

if [ ! -f "$TRAIN_FILE" ]; then
  echo "Could not find $TRAIN_FILE, exit."
  exit
fi

echo "== Train model $1 times"

for (( run=1; run<=$1; run++ ))
do
  echo
  echo "-- Start training run $run/$1"
  python train.py \
    --config_file conf/guacamol.args \
    --tokenizer_name ../../tokenizers/guacamol/char_wordpiece_176 \
    --random_seed true
#   python train.py \
#     --config_file conf/uspto50k.args \
#     --pre_tokenizer smarts \
#     --algorithm wordlevel \
#     --random_seed true
  sleep "$BREAK_TIME"
done
