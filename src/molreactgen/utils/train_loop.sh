#!/bin/bash
# Helper script to train a model n times

TRAIN_FILE="train.py"
BREAK_TIME=60

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
#  python train.py \
#    --config_file conf/guacamol.args \
#    --tokenizer_name ../../tokenizers/char_unigram_88 \
#    --learning_rate 0.0025 \
#    --per_device_train_batch_size 64 \
#    --random_seed true
  python train.py \
    --config_file conf/uspto50k.args \
    --random_seed true
  sleep "$BREAK_TIME"
done
