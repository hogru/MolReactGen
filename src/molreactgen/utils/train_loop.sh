#!/bin/bash
# Helper script to train a model n times

TRAIN_FILE="train.py"
BREAK_TIME=3

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
# for run in {1..5}
do
  echo
  echo "-- Start training run $run/$1"
  echo python3 train.py
  sleep "$BREAK_TIME"
done
