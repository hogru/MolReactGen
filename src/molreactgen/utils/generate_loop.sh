#!/bin/bash
# Helper script to generate items n times

GENERATE_FILE="utils/generate_smiles.sh"
CHECKPOINT_DIR="../../checkpoints/step_7-finetuning-runs/2023-05-07_03-20-38_experiment"
BREAK_TIME=30

if [ -z "$1" ]; then
  echo "Number of runs not supplied, exit."
  exit
fi

if [ ! -f "$GENERATE_FILE" ]; then
  echo "Could not find $GENERATE_FILE, exit."
  exit
fi

echo "== Generate items $1 times"

for (( run=1; run<=$1; run++ ))
do
  echo
  echo "-- Start generating run $run/$1"
  $GENERATE_FILE $CHECKPOINT_DIR
  sleep "$BREAK_TIME"
done
