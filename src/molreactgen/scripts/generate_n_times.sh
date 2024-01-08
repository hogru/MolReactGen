#!/bin/bash
# Sample script to generate SMILES or SMARTS from a single model
# Current directory should contain the python scripts
# i.e. src/molreactgen/

GENERATE_FILE="utils/2023-12-09_generate_smiles.sh"
# GENERATE_FILE="utils/generate_smarts.sh"
MODEL_DIR="../../checkpoints/guacamol/tokenizers/char_wordpiece_176/2023-10-30_14-29-16_experiment"
BREAK_TIME=15

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
  $GENERATE_FILE $MODEL_DIR
  sleep "$BREAK_TIME"
done
