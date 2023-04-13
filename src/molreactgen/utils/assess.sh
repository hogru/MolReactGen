#!/bin/bash
# Helper script to evaluate molecules from a set of generated molecules
# Directory with sub-directories of generated molecules expected
# Evaluates molecules based on variables defined below
# Current directory should contain the python scripts

ASSESS_FILE="assess.py"
REFERENCE_MOLECULES="../../data/prep/guacamol/csv/guacamol_v1_train.csv"
REFERENCE_STATS="../../data/prep/guacamol/fcd_stats/guacamol_train_fcd_stats.pkl"
GENERATED_FILE="generated_smiles.csv"
NUM_MOLS=10000
BREAK_TIME=30

if [ -z "$1" ]; then
  echo "No directory supplied, exit."
  exit
fi

if [ ! -f "$ASSESS_FILE" ]; then
    echo "Could not find $ASSESS_FILE, exit."
    exit
fi

if [ ! -f "$REFERENCE_MOLECULES" ]; then
    echo "Could not find $REFERENCE_MOLECULES, exit."
    exit
fi

if [ ! -f "$REFERENCE_STATS" ]; then
    echo "Could not find $REFERENCE_STATS, exit."
    exit
fi


echo "== Assessing $NUM_MOLS molecules for each generated molecules in $1"
echo "-- Reference molecules: $REFERENCE_MOLECULES"
echo "-- Reference stats: $REFERENCE_STATS"

for dir in $(find "$1" -mindepth 1 -maxdepth 2 -type d); do
    file="$dir/$GENERATED_FILE"
    if [ -f "$file" ]; then
        echo "---- Starting evaluation at $(date +%T) with molecules in $file ..."
        echo python assess.py stats \
        --generated "$file" \
        --reference "$REFERENCE_MOLECULES" \
        --stats "$REFERENCE_STATS" \
        --known "$KNOWN_FILE" \
        --num_molecules "$NUM_MOLS"
        date +"---- Finished evaluation at %T"
        echo "Pausing for $BREAK_TIME second(s)..."
        sleep "$BREAK_TIME"
    else
        echo "-- Could not find molecules in $dir"
    fi
done
