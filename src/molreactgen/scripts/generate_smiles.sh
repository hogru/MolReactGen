#!/bin/bash
# Sample script to generate SMILES from a single model
# Generates SMILES based on variables defined below
# Current directory should contain the python scripts
# i.e. src/molreactgen/

GENERATE_FILE="generate.py"
KNOWN_FILE="../../data/prep/guacamol/csv/guacamol_v1_train.csv"
NUM_MOLS=10000
NUM_BEAMS=1
TEMPERATURE=1.0
REPETITION_PENALTY=1.0

if [ -z "$1" ]; then
  echo "No directory supplied, exit."
  exit
fi

if [ ! -f "$GENERATE_FILE" ]; then
  echo "Could not find $GENERATE_FILE, exit."
  exit
fi

if [ ! -f "$KNOWN_FILE" ]; then
  echo "Could not find $KNOWN_FILE, exit."
  exit
fi


echo "== Generating $NUM_MOLS molecules for model in $1"
echo "-- Reference file: $KNOWN_FILE"
echo


echo "---- Starting generation at $(date +%T) with model in $1 ..."
python generate.py smiles \
--model "$1" \
--known "$KNOWN_FILE" \
--num "$NUM_MOLS" \
--num_beams "$NUM_BEAMS" \
--temperature "$TEMPERATURE" \
--repetition_penalty "$REPETITION_PENALTY" \
--random_seed
date +"---- Finished generation at %T"
