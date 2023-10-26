#!/bin/bash
# Helper script to generate molecules from a set of models
# Directory with sub-directories of models expected
# Generates molecules based on variables defined below
# Current directory should contain the python scripts

GENERATE_FILE="generate.py"
MODEL_FILE="pytorch_model.bin"
KNOWN_FILE="../../data/prep/uspto50k/csv/USPTO_50k_known.csv"
OPTIM_FILE="scheduler.pt"
NUM_RTS=10000
NUM_BEAMS=1
REPETITION_PENALTY=1.0
BREAK_TIME=15

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


echo "== Generating $NUM_RTS reaction templates for each model in $1"
echo "-- Reference file: $KNOWN_FILE"
echo

for dir in $(find "$1" -mindepth 0 -maxdepth 1 -type d); do
  if [ -f "$dir/$MODEL_FILE" ] && [ ! -f "$dir/$OPTIM_FILE" ]; then
    echo "---- Starting generation at $(date +%T) with model in $dir ..."
    python generate.py smarts \
    --model "$dir" \
    --known "$KNOWN_FILE" \
    --num "$NUM_RTS" \
    --num_beams "$NUM_BEAMS" \
    --repetition_penalty "$REPETITION_PENALTY"
    date +"---- Finished generation at %T"
    echo -e "\nPausing for $BREAK_TIME second(s)...\n"
    sleep "$BREAK_TIME"
  elif [ -f "$dir/$OPTIM_FILE" ]; then
    echo "-- Skipping model in $dir, seems to be a checkpoint"
  else
    echo "-- Could not find model in $dir"
  fi
done
