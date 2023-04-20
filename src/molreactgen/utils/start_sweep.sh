#!/bin/bash
# Helper script to start a wandb sweep
# Mainly used to set environment variables
# Current directory should contain the python training script

TRAIN_FILE="train.py"
DISABLE_FLAPPING=false
MAX_INITIAL_FAILURES=2
WAIT_TIME=15

if [ -z "$1" ]; then
  echo "No sweep id, exit."
  exit
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Could not find $TRAIN_FILE, exit."
    exit
fi

export WANDB_AGENT_DISABLE_FLAPPING="$DISABLE_FLAPPING"
export WANDB_AGENT_MAX_INITIAL_FAILURES="$MAX_INITIAL_FAILURES"

echo "== Starting a wandb sweep with sweep id $1"
echo "-- Disable Flapping: $WANDB_AGENT_DISABLE_FLAPPING"
echo "-- Max Initial Failures: $WANDB_AGENT_MAX_INITIAL_FAILURES"

echo "-- Is this correct? If no, press Ctrl+C to exit."
echo "-- Waiting for $WAIT_TIME second(s)..."
sleep "$WAIT_TIME"

echo "-- Starting wandb agent..."
wandb agent "hogru/MolReactGen/$1"
