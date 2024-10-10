#!/usr/bin/env bash


# Halt on error
set -e


OUTDIR="./results-v1"

SEED=123456789
LEXP="data/LearningExp_190501_S5_001_log.txt"
echo "Seed: $SEED"
echo "Starting run with experiment data: $LEXP"
python3 train.py --as_humans "$LEXP" --seed "$SEED" --debug --iterations 5 --outdir "tmp/"
