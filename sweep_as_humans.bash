#!/usr/bin/env bash


# Halt on error
set -e


OUTDIR="./results-v1"

# Iterate over different seeds here
for SEED in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000; do
	echo "Seed: $SEED"
	for LEXP in data/LearningExp_*.txt; do
		echo "Starting run with experiment data: $LEXP"
		python3 train.py --as_humans "$LEXP" --seed "$SEED" --tags 'complete-results-v1' 'Home' 'Restart' --outdir "$OUTDIR"
	done
done
