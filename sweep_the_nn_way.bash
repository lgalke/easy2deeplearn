#!/usr/bin/env bash


# Halt on error
set -e

echo "SCRIPT NOT READY"
exit 1


OUTDIR="./results-the-nn-way-v1"

# Iterate over different seeds here
for SEED in 20000; do
	echo "Seed: $SEED"
	for LEXP in data/LearningExp_*.txt; do
		echo "Starting run with experiment data: $LEXP"
		python3 train.py "$LEXP" --seed "$SEED" --project "tied-weights-con-loss" --tags 'sweep-1' --outdir "$OUTDIR"
		python3 train.py "$LEXP" --seed "$SEED" --project "tied-weights-con-loss" --tags 'sweep-2' --outdir "$OUTDIR"
	done
done
