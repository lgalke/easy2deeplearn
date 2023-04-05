#!/usr/bin/env bash


# Halt on error
set -e

NUM_ITER=1000
SEED=42000


OUTDIR="./results_ll_num_layers_effect"

# Iterate over different seeds here
for LEXP in data/LearningExp_*.txt; do
	for NUM_LAYERS in 3 2 1; do
		echo "Learning experiment: $LEXP";
		python3 train.py "$LEXP" --seed "$SEED" --project "language-learning" --tags "num_layers_effect" \
			--iterations "$NUM_ITER" --con_weight "0.1" --num_layers "$NUM_LAYERS" \
			--outdir "$OUTDIR"
	done
done
