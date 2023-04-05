#!/usr/bin/env bash


# Halt on error
set -e

NUM_ITER=1000
SEED=42000


OUTDIR="./results_ll_tied_weights_effect"

# Iterate over different seeds here
for LEXP in data/LearningExp_*.txt; do
	for TIED_WEIGHTS in "all" "within" "between" "none"; do
		echo "Learning experiment: $LEXP // TIED WEIGHTS: $TIED_WEIGHTS";
		python3 train.py "$LEXP" --seed "$SEED" --project "language-learning" --tags "tied_weights_effect" \
			--iterations "$NUM_ITER" --con_weight "0.1" --tied_weights "$TIED_WEIGHTS" \
			--outdir "$OUTDIR"
	done
done
