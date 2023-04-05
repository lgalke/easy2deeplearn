#!/usr/bin/env bash


# Halt on error
set -e

NUM_ITER=1000
SEED=42000


OUTDIR="./results_ll_con_loss_effect"

# Iterate over different seeds here
for CON_WEIGHT in "0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0"; do
        for LEXP in data/LearningExp_*.txt; do
                python3 train.py "$LEXP" --seed "$SEED" --project "language-learning" --tags "con_loss_effect" \
                        --iterations "$NUM_ITER" --con_weight "$CON_WEIGHT" \
                        --outdir "$OUTDIR"
        done
done
