#!/bin/sh
#$ -N hidden-size-effect-2
#$ -cwd 
#$ -q cuda.q
#$ -S /bin/bash
#$ -M lukas.galke@mpi.nl
#$ -m beas

# Halt on error
set -e

module load miniconda/3.2021.10
conda activate pytorch
pip install -r requirements.txt

NUM_ITER=1000
SEED=42000


OUTDIR="./results_ll_hidden_size_effect_2"


NUM_LAYERS=2

# Iterate over different seeds here
for LEXP in data/LearningExp_*.txt; do
	for HIDDEN_SIZE in 8 16 32 64 128; do
		echo "Learning experiment: $LEXP";
		python3 train.py "$LEXP" --seed "$SEED" --project "language-learning" --tags "hidden_size_effect_2" "two-layers" \
			--iterations "$NUM_ITER" --con_weight "0.1" --hidden_size $HIDDEN_SIZE --num_layers "$NUM_LAYERS" \
			--outdir "$OUTDIR"
	done
done
