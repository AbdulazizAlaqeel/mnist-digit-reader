#!/bin/bash
#SBATCH --job-name=mnist
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source /ibex/user/$USER/miniforge/bin/activate /ibex/user/$USER/conda-environments/ml
python -u mnist_train.py