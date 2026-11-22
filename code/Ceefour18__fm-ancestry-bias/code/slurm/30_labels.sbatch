#!/bin/bash
#SBATCH -J labels
#SBATCH --mem=400G
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00
#SBATCH -o logs/labels_%j.out
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fm-core
cd ~/fm-ancestry-bias
export OMP_NUM_THREADS=32
python scripts/30_labels.py randolph data/processed/randolph_qc.h5ad batch 2>&1 | grep -v -E "Warning|warn"
python scripts/30_labels.py prostate data/processed/prostate_qc.h5ad subject_id 2>&1 | grep -v -E "Warning|warn"
