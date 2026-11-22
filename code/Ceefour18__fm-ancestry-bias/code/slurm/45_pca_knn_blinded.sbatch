#!/bin/bash
#SBATCH -J pca_knn
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
#SBATCH -t 4:00:00
#SBATCH -o logs/pca_knn_%j.out
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fm-core
cd ~/fm-ancestry-bias
python scripts/45_pca_knn_blinded.py 2>&1 | grep -v -E "Warning|warn"
