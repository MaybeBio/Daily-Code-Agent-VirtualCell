#!/bin/bash
#SBATCH -J celltypist
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH -t 4:00:00
#SBATCH -o logs/celltypist_%j.out
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fm-core
cd ~/fm-ancestry-bias
python scripts/40_celltypist.py 2>&1 | grep -v -E "Warning|warn"
