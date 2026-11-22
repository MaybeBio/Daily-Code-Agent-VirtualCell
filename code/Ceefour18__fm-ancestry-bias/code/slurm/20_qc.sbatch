#!/bin/bash
#SBATCH -J qc
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH -t 6:00:00
#SBATCH -o logs/qc_%j.out
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fm-core
cd ~/fm-ancestry-bias
python scripts/20_qc.py 2>&1 | grep -v -E "UserWarning|warn_names|FutureWarning"
