#!/bin/bash
#SBATCH -J randolph_inspect
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH -t 4:00:00
#SBATCH -o logs/randolph_inspect_%j.out
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fm-R
cd ~/fm-ancestry-bias
Rscript scripts/11_randolph_inspect.R
