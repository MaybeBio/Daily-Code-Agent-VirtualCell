#!/bin/bash
# td-serve on one GPU under Slurm; the URL is written to $URL_FILE once the server accepts requests (~5 min with the
# warm-up). Run from the repository root:
#
#   MODEL=/path/to/Thalamus-Decision-9B PROFILES="mine.npz other.npz" sbatch --partition=<p> scripts/serve.sbatch
#
#SBATCH --job-name=td-serve
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=td-serve-%j.out
set -euo pipefail
MODEL=${MODEL:-ingenix/Thalamus-Decision-9B}
PORT=${PORT:-8000}
URL_FILE=${URL_FILE:-$PWD/td-serve.url}
PY=${PY:-uv run}
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec $PY td-serve --model "$MODEL" ${PROFILES:+--profiles $PROFILES} ${PROFILE_STORE:+--profile-store $PROFILE_STORE} \
  --host 0.0.0.0 --port $PORT --url-file "$URL_FILE" ${EXTRA:-}
