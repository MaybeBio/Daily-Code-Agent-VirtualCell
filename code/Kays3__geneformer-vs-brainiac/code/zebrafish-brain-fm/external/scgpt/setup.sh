#!/usr/bin/env bash
# Build this folder's own environment for scGPT and download the whole-human
# checkpoint that the scGPT README names. Safe to run again. Needs uv.
set -euo pipefail
cd "$(dirname "$0")"

uv sync
if [ ! -f checkpoint/best_model.pt ]; then
  uv run gdown --folder "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y" -O checkpoint
fi
for file in best_model.pt vocab.json args.json; do
  [ -s "checkpoint/$file" ] || { echo "Missing checkpoint/$file: the download did not complete." >&2; exit 1; }
done
( cd checkpoint && sha256sum best_model.pt vocab.json args.json > SHA256SUMS )
echo "scGPT is ready."
