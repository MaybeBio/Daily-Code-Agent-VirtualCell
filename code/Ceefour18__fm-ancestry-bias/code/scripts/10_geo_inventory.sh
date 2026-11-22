#!/usr/bin/env bash
set -euo pipefail
OUT=data/raw/GSE162632
mkdir -p "$OUT"
cd "$OUT"

curl -s https://ftp.ncbi.nlm.nih.gov/geo/series/GSE162nnn/GSE162632/suppl/ \
  | grep -oE 'href="[^"]+"|[0-9.]+[KMG] ' | paste - - | tee suppl_listing.txt

curl -sO https://ftp.ncbi.nlm.nih.gov/geo/series/GSE162nnn/GSE162632/matrix/GSE162632_series_matrix.txt.gz
zcat GSE162632_series_matrix.txt.gz | grep -E '^!Sample_(title|characteristics_ch1|source_name_ch1|relation)' \
  > sample_metadata_raw.txt
echo; echo "Sample metadata lines:"; cut -c1-200 sample_metadata_raw.txt | head -n 20

grep -iE 'SRA|dbgap|phs[0-9]+' sample_metadata_raw.txt | head -n 5 || echo "No SRA/dbGaP link in series matrix"
echo; echo "Done."