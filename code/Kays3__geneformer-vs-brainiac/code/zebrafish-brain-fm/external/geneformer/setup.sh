#!/usr/bin/env bash
# Fetch Geneformer at a pinned commit, download the gene dictionaries and the
# V2-104M model, verify each file, then build this folder's own environment.
# Adapted from quickstart/setup.sh of Kays3/geneformer-petadimension-starter (MIT).
# Safe to run again. Needs git, curl, sha256sum and uv. It does not need git-lfs.
set -euo pipefail
cd "$(dirname "$0")"

GENEFORMER_COMMIT="1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5"
REPO_URL="https://huggingface.co/ctheodoris/Geneformer"

if [ ! -d Geneformer ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --quiet "$REPO_URL" Geneformer
fi
git -C Geneformer checkout --quiet "$GENEFORMER_COMMIT"

cd Geneformer
git ls-files 'geneformer/*.pkl' 'Geneformer-V2-104M/*' | while read -r file; do
  if head -c 64 "$file" | grep -q '^version https://git-lfs'; then
    want_oid=$(sed -n 's/^oid sha256://p' "$file")
    want_size=$(sed -n 's/^size //p' "$file")
    echo "downloading $file ($((want_size / 1000000)) MB)"
    curl --fail --location --silent --show-error --retry 3 \
      -o "$file.part" "$REPO_URL/resolve/$GENEFORMER_COMMIT/$file"
    [ "$(sha256sum "$file.part" | cut -d' ' -f1)" = "$want_oid" ] || { echo "Checksum mismatch for $file" >&2; rm -f "$file.part"; exit 1; }
    mv "$file.part" "$file"
  fi
done
cd ..

size=$(wc -c < Geneformer/Geneformer-V2-104M/model.safetensors)
if [ "$size" -lt 100000000 ]; then
  echo "The model file is only $size bytes: the download did not complete." >&2
  exit 1
fi

uv sync
echo "Geneformer is ready."
