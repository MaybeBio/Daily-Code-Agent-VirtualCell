#!/usr/bin/env python3
"""Two-cell CellFM inference smoke: load checkpoint, encode, check IDs and finiteness."""
from pathlib import Path
import json
import subprocess
import sys

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / 'runtime_envs/cellfm/bin/python'
WEIGHTS = ROOT / 'weights/CellFM'
SOURCE = ROOT / 'public_code/CellFM'
SOURCE_DATA = ROOT / 'datasets/pbmc_12k.h5ad'
OUT = ROOT / 'output/adapter_smoke/cellfm'


def main():
    if not PYTHON.is_file():
        raise SystemExit(f'missing CellFM python: {PYTHON}')
    OUT.mkdir(parents=True, exist_ok=True)
    data = ad.read_h5ad(SOURCE_DATA)
    subset = data[:2].copy()
    inp = OUT / 'two_cells.h5ad'
    subset.write_h5ad(inp)
    settings = OUT / 'backend_settings.json'
    settings.write_text(json.dumps({
        'variant': 'base',
        'backend_device': 'CPU',
        'inference_batch_size': 1,
        'max_length': 2048,
        'organism': 'NCBITaxon:9606',
        'cpu_threads': 4,
    }, indent=2))
    output = OUT / 'foundation_embeddings.npz'
    cmd = [str(PYTHON), str(ROOT / 'scripts/extract_frontier_embeddings.py'),
           '--model', 'CellFM', '--input', str(inp), '--output', str(output),
           '--weights', str(WEIGHTS), '--source', str(SOURCE),
           '--settings', str(settings), '--seed', '41']
    print('RUN', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    with np.load(output, allow_pickle=False) as result:
        x, ids = result['embeddings'], result['cell_ids']
    expected = subset.obs_names.to_numpy(dtype=str)
    if x.ndim != 2 or x.shape[0] != 2 or x.shape[1] == 0:
        raise SystemExit(f'bad shape {x.shape}')
    if not np.isfinite(x).all():
        raise SystemExit('nonfinite embeddings')
    if list(ids) != list(expected):
        raise SystemExit(f'cell id mismatch {ids} vs {expected}')
    report = dict(shape=list(map(int, x.shape)), dtype=str(x.dtype),
                  l2=[float(np.linalg.norm(row)) for row in x],
                  cell_ids=list(map(str, ids)))
    (OUT / 'smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print('CELLFM_SMOKE_OK', report, flush=True)


if __name__ == '__main__':
    main()
