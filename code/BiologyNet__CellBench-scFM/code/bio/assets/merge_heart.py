"""Merge two prepared AnnData files on shared genes, tagging assay as batch."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse


def merge_prepared(paths, labels, target, provenance_extra, limit, seed):
    import anndata as ad
    pieces = []
    for path, label in zip(paths, labels):
        data = ad.read_h5ad(path)
        data.obs_names = pd.Index([f'{label}:{name}' for name in data.obs_names.astype(str)])
        data.obs['assay'] = str(label)
        data.obs['batch'] = data.obs['donor_id'].astype(str) + ':' + str(label) if 'donor_id' in data.obs else str(label)
        pieces.append(data)
    genes = pieces[0].var_names
    for part in pieces[1:]:
        genes = genes.intersection(part.var_names)
    if len(genes) < 1000:
        raise ValueError(f'Too few shared genes to merge: {len(genes)}')
    aligned = []
    for part in pieces:
        subset = part[:, genes].copy()
        aligned.append(subset)
    data = ad.concat(aligned, join='inner', index_unique=None)
    if not data.obs_names.is_unique:
        raise ValueError('Merged cell IDs are not unique')
    if data.n_obs > limit:
        from bio.assets.prepare_datasets import stratified_rows
        keys = ['donor_id', 'celltype'] if 'donor_id' in data.obs else ['celltype']
        rows = stratified_rows(data.obs, keys, limit, seed)
        data = data[rows].copy()
    if sparse.issparse(data.X):
        data.X = data.X.tocsr().astype(np.float32)
    manifest = dict(
        id='mouse_heart',
        cells=int(data.n_obs),
        genes=int(data.n_vars),
        sources=[{'path': str(p), 'assay': lab, 'cells': int(ad.read_h5ad(p, backed='r').n_obs)}
                 for p, lab in zip(paths, labels)],
        shared_genes=int(len(genes)),
        seed=seed,
        limit=limit,
        selection='inner-join genes; donor+celltype quotas if over limit; assay in batch',
        cell_id_sha256=hashlib.sha256('\n'.join(data.obs_names.astype(str)).encode()).hexdigest(),
        **provenance_extra,
    )
    data.uns['cellbench_provenance'] = json.dumps(manifest)
    target = Path(target)
    temporary = target.with_suffix('.prepared.h5ad')
    data.write_h5ad(temporary, compression='lzf')
    temporary.replace(target)
    target.with_suffix('.provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ('id', 'cells', 'genes', 'shared_genes', 'seed')}), flush=True)
    return manifest
