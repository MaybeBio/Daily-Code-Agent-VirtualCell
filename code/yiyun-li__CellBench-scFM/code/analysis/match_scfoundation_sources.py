"""Reproduce label-level overlap from scFoundation supplemental data and local h5ad files."""
from pathlib import Path
import re
import pandas as pd
import anndata as ad

root=Path(os.environ.get("CELLBENCH_ROOT", str(Path(__file__).resolve().parents[1])))
project=Path(__file__).resolve().parents[1]
u='https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-024-02305-7/MediaObjects/41592_2024_2305_MOESM4_ESM.xlsx'
x=pd.read_excel(u)
for dataset,study,filename in [('heart','HCA-HeartSingleCellsAndNucleiSeq','heart.h5ad'),('ms','HCA-Oligodendrocyte_MS','filtered_ms_adata.h5ad')]:
 src=x.loc[x.project_ID.eq(study),'sample_ID'].astype(str).tolist();a=ad.read_h5ad(root/'datasets'/filename,backed='r')
 field='sample' if dataset=='heart' else 'donor_id';labels=a.obs[field].astype(str);matched=[]
 for label in labels.unique():
  if dataset=='heart':found=any(label in s for s in src)
  else:found=any(re.search(r'(?<![A-Za-z0-9])'+re.escape(label)+r'(?![A-Za-z0-9])',s) for s in src)
  if found:matched.append(label)
 print(dataset,'labels',len(matched),'evaluation cells in these labels',int(labels.isin(matched).sum()),'labels',sorted(matched));a.file.close()
