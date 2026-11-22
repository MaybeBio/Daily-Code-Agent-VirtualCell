"""Exploratory, paired summaries. No significance tests or inferred intervals."""
from pathlib import Path
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parents[1]
rows=[]
for panel in ['organ','disease','species']:
 x=pd.read_csv(R/'data/exports'/panel/'annotation_scores.csv')
 a=x[x.model=='scFoundation'].set_index(['dataset','seed'])
 b=x[x.model=='scVI'].set_index(['dataset','seed'])
 assert a.index.equals(b.index) or set(a.index)==set(b.index)
 b=b.reindex(a.index)
 assert (a.n_train==b.n_train).all() and (a.n_test==b.n_test).all()
 for key in a.index:
  for metric in ['acc','f1_m']:
   rows.append(dict(panel=panel,dataset=key[0],seed=key[1],metric=metric,
       foundation=float(a.loc[key,metric]),baseline=float(b.loc[key,metric]),
       difference=float(a.loc[key,metric]-b.loc[key,metric]),
       n_train=int(a.loc[key,'n_train']),n_test=int(a.loc[key,'n_test'])))
y=pd.DataFrame(rows);assert len(y)==72 and np.isfinite(y.difference).all()
y.to_csv(R/'data/annotation_paired_metric_differences.csv',index=False)
g=y.groupby(['panel','dataset','metric'],as_index=False).agg(mean_difference=('difference','mean'),min_seed=('difference','min'),max_seed=('difference','max'),n_seeds=('seed','nunique'))
g.to_csv(R/'data/annotation_paired_metric_summary.csv',index=False)
# Parameter/rank data are generated from the checkpoint audit by summarize_parameter_audit.py.
print(g.to_string(index=False));print('Pandas',pd.__version__,'NumPy',np.__version__)
# Mean each metric across seeds, then datasets: no mixing of metric units.
out=[]
for panel in ['organ','disease','species']:
 z=pd.read_csv(R/'data/exports'/panel/'correction_scores.csv')
 for model in ['CellPLM','UCE','PCA_model','UMAP']:
  subset=z[z.model==model]
  assert subset.groupby('dataset').seed.nunique().eq(3).all() and subset.dataset.nunique()==4
  for metric in ['ARI','ASW_celltype','iLISI_scaled','kBET']:
   out.append(dict(panel=panel,model=model,metric=metric,equal_dataset_mean=float(subset.groupby('dataset')[metric].mean().mean())))
pd.DataFrame(out).to_csv(R/'data/batch_metric_means_selected.csv',index=False)
