from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
# Approximate, reported corpus scale; exact unique cells not consistently disclosed.
size={'CellFM':100,'GeneCompass':101.76842,'Geneformer':30,'LangCell':27.5,'Nicheformer':110,'TranscriptFormer':57,'UCE':36,'scFoundation':50,'scGPT':33,'scLong':48.024242}
x=pd.read_csv(R/'data/all_panel_ranks.csv');rows=[]
for (panel,task),g in x[x.model.isin(size)].groupby(['panel','task'],sort=True):
 z=g.copy();z['pretraining_cells_millions']=z.model.map(size);z=z.sort_values('model');rho=float(spearmanr(z.pretraining_cells_millions,z.mean_metric_rank).statistic); loo=[float(spearmanr(z.pretraining_cells_millions.drop(i),z.mean_metric_rank.drop(i)).statistic) for i in z.index];rows.append(dict(panel=panel,task=task,n_models=len(z),spearman_rho=rho,leave_one_out_min=min(loo),leave_one_out_max=max(loo)))
pd.DataFrame(rows).to_csv(R/'data/pretraining_size_associations.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(11,5))
for ax,(p,t) in zip(axs,[('organ','annotation'),('perturbation','perturbation')]):
 z=x[(x.panel==p)&(x.task==t)&x.model.isin(size)].copy();z['scale']=z.model.map(size);ax.scatter(z['scale'],z.mean_metric_rank,color='#24728b',s=35)
 for _,q in z.iterrows():ax.annotate(q.model,(q['scale'],q.mean_metric_rank),xytext=(4,4),textcoords='offset points',fontsize=7)
 ax.set_xscale('log');ax.invert_yaxis();ax.set_title(f'{p}: {t}');ax.set_xlabel('Reported pretraining corpus (million cells or cell-text pairs)');ax.set_ylabel('Mean paired rank (lower is better)');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(R/'figures/pretraining_size.pdf');fig.savefig(R/'figures/pretraining_size.png',dpi=180)
print(pd.DataFrame(rows).to_string(index=False))
