"""Full-model displays from existing benchmark exports; unknown attributes are explicit."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1];O=R/'figures';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],'font.size':8,
 'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,
 'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
def export(fig,name):
 fig.savefig(O/f'{name}.pdf');fig.savefig(O/f'{name}.svg');fig.savefig(O/f'{name}.png',dpi=300);plt.close(fig)
models=['CellFM','CellPLM','Geneformer','LangCell','Tabula','TranscriptFormer','scFoundation','scGPT','scLong',
        'GeneCompass','Nicheformer','UCE','scPRINT2','PCA_model','UMAP','scVI']
assert len(models)==len(set(models))==16
# 1. Dataset-level raw F1: mean seeds within each dataset/model, not across datasets.
blocks=[]
for panel in ['organ','disease','species']:
 z=pd.read_csv(R/'data/exports'/panel/'annotation_scores.csv');assert len(z)==192
 z=z.groupby(['dataset','model'],as_index=False).agg(macro_f1=('f1_m','mean'),n_seeds=('seed','nunique'));z['panel']=panel;blocks.append(z)
f=pd.concat(blocks,ignore_index=True);assert len(f)==192 and f.n_seeds.eq(3).all() and f.macro_f1.between(0,1).all()
f.to_csv(R/'data/all_model_annotation_f1.csv',index=False)
f.to_csv(O/'source_data/figure_all_model_annotation_f1.csv',index=False)
datasets=['bmmc','heart','hlca','kidney','ms','uc_epi','covid_lung','tnbc','mouse_marrow','mouse_lung','mouse_kidney','mouse_heart']
a=f.pivot(index='model',columns='dataset',values='macro_f1').loc[models,datasets];assert a.notna().all().all()
fig,ax=plt.subplots(figsize=(7.2,5.3));fig.subplots_adjust(left=.23,right=.88,bottom=.24,top=.90)
im=ax.imshow(a.to_numpy(),cmap='viridis',vmin=0,vmax=1,aspect='auto',interpolation='nearest');ax.set_yticks(range(16),models);ax.set_xticks(range(12),['BMMC','Heart','HLCA','Kidney','MS','UC epi.','COVID lung','TNBC','Marrow','Lung','Kidney','Heart'],rotation=90,ha='center');ax.tick_params(length=0,pad=3)
for v in [3.5,7.5]:ax.axvline(v,color='white',lw=1.4)
for v,label in [(1.5,'Human organs'),(5.5,'Human diseases'),(9.5,'Mouse organs')]:ax.text(v,-.9,label,ha='center',va='bottom',fontsize=7,clip_on=False)
bar=fig.colorbar(im,ax=ax,fraction=.03,pad=.025,ticks=[0,.25,.5,.75,1]);bar.ax.set_ylabel('Macro-F1 (higher is better)',fontsize=7)
export(fig,'all_models_dataset_f1')

# 2. Same organ, different datasets: display all 16 methods.
d=pd.read_csv(R/'data/dataset_mean_ranks.csv');pairs=[('bmmc','mouse_marrow'),('hlca','mouse_lung'),('kidney','mouse_kidney'),('heart','mouse_heart')];records=[]
for model in models:
 for human,mouse in pairs:
  h=d[(d.panel=='organ')&(d.task=='annotation')&(d.model==model)&(d.dataset==human)].mean_rank
  m=d[(d.panel=='species')&(d.task=='annotation')&(d.model==model)&(d.dataset==mouse)].mean_rank
  assert len(h)==len(m)==1;records.append({'model':model,'human_dataset':human,'mouse_dataset':mouse,'human_rank':h.iloc[0],'mouse_rank':m.iloc[0],'mouse_minus_human':m.iloc[0]-h.iloc[0]})
r=pd.DataFrame(records);assert len(r)==64;r.to_csv(R/'data/organ_matched_annotation_rank_shifts.csv',index=False);r.to_csv(O/'source_data/figure_species_rank_shift.csv',index=False)
a=r.pivot(index='model',columns='human_dataset',values='mouse_minus_human').loc[models,[p[0] for p in pairs]]
fig,ax=plt.subplots(figsize=(7.2,5.7));fig.subplots_adjust(left=.27,right=.89,bottom=.14,top=.93)
im=ax.imshow(a.to_numpy(),cmap='PuOr',vmin=-10,vmax=10,aspect='auto',interpolation='nearest');ax.set_yticks(range(16),models);ax.set_xticks(range(4),['Marrow','Lung','Kidney','Heart']);ax.tick_params(length=0)
for i in range(16):
 for j in range(4):
  v=a.iloc[i,j];ax.text(j,i,f'{v:+.1f}',ha='center',va='center',fontsize=7,color='white' if abs(v)>6 else '#222222')
for y in (8.5,12.5):ax.axhline(y,lw=1.2,color='white')
bar=fig.colorbar(im,ax=ax,fraction=.035,pad=.025,ticks=[-10,-5,0,5,10]);bar.ax.set_ylabel('Mouse rank − human rank',fontsize=7)
export(fig,'species_rank_shift')

# 3. All model ranks and independently audited instantiated-model counts.
rank=pd.read_csv(R/'data/all_panel_ranks.csv');cols=[('organ','annotation'),('disease','annotation'),('species','annotation')]
pivot=rank.pivot(index='model',columns=['panel','task'],values='mean_metric_rank').loc[models,cols];assert pivot.notna().all().all()
counts=pd.read_csv(R/'data/checkpoint_parameter_counts.csv').set_index('model').total_parameters.to_dict();assert len(counts)==13
labels={m:(f'{counts[m]/1e6:,.1f} M' if m in counts else 'Dataset-fitted') for m in models}
out=pivot.reset_index();out.columns=['model','human_organs','human_diseases','mouse_organs'];out['parameter_label']=out.model.map(labels);out.to_csv(R/'data/all_model_parameters_annotation_ranks.csv',index=False);out.to_csv(O/'source_data/figure_parameter_annotation.csv',index=False)
fig,ax=plt.subplots(figsize=(7.2,5.7));fig.subplots_adjust(left=.27,right=.64,bottom=.14,top=.91)
im=ax.imshow(pivot.to_numpy(),cmap='viridis_r',vmin=1,vmax=16,aspect='auto',interpolation='nearest');ax.set_yticks(range(16),models);ax.set_xticks(range(3),['Human organs','Human disease','Mouse organs'],rotation=30,rotation_mode='anchor',ha='right');ax.tick_params(length=0,pad=3)
for i,m in enumerate(models):ax.text(2.62,i,labels[m],ha='left',va='center',fontsize=7,clip_on=False)
ax.text(2.62,-.9,'Parameters',ha='left',va='bottom',fontsize=7,clip_on=False)
cax=fig.add_axes([.91,.22,.018,.58])
bar=fig.colorbar(im,cax=cax,ticks=[1,4,8,12,16]);bar.ax.set_ylabel('Mean rank',fontsize=7)
export(fig,'parameter_annotation')
