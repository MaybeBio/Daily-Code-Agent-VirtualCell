from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parents[1]
D=R/'data'
primary={'clustering':{'ari':1,'nmi':1,'asw':1,'acc':1},'annotation':{'acc':1,'f1_m':1,'balanced_accuracy':1,'precision_macro':1},'correction':{'ARI':1,'ASW_celltype':1,'iLISI_scaled':1,'kBET':1},'reconstruction':{'Pearson_expr':1,'Spearman_expr':1,'MAE_expr':0,'TopK_expression_overlap':1},'perturbation':{'Pearson_delta':1,'Spearman_delta':1,'MAE_delta':0,'DES':1}}
rows=[]
for panel in ['organ','disease','species','perturbation']:
 for path in sorted((D/'exports'/panel).glob('*_scores.csv')):
  task=path.name.replace('_scores.csv',''); df=pd.read_csv(path)
  assert len(df)==192 and df.groupby(['dataset','seed']).model.nunique().eq(16).all()
  for met,high in primary[task].items():
   z=df[['dataset','seed','model',met]].copy();z['metric']=met;z['value']=z.pop(met);z['rank']=z.groupby(['dataset','seed']).value.rank(ascending=not high,method='average');z['task']=task;z['panel']=panel;rows.append(z)
a=pd.concat(rows,ignore_index=True);a.to_csv(D/'paired_metric_ranks.csv',index=False)
b=a.groupby(['panel','task','dataset','model'],as_index=False).agg(mean_rank=('rank','mean'));b.to_csv(D/'dataset_mean_ranks.csv',index=False)
ref=pd.read_csv(D/'all_panel_ranks.csv');our=b.groupby(['panel','task','model'],as_index=False).mean(numeric_only=True);c=our.merge(ref,on=['panel','task','model']);assert np.allclose(c.mean_rank,c.mean_metric_rank)
panels=['organ','disease','species','perturbation']; tasks=['clustering','annotation','correction','reconstruction','perturbation']; models=sorted(a.model.unique()); cols=[(p,t) for p in panels for t in tasks if ((a.panel==p)&(a.task==t)).any()]
grid=np.array([[float(our.loc[(our.panel==p)&(our.task==t)&(our.model==m),'mean_rank'].iloc[0]) for p,t in cols] for m in models]);fig,ax=plt.subplots(figsize=(13,7));im=ax.imshow(grid,cmap='viridis_r',vmin=1,vmax=16,aspect='auto');ax.set_xticks(range(len(cols)),[f'{p}\n{t}' for p,t in cols],rotation=60,ha='right',fontsize=8);ax.set_yticks(range(len(models)),models,fontsize=9);fig.colorbar(im,ax=ax,label='Mean paired rank (lower is better)');fig.tight_layout();fig.savefig(R/'figures'/'panel_ranks.pdf');fig.savefig(R/'figures'/'panel_ranks.png',dpi=180);plt.close(fig)
for p,t,ma,mb in [('organ','clustering','UCE','PCA_model'),('disease','clustering','UCE','PCA_model'),('species','clustering','UCE','PCA_model'),('perturbation','perturbation','scLong','scFoundation')]:
 z=b[(b.panel==p)&(b.task==t)].pivot(index='dataset',columns='model',values='mean_rank');print(p,t,ma,mb,[(i,round(v[ma],3),round(v[mb],3)) for i,v in z.iterrows()])
protocols=json.loads((D/'model_protocols.json').read_text());pmap={p['model']:p for p in protocols};records=[]
for m in models:
 p=pmap.get(m,{});records.append({'model':m,'config':p.get('config','data/configs (baseline)'), 'protocol':p.get('protocol','非预训练基线；训练集拟合'), 'input':p.get('stages',{}).get('data','待核实'), 'embedding':p.get('stages',{}).get('embedding','待核实'), 'checkpoint':'见具体配置；版本和哈希待逐项核查','architecture':'待核查','training_objective':'待核查','total_parameters':'未知','updated_parameters':'待核查','pretraining_size_unit':'未知','pretraining_species':'未知','context_length':'未知','source':'data/model_protocols.json; data/configs/', 'status':'待核查；不以缺失推断未公开'})
pd.DataFrame(records).to_csv(D/'model_attributes_working.csv',index=False)
print('paired rows',len(a),'dataset rows',len(b),'rank reference rows',len(ref))
