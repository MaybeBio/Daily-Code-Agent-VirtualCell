"""Compile checkpoint-level count reports and exploratory parameter/rank summaries."""
from pathlib import Path
import json
import pandas as pd
R=Path(__file__).resolve().parents[1]
files={
 'CellFM':'weights/CellFM/base_weight.ckpt',
 'CellPLM':'weights/CellPLM/20230926_85M.best.ckpt',
 'GeneCompass':'weights/GeneCompass/GeneCompass_Small/pytorch_model.bin',
 'Geneformer':'weights/Geneformer/gf-12L-30M-i2048/pytorch_model.bin',
 'LangCell':'weights/LangCell/cell_bert/pytorch_model.bin',
 'Nicheformer':'weights/Nicheformer/model.safetensors',
 'Tabula':'weights/Tabula/pancreas.pth',
 'TranscriptFormer':'weights/TranscriptFormer/tf_sapiens/model_weights.pt',
 'UCE':'weights/UCE/model_files/4layer_model.torch',
 'scFoundation':'weights/scFoundation/models.ckpt',
 'scGPT':'weights/scGPT/human/best_model.pt',
 'scLong':'weights/scLong/gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth',
 'scPRINT2':'weights/scPRINT2/small-v2.ckpt',
}
manifest=Path('payload.sha256')
digests={}
if manifest.exists():
 for line in manifest.read_text().splitlines():
  sha,path=line.split(None,1);digests[path.strip().lstrip('*')]=sha
else:
 existing=R/'data/checkpoint_parameter_counts.csv'
 if existing.exists():
  for row in pd.read_csv(existing).itertuples():
   if isinstance(row.manifest_sha256,str):digests[row.checkpoint]=row.manifest_sha256
rows=[]
for model,checkpoint in files.items():
 d=json.loads((R/f'data/parameter_audit/{model}.json').read_text());assert d['model']==model and d['total_parameters']>0
 load=d.get('load_audit') or {};rows.append(dict(model=model,total_parameters=d['total_parameters'],registered_buffer_elements=d.get('registered_buffer_elements'),checkpoint=checkpoint,manifest_sha256=digests.get(checkpoint,''),notes=d['notes'],matched_keys=load.get('matched_keys'),missing_keys=len(load.get('missing_keys',[])),unexpected_keys=len(load.get('unexpected_keys',[]))))
df=pd.DataFrame(rows);assert len(df)==13 and df.model.nunique()==13
df.to_csv(R/'data/checkpoint_parameter_counts.csv',index=False)
rank=pd.read_csv(R/'data/all_panel_ranks.csv');out=[];correlations=[]
for panel,task in [('organ','annotation'),('disease','annotation'),('species','annotation'),('perturbation','perturbation')]:
 z=rank[(rank.panel==panel)&(rank.task==task)].merge(df[['model','total_parameters']],on='model',validate='one_to_one');assert len(z)==13
 rho=z.total_parameters.corr(z.mean_metric_rank,method='spearman')
 loo=[z[z.model!=m].total_parameters.corr(z[z.model!=m].mean_metric_rank,method='spearman') for m in z.model]
 correlations.append(dict(panel=panel,task=task,n_models=len(z),spearman_rho=rho,leave_one_out_min=min(loo),leave_one_out_max=max(loo)))
 for _,v in z.iterrows():out.append(dict(model=v.model,parameter_millions=v.total_parameters/1e6,parameter_scope='instantiated representation model, see checkpoint audit',panel=panel,task=task,mean_rank=v.mean_metric_rank))
pd.DataFrame(out).to_csv(R/'data/model_parameter_annotation_ranks.csv',index=False)
pd.DataFrame(correlations).to_csv(R/'data/parameter_rank_associations.csv',index=False)
attrs=pd.read_csv(R/'data/model_attributes_working.csv').fillna('未知')
for _,v in df.iterrows():attrs.loc[attrs.model==v.model,'total_parameters']=f'{v.total_parameters:,} locally audited instantiated-model parameters; see data/checkpoint_parameter_counts.csv for scope'
attrs.to_csv(R/'data/model_attributes_working.csv',index=False)
print(df[['model','total_parameters','registered_buffer_elements']].to_string(index=False));print(pd.DataFrame(correlations).to_string(index=False))
