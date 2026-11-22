from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parents[1];df=pd.read_csv(R/'data/dataset_mean_ranks.csv');out=[]
for panel,excluded in [('organ','heart'),('disease','ms')]:
 for task in ['annotation','clustering','correction','reconstruction']:
  z=df[(df.panel==panel)&(df.task==task)&(df.dataset!=excluded)].groupby('model',as_index=False).mean(numeric_only=True).sort_values('mean_rank');z['position']=z.mean_rank.rank(method='min').astype(int);z['panel']=panel;z['task']=task;z['excluded_dataset']=excluded;out.append(z)
r=pd.concat(out);r.to_csv(R/'data/overlap_exclusion_sensitivity.csv',index=False)
for (panel,task),g in r.groupby(['panel','task']):
 print(panel,task,'winner',g.iloc[0]['model'],round(g.iloc[0].mean_rank,3),'scFoundation',round(g.loc[g.model=='scFoundation','mean_rank'].iloc[0],3),'position',g.loc[g.model=='scFoundation','position'].iloc[0])
