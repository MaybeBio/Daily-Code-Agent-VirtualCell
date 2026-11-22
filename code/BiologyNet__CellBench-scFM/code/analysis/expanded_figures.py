"""Expanded analysis figures. Inputs are saved by expanded_analysis.py."""
from pathlib import Path
import os,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
R=Path(__file__).resolve().parents[1];O=R/'figures';Q=O/'qa';Q.mkdir(exist_ok=True)
if os.getenv('NATURE_FIGURE_SCRIPTS'):
 sys.path.insert(0,os.environ['NATURE_FIGURE_SCRIPTS'])
 from audit_panel_alignment import require_matplotlib_panel_alignment
else:require_matplotlib_panel_alignment=None
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],'font.size':8,
 'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,
 'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
B,OCH='#0072B2','#D55E00'
def export(fig,name,axes):
 fig.canvas.draw()
 if require_matplotlib_panel_alignment:
  require_matplotlib_panel_alignment(fig,axes=axes,json_out=Q/f'{name}.alignment.json',strict=True)
 fig.savefig(O/f'{name}.pdf');fig.savefig(O/f'{name}.svg');fig.savefig(O/f'{name}.png',dpi=300);plt.close(fig)

# Paired raw accuracy and macro-F1 differences for identical dataset and seed.
x=pd.read_csv(R/'data/annotation_paired_metric_differences.csv')
order=[('organ','bmmc'),('organ','heart'),('organ','hlca'),('organ','kidney'),('disease','ms'),('disease','uc_epi'),('disease','covid_lung'),('disease','tnbc'),('species','mouse_marrow'),('species','mouse_lung'),('species','mouse_kidney'),('species','mouse_heart')]
fig,axs=plt.subplots(1,2,figsize=(7.2,5.2),sharey=True,sharex=True);fig.subplots_adjust(left=.19,right=.98,bottom=.16,top=.90,wspace=.14)
for j,(ax,metric,title) in enumerate(zip(axs,['acc','f1_m'],['Accuracy','Macro-F1'])):
 for i,(panel,ds) in enumerate(order):
  z=x[(x.panel==panel)&(x.dataset==ds)&(x.metric==metric)].sort_values('seed');assert len(z)==3
  y=11-i; vals=z.difference.to_numpy();ax.plot([vals.min(),vals.max()],[y,y],color='#b4b4b4',lw=1,zorder=1)
  ax.scatter(vals,[y-.1,y,y+.1],s=11,color='#858585',zorder=2)
  ax.scatter([vals.mean()],[y],marker='D',s=24,color=B,zorder=3)
 ax.axvline(0,color='#555555',lw=.7);ax.axhline(7.5,color='#cccccc',lw=.6);ax.axhline(3.5,color='#cccccc',lw=.6)
 ax.set_yticks(range(11,-1,-1),[d for _,d in order]);ax.set_xlim(-.033,.158);ax.set_xticks([-.02,0,.05,.10,.15]);ax.grid(axis='x',lw=.4,color='#e5e5e5');ax.set_xlabel('scFoundation − scVI (metric units)');ax.set_title(title,loc='left',pad=7)
 ax.text(-.12,1.11,'AB'[j],transform=ax.transAxes,fontsize=10,fontweight='bold',va='top')
fig.text(.52,.035,'Small circles: individual seeds    Diamond: mean    Line: seed range',ha='center',fontsize=7)
export(fig,'annotation_raw_differences',list(axs))

# Corpus size versus rank across all three annotation panels and perturbation.
sizes={'CellFM':100,'GeneCompass':101.76842,'Geneformer':30,'LangCell':27.5,'Nicheformer':110,'TranscriptFormer':57,'UCE':36,'scFoundation':50,'scGPT':33,'scLong':48.024242}
assert all(np.isfinite(v) and v>0 for v in sizes.values())
r=pd.read_csv(R/'data/all_panel_ranks.csv');fig,axs=plt.subplots(2,2,figsize=(7.2,6.2),sharex=True,sharey=True)
fig.subplots_adjust(left=.11,right=.98,bottom=.29,top=.93,wspace=.15,hspace=.38)
markers=['o','s','^','D','v','P','X','<','>','*']
colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#332288','#882255','#44AA99','#555555']
styles={model:(markers[i],colors[i]) for i,model in enumerate(sizes)}
sets=[('organ','annotation','Human organ annotation'),('disease','annotation','Human disease annotation'),('species','annotation','Mouse organ annotation'),('perturbation','perturbation','Perturbation prediction')]
source=[]
for j,(ax,(panel,task,title)) in enumerate(zip(axs.flat,sets)):
 z=r[(r.panel==panel)&(r.task==task)&r.model.isin(sizes)].copy();assert len(z)==10;z['reported_size_millions']=z.model.map(sizes)
 source.append(z[['panel','task','model','reported_size_millions','mean_metric_rank']])
 for row in z.itertuples():
  marker,color=styles[row.model];ax.scatter(row.reported_size_millions,row.mean_metric_rank,s=29,marker=marker,color=color,edgecolors='white',linewidths=.35,zorder=3)
 ax.set_xscale('log');ax.set_xlim(22,130);ax.set_xticks([30,50,100],['30','50','100']);ax.xaxis.set_minor_formatter(NullFormatter());ax.set_ylim(16.5,.5);ax.set_yticks([1,4,8,12,16]);ax.grid(color='#e4e4e4',lw=.5);ax.set_title(title,loc='left',pad=7);ax.text(-.12,1.14,'ABCD'[j],transform=ax.transAxes,fontsize=10,fontweight='bold',va='top')
 if j%2==0:ax.set_ylabel('Mean rank (lower is better)')
 if j>=2:ax.set_xlabel('Reported pretraining size (million)')
handles=[Line2D([0],[0],marker=styles[m][0],color='none',markerfacecolor=styles[m][1],markeredgecolor='white',markersize=6,label=m) for m in sizes]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.095),ncol=5,frameon=False,fontsize=6.5,columnspacing=1.0,handletextpad=.3)
fig.text(.5,.035,'Not plotted: CellPLM, Tabula, scPRINT2 (checkpoint corpus unverified); PCA, UMAP, scVI (dataset-fitted baselines)',ha='center',fontsize=6.5)
pd.concat(source).to_csv(R/'data/corpus_size_panel_points.csv',index=False)
export(fig,'corpus_size_four_panels',list(axs.flat))

# Checkpoint-level model parameters vs. task rank. PCA, UMAP and scVI are
# fitted per dataset rather than assigned an invented pretrained-model size.
z=pd.read_csv(R/'data/model_parameter_annotation_ranks.csv')
model_order=['CellFM','CellPLM','GeneCompass','Geneformer','LangCell','Nicheformer','Tabula','TranscriptFormer','UCE','scFoundation','scGPT','scLong','scPRINT2']
assert set(z.model)==set(model_order)
markers=['o','s','^','D','v','P','X','<','>','*','h','d','p']
colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#332288','#882255','#44AA99','#555555','#117733','#AA4499','#AA8800']
styles={m:(markers[i],colors[i]) for i,m in enumerate(model_order)}
fig,axs=plt.subplots(2,2,figsize=(7.2,6.6),sharex=True,sharey=True);fig.subplots_adjust(left=.11,right=.98,bottom=.30,top=.94,wspace=.15,hspace=.38)
for j,(ax,(panel,task,title)) in enumerate(zip(axs.flat,sets)):
 sub=z[(z.panel==panel)&(z.task==task)];assert len(sub)==13
 for row in sub.itertuples():
  marker,color=styles[row.model];ax.scatter(row.parameter_millions,row.mean_rank,s=29,marker=marker,color=color,edgecolors='white',linewidths=.35,zorder=3)
 ax.set_xscale('log');ax.set_xlim(10,1400);ax.set_xticks([10,30,100,300,1000],['10','30','100','300','1,000']);ax.xaxis.set_minor_formatter(NullFormatter());ax.set_ylim(16.5,.5);ax.set_yticks([1,4,8,12,16]);ax.grid(color='#e4e4e4',lw=.5);ax.set_title(title,loc='left',pad=7);ax.text(-.12,1.14,'ABCD'[j],transform=ax.transAxes,fontsize=10,fontweight='bold',va='top')
 if j%2==0:ax.set_ylabel('Mean rank (lower is better)')
 if j>=2:ax.set_xlabel('Instantiated-model parameters (million)')
handles=[Line2D([0],[0],marker=styles[m][0],color='none',markerfacecolor=styles[m][1],markeredgecolor='white',markersize=6,label=m) for m in model_order]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.09),ncol=5,frameon=False,fontsize=6.5,columnspacing=.9,handletextpad=.3)
fig.text(.5,.035,'PCA, UMAP, scVI: dataset-fitted baselines; no single pretrained-model parameter count',ha='center',fontsize=6.5)
z.to_csv(O/'source_data/figure_checkpoint_parameters_rank.csv',index=False)
export(fig,'checkpoint_parameters_rank',list(axs.flat))
