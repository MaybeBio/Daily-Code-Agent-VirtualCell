"""Render publication figures from checked-in, independently audited rank tables.

Optional figure alignment audit: set NATURE_FIGURE_SCRIPTS to the scripts directory
of nature-skills/nature-figure. Plotting itself has no dependency on that checkout.
"""
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
QA = OUT / 'qa'
OUT.mkdir(exist_ok=True)
QA.mkdir(exist_ok=True)
if os.environ.get('NATURE_FIGURE_SCRIPTS'):
    sys.path.insert(0, os.environ['NATURE_FIGURE_SCRIPTS'])
    from audit_panel_alignment import require_matplotlib_panel_alignment
else:
    require_matplotlib_panel_alignment = None

plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial'],
    'font.size':8, 'axes.labelsize':8, 'axes.titlesize':9, 'xtick.labelsize':7,
    'ytick.labelsize':7, 'pdf.fonttype':42, 'svg.fonttype':'none',
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':0.6})
BLUE, ORANGE, INK = '#0072B2', '#D55E00', '#222222'

def export(fig, name, axes=None):
    fig.canvas.draw()
    if require_matplotlib_panel_alignment is not None and axes is not None:
        require_matplotlib_panel_alignment(fig, axes=axes,
            json_out=QA / f'{name}.alignment.json', strict=True)
    fig.savefig(OUT / (name+'.pdf'))
    fig.savefig(OUT / (name+'.png'), dpi=300)
    fig.savefig(OUT / (name+'.svg'))
    plt.close(fig)

rank = pd.read_csv(ROOT/'data/all_panel_ranks.csv')
models = sorted(rank.model.unique(), key=str.casefold)
cols = [(p,t) for p in ['organ','disease','species'] for t in ['clustering','annotation','correction','reconstruction']] + [('perturbation','perturbation')]
assert len(models)==16 and len(cols)==13 and len(rank)==208
piv = rank.pivot(index='model', columns=['panel','task'], values='mean_metric_rank').loc[models,cols]
assert np.isfinite(piv.to_numpy()).all()
fig, ax = plt.subplots(figsize=(7.2,5.0))
fig.subplots_adjust(left=.185,right=.89,bottom=.25,top=.90)
im=ax.imshow(piv.to_numpy(),cmap='viridis_r',vmin=1,vmax=16,aspect='auto',interpolation='nearest')
ax.set_yticks(range(16),models)
ax.set_xticks(range(13),['Clust.','Annot.','Batch','Recon.']*3+['Pert.'],rotation=55,rotation_mode='anchor',ha='right')
ax.tick_params(length=0,pad=3)
for x in (3.5,7.5,11.5): ax.axvline(x,color='white',lw=1.5)
for x,label in [(1.5,'Human organs'),(5.5,'Human disease'),(9.5,'Mouse organs'),(12,'Perturb.')]:
    ax.text(x,-.9,label,ha='center',va='bottom',fontsize=7,clip_on=False)
bar=fig.colorbar(im,ax=ax,fraction=.028,pad=.025,ticks=[1,4,8,12,16]);bar.ax.set_ylabel('Mean rank (lower is better)',fontsize=7)
export(fig,'panel_ranks')

# Paired dataset ranks: average over metrics and seeds as in the exported table.
dat = pd.read_csv(ROOT/'data/dataset_mean_ranks.csv')
fig, axs=plt.subplots(1,2,figsize=(7.2,3.35),sharex=True)
fig.subplots_adjust(left=.14,right=.98,bottom=.25,top=.83,wspace=.48)
settings=[('organ','clustering','UCE','PCA_model',['bmmc','heart','hlca','kidney'],'Human organ clustering'),
          ('perturbation','perturbation','scLong','scFoundation',['norman','adamson','replogleK562','replogleRPE1'],'Perturbation prediction')]
for j,(ax,(panel,task,m1,m2,datasets,title)) in enumerate(zip(axs,settings)):
    sub=dat[(dat.panel==panel)&(dat.task==task)].pivot(index='dataset',columns='model',values='mean_rank').loc[datasets]
    assert sub[[m1,m2]].notna().all().all()
    yy=np.arange(len(datasets))[::-1]
    for y,(_,row) in zip(yy,sub.iterrows()):
        ax.plot([row[m1],row[m2]],[y,y],color='#9b9b9b',lw=1,zorder=1)
        ax.scatter(row[m1],y,s=29,color=BLUE,zorder=2)
        ax.scatter(row[m2],y,s=29,color=ORANGE,marker='s',zorder=2)
    ax.set_yticks(yy,datasets);ax.set_xlim(.5,16.5);ax.set_xticks([1,4,8,12,16]);ax.grid(axis='x',color='#dddddd',lw=.5)
    ax.set_title(title,loc='left',pad=9);ax.set_xlabel('Mean paired rank (lower is better)')
    ax.legend([Line2D([],[],marker='o',color=BLUE,linestyle='',markersize=5),
               Line2D([],[],marker='s',color=ORANGE,linestyle='',markersize=5)],
              [m1,m2],loc='upper center',bbox_to_anchor=(.5,-.29),ncol=2,fontsize=7,frameon=False,handletextpad=.3,columnspacing=1.0)
    ax.text(-.16,1.14,'AB'[j],transform=ax.transAxes,fontsize=10,fontweight='bold',va='top')
export(fig,'dataset_pairs',list(axs))

# Ten checkpoint-level corpus estimates, as specified in factor_associations.py.
size={'CellFM':100,'GeneCompass':101.76842,'Geneformer':30,'LangCell':27.5,
      'Nicheformer':110,'TranscriptFormer':57,'UCE':36,'scFoundation':50,'scGPT':33,'scLong':48.024242}
assert all(np.isfinite(value) and value > 0 for value in size.values())
fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),sharex=True,sharey=True)
fig.subplots_adjust(left=.095,right=.98,bottom=.22,top=.85,wspace=.16)
for j,(ax,(panel,task,title)) in enumerate(zip(axs,[('organ','annotation','Human organ annotation'),('perturbation','perturbation','Perturbation prediction')])):
    sub=rank[(rank.panel==panel)&(rank.task==task)&rank.model.isin(size)].copy()
    assert len(sub)==10 and sub.model.nunique()==10
    ax.scatter(sub.model.map(size),sub.mean_metric_rank,s=26,color=BLUE,edgecolor='white',linewidth=.45,zorder=3)
    ax.set_xscale('log');ax.set_xticks([30,50,100],['30','50','100']);ax.xaxis.set_minor_formatter(NullFormatter());ax.set_xlim(22,130)
    ax.set_ylim(16.5,.5);ax.set_yticks([1,4,8,12,16]);ax.grid(color='#e4e4e4',lw=.5,zorder=0)
    ax.set_title(title,loc='left',pad=9);ax.set_xlabel('Reported pretraining size (million)')
    if j==0:ax.set_ylabel('Mean rank (lower is better)')
    ax.text(-.14,1.14,'AB'[j],transform=ax.transAxes,fontsize=10,fontweight='bold',va='top')
export(fig,'pretraining_size',list(axs))
