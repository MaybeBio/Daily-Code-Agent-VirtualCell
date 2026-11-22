"""Recompute four-task evidence and draw the agreed manuscript/review layout.

Inputs are read-only benchmark exports. No synthetic observations are used.
Run with Python, NumPy, matplotlib, python-pptx, lxml and PyMuPDF.
"""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator, LogLocator

R = Path(__file__).resolve().parents[1]
QA = R / 'analysis' / 'figure_qa'
sys.path.insert(0, str(QA))
from audit_panel_alignment import require_matplotlib_panel_alignment

MODELS = ['CellFM','CellPLM','GeneCompass','Geneformer','LangCell','Nicheformer',
          'Tabula','TranscriptFormer','UCE','scFoundation','scGPT','scLong','scPRINT2',
          'PCA_model','UMAP','scVI']
SEEDS = [41,42,43]
PANELS = ['organ','disease','species']
PANEL_NAMES = ['Human organs','Human diseases','Mouse organs']
DATASETS = [['bmmc','hlca','heart','kidney'],['ms','uc_epi','covid_lung','tnbc'],
            ['mouse_marrow','mouse_lung','mouse_heart','mouse_kidney']]
DS = sum(DATASETS, [])
DS_NAMES = ['BM','HL','AH','AK','MS','UC','C19','BC',
            'MM','ML','MH','MK']
TASKS = ['clustering','correction','annotation','reconstruction']
TASK_NAMES = ['Unsupervised clustering','Native-embedding batch assessment',
              'Supervised annotation','Supervised reconstruction']
METRICS = {
 'clustering':[('ari','ARI',True),('nmi','NMI',True),('asw','ASW',True),('acc','Accuracy',True)],
 'correction':[('iLISI_scaled','Scaled iLISI',True),('kBET','kBET',True),('cLISI_scaled','Scaled cLISI',True),('Graph_connectivity','Graph connectivity',True)],
 'annotation':[('acc','Accuracy',True),('f1_m','Macro-F1',True),('balanced_accuracy','Balanced acc.',True),('precision_macro','Macro-precision',True)],
 'reconstruction':[('Pearson_expr','Pearson r',True),('Spearman_expr','Spearman rho',True),('MAE_expr','MAE',False),('TopK_expression_overlap','Top-50 overlap',True)]}
SIZES = {'CellFM':100,'GeneCompass':101.76842,'Geneformer':30,'LangCell':27.5,
         'Nicheformer':110,'TranscriptFormer':57,'UCE':36,'scFoundation':50,'scGPT':33,'scLong':48.024242}
ENCODING = ['A','V','R+V','R','R','R+C','V','V','S+P','F+V','B','F+V','A+C']
SPECIES = ['Human','Human (family)','Human + mouse','Human','Human','Human + mouse',
           'Human (family)','Human','Multispecies (family)','Human','Human','Human','Multispecies (family)']
MULTI = {2,5,8,12}
COLORS = ['#477F9B','#AD8156','#668D76','#947BA6']
SEQ = LinearSegmentedColormap.from_list('score',['#F3F6F8','#C0D6DB','#6F9BA8','#264D66'])
RANK = LinearSegmentedColormap.from_list('rank',['#E4EFF0','#9FBCC5','#526A85'])
DIV = LinearSegmentedColormap.from_list('difference',['#6A89B0','#F7F7F5','#B87865'])
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],
 'font.size':6.4,'axes.titlesize':7,'axes.labelsize':6.4,'xtick.labelsize':5.7,
 'ytick.labelsize':6.1,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.5,
 'xtick.major.width':.4,'ytick.major.width':.4,'text.color':'#263C4B',
 'axes.labelcolor':'#344B59','axes.edgecolor':'#98A7AF','savefig.dpi':600})

def readcsv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def writecsv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def ranks(v, high=False):
    v=np.asarray(v); z=-v if high else v
    out=np.empty(len(z),float)
    order=np.argsort(z,kind='stable')
    start=0
    while start<len(order):
        stop=start+1
        while stop<len(order) and z[order[stop]]==z[order[start]]:stop+=1
        out[order[start:stop]]=(start+1+stop)/2
        start=stop
    return out

def rho(x,y):
    return float(np.corrcoef(ranks(x),ranks(y))[0,1])

def compute():
    raw={}; ranked={}; source_hashes={}; runrows=[]
    for t in TASKS:
        a=np.empty((12,3,16,4))
        for pi,p in enumerate(PANELS):
            f=R/'data'/'exports'/p/(t+'_scores.csv')
            rows=readcsv(f); assert len(rows)==192
            source_hashes[str(f.relative_to(R))]=hashlib.sha256(f.read_bytes()).hexdigest()
            lookup={(r['dataset'],int(r['seed']),r['model']):r for r in rows}
            assert len(lookup)==192
            for di,d in enumerate(DATASETS[pi]):
                for si,s in enumerate(SEEDS):
                    for mi,m in enumerate(MODELS):
                        r=lookup[d,s,m]
                        a[pi*4+di,si,mi]=[float(r[k]) for k,_,_ in METRICS[t]]
                        runrows.append(dict(panel=p,task=t,dataset=d,model=m,seed=s,
                                            **{k:float(r[k]) for k,_,_ in METRICS[t]}))
        assert np.isfinite(a).all()
        raw[t]=a;b=np.empty_like(a)
        for di,si,ki in itertools.product(range(12),range(3),range(4)):
            b[di,si,:,ki]=ranks(a[di,si,:,ki],METRICS[t][ki][2])
        ranked[t]=b
    reference={(r['panel'],r['task'],r['model']):float(r['mean_metric_rank'])
               for r in readcsv(R/'data'/'all_panel_ranks.csv')}
    summary=[]
    for t in TASKS:
        for pi,p in enumerate(PANELS):
            mr=ranked[t][pi*4:pi*4+4].mean(axis=(0,1,3))
            for mi,m in enumerate(MODELS):
                assert np.isclose(mr[mi],reference[p,t,m],atol=1e-12)
                summary.append(dict(panel=p,task=t,model=m,mean_rank=float(mr[mi])))
    assert sum(a.size for a in raw.values())==9216
    writecsv(R/'data'/'layout_panel_ranks.csv',summary)
    params={r['model']:int(r['total_parameters']) for r in readcsv(R/'data'/'checkpoint_parameter_counts.csv')}
    associations=[]
    for ti,t in enumerate(TASKS):
        for pi,p in enumerate(['all']+PANELS):
            b=ranked[t] if p=='all' else ranked[t][(pi-1)*4:pi*4]
            mr=b.mean(axis=(0,1,3))
            for factor,info in [('parameters',params),('corpus',SIZES)]:
                ids=[i for i,m in enumerate(MODELS[:13]) if m in info]
                x=np.array([info[MODELS[i]] for i in ids]);y=mr[ids]
                rr=rho(x,y);loo=[rho(np.delete(x,j),np.delete(y,j)) for j in range(len(ids))]
                associations.append(dict(factor=factor,panel=p,task=t,n_models=len(ids),
                    spearman_rho=rr,leave_one_out_min=min(loo),leave_one_out_max=max(loo)))
    writecsv(R/'data'/'layout_attribute_associations.csv',associations)
    protocol=[dict(model=m,input_code=ENCODING[i],species=SPECIES[i],
                   parameters=params[m],corpus_millions=SIZES.get(m,'')) for i,m in enumerate(MODELS[:13])]
    writecsv(R/'data'/'layout_model_attributes.csv',protocol)
    stats={'input_hashes':source_hashes,'evaluations':2304,'primary_values':9216,
           'panel_rank_values_checked':192,'tasks':TASKS,'panels':PANELS,'seeds':SEEDS,
           'individual_seed_markers':False,'model_count':16,'pretrained_count':13,
           'panel_summary':summary,'associations':associations}
    (R/'data'/'layout_verification.json').write_text(json.dumps(stats,indent=2))
    return raw,ranked,params,stats

def display(m):return 'PCA' if m=='PCA_model' else m

def letter(ax, s, title):
    ax.text(0,1.035,s+'  '+title,transform=ax.transAxes,ha='left',va='bottom',
            fontsize=7,weight='bold')

def heat(ax,a,xlabels,ylabels,cmap,vmin,vmax):
    # pcolormesh keeps all heatmap cells vector paths in the PDF/SVG.
    im=ax.pcolormesh(np.arange(a.shape[1]+1)-.5,np.arange(a.shape[0]+1)-.5,
                     a,cmap=cmap,vmin=vmin,vmax=vmax,shading='flat',rasterized=False)
    ax.set_xlim(-.5,a.shape[1]-.5);ax.set_ylim(a.shape[0]-.5,-.5)
    ax.set_xticks(range(a.shape[1]),xlabels,rotation=65,ha='right',rotation_mode='anchor')
    if ylabels:ax.set_yticks(range(a.shape[0]),ylabels)
    else:ax.set_yticks([])
    ax.tick_params(length=0,pad=2)
    for spine in ax.spines.values():spine.set_visible(False)
    return im

def cbar(fig,im,rect,label,ticks=None):
    ax=fig.add_axes(rect);cb=fig.colorbar(im,cax=ax,orientation='horizontal',ticks=ticks)
    if cb.solids is not None:cb.solids.set_rasterized(False)
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=5.6,pad=1,length=2)
    cb.set_label(label,fontsize=5.8,labelpad=1)
    return ax

def label_points(ax,x,y,labels,color):
    """Keep quantitative points fixed; place identifiers in free display space."""
    ax.scatter(x,y,s=5,color=color,zorder=3)
    ax.figure.canvas.draw();renderer=ax.figure.canvas.get_renderer()
    from matplotlib.transforms import Bbox
    occupied=[Bbox.from_extents(px-4,py-4,px+4,py+4)
              for px,py in ax.transData.transform(np.c_[x,y])]
    area=ax.get_window_extent(renderer).padded(-2)
    offsets=[(dx,dy) for dy in [7,-7,13,-13,20,-20,27,-27]
             for dx in [0,8,-8,16,-16,24,-24]]
    for xx,yy,label in zip(x,y,labels):
        for dx,dy in offsets:
            text=ax.annotate(str(label),(xx,yy),xytext=(dx,dy),textcoords='offset points',
                fontsize=6,ha='center',va='center',color=color,zorder=4)
            bb=text.get_window_extent(renderer).padded(3.0)
            if (area.contains(bb.x0,bb.y0) and area.contains(bb.x1,bb.y1)
                and not any(bb.overlaps(other) for other in occupied)):
                occupied.append(bb);break
            text.remove()
        else:raise RuntimeError('No collision-free point label position')

RECORDS=[]
def save(fig,slug,caption,axes=None):
    folder=R/'figures'/'task_layout'/slug;folder.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(folder/'alignment.json'),
        overlay_svg=str(folder/'alignment.svg'),exclude_axes=[ax for ax in fig.axes if axes and ax not in axes],
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(folder/(slug+'.pdf'))
    fig.savefig(folder/(slug+'.svg'))
    with plt.rc_context({'svg.fonttype':'path'}):fig.savefig(folder/(slug+'_outlined.svg'))
    fig.savefig(folder/(slug+'.png'),dpi=600)
    fig.savefig(folder/'preview.png',dpi=300)
    (folder/'caption.txt').write_text(caption+'\n')
    RECORDS.append(dict(slug=slug,folder=str(folder.relative_to(R)),caption=caption,
                        width_inches=float(fig.get_figwidth()),height_inches=float(fig.get_figheight())))
    plt.close(fig); print('Rendered',slug,flush=True)

def overview():
    fig,axs=plt.subplots(2,2,figsize=(6.48,4.7))
    fig.subplots_adjust(left=.035,right=.99,bottom=.06,top=.90,wspace=.10,hspace=.20)
    fig.text(.035,.965,'Benchmark design and evidence structure',fontsize=12,weight='bold',va='top')
    blocks=[('A','Models','13 pretrained checkpoints\n+ PCA, UMAP and scVI\n\nFrozen pretrained representations\nDataset-fitted baselines'),
      ('B','Evaluation data','Human organs: marrow, lung, heart, kidney\nHuman diseases: MS, UC, COVID-19, TNBC\nMouse organs: marrow, lung, heart, kidney\n\n12 datasets; 3 seeds (41, 42, 43)'),
      ('C','Four tasks','Unsupervised clustering\nNative-embedding batch assessment\nSupervised annotation\nSupervised expression reconstruction\n\n4 primary metrics per task'),
      ('D','Readout and interpretation','Within-dataset / seed / metric ranks\nEqual weighting of datasets and metrics\nRaw scores and seed variability retained\n\nMain: task summaries and model attributes\nSupplement: 16 full boxplot grids + audits')]
    for ax,(lab,title,body) in zip(axs.flat,blocks):
        ax.set(xlim=(0,1),ylim=(0,1));ax.set_xticks([]);ax.set_yticks([])
        ax.set_facecolor('#EFF4F6')
        for sp in ax.spines.values():sp.set_visible(False)
        ax.text(.055,.91,lab+'  '+title,weight='bold',fontsize=9,va='top')
        ax.text(.055,.71,body,fontsize=7.1,va='top',linespacing=1.7)
    save(fig,'fig01_framework','Benchmark design. Sixteen methods were evaluated on twelve datasets in three panels, using seeds 41, 42 and 43. The four tasks each have four primary metrics. This manuscript excludes perturbation prediction. Rankings are calculated within matched dataset--seed--metric blocks; raw scores and seed variability are retained in the supplementary figures. Attribute comparisons describe checkpoint packages and do not identify causal effects.',list(axs.flat))

def task_figures(raw,ranked):
    for ti,t in enumerate(TASKS):
        fig=plt.figure(figsize=(6.48,6.2))
        gs=fig.add_gridspec(2,2,left=.16,right=.98,bottom=.16,top=.84,
            width_ratios=[3.15,1],height_ratios=[2.65,1.05],hspace=.85,wspace=.20)
        a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
        fig.text(.035,.974,TASK_NAMES[ti],fontsize=12,weight='bold',va='top')
        fig.text(.035,.935,'16 methods | 12 datasets | 3 seeds | all four primary metrics retained',fontsize=6.8)
        metric,label,high=METRICS[t][0]
        im=heat(a,raw[t].mean(axis=1)[:,:,0].T,DS_NAMES,
                [f'{i+1}  {display(m)}' for i,m in enumerate(MODELS)],SEQ,0,1)
        letter(a,'A',label+' by dataset')
        a.texts[-1].set_y(1.13)
        for edge in [3.5,7.5]:a.axvline(edge,color='white',lw=1.5)
        for center,name in [(1.5,'Human organs'),(5.5,'Diseases'),(9.5,'Mouse organs')]:
            a.text(center,-.88,name,ha='center',va='bottom',fontsize=5.7)
        im2=heat(b,ranked[t].mean(axis=(0,1)),[x[1] for x in METRICS[t]],[],RANK,1,16)
        b.tick_params(axis='x',labelrotation=90)
        for text in b.get_xticklabels():
            text.set_ha('center');text.set_va('top');text.set_rotation_mode('default')
        letter(b,'B','Metric ranks')
        b.texts[-1].set_y(1.13)
        for ax in [a,b]:ax.axhline(12.5,color='white',lw=1.4)
        # Separate color scales: original metric units versus within-block ranks.
        cbar(fig,im,[.16,.36,.30,.010],label+' (higher is better)',[0,.5,1])
        cbar(fig,im2,[.77,.36,.20,.010],'Mean rank (lower is better)',[1,8,16])
        if t=='clustering':
            z=ranked[t].mean(axis=(1,3))
            delta=(z[8:12]-z[:4]).T
            im3=heat(c,delta.T,[str(i+1) for i in range(16)],['Marrow','Lung','Heart','Kidney'],DIV,-8,8)
            c.tick_params(axis='x',labelrotation=0)
            letter(c,'C','Organ-matched mouse minus human mean rank')
            c.set_xlabel('Method number (as in A)',labelpad=3)
            cbar(fig,im3,[.16,.045,.38,.012],'Negative: better relative mouse rank',[-8,0,8])
            extra='C: mouse-minus-human mean ranks for four organ-matched dataset pairs; negative values indicate better relative mouse rank. Organs, not individuals, are matched.'
        elif t=='correction':
            r=ranked[t].mean(axis=0) # seeds x models x metrics
            x=r[:,:,:2].mean(axis=2);y=r[:,:,2:].mean(axis=2)
            for mi,m in enumerate(MODELS):
                xx=x[:,mi];yy=y[:,mi]
                color='#477F9B' if mi<13 else '#AD8156'
                c.errorbar(xx.mean(),yy.mean(),xerr=[[xx.mean()-xx.min()],[xx.max()-xx.mean()]],
                    yerr=[[yy.mean()-yy.min()],[yy.max()-yy.mean()]],fmt='none',ecolor=color,elinewidth=.6)
                c.plot(xx.mean(),yy.mean(),'o',color=color,markersize=2.1)
                if m in ['CellPLM','UCE','PCA_model','UMAP']:
                    dy={'CellPLM':1.5,'UCE':-1.5,'PCA_model':-1.65,'UMAP':1.5}[m]
                    dx=1.8 if m=='UCE' else 0
                    c.text(xx.mean()+dx,yy.mean()+dy,display(m),fontsize=5.8,ha='center',color=color)
            c.set(xlim=(0,17),ylim=(0,17),xlabel='Batch-mixing mean rank',ylabel='Biology mean rank')
            c.set_xticks([1,4,8,12,16]);c.set_yticks([1,8,16])
            letter(c,'C','Batch mixing and biological conservation')
            extra='C: mean ranks for batch mixing (scaled iLISI and kBET) versus biological conservation (ARI and cell-type ASW), averaged equally over twelve datasets. Each point is one method, with selected methods labelled; crossed ranges show the minimum--maximum of three seed summaries, without individual seed markers. Blue denotes pretrained methods and brown dataset-fitted baselines. Lower is better on both axes.'
        elif t=='annotation':
            z=raw[t][:,:,9,:2]-raw[t][:,:,15,:2]
            # Macro-F1 then accuracy: paired raw differences, never averaged together.
            im3=heat(c,z.mean(axis=1).T,DS_NAMES,['Macro-F1','Accuracy'],DIV,-.15,.15)
            letter(c,'C','Paired score difference: scFoundation minus scVI')
            cbar(fig,im3,[.16,.045,.38,.012],'Positive: scFoundation higher',[-.15,0,.15])
            extra='C: paired scFoundation-minus-scVI differences, averaged over three seeds separately for macro-F1 and accuracy. Exported training/test counts match; exact cell identity was not independently compared. The MS comparison has documented study-level pretraining overlap.'
        else:
            z=np.array([[rho(ranked[t][di,si,:,0],ranked[t][di,si,:,2]) for si in range(3)] for di in range(12)])
            mean=z.mean(axis=1)
            c.bar(np.arange(12),mean,color='#85A7B5',width=.60)
            c.errorbar(np.arange(12),mean,yerr=[mean-z.min(axis=1),z.max(axis=1)-mean],fmt='none',
                       ecolor='#344B59',elinewidth=.6,capsize=1.5)
            c.axhline(0,color='#99A8AD',lw=.5)
            c.set_xticks(range(12),DS_NAMES,rotation=65,ha='right',rotation_mode='anchor')
            c.set(ylim=(-1.05,1.05),ylabel='Spearman rho')
            c.set_yticks([-1,0,1])
            letter(c,'C','Agreement of Pearson and MAE method rankings')
            extra='C: Spearman agreement between the sixteen-method Pearson and MAE rankings within each dataset and seed; bars show the three-seed mean and whiskers the observed seed range. MAE ranks use lower-is-better direction. Raw MAEs are not pooled across datasets.'
        caption=(TASK_NAMES[ti]+'. A: three-seed mean '+label+' for every method and dataset; higher is better. '
          'B: within-dataset--seed ranks averaged equally over twelve datasets and three seeds, separately for each of the four metrics; lower is better. '
          'The white horizontal divider separates thirteen pretrained methods from PCA, UMAP and scVI. '+extra+
          ' Seed ranges are descriptive, not confidence intervals. Complete seed boxplots are provided in the supplement. No hypothesis tests are shown.')
        save(fig,f'fig{ti+2:02d}_{t}',caption,[a,b,c])

def attributes(ranked,params,stats):
    fig,axs=plt.subplots(2,2,figsize=(6.48,6.5))
    fig.subplots_adjust(left=.22,right=.975,bottom=.12,top=.87,wspace=.52,hspace=.62)
    fig.text(.035,.975,'Model attributes and task-dependent performance',fontsize=11.5,weight='bold',va='top')
    fig.text(.035,.939,'Checkpoint-level associations | no causal attribution | pretrained models only',fontsize=6.7)
    a,b,c,d=axs.flat
    mr=np.array([ranked[t].mean(axis=(0,1,3))[:13] for t in TASKS]).T
    im=heat(a,mr,['Clust.','Batch','Annot.','Recon.'],
        [f'{i+1} {m} [{ENCODING[i]}]' for i,m in enumerate(MODELS[:13])],RANK,1,16)
    a.tick_params(axis='y',labelsize=5.7)
    letter(a,'A','Input protocol and ranks')
    for ti,t in enumerate(TASKS):
        x=np.array([params[m]/1e6 for m in MODELS[:13]])
        b.scatter(x,mr[:,ti],s=8,color=COLORS[ti],label=['Clustering','Batch','Annotation','Reconstruction'][ti])
        ids=[i for i,m in enumerate(MODELS[:13]) if m in SIZES]
        xx=np.array([SIZES[MODELS[i]] for i in ids])
        assert (xx>0).all() and (x>0).all()
        c.scatter(xx,mr[ids,ti],s=8,color=COLORS[ti])
    for ax in [b,c]:
        ax.set_xscale('log');ax.set_ylim(16,1);ax.set_yticks([1,8,16]);ax.set_ylabel('Mean task rank')
    b.set_xlabel('Audited parameters (millions)');c.set_xlabel('Reported corpus (millions)')
    b.set_xticks([10,100,1000],['10','100','1000']);c.set_xticks([30,50,100],['30','50','100'])
    b.xaxis.set_minor_locator(plt.NullLocator());c.xaxis.set_minor_locator(plt.NullLocator())
    letter(b,'B','Model size (13 models / task)');letter(c,'C','Corpus size (10 models / task)')
    x=ranked['annotation'][:4].mean(axis=(0,1,3))[:13]
    y=ranked['annotation'][8:12].mean(axis=(0,1,3))[:13]
    for mi in range(13):
        color='#AD8156' if mi in MULTI else '#477F9B'
        d.text(x[mi],y[mi],str(mi+1),ha='center',va='center',color=color,fontsize=6.4,weight='bold')
    d.set(xlim=(0,17),ylim=(17,0),xlabel='Human-organ annotation rank',ylabel='Mouse-organ annotation rank')
    d.set_xticks([1,8,16]);d.set_yticks([1,8,16]);letter(d,'D','Species context (13 models)')
    fig.legend(*b.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.57,.035),
               ncol=4,frameon=False,fontsize=5.9,columnspacing=.9,handletextpad=.3)
    save(fig,'fig06_attributes',
       'Exploratory checkpoint-attribute comparisons. A: four-task mean ranks across twelve datasets for thirteen pretrained models, with protocol codes: A, gene-aligned input; V, gene identity and expression values; R, ranked genes; C, context; S, sampled genes; P, protein-derived features; F, fixed gene panel; B, expression bins. Codes describe implementations and are not mutually exclusive tokenizer classes. B and C: one point per checkpoint and task; the same checkpoint appears in four task series. Corpus estimates are available for ten models and mix cells with LangCell cell--text pairs, with nonuniform deduplication and checkpoint attribution. D: human-organ versus mouse-organ annotation ranks; numbers identify the models in A. Blue denotes predominantly human pretraining and brown reported human/mouse or multispecies pretraining; some reports are family-level. Dataset-fitted baselines remain in the rankings but are excluded from attribute associations. No regression, confidence interval, hypothesis test or causal effect is claimed. Full model-labelled size plots and leave-one-model-out summaries appear in the supplement.',list(axs.flat))

def full_boxplots(raw):
    for ti,t in enumerate(TASKS):
        for ki,(metric,label,high) in enumerate(METRICS[t]):
            si=ti*4+ki+1
            fig,axs=plt.subplots(3,4,figsize=(6.48,6.0))
            fig.subplots_adjust(left=.095,right=.988,bottom=.26,top=.88,wspace=.24,hspace=.58)
            fig.text(.035,.97,f'S{si}  '+TASK_NAMES[ti],fontsize=10.5,weight='bold',va='top')
            fig.text(.035,.932,label+' | '+('Higher' if high else 'Lower')+' is better | 3 seeds per box',fontsize=6.7)
            amin=float(raw[t][:,:,:,ki].min());amax=float(raw[t][:,:,:,ki].max())
            for di,ax in enumerate(axs.flat):
                bp=ax.boxplot(raw[t][di,:,:,ki],positions=np.arange(1,17),widths=.65,whis=(0,100),
                    showfliers=False,patch_artist=True,manage_ticks=False,
                    boxprops={'linewidth':.4},medianprops={'color':'#233B49','linewidth':.6},
                    whiskerprops={'color':'#607D8B','linewidth':.4},capprops={'color':'#607D8B','linewidth':.4})
                for mi,box in enumerate(bp['boxes']):
                    box.set_facecolor('#8DACB8' if mi<13 else '#B8AEA0');box.set_edgecolor('#6B8792')
                assert not ax.collections,'Seed markers must remain hidden'
                ax.set_title(DS_NAMES[di],loc='left',fontsize=6.1,pad=4)
                ax.set_xlim(.3,16.7);ax.set_xticks(range(1,17),range(1,17),rotation=90)
                ax.tick_params(axis='x',labelsize=5.2,length=0,pad=2)
                ax.tick_params(axis='y',labelsize=5.3,length=2,pad=2)
                ax.axvline(13.5,color='#D4DDE1',lw=.5,ls='--')
                if not high:
                    assert amin>0
                    ax.set_yscale('log');ax.set_ylim(amin/1.12,amax*1.12)
                    ax.yaxis.set_major_locator(LogLocator(numticks=3))
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'1e{int(round(np.log10(v)))}'))
                    ax.yaxis.set_minor_locator(plt.NullLocator())
                else:
                    pad=max(.025,(amax-amin)*.06)
                    ax.set_ylim(min(0,amin-pad),min(1,max(amax+pad,.1)))
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                if di%4:ax.set_yticklabels([])
                ax.set_axisbelow(True);ax.yaxis.grid(color='#E7ECEE',lw=.35)
            for pi,name in enumerate(PANEL_NAMES):
                bb=axs[pi,0].get_position()
                fig.text(.018,(bb.y0+bb.y1)/2,name,rotation=90,fontsize=6.2,ha='center',va='center',weight='bold')
            for mi,m in enumerate(MODELS):
                fig.text(.095+(mi%4)*.225,.178-(mi//4)*.027,f'{mi+1}  {display(m)}',fontsize=6.1)
            fig.text(.095,.035,'Box: Q1–Q3 | Line: median | Whiskers: min–max | Seeds: 41, 42, 43',fontsize=5.8)
            caption=(TASK_NAMES[ti]+' assessed by '+label+'. Rows show human organs, human diseases and mouse organs; columns identify four datasets within each row. '
                'Numbers 1–16 identify methods in the key. Each box uses seeds 41, 42 and 43, with linearly interpolated quartiles, median line and observed minimum--maximum whiskers. Individual seed markers and outlier markers are hidden. '
                'These are repeated runs/splits, not independent biological replicates or confidence intervals. '+
                ('MAE uses a logarithmic axis and dataset-defined expression scales; compare methods within datasets. ' if not high else '')+
                'No hypothesis tests are displayed.')
            slug=f's{si:02d}_{t}_{metric}'
            save(fig,slug,caption,list(axs.flat))
            rows=[]
            for di,seed,mi in itertools.product(range(12),range(3),range(16)):
                rows.append(dict(panel=PANELS[di//4],dataset=DS[di],task=t,model=MODELS[mi],seed=SEEDS[seed],
                                 metric=metric,value=float(raw[t][di,seed,mi,ki])))
            writecsv(R/'figures'/'task_layout'/slug/'source_data.csv',rows)

def supplemental_audits(raw,ranked,params):
    fig,axs=plt.subplots(2,2,figsize=(6.48,5.5))
    fig.subplots_adjust(left=.17,right=.98,bottom=.18,top=.88,wspace=.38,hspace=.62)
    fig.text(.035,.975,'Seed variation in within-block method ranks',fontsize=11,weight='bold',va='top')
    for ti,(ax,t) in enumerate(zip(axs.flat,TASKS)):
        z=ranked[t].mean(axis=3).max(axis=1)-ranked[t].mean(axis=3).min(axis=1)
        heat(ax,z.T,DS_NAMES,[display(m) for m in MODELS] if ti%2==0 else [],SEQ,0,8)
        ax.tick_params(axis='x',labelrotation=90)
        for text in ax.get_xticklabels():
            text.set_ha('center');text.set_va('top');text.set_rotation_mode('default')
        letter(ax,chr(65+ti),TASK_NAMES[ti].replace('Native-embedding batch assessment','Batch assessment'))
        ax.tick_params(axis='x',labelsize=5.2);ax.tick_params(axis='y',labelsize=5.4)
    save(fig,'s17_seed_stability','Maximum minus minimum seed-specific mean metric rank for each method and dataset. Each seed rank averages four within-dataset primary-metric ranks; the plotted range spans seeds 41, 42 and 43. Higher values indicate more variation. Color scale: 0–8 rank positions. No independent-study replication or confidence interval is implied.',list(axs.flat))
    fig,axs=plt.subplots(2,2,figsize=(6.48,5.0))
    fig.subplots_adjust(left=.15,right=.96,bottom=.16,top=.88,wspace=.60,hspace=.66)
    fig.text(.035,.975,'Agreement among primary metric rankings',fontsize=11,weight='bold',va='top')
    for ti,(ax,t) in enumerate(zip(axs.flat,TASKS)):
        a=np.array([[np.mean([rho(ranked[t][di,si,:,i],ranked[t][di,si,:,j])
             for di in range(12) for si in range(3)]) for j in range(4)] for i in range(4)])
        heat(ax,a,[v[1] for v in METRICS[t]],[v[1] for v in METRICS[t]],DIV,-1,1)
        ax.tick_params(axis='both',labelsize=5.4)
        letter(ax,chr(65+ti),['Clustering','Batch assessment','Annotation','Reconstruction'][ti])
    save(fig,'s18_metric_agreement','Spearman correlation of method rankings between every pair of primary metrics, calculated within each of 36 dataset--seed blocks and then averaged equally across twelve datasets and three seeds. MAE is oriented lower-is-better before ranking. Color scale: -1 to 1. This is descriptive agreement, with no hypothesis test.',list(axs.flat))
    fig,axs=plt.subplots(1,2,figsize=(6.48,4.2))
    fig.subplots_adjust(left=.18,right=.98,bottom=.18,top=.86,wspace=.35)
    fig.text(.035,.975,'Annotation sensitivity to documented pretraining overlap',fontsize=10.6,weight='bold',va='top')
    s19map=LinearSegmentedColormap.from_list('s19',['#C5D0C2','#F7F6F2'])
    for pi,ax in enumerate(axs):
        ix=list(range(pi*4,pi*4+4));ix.remove(2 if pi==0 else 4)
        full=ranked['annotation'][pi*4:pi*4+4].mean(axis=(0,1,3))
        excl=ranked['annotation'][ix].mean(axis=(0,1,3))
        heat(ax,np.c_[full,excl],['Full','Excluded'],[display(m) for m in MODELS] if pi==0 else [],s19map,1,16)
        for mi in range(16):
            for j,v in enumerate([full[mi],excl[mi]]):
                rgb=np.array(s19map((np.clip(v,1,16)-1)/15)[:3])
                lin=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
                lum=float(lin @ np.array([.2126,.7152,.0722]))
                ax.text(j,mi,f'{v:.1f}',ha='center',va='center',fontsize=5.8,
                        color='#FFFFFF' if lum<0.45 else '#263C4B')
        letter(ax,chr(65+pi),['Human organs: heart excluded','Diseases: MS excluded'][pi])
    save(fig,'s19_overlap_sensitivity','Full-panel and leave-study-out annotation ranks for all sixteen methods. The human-organ comparison excludes heart; the disease comparison excludes MS. Remaining datasets are equally weighted. The audit established source-study and sample/donor label matches, not a cell-barcode overlap fraction. These exclusions do not establish that the remaining data were absent from all pretraining corpora.',list(axs))
    for factor,info,number in [('parameters',params,20),('corpus',SIZES,21)]:
        fig,axs=plt.subplots(2,2,figsize=(6.48,5.4))
        fig.subplots_adjust(left=.13,right=.98,bottom=.13,top=.87,wspace=.32,hspace=.56)
        fig.text(.035,.975,('Audited parameter count' if factor=='parameters' else 'Reported pretraining corpus')+' and task ranks',fontsize=11,weight='bold',va='top')
        for ti,(ax,t) in enumerate(zip(axs.flat,TASKS)):
            ids=[i for i,m in enumerate(MODELS[:13]) if m in info]
            x=np.array([info[MODELS[i]]/(1e6 if factor=='parameters' else 1) for i in ids])
            y=ranked[t].mean(axis=(0,1,3))[ids]
            assert (x>0).all()
            ax.set_xscale('log');ax.set(xlim=(x.min()*.75,x.max()*1.35),ylim=(16.5,.5),ylabel='Mean task rank',xlabel='Millions')
            ticks=[10,100,1000] if factor=='parameters' else [30,50,100]
            ax.set_xticks(ticks,[str(v) for v in ticks]);ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.set_yticks([1,4,8,12,16])
            letter(ax,chr(65+ti),['Clustering','Batch assessment','Annotation','Reconstruction'][ti])
            label_points(ax,x,y,[i+1 for i in ids],COLORS[ti])
        save(fig,f's{number:02d}_{factor}',
             ('Audited instantiated-model parameters' if factor=='parameters' else 'Reported corpus size')+
             ' versus mean task rank, averaged equally over twelve datasets, three seeds and four primary metrics. Numbers identify the thirteen pretrained methods in Figure 6A. '+
             ('All thirteen checkpoints are included. Parameters include different model components, including frozen weights; they do not measure active compute.' if factor=='parameters' else
              'Ten models with recorded estimates are included; CellPLM, Tabula and scPRINT2 lack comparable verified checkpoint-specific values. LangCell uses cell--text pairs; other values primarily count cells. Some estimates are family-level or lower bounds.')+
             ' No scaling-law fit, confidence interval or hypothesis test is shown. Baselines remain in the ranks but are excluded from the attribute axis.',list(axs.flat))

def review_deck():
    from pptx import Presentation
    from pptx.util import Inches
    from pptx.opc.package import Part
    from pptx.opc.packuri import PackURI
    from pptx.opc.constants import RELATIONSHIP_TYPE as RT
    from pptx.oxml.xmlchemy import OxmlElement
    from pptx.oxml.ns import qn
    from lxml import etree
    import pymupdf
    prs=Presentation();prs.slide_width=Inches(8.2);prs.slide_height=Inches(8.2)
    combined=pymupdf.open()
    for index,r in enumerate(RECORDS,1):
        f=R/r['folder'];slug=r['slug'];slide=prs.slides.add_slide(prs.slide_layouts[6])
        scale=min(7.9/r['width_inches'],7.9/r['height_inches'])
        w=r['width_inches']*scale;h=r['height_inches']*scale
        pic=slide.shapes.add_picture(str(f/(slug+'.png')),Inches((8.2-w)/2),Inches((8.2-h)/2),width=Inches(w),height=Inches(h))
        part=Part(PackURI(f'/ppt/media/vector{index:02d}.svg'),'image/svg+xml',prs.part.package,(f/(slug+'_outlined.svg')).read_bytes())
        rid=slide.part.relate_to(part,RT.IMAGE);exts=OxmlElement('a:extLst');ext=OxmlElement('a:ext')
        ext.set('uri','{96DAC541-7B7A-43D3-8B79-37D633B846F1}')
        ns='http://schemas.microsoft.com/office/drawing/2016/SVG/main'
        element=etree.Element('{'+ns+'}svgBlip',nsmap={'asvg':ns});element.set(qn('r:embed'),rid)
        ext.append(element);exts.append(ext);pic._pic.blipFill.blip.append(exts)
        slide.notes_slide.notes_text_frame.text=r['caption']
        with pymupdf.open(f/(slug+'.pdf')) as doc:combined.insert_pdf(doc)
    prs.save(R/'manuscript'/'CellBench_main_and_supplement_figures.pptx')
    combined.save(R/'manuscript'/'CellBench_main_and_supplement_figures.pdf')
    combined.close()
    (R/'figures'/'task_layout'/'manifest.json').write_text(json.dumps(RECORDS,indent=2))
    print('Saved vector review deck:',len(RECORDS),'slides',flush=True)

def export_figure_sources(raw,ranked,params,stats):
    root=R/'figures'/'task_layout'
    (root/'fig01_framework'/'source_data.json').write_text(json.dumps({
        k:v for k,v in stats.items() if k not in ['panel_summary','associations']},indent=2))
    for ti,t in enumerate(TASKS):
        rows=[]
        for di,si,mi,ki in itertools.product(range(12),range(3),range(16),range(4)):
            rows.append(dict(panel=PANELS[di//4],dataset=DS[di],seed=SEEDS[si],model=MODELS[mi],
                metric=METRICS[t][ki][0],value=float(raw[t][di,si,mi,ki]),
                within_block_rank=float(ranked[t][di,si,mi,ki])))
        writecsv(root/f'fig{ti+2:02d}_{t}'/'source_data.csv',rows)
    rows=[]
    for ti,t in enumerate(TASKS):
        mean=ranked[t].mean(axis=(0,1,3))
        human=ranked['annotation'][:4].mean(axis=(0,1,3))
        mouse=ranked['annotation'][8:12].mean(axis=(0,1,3))
        for mi,m in enumerate(MODELS[:13]):
            rows.append(dict(method_id=mi+1,model=m,task=t,mean_rank=float(mean[mi]),
                input_code=ENCODING[mi],pretraining_species=SPECIES[mi],parameters=params[m],
                reported_corpus_millions=SIZES.get(m,''),human_annotation_rank=float(human[mi]),
                mouse_annotation_rank=float(mouse[mi])))
    writecsv(root/'fig06_attributes'/'source_data.csv',rows)
    writecsv(root/'s20_parameters'/'source_data.csv',rows)
    writecsv(root/'s21_corpus'/'source_data.csv',[r for r in rows if r['reported_corpus_millions']!=''])
    stable=[];agreement=[];overlap=[]
    for t in TASKS:
        seedranks=ranked[t].mean(axis=3)
        for di,mi in itertools.product(range(12),range(16)):
            z=seedranks[di,:,mi]
            stable.append(dict(task=t,dataset=DS[di],model=MODELS[mi],min_rank=float(z.min()),
                              max_rank=float(z.max()),rank_range=float(z.max()-z.min())))
        for i,j in itertools.product(range(4),range(4)):
            z=[rho(ranked[t][di,si,:,i],ranked[t][di,si,:,j]) for di in range(12) for si in range(3)]
            agreement.append(dict(task=t,metric1=METRICS[t][i][0],metric2=METRICS[t][j][0],
                                  mean_spearman_rho=float(np.mean(z)),blocks=len(z)))
    for pi in range(2):
        ix=list(range(pi*4,pi*4+4));ix.remove(2 if pi==0 else 4)
        full=ranked['annotation'][pi*4:pi*4+4].mean(axis=(0,1,3))
        reduced=ranked['annotation'][ix].mean(axis=(0,1,3))
        for mi,m in enumerate(MODELS):
            overlap.append(dict(panel=PANELS[pi],model=m,full_rank=float(full[mi]),
                                excluded_dataset=['heart','ms'][pi],excluded_rank=float(reduced[mi])))
    writecsv(root/'s17_seed_stability'/'source_data.csv',stable)
    writecsv(root/'s18_metric_agreement'/'source_data.csv',agreement)
    writecsv(root/'s19_overlap_sensitivity'/'source_data.csv',overlap)

if __name__=='__main__':
    raw,ranked,params,stats=compute()
    overview();task_figures(raw,ranked);attributes(ranked,params,stats)
    full_boxplots(raw);supplemental_audits(raw,ranked,params)
    export_figure_sources(raw,ranked,params,stats)
    review_deck()
