#!/usr/bin/env python3
"""Build the manuscript appendix and a seed-level, filterable Excel workbook.

Only project exports are used; raw metrics are averaged within model/dataset
across seeds for the PDF and kept separately per seed in the workbook.
"""
from pathlib import Path
import json,math,re
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

R=Path(__file__).resolve().parents[1];D=R/'data';O=D/'supplementary_tables';O.mkdir(exist_ok=True)
panels=json.loads((D/'experiment_panels.json').read_text())
models=['CellFM','CellPLM','GeneCompass','Geneformer','LangCell','Nicheformer','Tabula','TranscriptFormer','UCE','scFoundation','scGPT','scLong','scPRINT2','PCA_model','UMAP','scVI']
titles={'organ':'Human organs','disease':'Human diseases','species':'Mouse organs','perturbation':'Perturbation'}
metrics={
 'clustering':[('ari','ARI',True),('nmi','NMI',True),('asw','ASW',True),('acc','ACC',True)],
 'annotation':[('acc','ACC',True),('f1_m','Macro-F1',True),('balanced_accuracy','Balanced ACC',True),('precision_macro','Macro-precision',True)],
 'correction':[('ARI','ARI',True),('ASW_celltype','Cell-type ASW',True),('iLISI_scaled','Scaled iLISI',True),('kBET','kBET',True)],
 'reconstruction':[('Pearson_expr','Pearson',True),('Spearman_expr','Spearman',True),('MAE_expr','MAE',False),('TopK_expression_overlap','Top-50 overlap',True)],
 'perturbation':[('Pearson_delta','Pearson',True),('Spearman_delta','Spearman',True),('MAE_delta','MAE',False),('DES','DES',True)]}
labeltask={'clustering':'clustering','annotation':'cell annotation','correction':'native-embedding batch assessment','reconstruction':'expression reconstruction','perturbation':'perturbation prediction'}
escape_map={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
def esc(x):
 s=str(x);return ''.join(escape_map.get(c,c) for c in s)
def mname(x):return 'PCA' if x=='PCA_model' else x
def fmt(x,n=3):
 if x is None or pd.isna(x):return '--'
 return f'{float(x):.{n}f}'
def nfmt(x):return f'{int(x):,}' if pd.notna(x) else '--'
tex=[];wb=Workbook();index=wb.active;index.title='Contents';index_rows=[]

def sheet(name,title,frame,note):
 assert len(name)<=31 and name not in wb.sheetnames
 w=wb.create_sheet(name);w.append([title]);w.append([note]);w.append([]);w.append(list(frame.columns))
 for row in frame.itertuples(index=False,name=None):
  w.append([None if pd.isna(v) else (v.item() if hasattr(v,'item') else v) for v in row])
 w.freeze_panes='C5';w.auto_filter.ref=f'A4:{get_column_letter(len(frame.columns))}{w.max_row}'
 w.row_dimensions[1].height=27;w.row_dimensions[2].height=30
 for c in w[1]:c.font=Font(name='Aptos',size=14,bold=True,color='173754')
 w.cell(2,1).font=Font(name='Aptos',size=10,italic=True,color='576B7C');w.cell(2,1).alignment=Alignment(wrap_text=True,vertical='center')
 for c in w[4]:c.fill=PatternFill('solid',fgColor='244C68');c.font=Font(name='Aptos',bold=True,color='FFFFFF',size=10);c.alignment=Alignment(vertical='center',wrap_text=True)
 w.row_dimensions[4].height=29
 for row in w.iter_rows(min_row=5):
  for c in row:
   c.font=Font(name='Aptos',size=10,color='223447');c.alignment=Alignment(vertical='top',wrap_text=False)
   if c.row%2==0:c.fill=PatternFill('solid',fgColor='F1F6F8')
   if isinstance(c.value,float):c.number_format='0.0000'
 for col in range(1,len(frame.columns)+1):
  values=[str(frame.columns[col-1])]+[str(v) for v in frame.iloc[:100,col-1] if pd.notna(v)]
  w.column_dimensions[get_column_letter(col)].width=min(44,max(13,max(map(len,values))+2))
 w.sheet_view.showGridLines=False;w.sheet_properties.pageSetUpPr.fitToPage=True;w.page_setup.fitToWidth=1;w.print_title_rows='1:4'
 index_rows.append((name,title,note))

def begin(caption,cols,heads,label,need=r'.47\textheight'):
 tex.extend([r'\suppneedspace{'+need+'}',r'\begingroup',r'\footnotesize',r'\setlength{\tabcolsep}{4pt}',r'\renewcommand{\arraystretch}{1.12}',r'\begin{longtable}{'+cols+'}',r'\caption{'+caption+r'}\label{'+label+r'}\\',r'\toprule',heads+r'\\',r'\midrule',r'\endfirsthead',r'\multicolumn{'+str(heads.count('&')+1)+r'}{l}{\small\itshape Table \thetable\ (continued)}\\',r'\toprule',heads+r'\\',r'\midrule',r'\endhead',r'\midrule',r'\multicolumn{'+str(heads.count('&')+1)+r'}{r}{\small\itshape Continued on next page}\\',r'\endfoot',r'\bottomrule',r'\endlastfoot'])
def end():tex.extend([r'\end{longtable}',r'\endgroup',''])
def row(*items):tex.append(' & '.join(str(i) for i in items)+r' \\')

def source_link(url):
 if not isinstance(url,str) or not url.startswith('http'):return '--'
 return r'\href{'+url.split(' ; ')[0]+r'}{source}'

# Dataset coverage and provenance, matching Norman/Adamson to their prepared full records.
prov=pd.read_csv(D/'dataset_provenance.csv').set_index('dataset');assets={x['id']:x for x in json.loads((D/'dataset_assets.json').read_text())}
drows=[]
for panel in panels:
 for ds in panels[panel]['datasets']:
  key={'norman':'norman_full','adamson':'adamson_full'}.get(ds,ds);v=prov.loc[key]
  source=v['source'];source='https://cellxgene.cziscience.com/collections/0b9d8a04-bb9d-44da-aa27-705bb65b54eb' if ds=='mouse_heart' else source
  drows.append(dict(panel=titles[panel],dataset=ds,prepared_cells=int(v.cells),genes=int(v.genes),source=source,preparation_record=v.provenance))
df=pd.DataFrame(drows);assert len(df)==16;df.to_csv(O/'S01_datasets.csv',index=False)
sheet('S01 Datasets','Dataset provenance and dimensions',df,'Prepared cell and gene totals, before dataset-specific train/validation/test splits. Source links identify dataset provenance.')
begin('Prepared evaluation datasets and source records. Cell and gene counts describe prepared matrices before downstream splits. Norman and Adamson use the prepared full-study records; mouse heart combines two assay sources from the same collection.',r'p{2.7cm}p{3.5cm}rrl',r'Panel & Dataset & Cells & Genes & Source','tab:supp_datasets')
for v in df.itertuples():row(esc(v.panel),esc(v.dataset),nfmt(v.prepared_cells),nfmt(v.genes),source_link(v.source))
end()
# Models and experiment-specific parameter counts.
attrs=pd.read_csv(D/'model_attributes_working.csv').set_index('model');counts=pd.read_csv(D/'checkpoint_parameter_counts.csv').set_index('model')
contexts={'CellFM':'2,048','CellPLM':'Unknown','GeneCompass':'2,048','Geneformer':'2,048','LangCell':'2,048 (family)','Nicheformer':'1,500','Tabula':'Unknown','TranscriptFormer':'1,023','UCE':'1,536','scFoundation':'19,264 genes','scGPT':'1,200','scLong':'27,875','scPRINT2':'2,000'}
rows=[]
for model in models:
 if model in counts.index:
  a=attrs.loc[model];v=counts.loc[model];rows.append(dict(model=model,architecture=a.architecture,parameter_count=int(v.total_parameters),context=contexts[model],checkpoint=v.checkpoint,source=a.source))
 else:rows.append(dict(model=model,architecture='Dataset-fitted baseline',parameter_count=None,context='Dataset-specific',checkpoint='No pretrained checkpoint',source=''))
md=pd.DataFrame(rows);md.to_csv(O/'S02_models.csv',index=False)
sheet('S02 Models','Model architecture and checkpoint parameters',md,'Registered instantiated-model parameters include frozen weights and omit buffers; see parameter_audit.md for model-specific scope.')
begin('Model architecture, instantiated-model parameter counts and context in the benchmark. Counts include registered frozen parameters, exclude buffers and shared-storage duplicates, and can include unused components. PCA, UMAP and scVI are dataset-fitted. See the checkpoint audit for component-specific exclusions and provenance.',r'p{3.0cm}>{\raggedright\arraybackslash}p{5.2cm}rl',r'Method & Architecture & Parameters (M) & Context','tab:supp_models',need=r'.55\textheight')
for v in md.itertuples():row(esc(mname(v.model)),esc(v.architecture),f'{v.parameter_count/1e6:,.1f}' if pd.notna(v.parameter_count) else '--',esc(v.context))
end()
# Corpus composition and tokenization, exactly as recorded with source links.
inputs={'CellFM':'Gene-aligned expression, CLS','CellPLM':'Gene expression and cell-level embedding','GeneCompass':'Ranked genes, values and knowledge priors','Geneformer':'Median-normalized ranked genes','LangCell':'Ranked genes and cell projection','Nicheformer':'Ranked genes with context tokens','Tabula':'Tabular gene and value encoding','TranscriptFormer':'Gene identity, expression and assay token','UCE':'Sampled genes and protein-derived features','scFoundation':'Fixed gene index and expression values','scGPT':'Gene identity and binned expression','scLong':'Fixed gene panel and expression values','scPRINT2':'Gene-aligned counts and context'}
corpus=pd.DataFrame([dict(model=m,input_protocol=inputs[m],corpus=attrs.loc[m,'pretraining_size_unit'],species=attrs.loc[m,'pretraining_species'],source=attrs.loc[m,'source']) for m in models if m in counts.index]);corpus.to_csv(O/'S03_corpora.csv',index=False)
sheet('S03 Corpora','Pretraining corpus and species',corpus,'Family-level reports are not automatically specific to the evaluated checkpoint; corpus units and deduplication differ.')
# English compact versions mirror the manuscript attribute table.
cr={'CellFM':('~100M cells (family)','Human'),'CellPLM':('>9M scRNA + >2M spatial (family)','Human (paper)'), 'GeneCompass':('101.8M cells after QC','Human, mouse'),'Geneformer':('~30M cells','Human'),'LangCell':('~27.5M cell--text pairs','Human'),'Nicheformer':('>110M cells','Human, mouse'),'Tabula':('15M profiles (family)','Human (paper)'),'TranscriptFormer':('57M cells','Human'),'UCE':('>36M cells (family)','Eight species (family)'),'scFoundation':('>50M cells','Human'),'scGPT':('~33M cells','Human'),'scLong':('48,024,242 unique cells','Human'),'scPRINT2':('>350M cells (family)','16 species (family)')}
begin('Reported pretraining corpus and species. Family-level values are not independently verified for every checkpoint. LangCell counts cell--text pairs; other sources primarily report cells. Deduplication and quality-control definitions differ.',r'p{3.2cm}p{5.5cm}p{3.2cm}l',r'Method & Reported corpus & Pretraining species & Source','tab:supp_corpora')
for m in cr:
 size,sp=cr[m];row(esc(m),esc(size),esc(sp),source_link(attrs.loc[m,'source']))
end()
# Panel-level full rankings for all methods.
ranks=pd.read_csv(D/'all_panel_ranks.csv');dataset_ranks=pd.read_csv(D/'dataset_mean_ranks.csv')
for panel in panels:
 tasks=panels[panel]['tasks'];out=[]
 for m in models:
  d={'model':m}
  for task in tasks:
   z=ranks[(ranks.panel==panel)&(ranks.task==task)&(ranks.model==m)];assert len(z)==1;d[task]=float(z.mean_metric_rank.iloc[0])
  out.append(d)
 frame=pd.DataFrame(out);name=f'S{4+list(panels).index(panel):02d}'
 frame.to_csv(O/f'{name}_{panel}_ranks.csv',index=False);sheet(name+' Ranks',titles[panel]+' panel mean ranks',frame,'Equal-weight mean of within-dataset, within-seed metric ranks; rank 1 is best.')
 heads=['Method']+[labeltask[t].replace('native-embedding batch assessment','Batch assessment').replace('cell annotation','Annotation').replace('expression reconstruction','Reconstruction').replace('perturbation prediction','Perturbation').title() for t in tasks]
 colspec=r'p{3.7cm}'+''.join('r' for _ in tasks)
 begin(titles[panel]+' panel: mean model ranks for each task. Ranks are calculated within each dataset, seed and metric over all 16 methods, then averaged equally across four metrics, three seeds and four datasets. Lower is better; ties retain average ranks.',colspec,' & '.join(map(esc,heads)),'tab:supp_rank_'+panel)
 for v in frame.itertuples(index=False,name=None):row(esc(mname(v[0])),*[fmt(x) for x in v[1:]])
 end()
# All task/panel primary metrics, 16 methods x four datasets; seed-level values in workbook.
for panel in panels:
 for task in panels[panel]['tasks']:
  path=D/'exports'/panel/f'{task}_scores.csv';raw=pd.read_csv(path);metric=metrics[task];names=[m[0] for m in metric]
  assert len(raw)==192 and set(raw.model)==set(models) and set(raw.dataset)==set(panels[panel]['datasets'])
  assert raw.groupby(['dataset','model']).seed.nunique().eq(3).all() and np.isfinite(raw[names].to_numpy(dtype=float)).all()
  raw=raw[['dataset','model','seed']+names].copy();raw['dataset']=pd.Categorical(raw.dataset,categories=panels[panel]['datasets'],ordered=True);raw['model']=pd.Categorical(raw.model,categories=models,ordered=True);raw=raw.sort_values(['dataset','model','seed'])
  means=raw.groupby(['dataset','model'],observed=True,as_index=False)[names].mean();assert len(means)==64
  means.to_csv(O/f'{panel}_{task}_three_seed_means.csv',index=False)
  tabname={'organ':'Org','disease':'Dis','species':'Mouse','perturbation':'Pert'}[panel]+' '+{'clustering':'Cluster','annotation':'Annot','correction':'Batch','reconstruction':'Recon','perturbation':'Predict'}[task]
  sheet(tabname,titles[panel]+' / '+labeltask[task]+' — seed-level primary metrics',raw,'Three seeds (41, 42, 43) for each model and dataset; metric names and directions are documented in the PDF appendix.')
  directions=', '.join(f'{m[1]} '+('higher' if m[2] else 'lower') for m in metric)
  caption=f'{titles[panel]}: {labeltask[task]} primary metrics for every model and dataset. Each entry is the mean of seeds 41, 42 and 43 for one dataset and method; individual seed values are in the accompanying workbook and source exports. Direction: {directions}. No confidence interval is implied.'
  heads=r'Dataset & Method & '+' & '.join(esc(m[1])+(' $\\uparrow$' if m[2] else ' $\\downarrow$') for m in metric)
  begin(caption,r'p{3.0cm}p{3.0cm}rrrr',heads,'tab:supp_'+panel+'_'+task,need=r'.30\textheight')
  for ds in panels[panel]['datasets']:
   zz=means[means.dataset==ds];assert len(zz)==16
   for j,v in enumerate(zz.itertuples(index=False,name=None)):
    if j==0:tex.append(r'\rowcolor{suppband}')
    row(esc(ds),esc(mname(v[1])),*[fmt(n,4) for n in v[2:]])
  end()
# Paired scFoundation vs scVI annotation.
pair=pd.read_csv(D/'annotation_paired_metric_summary.csv');assert len(pair)==24
sheet('Paired annotation','scFoundation minus scVI annotation metrics',pair,'Within-dataset and within-seed differences; min–max is the observed range across three seeds, not a confidence interval.')
begin('Within-dataset annotation differences, scFoundation minus scVI. The mean and minimum--maximum range describe three seeds for each metric separately; the range is not a confidence interval. Equal train/test counts were verified, but exact test-cell identity was not independently checked.',r'p{3cm}p{3cm}p{2.0cm}rrr',r'Panel & Dataset & Metric & Mean & Minimum & Maximum','tab:supp_pair')
for v in pair.itertuples():row(esc(titles[v.panel]),esc(v.dataset),esc('Macro-F1' if v.metric=='f1_m' else 'Accuracy'),fmt(v.mean_difference,4),fmt(v.min_seed,4),fmt(v.max_seed,4))
end()
# Mouse-human matched organ difference, all 16 methods.
shift=pd.read_csv(D/'organ_matched_annotation_rank_shifts.csv');assert len(shift)==64
s=shift.pivot(index='model',columns='human_dataset',values='mouse_minus_human').reindex(models)[['bmmc','hlca','kidney','heart']].reset_index()
s.to_csv(O/'organ_mouse_rank_shifts.csv',index=False);sheet('Organ rank shifts','Organ-matched mouse minus human annotation rank',s,'Tissue-matched datasets, not matched individuals; negative means better relative mouse rank.')
begin('Organ-matched mouse-minus-human annotation rank differences. Values compare separate datasets matched by organ, not paired donors. Negative values denote a better relative rank in the mouse dataset; different cell composition, source and assay may affect the comparison.',r'p{3.8cm}rrrr',r'Method & Marrow & Lung & Kidney & Heart','tab:supp_species')
for v in s.itertuples(index=False,name=None):row(esc(mname(v[0])),*[f'{float(x):+.3f}' for x in v[1:]])
end()
# Factor correlations / sensitivity analyses.
pr=pd.read_csv(D/'parameter_rank_associations.csv');pre=pd.read_csv(D/'pretraining_size_associations.csv');selection=[('organ','annotation'),('disease','annotation'),('species','annotation'),('perturbation','perturbation')]
records=[]
for factor,d in [('Parameters',pr),('Pretraining corpus',pre)]:
 for panel,task in selection:
  z=d[(d.panel==panel)&(d.task==task)];assert len(z)==1;v=z.iloc[0];records.append(dict(factor=factor,panel=titles[panel],n_models=int(v.n_models),rho=v.spearman_rho,leave_one_out_min=v.leave_one_out_min,leave_one_out_max=v.leave_one_out_max))
a=pd.DataFrame(records);sheet('Associations','Checkpoint-level exploratory associations',a,'Leave-one-model-out minimum and maximum are sensitivity summaries, not confidence intervals; factor definitions and model components differ.')
begin('Exploratory associations between model attributes and panel ranks. Spearman correlations use one observation per pretrained method. Leave-one-model-out ranges describe sensitivity to model selection, not confidence intervals. Corpus units and parameter scopes differ across models.',r'p{3.1cm}p{3.4cm}rrrr',r'Factor & Panel & $n$ & $\rho$ & Min. & Max.','tab:supp_correlations')
for v in a.itertuples():row(esc(v.factor),esc(v.panel),v.n_models,fmt(v.rho),fmt(v.leave_one_out_min),fmt(v.leave_one_out_max))
end()
# Complete matched-study exclusion sensitivity, retaining all 16 methods.
x=pd.read_csv(D/'overlap_exclusion_sensitivity.csv');out=[]
for m in models:
 r={'model':m}
 for panel,tag in [('organ','heart'),('disease','ms')]:
  z=x[(x.model==m)&(x.panel==panel)&(x.task=='annotation')&(x.excluded_dataset==tag)];assert len(z)==1;r[panel+'_rank']=z.iloc[0].mean_rank;r[panel+'_position']=int(z.iloc[0].position)
 out.append(r)
y=pd.DataFrame(out);sheet('Overlap sensitivity','Ranks after excluding overlapping studies',y,'Within-panel rank summaries exclude heart (human-organ) or MS (human-disease); remaining three datasets are equally weighted.')
begin('Annotation ranks after excluding studies with documented pretraining-source overlap for scFoundation. Each value averages the remaining three datasets, three seeds and four metrics. Positions order the 16 methods in each sensitivity analysis; excluding these studies does not establish that other data are unseen.',r'p{3.8cm}rrrr',r'Method & Heart excluded & Position & MS excluded & Position','tab:supp_overlap')
for v in y.itertuples():row(esc(mname(v.model)),fmt(v.organ_rank),v.organ_position,fmt(v.disease_rank),v.disease_position)
end()
# Split sizes from model-independent annotation/perturbation exports.
split_rows=[]
for panel in panels:
 task='perturbation' if panel=='perturbation' else 'annotation'
 z=pd.read_csv(D/'exports'/panel/f'{task}_scores.csv')
 columns=['n_train','n_val','n_test']+(['n_train_conditions','n_val_conditions','n_test_conditions'] if panel=='perturbation' else ['n_train_donors','n_val_donors','n_test_donors'])
 assert z.groupby(['dataset','seed'])[columns].nunique(dropna=False).eq(1).all().all()
 for ds in panels[panel]['datasets']:
  for seed in [41,42,43]:
   q=z[(z.dataset==ds)&(z.seed==seed)].iloc[0]
   split_rows.append(dict(panel=titles[panel],dataset=ds,seed=seed,train_cells=int(q[columns[0]]),val_cells=int(q[columns[1]]),test_cells=int(q[columns[2]]),unit='conditions' if panel=='perturbation' else 'donors',train_units=int(q[columns[3]]),val_units=int(q[columns[4]]),test_units=int(q[columns[5]])))
splits=pd.DataFrame(split_rows);assert len(splits)==48;splits.to_csv(O/'split_counts_by_seed.csv',index=False)
sheet('Split counts','Cell and donor or condition split counts',splits,'Counts are shared across model exports within each dataset and seed. Equal counts do not establish identical cell IDs; donors may recur under fallback splitting.')
begin('Exported train, validation and test counts by dataset and seed. The three cell counts are followed by donor counts for organ and disease panels or perturbation-condition counts for the perturbation panel. Counts agree across methods within each block but do not establish identical cell identities. Donor counts may overlap under fallback splitting.',r'p{2.4cm}p{3cm}rp{2.8cm}p{2.2cm}p{2.3cm}',r'Panel & Dataset & Seed & Cells: train/val/test & Unit & Units: train/val/test','tab:supp_splits',need=r'.35\textheight')
for v in splits.itertuples():
 row(esc(v.panel),esc(v.dataset),v.seed,f'{v.train_cells:,} / {v.val_cells:,} / {v.test_cells:,}',esc(v.unit),f'{v.train_units} / {v.val_units} / {v.test_units}')
end()
# Workbook retains all model/dataset overlap records and seed-level scores.
over=pd.read_csv(D/'pretraining_overlap_audit.csv');sheet('Overlap evidence','Pretraining and evaluation overlap evidence',over,'Unresolved does not mean no overlap; study/sample evidence does not establish cell-level identity.')
index.append(['Supplementary tables: evidence index']);index.append(['PDF appendix and per-seed primary metrics from the benchmark. No test statistics or confidence intervals are inferred.']);index.append([]);index.append(['Sheet','Contents','Interpretation'])
for r in index_rows:index.append(r)
index.column_dimensions['A'].width=24;index.column_dimensions['B'].width=55;index.column_dimensions['C'].width=85;index.freeze_panes='B5';index.sheet_view.showGridLines=False
for c in index[1]:c.font=Font(name='Aptos',size=14,bold=True,color='173754')
for c in index[4]:c.fill=PatternFill('solid',fgColor='244C68');c.font=Font(name='Aptos',bold=True,color='FFFFFF')
for rowcells in index.iter_rows(min_row=5):
 for c in rowcells:c.font=Font(name='Aptos',size=10,color='223447');c.alignment=Alignment(vertical='top',wrap_text=True)
 rowcells[0].hyperlink=f"#'{rowcells[0].value}'!A1";rowcells[0].font=Font(name='Aptos',size=10,color='14749A',underline='single')
 if rowcells[0].row%2==0:
  for c in rowcells:c.fill=PatternFill('solid',fgColor='F1F6F8')
(R/'appendix'/'supplementary_tables.tex').write_text('\n'.join(tex).rstrip()+'\n')
wb.save(R/'appendix'/'supplementary_tables.xlsx')
print('Wrote appendix with',len([x for x in tex if x.startswith('\\begin{longtable}')]),'tables and',len(wb.sheetnames),'workbook sheets')
