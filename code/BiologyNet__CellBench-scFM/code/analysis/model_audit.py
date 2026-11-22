from pathlib import Path
import csv,json
import pandas as pd
R=Path(__file__).resolve().parents[1]
# Numbers labeled family-level are not automatically checkpoint-specific.
A={
'CellFM':('base_weight.ckpt','ERetNet','masked gene expression','~100M human cells (article family corpus)','human','~800M (article largest model; exact base checkpoint unverified)','2048','https://www.nature.com/articles/s41467-025-59926-5'),
'CellPLM':('20230926_85M.best.ckpt','OmicsFormer; Gaussian-mixture prior','masked cell/gene modeling','>9M scRNA-seq + >2M spatial cells (paper; checkpoint match not explicit)','human (paper)','82,402,543 state-dict elements incl. buffers; unique parameters unverified','not established','https://proceedings.iclr.cc/paper_files/paper/2024/file/173bcdbd820edbe43284e0941109ada9-Paper-Conference.pdf'),
'GeneCompass':('GeneCompass_Small/pytorch_model.bin','6-layer knowledge-informed BERT','masked modeling with biological priors','101,768,420 cells after QC; 126M collected (paper)','human and mouse','173,495,551 distinct checkpoint tensor elements incl. 155,314,176 prior buffer elements; parameter count unverified','2048','https://www.nature.com/articles/s41422-024-01034-y ; https://github.com/xCompass-AI/GeneCompass'),
'Geneformer':('gf-12L-30M-i2048','12-layer BERT masked language model','masked gene prediction','~30M human cell transcriptomes (Genecorpus-30M)','human','39,591,250 architecture parameters (local config, tied weights counted once)','2048','https://www.nature.com/articles/s41586-023-06139-9'),
'LangCell':('cell_bert/pytorch_model.bin + cell_proj.bin','Geneformer-initialized BERT cell encoder with projection','masked gene modeling and cell-text alignment','~27.5M human scRNA-cell/text pairs, scLibrary','human','39,302,656 cell_bert state-dict elements including buffers; projection separate','2048 tokenizer family; local exact limit verify','https://proceedings.mlr.press/v235/zhao24u.html'),
'Nicheformer':('model.safetensors revision 0ba4ba162f9ad3d6c8b713761954c31be2fa69ed','12-layer transformer','masked gene tokens','>110M cells human/mouse, SpatialCorpus-110M','human and mouse','49.3M (paper)','1500','https://www.nature.com/articles/s41592-025-02814-z'),
'Tabula':('pancreas.pth; sha256 ed6a3ef16d042d6deacb4bc044daf6d93c57ddb3182559fb8bbd0149d06a965c','tabular transformer FlashAttention','contrastive tabular self-supervision','15M profiles in article; pancreas.pth exact training subset unverified','human (paper; exact checkpoint provenance unverified)','13,225,713 state-dict elements incl. buffers; parameter count unverified','not established','https://papers.nips.cc/paper_files/paper/2025/hash/95d590995a8722259c61e094b62b25ac-Abstract-Conference.html'),
'TranscriptFormer':('tf_sapiens/model_weights.pt','12-layer autoregressive transformer with ESM gene embeddings','gene identity and count generation','57M human cells (TF-Sapiens)','human','~429M total: ~368M trainable + ~61M frozen in original model; 429,870,637 checkpoint tensor elements','1023 benchmark; 2047 released model','https://github.com/czi-ai/transcriptformer'),
'UCE':('model_files/4layer_model.torch','4-layer transformer with ESM protein embeddings','masked expression/cell modeling','>36M cells; >300 datasets, eight species (UCE publication)','human, mouse, zebrafish, mouse lemur, crab-eating macaque, rhesus macaque, frog, pig','850,867,457 checkpoint tensor elements incl. buffers; 4-layer parameter count unverified','1536 benchmark configuration','https://doi.org/10.1038/s41586-026-10689-z'),
'scFoundation':('models.ckpt: cell','asymmetric masked autoencoder','read-depth-aware masked expression reconstruction','>50M human cells','human','~100M (paper); 119,354,430 cell state-dict tensor elements incl. buffers','19264-gene input panel','https://www.nature.com/articles/s41592-024-02305-7'),
'scGPT':('human/best_model.pt','generative transformer','masked expression prediction','33M normal human cells (whole-human checkpoint)','human','51,330,049 state-dict elements incl. buffers; exact unique parameters unverified','1200 benchmark','https://github.com/bowang-lab/scGPT'),
'scLong':('gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth','dual Performer and gene-ontology graph','masked value reconstruction','48,024,242 unique cells before train/validation split','human','~1B (paper); 1,006,869,568 model-state tensor elements incl. buffers','27875 benchmark configuration; paper 27874 genes','https://www.nature.com/articles/s41467-026-69102-y'),
'scPRINT2':('small-v2.ckpt','scPRINT-2 transformer','expression reconstruction and metadata tasks','>350M cells / 16 organisms (family-level official documentation; exact checkpoint corpus verify)','16 organisms including human and mouse (family-level)','187,755,572 state-dict tensor elements incl. buffers; parameter count unverified','2000 benchmark configuration','https://github.com/cantinilab/scPRINT-2'),
}
x=pd.read_csv(R/'data/model_attributes_working.csv').fillna('未知')
for i,row in x.iterrows():
 if row.model in A:
  ck,arch,obj,size,sp,param,ctx,url=A[row.model];x.loc[i,['checkpoint','architecture','training_objective','pretraining_size_unit','pretraining_species','total_parameters','context_length','source','status']]=[ck,arch,obj,size,sp,param,ctx,url,'已核验公开资料及本地配置；各字段精度见具体说明'];x.loc[i,'updated_parameters']='预训练骨干 0；下游任务头参数随任务、嵌入维度和输出维度变化，逐运行未统计'
 else:x.loc[i,'updated_parameters']='基线在数据集训练部分拟合；无预训练骨干，参数随数据集变化'
countfile=R/'data/checkpoint_parameter_counts.csv'
if countfile.exists():
 for _,v in pd.read_csv(countfile).iterrows():
  x.loc[x.model==v.model,'total_parameters']=f'{int(v.total_parameters):,} locally audited instantiated-model parameters; see data/checkpoint_parameter_counts.csv for scope'
x.to_csv(R/'data/model_attributes_working.csv',index=False)
records=[]
for path in sorted((R/'data/provenance').glob('*.provenance.json')):
 d=json.loads(path.read_text());records.append({'dataset':d.get('id'),'cells':d.get('cells'),'genes':d.get('genes',d.get('shared_genes')),'source':d.get('source','merged mouse heart; see component provenance'),'url':d.get('url',''),'source_cells':d.get('source_cells',''),'selection':d.get('selection',''),'provenance':str(path.relative_to(R))})
pd.DataFrame(records).to_csv(R/'data/dataset_provenance.csv',index=False)
models=list(A);panels=json.loads((R/'data/experiment_panels.json').read_text()); rows=[]
for model in models:
 for panel,part in panels.items():
  for ds in part['datasets']:
   status='unresolved: no checkpoint-specific training sample list matched'; evidence='No study/cell identity match established; source and sampling alone do not establish inclusion.'
   if model=='scFoundation' and ds in ['heart','ms']:
    status='confirmed study-level overlap; shared sample/donor labels, exact cells unverified';evidence='scFoundation Supplementary Data 1 and 2: HCA-HeartSingleCellsAndNucleiSeq and HCA-Oligodendrocyte_MS; source-study and labels matched.'
   if model=='scPRINT2' and ds in ['hlca','heart','kidney','ms','covid_lung','tnbc','mouse_marrow','mouse_lung','mouse_kidney','mouse_heart']:
    status='shared repository; likely overlap according to developer, unconfirmed for exact cells';evidence='Developer states data present in CELLxGENE or scBaseCount were likely used, with quality and validation/test exclusions.'
   if model=='TranscriptFormer' and panel in ['organ','disease']:
    status='shared repository for several studies; exact inclusion unconfirmed';evidence='TF-Sapiens human training sources include 644 CELLxGENE datasets; complete sample identity match not established.'
   if model=='scGPT' and ds in ['hlca','heart','kidney','ms','covid_lung','tnbc']:
    status='shared repository; exact census release and inclusion unconfirmed';evidence='whole-human model pretraining used CELLxGENE census 2023-05-15; source study membership and actual cells not established.'
   if model=='TranscriptFormer' and panel=='species':status='species not in TF-Sapiens pretraining; tissue/cell-level training overlap not expected';evidence='Official TF-Sapiens trained on human only; mouse input uses project-specific species policy.'
   rows.append({'model':model,'panel':panel,'dataset':ds,'status':status,'evidence':evidence})
pd.DataFrame(rows).to_csv(R/'data/pretraining_overlap_audit.csv',index=False)
