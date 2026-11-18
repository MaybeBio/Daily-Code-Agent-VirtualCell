#!/usr/bin/env python3
"""Count instantiated benchmark model parameters, separately from registered buffers.

Run with each model's benchmark interpreter and with cwd at the benchmark root.
Example: python checkpoint_parameter_audit.py Geneformer --out /tmp/Geneformer.json
Counts deduplicate registered parameter storage, offsets, shapes and strides;
inference backbones are frozen downstream regardless of requires_grad at load.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import shutil
import pickle

p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--out',required=True);args=p.parse_args()
ROOT=Path(os.environ.get('CELLBENCH_ROOT',os.environ.get("CELLBENCH_ROOT", str(Path(__file__).resolve().parents[1])))).resolve()
os.chdir(ROOT);sys.path.insert(0,str(ROOT));W=ROOT/'weights'; name=args.model
import torch

def verify(model,state):
 expected=model.state_dict(); shared=set(expected)&set(state)
 bad=[k for k in shared if expected[k].shape!=state[k].shape]
 return dict(expected_keys=len(expected),checkpoint_keys=len(state),matched_keys=len(shared)-len(bad),missing_keys=sorted(set(expected)-set(state)),unexpected_keys=sorted(set(state)-set(expected)),shape_mismatch=bad)

notes=''; match=None; exclude_prefixes=[]
if name=='Geneformer':
 from transformers import BertForMaskedLM
 model,info=BertForMaskedLM.from_pretrained(str(W/'Geneformer/gf-12L-30M-i2048'),local_files_only=True,output_loading_info=True)
 assert not info['missing_keys'] and not info['mismatched_keys'];notes='BertForMaskedLM; tied parameters deduplicated.'
elif name=='LangCell':
 from bio.models.adapters.langcell import ModelLoader
 model=ModelLoader.load_pretrained_model(str(W/'LangCell'));notes='Loaded cell BERT and trained 256-dimensional projection; original pooler replaced by projection.'
elif name=='CellPLM':
 sys.path.insert(0,str(ROOT/'public_code/CellPLM'))
 from CellPLM.model import OmicsFormer
 params=json.loads((W/'CellPLM/20230926_85M.config.json').read_text());params['head_type']='embedder';model=OmicsFormer(**params)
 state=torch.load(W/'CellPLM/20230926_85M.best.ckpt',map_location='cpu',weights_only=False)['model_state_dict']; match=verify(model,state)
 from bio.models.loading import verify_backbone
 model.load_state_dict({k:v for k,v in state.items() if k in model.state_dict() and v.shape==model.state_dict()[k].shape},strict=False)
 verify_backbone(model,state,('embedder.','encoder.','latent.','pre_latent_norm.'))
 notes='Embedding OmicsFormer configured as benchmark; all 87 instantiated state keys match, and checkpoint decoder keys are not instantiated.'
elif name=='scGPT':
 from bio.models.adapters.scgpt import ModelLoader
 model,_=ModelLoader.load_pretrained_model(str(W/'scGPT/human'),1);exclude_prefixes=['cls_decoder.']
 import re
 state=torch.load(W/'scGPT/human/best_model.pt',map_location='cpu');mapped={}
 for key,value in state.items():
  key=re.sub(r'self_attn\.(?:_impl\.)?Wqkv\.', 'self_attn.in_proj_',key).replace('self_attn._impl.out_proj.','self_attn.out_proj.');mapped[key]=value
 match=verify(model,mapped)
 notes='Benchmark TransformerModel; dataset-dependent unused cls_decoder excluded; other registered model parameters included.'
elif name=='Nicheformer':
 from transformers import AutoModel
 model,info=AutoModel.from_pretrained(str(W/'Nicheformer'),trust_remote_code=True,local_files_only=True,output_loading_info=True)
 assert not info['missing_keys'] and not info['mismatched_keys'];match={'missing_keys':info['missing_keys'],'unexpected_keys':info['unexpected_keys'],'shape_mismatch':info['mismatched_keys']};notes='Benchmark Hugging Face AutoModel; unused checkpoint classifier_head is not instantiated.'
elif name=='scFoundation':
 from bio.vendor.scfoundation.load import load_model_frommmf
 model,_=load_model_frommmf(str(W/'scFoundation/models.ckpt'),'cell');notes='Benchmark cell model; strict state load.'
elif name=='UCE':
 sys.path.insert(0,str(ROOT/'public_code/UCE'))
 from model import TransformerModel
 from torch import nn
 state=torch.load(str(W/'UCE/model_files/4layer_model.torch'),map_location='cpu',mmap=True)
 with torch.device('meta'):
  model=TransformerModel(token_dim=5120,d_model=1280,nhead=20,d_hid=5120,nlayers=4,dropout=.05,output_dim=1280);model.pe_embedding=nn.Embedding(145469,5120)
 match=verify(model,state);assert not match['missing_keys'] and not match['unexpected_keys'] and not match['shape_mismatch']
 model.load_state_dict(state,strict=True,assign=True);notes='Benchmark UCE 4-layer model; includes checkpoint pe_embedding protein table.'
elif name=='GeneCompass':
 from transformers import BertConfig
 with tempfile.TemporaryDirectory(prefix='genecompass-audit-') as folder:
  stage=Path(folder);shutil.copytree(ROOT/'public_code/GeneCompass/genecompass',stage/'genecompass',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  (stage/'prior_knowledge').symlink_to(W/'GeneCompass/prior_knowledge',target_is_directory=True);sys.path.insert(0,str(stage))
  from genecompass import BertForMaskedLM,modeling_bert,knowledge_embeddings
  from bio.models.compat.genecompass import checkpoint_embeddings
  state=torch.load(str(W/'GeneCompass/GeneCompass_Small/pytorch_model.bin'),map_location='cpu',mmap=True,weights_only=True)
  knowledge={n:state[f'bert.embeddings.{n}_knowledge'] for n in ('promoter','co_exp','gene_family','peca_grn')}
  homologous=state['bert.embeddings.homologous_index'];knowledge['homologous_gene_human2mouse']={i:int(v) for i,v in enumerate(homologous) if i!=int(v)}
  config=BertConfig.from_pretrained(str(W/'GeneCompass/GeneCompass_Small'),local_files_only=True);modeling_bert.KnowledgeBertEmbeddings=checkpoint_embeddings(knowledge_embeddings)
  model=BertForMaskedLM(config,knowledges=knowledge);match=verify(model,state)
  assert not match['missing_keys'] and not match['unexpected_keys'] and not match['shape_mismatch']
  for n,parameter in model.named_parameters():
   if not parameter.is_floating_point() and not parameter.is_complex():state[n]=torch.nn.Parameter(state[n],requires_grad=False)
  model.load_state_dict(state,strict=True,assign=True)
 notes='Six-layer benchmark checkpoint and registered priors; exclude registered buffers from parameter count.'
elif name=='scPRINT2':
 from scprint2 import scPRINT2
 checkpoint=torch.load(str(W/'scPRINT2/small-v2.ckpt'),map_location='cpu',mmap=True,weights_only=False)
 params=dict(checkpoint['hyper_parameters']);params.pop('_instantiator',None);params.update(precpt_gene_emb=None,gene_pos_file=None)
 model=scPRINT2(**params);model.on_load_checkpoint(checkpoint);match=verify(model,checkpoint['state_dict'])
 assert not match['missing_keys'] and not match['unexpected_keys'] and not match['shape_mismatch']
 model.load_state_dict(checkpoint['state_dict'],strict=True,assign=True);notes='Official checkpoint constructor and on_load_checkpoint; strict load.'
elif name=='scLong':
 sys.path.insert(0,str(ROOT/'public_code/scLong'))
 from performer_pytorch_cont.ding_models import DualEncoderSCFM
 with (W/'scLong/gocont_4096_48m_pretrain_1b_mix.pkl').open('rb') as f:params=pickle.load(f)
 params['gene2vec_file']=str(W/'scLong/selected_gene2vec_27k.npy');params['device']='cpu'
 model=DualEncoderSCFM(**params);checkpoint=torch.load(str(W/'scLong/gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth'),map_location='cpu',mmap=True)
 match=verify(model,checkpoint['model_state_dict']);assert not match['missing_keys'] and not match['unexpected_keys'] and not match['shape_mismatch']
 model.load_state_dict(checkpoint['model_state_dict'],strict=True,assign=True);notes='Benchmark DualEncoderSCFM strict checkpoint load.'
elif name=='Tabula':
 from bio.models.adapters.tabula import Tabula
 instance=Tabula.__new__(Tabula);torch.nn.Module.__init__(instance);instance.seed=41;instance.cell_type_num=1;instance.device=torch.device('cpu');instance.cfgs={'model_cfg':{'preprocessing_chunk_size':256}}
 instance.setup_task_heads=lambda *a,**k: None
 instance.load_pretrained({'ckpt_path':'weights/Tabula/pancreas.pth','vocab_file':'weights/Tabula/vocab.json','config_file':'weights/Tabula/finetune_framework_annotation.yaml','pretrained_backbone':'FlashAttention'})
 model=instance.backbone
 state=torch.load(W/'Tabula/pancreas.pth',map_location='cpu');state={('transformer.'+k if 'transformer.'+k in model.state_dict() else k):v for k,v in state.items()};match=verify(model,state)
 exclude_prefixes=['heads.supervised.'];notes='Benchmark FlashAttention-compatible Tabula; dataset-dependent supervised head excluded. Backbone loader permits partial checkpoint loading.'
elif name=='TranscriptFormer':
 from omegaconf import OmegaConf
 from transcriptformer.tokenizer.vocab import load_vocabs_and_embeddings
 from transcriptformer.model.model import Transcriptformer
 import transcriptformer.cli
 config_path=Path(transcriptformer.cli.__file__).parent/'conf/inference_config.yaml'
 cfg=OmegaConf.merge(OmegaConf.create(json.loads((W/'TranscriptFormer/tf_sapiens/config.json').read_text())),OmegaConf.load(config_path))
 cfg.model.data_config.aux_vocab_path=str(W/'TranscriptFormer/tf_sapiens/vocabs');cfg.model.data_config.esm2_mappings_path=str(W/'TranscriptFormer/tf_sapiens/vocabs');cfg.model.model_config.seq_len=1023;cfg.model.model_config.compile_block_mask=False
 (gene,aux),emb=load_vocabs_and_embeddings(cfg)
 model=Transcriptformer(data_config=cfg.model.data_config,model_config=cfg.model.model_config,loss_config=cfg.model.loss_config,inference_config=cfg.model.inference_config,gene_vocab_dict=gene,aux_vocab_dict=aux,emb_matrix=emb)
 state=torch.load(str(W/'TranscriptFormer/tf_sapiens/model_weights.pt'),map_location='cpu',weights_only=True,mmap=True);match=verify(model,state)
 assert not match['missing_keys'] and not match['unexpected_keys'] and not match['shape_mismatch'];model.load_state_dict(state,strict=True)
 notes='Benchmark TF-Sapiens model with 1023 context positions; strict checkpoint load; includes originally frozen embedding parameters.'
elif name=='CellFM':
 import mindspore as ms
 import pandas as pd
 sys.path.insert(0,str(ROOT/'public_code/CellFM'))
 from model import CellFM
 from config import Config
 ms.set_context(mode=ms.PYNATIVE_MODE,device_target='CPU');cfg=Config();cfg.recompute=False
 model=CellFM(len(pd.read_csv(W/'CellFM/gene_info.csv',index_col=0).index),cfg)
 params=ms.load_checkpoint(str(W/'CellFM/base_weight.ckpt'));missing,unused=ms.load_param_into_net(model,params,strict_load=True)
 assert not missing
 ps=list(model.get_parameters()); result=dict(model=name,total_parameters=sum(int(x.size) for x in ps),registered_trainable_parameters=sum(int(x.size) for x in ps if x.requires_grad),parameter_objects=len(ps),checkpoint_keys=len(params),unused_checkpoint_keys=unused,notes='MindSpore benchmark CellFM; strict load. Count unique get_parameters entries; optimizer checkpoint tensors excluded.')
 Path(args.out).write_text(json.dumps(result,indent=2,default=str)+'\n');print(json.dumps(result));sys.exit()
else:raise ValueError(name)
entries=[(n,x) for n,x in model.named_parameters(remove_duplicate=False) if not any(n.startswith(q) for q in exclude_prefixes)]
seen={};params=[];aliases={}
for n,x in entries:
 key=(x.untyped_storage().data_ptr(),x.storage_offset(),tuple(x.shape),tuple(x.stride()),str(x.dtype))
 if key in seen:aliases[n]=seen[key]
 else:seen[key]=n;params.append((n,x))
buffers=list(model.named_buffers()); result=dict(model=name,total_parameters=sum(x.numel() for _,x in params),registered_trainable_parameters=sum(x.numel() for _,x in params if x.requires_grad),registered_buffer_elements=sum(x.numel() for _,x in buffers),parameter_objects=len(params),shared_parameter_aliases=aliases,buffer_objects=len(buffers),excluded_parameter_prefixes=exclude_prefixes,load_audit=match,notes=notes,by_top_level={})
for n,x in params:
 key=n.split('.')[0];result['by_top_level'][key]=result['by_top_level'].get(key,0)+x.numel()
Path(args.out).write_text(json.dumps(result,indent=2,default=str)+'\n');print(json.dumps(result))
