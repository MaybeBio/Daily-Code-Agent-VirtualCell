"""Standard foundation adapter: prepare_inputs → load_pretrained → encode_batch.

BaseModel owns downstream evaluation; this class owns extraction and cache policy.
Native preprocessing and pooling remain explicit in each model implementation.
"""
import json
from pathlib import Path

import numpy as np
import torch

from bio.io.embeddings import SCHEMA, atomic_pickle, fingerprint, validate_result, expression_digest
from bio.io.paths import output_path
from .base import BaseModel


class FoundationAdapter(BaseModel):
    def setup_task_heads(self, dim):
        from torch import optim
        from .heads import Decoder, PerturbDecoder, Pooler_classify
        self.classify = Pooler_classify(dim, self.cell_type_num).to(self.device)
        self.optimizer_classify = optim.Adam(self.classify.parameters(), lr=1e-3)
        self.restruct = Decoder(dim, self.cell_dim_num).to(self.device)
        self.optimizer_restruct = optim.Adam(self.restruct.parameters(), lr=1e-3)
        if self.task == 'perturbation':
            self.pert_decoder = PerturbDecoder(emb_dim=dim, pert_num=self.pert_num,
                                              gene_num=self.cell_dim_num, pert_emb_dim=128).to(self.device)
            if getattr(self, 'condition_gene_index', None) is not None:
                self.pert_decoder.bind_genes(self.condition_gene_index)
            self.optimizer_pert = optim.Adam(self.pert_decoder.parameters(), lr=1e-3)
    def load_pretrained(self, model_cfg):
        raise NotImplementedError

    def prepare_inputs(self, data_cfg, train=True):
        raise NotImplementedError

    def encode_batch(self, batch):
        raise NotImplementedError

    # Preserve the downstream framework's entry points.
    def build_network(self, model_cfg):
        return self.load_pretrained(model_cfg)

    def get_loader(self, data_cfg, train=True):
        return self.prepare_inputs(data_cfg, train=True)

    def forward(self, batch):
        return self.encode_batch(batch)

    def get_embedding_dir_and_filename(self):
        if not hasattr(self, '_embedding_fingerprint'):
            self._embedding_fingerprint, self._embedding_provenance = fingerprint(self.cfgs)
        directory = output_path(self.cfgs, 'embeddings', f'{self.dataset_name}_{self.model_name}',
                                'v2', self._embedding_fingerprint)
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory), str(directory / 'all.pkl')

    def read_model_input(self):
        """Read full counts independently of the evaluation target's HVG/log data."""
        import anndata as ad
        from bio.data.foundation import prepare_input
        directory, _ = self.get_embedding_dir_and_filename()
        path = Path(directory) / 'foundation_input.h5ad'
        prepare_input(self.cfgs, self.adata_all, path)
        data = ad.read_h5ad(path)
        # Labels remain evaluation metadata, never arguments to pretrained encoders.
        data.obs = self.adata_all.obs.copy()
        data.obs['cell_id'] = np.arange(data.n_obs)
        data.var['gene_name'] = data.var_names
        data.var['gene_ids'] = data.var_names
        return data

    def evaluation_expression(self):
        x = self.adata_all.X
        if getattr(self, '_expression_source', None) is not x:
            self._expression_source = x
            self._expression_values = (x.toarray().astype(np.float32, copy=False)
                                       if hasattr(x, 'toarray') else np.asarray(x, dtype=np.float32))
        return self._expression_values

    def _compute_embeddings(self):
        backbone = getattr(self, 'backbone', getattr(self, 'model', None))
        if hasattr(backbone, 'eval'):
            backbone.eval()
        expected = self.evaluation_expression()
        embeddings, labels, identities = [], [], []
        def array(value):
            return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        with torch.inference_mode():
            for batch in self.all_loader:
                if isinstance(batch, dict) and 'cell_id' in batch:
                    ids = array(batch['cell_id']).reshape(-1)
                elif isinstance(batch, list) and batch and isinstance(batch[0], dict):
                    ids = np.asarray([int(item['cell_id']) for item in batch])
                else:
                    raise ValueError(f'{self.model_name}: inference batch has no source-row IDs')
                x, embedding, label = self.encode_batch(batch)
                x = array(x)
                if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
                    raise ValueError('Inference batch must preserve integer source-row IDs')
                if np.any(ids < 0) or np.any(ids >= len(expected)):
                    raise ValueError('Inference batch contains out-of-range source rows')
                if len(ids) and np.array_equal(ids, np.arange(ids[0], ids[0] + len(ids))):
                    target = (expected if ids[0] == 0 and len(ids) == len(expected)
                              else expected[ids[0]:ids[0] + len(ids)])
                else:
                    target = expected[ids]
                if x is not target and not np.array_equal(x, target):
                    raise ValueError('Encoder changed the evaluation expression targets')
                embeddings.append(array(embedding)); labels.append(array(label))
                identities.append(ids)
        embeddings, labels, identities = (parts[0] if len(parts) == 1 else np.concatenate(parts)
                                         for parts in (embeddings, labels, identities))
        raw = expected if np.array_equal(identities, np.arange(len(expected))) else expected[identities]
        return raw, embeddings, labels, identities

    def save_embeddings(self, embeddings, labels, raw, cell_ids):
        raw, embeddings, labels, cell_ids = validate_result(raw, embeddings, labels, cell_ids, self.adata_all.n_obs)
        _, expected_labels = np.unique(self.adata_all.obs['celltype'].astype(str), return_inverse=True)
        if not np.array_equal(labels, expected_labels):
            raise ValueError(f'{self.model_name}: model output labels are not aligned to source cells')
        if raw.shape != self.adata_all.shape:
            raise ValueError('Downstream expression targets must keep the evaluation gene space')
        # Evaluation counts already live in the signed input dataset. Verify
        # identity, then avoid writing a multi-GB copy for every model and seed.
        for start in range(0, len(raw), 256):
            original = self.adata_all.X[start:start + 256]
            original = original.toarray() if hasattr(original, 'toarray') else original
            if not np.array_equal(raw[start:start + 256], np.asarray(original, dtype=np.float32)):
                raise ValueError('Encoder changed the evaluation expression targets')
        directory, path = self.get_embedding_dir_and_filename()
        atomic_pickle(path, dict(schema=SCHEMA, signature=self._embedding_fingerprint,
                                raw_shape=list(raw.shape), raw_sha256=expression_digest(raw),
                                cell_embs=embeddings, labels=labels, cell_ids=cell_ids))
        manifest = dict(schema=SCHEMA, signature=self._embedding_fingerprint, shape=list(embeddings.shape),
                        provenance=self._embedding_provenance)
        Path(directory, 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')

    def load_embeddings(self):
        import pickle
        _, filename = self.get_embedding_dir_and_filename()
        path = Path(filename)
        if not path.is_file():
            return None
        with path.open('rb') as f:
            data = pickle.load(f)
        if data.get('schema') != SCHEMA or data.get('signature') != self._embedding_fingerprint:
            return None
        if data.get('raw_shape') != list(self.adata_all.shape):
            raise ValueError('Cached target shape does not match the evaluation dataset')
        result = validate_result(self.evaluation_expression(), data['cell_embs'], data['labels'], data['cell_ids'], self.adata_all.n_obs)
        if expression_digest(result[0]) != data.get('raw_sha256'):
            raise ValueError('Cached expression targets do not match the signed evaluation dataset')
        _, expected_labels = np.unique(self.adata_all.obs['celltype'].astype(str), return_inverse=True)
        if result[0].shape != self.adata_all.shape or not np.array_equal(result[2], expected_labels):
            raise ValueError('Cached targets/labels do not match the evaluation dataset')
        return result

    def extract_cell_embeddings(self):
        result = self.load_embeddings()
        if result is None:
            raw, embeddings, labels, identities = self._compute_embeddings()
            raw, embeddings, labels, identities = validate_result(
                raw, embeddings, labels, identities, self.adata_all.n_obs)
            self.save_embeddings(embeddings, labels, raw, identities)
            result = raw, embeddings, labels, identities
        if hasattr(self, '_heads'):
            self._heads(result[1].shape[1])
        return result
