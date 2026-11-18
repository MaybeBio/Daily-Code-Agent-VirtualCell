"""Statistical baselines with the same extraction and cache contract as FMs.

These methods fit the supplied dataset; they do not load foundation checkpoints.
Fit and HVG use the evaluation train split only, then the encoder is applied to
every cell so clustering and correction still see a full embedding matrix.
"""
import numpy as np

from bio.models.adapter import FoundationAdapter

__all__ = ['PCA_model', 'UMAP', 'scVI']


class StatisticalAdapter(FoundationAdapter):
    def load_pretrained(self, model_cfg):
        # There is no pretrained checkpoint for these dataset-fitted baselines.
        self._head_dim = None

    def prepare_inputs(self, data_cfg, train=True):
        import scanpy as sc
        from bio.data.catalog import CURATED
        from bio.data.preprocess import apply_train_hvg, train_cell_mask
        self.adata_all = self.read_adata(data_cfg['dataset_name'])
        n_hvg = int(data_cfg.get('hvg', 3000) or 3000)
        split_by = self.cfgs.get('evaluation_cfg', {}).get('split_by', 'donor')
        perturbation = bool(CURATED.get(self.dataset_name, {}).get('perturbation'))
        perturbation_split = self.cfgs.get('evaluation_cfg', {}).get('perturbation_split', 'auto')
        # Encoder features may be restricted to HVGs; targets retain the full,
        # unchanged expression matrix used by every foundation adapter.
        encoder_data, hvg_report = apply_train_hvg(
            self.adata_all, n_top=n_hvg, seed=self.seed, split_by=split_by,
            perturbation=perturbation, perturbation_split=perturbation_split, task=self.task)
        mask, fit_info = train_cell_mask(
            self.adata_all, seed=self.seed, split_by=split_by,
            perturbation=perturbation, perturbation_split=perturbation_split, task=self.task)
        if int(mask.sum()) < 4:
            raise ValueError('Train split is too small to fit a statistical baseline')
        hvg_report.update(fit_split_mode=fit_info.get('mode'), n_fit_cells=int(mask.sum()))
        self.cell_dim_num = self.adata_all.n_vars
        self.cfgs.setdefault('evaluation_cfg', {}).update(hvg_report)
        work = encoder_data.copy()
        if np.asarray(work.X.max()).item() > 10:
            sc.pp.normalize_total(work, target_sum=1e4)
            sc.pp.log1p(work)
        x = work.X.toarray().astype(np.float32) if hasattr(work.X, 'toarray') else np.asarray(work.X, dtype=np.float32)
        x_train = x[mask]
        dim = int(self.cfgs['model_cfg'].get('embedding_dim', 256))
        if self.model_name == 'PCA_model':
            from sklearn.decomposition import PCA
            dim = min(dim, x_train.shape[0], x_train.shape[1])
            self.model = PCA(n_components=dim, random_state=self.seed)
            self.model.fit(x_train)
            embedding = self.model.transform(x)
        elif self.model_name == 'UMAP':
            import umap
            dim = min(dim, x_train.shape[1], max(2, int(mask.sum()) - 2))
            self.model = umap.UMAP(n_components=dim, random_state=self.seed,
                                   n_neighbors=min(15, int(mask.sum()) - 1), init='random')
            self.model.fit(x_train)
            embedding = self.model.transform(x)
        else:
            import scvi
            data = encoder_data.copy()
            scvi.settings.seed = self.seed
            train_ad = data[mask].copy()
            scvi.model.SCVI.setup_anndata(train_ad)
            dim = min(dim, max(2, int(mask.sum()) - 1))
            self.scvi_model = scvi.model.SCVI(train_ad, n_layers=2, n_latent=dim, gene_likelihood='nb')
            # Legacy scvi-tools derives accelerator/devices from use_gpu.
            # Passing both raises duplicate-keyword TypeError.
            import inspect
            if 'use_gpu' in inspect.signature(self.scvi_model.train).parameters:
                device_options = {'use_gpu': self.device == 'cuda'}
            else:
                device_options = dict(accelerator='gpu' if self.device == 'cuda' else 'cpu', devices=1)
            self.scvi_model.train(max_epochs=self.cfgs['model_cfg'].get('max_epochs', 400),
                                  **device_options, logger=False, enable_checkpointing=False)
            query = data.copy()
            scvi.model.SCVI.prepare_query_anndata(query, self.scvi_model)
            embedding = self.scvi_model.get_latent_representation(query)
        _, labels = np.unique(self.adata_all.obs['celltype'].astype(str), return_inverse=True)
        self._heads(embedding.shape[1])
        self.all_loader = [dict(embedding=embedding, original=self.evaluation_expression(), label=labels,
                                cell_id=np.arange(len(x)))]

    def _heads(self, dim):
        if self._head_dim != dim:
            self.setup_task_heads(dim)
            self._head_dim = dim

    def encode_batch(self, batch):
        return batch['original'], batch['embedding'], batch['label']

    def forward_annotation(self, cell_emb):
        return self.classify(cell_emb)

    def forward_reconstruction(self, cell_emb):
        return self.restruct(cell_emb)

    def forward_perturbation(self, ctrl_emb, pert_id, ctrl_expr):
        return (ctrl_expr + self.pert_decoder(ctrl_emb, pert_id)).clamp_min(0)

    def full_finetuning_annotation(self, cfgs):
        raise NotImplementedError('Statistical baselines have no pretrained backbone to finetune')


class PCA_model(StatisticalAdapter):
    pass


class UMAP(StatisticalAdapter):
    pass


class scVI(StatisticalAdapter):
    pass
