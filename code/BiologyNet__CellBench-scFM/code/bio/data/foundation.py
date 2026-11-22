"""Data and identity contract shared by the six isolated foundation-model runners."""

from bio.io.paths import PROJECT_ROOT, project_path

MODELS = ('UCE', 'scPRINT2', 'TranscriptFormer', 'CellFM', 'GeneCompass', 'scLong')
# These released encoders have native mouse vocabularies. Human-only encoders
# use the user-approved one-to-one ortholog projection for the mouse panel.
NATIVE_MOUSE_MODELS = frozenset({'UCE', 'scPRINT2', 'Nicheformer', 'GeneCompass'})
from bio.data.catalog import CURATED
DATASETS = {name: spec['filename'] for name, spec in CURATED.items()}

REQUIRED = {
    'UCE': ['model_files/4layer_model.torch', 'model_files/all_tokens.torch',
            'model_files/species_offsets.pkl', 'model_files/species_chrom.csv',
            'model_files/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
            'model_files/new_species_protein_embeddings.csv'],
    'scPRINT2': ['small-v2.ckpt'],
    'TranscriptFormer': ['tf_sapiens/config.json', 'tf_sapiens/model_weights.pt',
                         'tf_sapiens/vocabs/homo_sapiens_gene.h5', 'tf_sapiens/vocabs/assay_vocab.json'],
    'CellFM': ['base_weight.ckpt', 'gene_info.csv', 'updated_hgcn.tsv'],
    # Released checkpoints contain the four prior matrices and homologous index.
    # External prior pickles are needed for rebuilding/training, not frozen inference.
    'GeneCompass': ['GeneCompass_Small/config.json', 'GeneCompass_Small/pytorch_model.bin',
                    'scdata/dict/human_gene_median_after_filter.pickle',
                    'prior_knowledge/human_mouse_tokens.pickle',
                    'prior_knowledge/h&m_token1000W.pickle'],
    'scLong': ['selected_genes_27k.txt', 'selected_gene2vec_27k.npy',
               'human_ens_gene2go_graph.csv', 'gocont_4096_48m_pretrain_1b_mix.pkl',
               'gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth'],
}


def check_resources(model, weights=None):
    weights = project_path(weights or f'weights/{model}')
    return [name for name in REQUIRED[model]
            if not (weights / name).is_file() or (weights / name).stat().st_size == 0]


def input_path(cfgs):
    data = cfgs['data_cfg']
    if data.get('foundation_input_path'):
        return project_path(data['foundation_input_path'])
    name = data['dataset_name']
    if name not in DATASETS:
        raise ValueError(f'Configure data_cfg.foundation_input_path for {name}')
    return project_path(data.get('root_path', 'datasets')) / DATASETS[name]


def align_embeddings(embeddings, observed, expected):
    import numpy as np
    observed, expected = list(map(str, observed)), list(map(str, expected))
    if len(set(observed)) != len(observed) or len(set(expected)) != len(expected):
        raise ValueError('Duplicate cell identifiers in embedding input/output')
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != len(observed) or x.shape[1] == 0:
        raise ValueError('Embedding matrix does not match its cell identifiers')
    if set(observed) != set(expected):
        raise ValueError('Backend dropped or added cells; refuse misaligned benchmark results')
    order = {name: i for i, name in enumerate(observed)}
    x = x[[order[name] for name in expected]]
    if not np.isfinite(x).all():
        raise ValueError('Backend produced nonfinite embeddings')
    return x


def prepare_input(cfgs, evaluation, destination):
    """Load full genes before benchmark HVG/log transforms; never pass reference labels."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    source = ad.read_h5ad(input_path(cfgs))
    dataset_name = cfgs['data_cfg']['dataset_name']
    if dataset_name in ('uc_epi', 'uc_imm'):
        sidecar = input_path(cfgs).parent / f'{dataset_name}_gene.tsv'
        if sidecar.is_file():
            genes = pd.read_csv(sidecar, sep='\t', header=None, dtype=str).iloc[:, 0].to_numpy()
            if len(genes) == source.n_vars:
                source.var['gene_name'] = genes
        if 'gene_name' not in source.var:
            source.var['gene_name'] = source.var_names.astype(str)
    source_names, eval_names = source.obs_names.astype(str), evaluation.obs_names.astype(str)
    if source.n_obs != evaluation.n_obs or not np.array_equal(source_names, eval_names):
        if source_names.is_unique and set(map(str, eval_names)).issubset(set(map(str, source_names))):
            source = source[eval_names].copy()
        elif not (np.array_equal(source_names, np.arange(source.n_obs).astype(str))
                  and cfgs['data_cfg']['dataset_name'] in ('uc_epi', 'uc_imm')):
            raise ValueError('Foundation input cell order cannot be matched to evaluation data')
        if source.n_obs != evaluation.n_obs:
            raise ValueError('Foundation input and evaluation dataset have different cell counts')
    config = cfgs['model_cfg']
    layer = config.get('expression_layer', 'auto')
    if layer == 'raw' or (layer == 'auto' and 'counts' not in source.layers and source.raw is not None):
        if source.raw is None:
            raise ValueError('expression_layer=raw but dataset has no .raw')
        x, var = source.raw.X, source.raw.var.copy()
    elif layer not in ('auto', 'X'):
        x, var = source.layers[layer], source.var.copy()
    elif layer == 'auto' and 'counts' in source.layers:
        x, var = source.layers['counts'], source.var.copy()
    else:
        x, var = source.X, source.var.copy()
    # The inner-join heart assembly dropped var metadata. Recover identifiers
    # by exact Ensembl index from its prepared 10x source, without changing X.
    if dataset_name == 'mouse_heart' and not any(k in var for k in ('gene_name', 'gene_symbols', 'feature_name')):
        reference_path = input_path(cfgs).parent / 'mouse_heart_10x.h5ad'
        if reference_path.is_file():
            reference = ad.read_h5ad(reference_path, backed='r')
            try:
                aligned = reference.var.reindex(var.index)
                for key in ('gene_name', 'gene_symbols', 'feature_name', 'ensembl_id'):
                    if key in aligned:
                        var[key] = aligned[key].to_numpy()
            finally:
                reference.file.close()
    x = sp.csr_matrix(x, dtype=np.float32)
    values = x.data
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Expression matrix contains negative or nonfinite values')
    counts = np.allclose(values, np.rint(values), atol=1e-5, rtol=0)
    if not counts and config['model'] != 'scLong':
        raise ValueError('Model requires raw integer counts. Set expression_layer to a genuine counts layer '
                         'or foundation_input_path to the original count h5ad; normalized data cannot be rounded.')
    if config['model'] == 'scLong' and not counts and not config.get('input_is_log1p', False):
        raise ValueError('For normalized scLong input explicitly set input_is_log1p: true')
    # Prefer supplied stable IDs and symbols, with a pinned HGNC fallback from CellFM.
    def column(candidates, fallback):
        for key in candidates:
            if key and key in var:
                return var[key].astype(object).fillna('').astype(str).to_numpy()
        return np.asarray(fallback, dtype=str)
    symbols = column([config.get('symbol_column'), 'gene_symbols', 'gene_name', 'feature_name', 'gene_symbol', 'Gene Symbol', 'Gene'], var.index)
    ensembl = column([config.get('ensembl_column'), 'ensembl_id', 'gene_ids', 'gene_id', 'ensembl'], var.index)
    ensembl = np.asarray([g.split('.')[0] if g.startswith('ENS') else '' for g in ensembl])
    mouse = dataset_name.startswith('mouse_') or any(g.startswith('ENSMUSG') for g in ensembl)
    projected = mouse and config['model'] not in NATIVE_MOUSE_MODELS
    projection = dict(source_organism='NCBITaxon:10090' if mouse else 'NCBITaxon:9606',
                      mode='human_one_to_one_orthologs' if projected else 'native',
                      genes_before=int(len(ensembl)))
    if projected:
        from bio.data.orthologs import mouse_to_human_ensembl
        orthologs = mouse_to_human_ensembl()
        ensembl = np.asarray([orthologs.get(g, '') for g in ensembl])
        # Symbols must come from the mapped human IDs, never uppercased mouse
        # symbols or guessed orthologs. Unmapped genes are dropped below.
        symbols = np.asarray([''] * len(ensembl))
        projection['mapped_genes'] = int(sum(bool(g) for g in ensembl))
        projection['mapped_fraction'] = projection['mapped_genes'] / max(1, len(ensembl))
    table = PROJECT_ROOT / 'weights/CellFM/updated_hgcn.tsv'
    if table.is_file():
        hgnc = pd.read_csv(table, sep='\t', dtype=str).fillna('')
        hgnc = hgnc[hgnc['Status'] == 'Approved']
        mapping = dict(zip(hgnc['Approved symbol'], hgnc['Ensembl gene ID']))
        inverse = {v: k for k, v in mapping.items() if v}
        ensembl = np.asarray([e or mapping.get(s, '') for e, s in zip(ensembl, symbols)])
        symbols = np.asarray([inverse.get(e, s) if not s or s.startswith('ENS') else s for e, s in zip(ensembl, symbols)])
    use_ensembl = config['model'] in ('TranscriptFormer', 'GeneCompass', 'scLong', 'scPRINT2', 'CellPLM', 'Geneformer', 'LangCell', 'Nicheformer')
    genes = ensembl if use_ensembl else symbols
    keep = np.array([bool(g) and g.lower() not in ('nan', 'none') for g in genes])
    if keep.sum() < 100:
        raise ValueError(f'Only {keep.sum()} genes have usable identifiers; configure the gene ID columns')
    x, genes = x[:, keep], genes[keep]
    # Merge duplicate identifiers by summing counts, without suffixing biological IDs.
    unique, inverse = np.unique(genes, return_inverse=True)
    transform = sp.csr_matrix((np.ones(len(genes)), (np.arange(len(genes)), inverse)), shape=(len(genes), len(unique)))
    x = (x @ transform).tocsr().astype(np.float32)
    if np.any(np.asarray(x.sum(axis=1)).ravel() <= 0):
        raise ValueError('One or more cells have no expression after gene ID mapping')
    if config['model'] == 'scLong' and counts:
        import scanpy as sc
        temp = ad.AnnData(x)
        sc.pp.normalize_total(temp, target_sum=1e4)
        sc.pp.log1p(temp)
        x = temp.X
    obs = pd.DataFrame(index=[f'cellbench_{i}' for i in range(source.n_obs)])
    obs['_cellbench_cell_id'] = obs.index.astype(str)
    obs['organism_ontology_term_id'] = 'NCBITaxon:10090' if mouse and not projected else config.get('organism', 'NCBITaxon:9606')
    if config['model'] == 'TranscriptFormer':
        assay_column = config.get('assay_column', 'assay')
        obs['assay'] = (source.obs[assay_column].astype(str).to_numpy()
                        if assay_column in source.obs else config.get('assay', 'unknown'))
    result = ad.AnnData(x, obs=obs, var=pd.DataFrame(index=unique))
    result.uns['cellbench_gene_projection'] = projection
    if use_ensembl:
        result.var['ensembl_id'] = result.var_names
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(destination, compression='lzf')
    return result.obs_names.to_numpy(dtype=str)
