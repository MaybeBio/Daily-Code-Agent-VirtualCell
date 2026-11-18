"""Stage 1 node: turn counts into ranked gene tokens."""

from pathlib import Path
from typing import Any, Dict

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)


class RankTokenizerNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="zfbrain_rank_tokenizer",
        stage="analysis",
        tool="PyTorch",
        model_source="https://github.com/Kays3/geneformer-vs-brainiac/tree/main/zebrafish-brain-fm",
        description="Encode each cell as its genes ranked from highest to lowest scaled expression. Counts are scaled to 10,000 per cell and divided by the gene's median non-zero value among the training cells; the scaling uses training cells only.",
        parameters={
            "seq_len": ParameterDefinition(default_value=2048, description="Most genes kept per cell. Shorter cells are padded with -1."),
            "min_cells": ParameterDefinition(default_value=3, description="A gene enters the vocabulary if at least this many training cells express it."),
            "out_path": ParameterDefinition(default_value="data/derived/atlas_tokens.pt", description="File to write."),
        },
        inputs={
            "data_path": PortDefinition(type=PortType.STR, description="AnnData file with raw counts and obs columns 'split' and 'celltype'."),
        },
        outputs={
            "tokens_path": PortDefinition(type=PortType.STR, description="File with 'tokens' (cells x seq_len, -1 is padding), 'is_heldout', 'cell_type' codes, 'cell_type_names', 'vocab_size', 'gene_ids'."),
            "vocab_size": PortDefinition(type=PortType.INT, description="Genes in the vocabulary."),
            "median_genes_per_cell": PortDefinition(type=PortType.FLOAT, description="Median number of gene tokens per cell, before padding."),
        },
        methods={
            "tokenize": MethodDefinition(description="Scale, rank and save.", inputs=["data_path"], outputs=["tokens_path", "vocab_size", "median_genes_per_cell"]),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("tokenize", self.tokenize, method_key="tokenize")

    def tokenize(self, data_path) -> Dict[str, Any]:
        p = self._parameters
        seq_len = int(p["seq_len"])
        adata = ad.read_h5ad(data_path)
        counts = sp.csr_matrix(adata.X, dtype=np.float64)
        totals = np.asarray(counts.sum(axis=1)).ravel()
        totals[totals == 0] = 1.0
        scaled = (sp.diags(1e4 / totals) @ counts).tocsr()
        is_train = (adata.obs["split"] == "train").to_numpy()

        by_gene = scaled[is_train].tocsc()
        medians = np.zeros(by_gene.shape[1])
        for gene in np.flatnonzero(np.diff(by_gene.indptr) >= int(p["min_cells"])):
            medians[gene] = np.median(by_gene.data[by_gene.indptr[gene]:by_gene.indptr[gene + 1]])
        in_vocab = medians > 0
        token_of_gene = np.where(in_vocab, np.cumsum(in_vocab) - 1, -1)

        tokens = np.full((scaled.shape[0], seq_len), -1, dtype=np.int32)
        lengths = np.zeros(scaled.shape[0], dtype=np.int64)
        for cell in range(scaled.shape[0]):
            span = slice(scaled.indptr[cell], scaled.indptr[cell + 1])
            genes, values = scaled.indices[span], scaled.data[span]
            keep = in_vocab[genes] & (values > 0)
            genes, values = genes[keep], values[keep] / medians[genes[keep]]
            order = np.argsort(-values, kind="stable")[:seq_len]
            tokens[cell, :len(order)] = token_of_gene[genes[order]]
            lengths[cell] = len(order)

        names = sorted(adata.obs["celltype"].unique())
        code = {name: i for i, name in enumerate(names)}
        out = Path(p["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "tokens": torch.from_numpy(tokens),
            "is_heldout": torch.from_numpy(~is_train),
            "cell_type": torch.tensor([code[c] for c in adata.obs["celltype"]], dtype=torch.long),
            "cell_type_names": names,
            "vocab_size": int(in_vocab.sum()),
            "gene_ids": [g for g, k in zip(adata.var_names, in_vocab) if k],
        }, out)
        return {"tokens_path": str(out), "vocab_size": int(in_vocab.sum()), "median_genes_per_cell": float(np.median(lengths))}
