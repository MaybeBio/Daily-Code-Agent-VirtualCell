"""Torch dataset, masking and collation.

Holds the one piece of state that is easiest to get wrong: everything fitted
on data -- HVG selection, the gene vocabulary, the rank tokenizer's medians,
label encoders -- is fitted on the **training split only** and then applied
to val/test.  Fitting on the pooled dataset is a quiet form of leakage that
inflates every downstream number.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from ..core.registry import SOURCES, SPLITTERS, TOKENIZERS
from ..core.types import CellBatch
from .preprocessing import (
    GeneVocabulary,
    ensure_obs_columns,
    is_raw_counts,
    normalize_counts,
    select_hvg,
)
from .splits import audit_leakage

log = logging.getLogger(__name__)

__all__ = ["CellDataset", "SingleCellDataModule", "collate_cells"]


class CellDataset(Dataset):
    """Tokenizes one cell at a time.

    ``counts`` stays sparse; a row is densified only when requested, which is
    what keeps memory flat as the gene panel grows.
    """

    def __init__(
        self,
        counts: sp.spmatrix,
        normalized: sp.spmatrix,
        obs: pd.DataFrame,
        tokenizer,
        vocab: GeneVocabulary,
        label_maps: Dict[str, Dict[str, int]],
        mask_ratio: float = 0.15,
        zero_fraction: float = 0.2,
        seed: int = 0,
        train: bool = True,
        return_counts: bool = False,
    ):
        self.counts = counts.tocsr()
        self.normalized = normalized.tocsr()
        self.obs = obs.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.label_maps = label_maps
        self.mask_ratio = float(mask_ratio)
        self.zero_fraction = float(zero_fraction)
        self.train = train
        self.return_counts = return_counts
        self.seed = seed
        self.epoch = 0
        self._gene_ids_all = vocab.ids()  # token id per column

    def __len__(self) -> int:
        return self.normalized.shape[0]

    def set_epoch(self, epoch: int) -> None:
        """Advance the masking stream (training split only).

        Training masks are resampled every epoch so the model does not see
        the same corrupted view of a cell repeatedly; evaluation masks ignore
        the epoch and stay fixed, so validation loss is comparable across
        epochs and runs.  With ``num_workers > 0`` and persistent workers the
        worker copies do not see this call; use ``num_workers=0`` or
        non-persistent workers if per-epoch masking matters.
        """
        self.epoch = int(epoch)

    def _rng(self, i: int) -> np.random.Generator:
        epoch = self.epoch if self.train else 0
        return np.random.default_rng((self.seed, epoch, i))

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row = np.asarray(self.normalized[i].todense()).ravel()
        tok = self.tokenizer(row, self._gene_ids_all)
        gene_ids = tok["gene_ids"].astype(np.int64)
        values = tok["values"].astype(np.float32)
        bins = tok.get("value_bins")
        rng = self._rng(i)

        item: Dict[str, Any] = {
            "gene_ids": gene_ids,
            "values": values,
            "value_bins": None if bins is None else bins.astype(np.int64),
            "target_values": values.copy(),
            "target_bins": None if bins is None else bins.astype(np.int64).copy(),
        }

        if self.return_counts:
            craw = np.asarray(self.counts[i].todense()).ravel()
            col = gene_ids - self.vocab.N_SPECIAL
            item["counts"] = craw[col].astype(np.float32)

        # masked-value corruption
        n = gene_ids.size
        if self.mask_ratio > 0 and n > 0:
            k = max(1, int(round(self.mask_ratio * n)))
            pos = rng.choice(n, size=min(k, n), replace=False)
            m = np.zeros(n, dtype=bool)
            m[pos] = True
            item["mask_positions"] = m
            vals = values.copy()
            vals[m] = 0.0
            item["values"] = vals
            if bins is not None:
                bb = item["value_bins"].copy()
                bb[m] = 0  # 0 is the reserved "unknown/zero" bin
                item["value_bins"] = bb
        else:
            item["mask_positions"] = np.zeros(n, dtype=bool)

        for col, mapping in self.label_maps.items():
            raw = str(self.obs.iloc[i][col])
            item[f"label_{col}"] = mapping.get(raw, -100)
        item["_index"] = i
        return item


def collate_cells(items: Sequence[Dict[str, Any]], pad_to: Optional[int] = None) -> CellBatch:
    """Right-pad a list of variable-length cells into a ``CellBatch``."""
    B = len(items)
    L = max(1, max(it["gene_ids"].size for it in items))
    if pad_to:
        L = max(L, pad_to)
    gi = np.zeros((B, L), dtype=np.int64)
    vv = np.zeros((B, L), dtype=np.float32)
    tv = np.zeros((B, L), dtype=np.float32)
    pm = np.zeros((B, L), dtype=bool)
    mp = np.zeros((B, L), dtype=bool)
    has_bins = items[0]["value_bins"] is not None
    vb = np.zeros((B, L), dtype=np.int64) if has_bins else None
    tb = np.zeros((B, L), dtype=np.int64) if has_bins else None
    has_counts = "counts" in items[0]
    cc = np.zeros((B, L), dtype=np.float32) if has_counts else None

    for b, it in enumerate(items):
        n = it["gene_ids"].size
        gi[b, :n] = it["gene_ids"]
        vv[b, :n] = it["values"]
        tv[b, :n] = it["target_values"]
        pm[b, :n] = True
        mp[b, :n] = it["mask_positions"]
        if has_bins:
            vb[b, :n] = it["value_bins"]
            tb[b, :n] = it["target_bins"]
        if has_counts:
            cc[b, :n] = it["counts"]

    targets: Dict[str, torch.Tensor] = {"values": torch.from_numpy(tv)}
    if has_bins:
        targets["value_bins"] = torch.from_numpy(tb)
    if has_counts:
        targets["counts"] = torch.from_numpy(cc)
    for key in items[0]:
        if key.startswith("label_"):
            name = key[len("label_"):]
            targets[name] = torch.tensor([it[key] for it in items], dtype=torch.long)

    return CellBatch(
        gene_ids=torch.from_numpy(gi),
        values=torch.from_numpy(vv),
        value_bins=None if vb is None else torch.from_numpy(vb),
        pad_mask=torch.from_numpy(pm),
        mask_positions=torch.from_numpy(mp),
        targets=targets,
        meta={"index": np.array([it["_index"] for it in items])},
    )


class SingleCellDataModule:
    """End-to-end: source -> QC -> split -> fit-on-train -> dataloaders."""

    def __init__(
        self,
        source_cfg: Dict[str, Any],
        splitter_cfg: Dict[str, Any],
        tokenizer_cfg: Dict[str, Any],
        n_hvg: int = 2000,
        batch_size: int = 64,
        num_workers: int = 0,
        mask_ratio: float = 0.15,
        min_genes_per_cell: int = 10,
        min_cells_per_gene: int = 3,
        label_columns: Sequence[str] = ("cell_type", "batch_id"),
        return_counts: bool = False,
        seed: int = 0,
        fail_on_leakage: bool = True,
    ):
        self.source_cfg = dict(source_cfg)
        self.splitter_cfg = dict(splitter_cfg)
        self.tokenizer_cfg = dict(tokenizer_cfg)
        self.n_hvg = n_hvg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mask_ratio = mask_ratio
        self.min_genes_per_cell = min_genes_per_cell
        self.min_cells_per_gene = min_cells_per_gene
        self.label_columns = list(label_columns)
        self.return_counts = return_counts
        self.seed = seed
        self.fail_on_leakage = fail_on_leakage

        self.adata = None
        self.vocab: Optional[GeneVocabulary] = None
        self.tokenizer = None
        self.splits: Dict[str, np.ndarray] = {}
        self.label_maps: Dict[str, Dict[str, int]] = {}
        self.leakage = None
        self.datasets: Dict[str, CellDataset] = {}

    # -- setup -----------------------------------------------------------
    def setup(self) -> "SingleCellDataModule":
        source = SOURCES.build(self.source_cfg)
        adata = source.load()
        ensure_obs_columns(adata)
        self.provenance = source.provenance
        self.is_synthetic = bool(adata.uns.get("synthetic", False))

        X = adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = X.tocsr()
        if not is_raw_counts(X):
            log.warning(
                "input does not look like raw counts; count-likelihood heads "
                "(zinb) will be misspecified on this data."
            )

        # QC -- applied before splitting because it depends on no labels
        genes_per_cell = np.diff(X.indptr)
        keep_cells = genes_per_cell >= self.min_genes_per_cell
        cells_per_gene = np.asarray((X > 0).sum(axis=0)).ravel()
        keep_genes = cells_per_gene >= self.min_cells_per_gene
        if (~keep_cells).any() or (~keep_genes).any():
            log.info(
                "QC: dropping %d cells (<%d genes) and %d genes (<%d cells)",
                int((~keep_cells).sum()), self.min_genes_per_cell,
                int((~keep_genes).sum()), self.min_cells_per_gene,
            )
        adata = adata[keep_cells][:, keep_genes].copy()
        X = sp.csr_matrix(adata.X).tocsr()
        obs = adata.obs.reset_index(drop=True)

        # split FIRST, then fit everything on train only
        splitter = SPLITTERS.build(self.splitter_cfg)
        self.splits = splitter.split(obs)
        self.leakage = audit_leakage(obs, self.splits)
        log.info("\n%s", self.leakage)
        if self.fail_on_leakage and not self.leakage.clean:
            raise RuntimeError(
                "leakage audit failed:\n"
                + "\n".join(self.leakage["violations"])
                + "\nSet data.fail_on_leakage=false only to measure the "
                "size of the leak as a deliberate control."
            )

        train_idx = self.splits["train"]

        # HVG on train only, ranked within batch
        hvg = select_hvg(
            X[train_idx], n_top=self.n_hvg,
            batch=obs["batch_id"].to_numpy()[train_idx],
        )
        gene_names = adata.var_names.astype(str).to_numpy()[hvg]
        self.vocab = GeneVocabulary(gene_names)
        X = X[:, hvg].tocsr()
        adata = adata[:, hvg].copy()

        Xn, lib = normalize_counts(X, copy=True)
        obs["library_size"] = lib

        # tokenizer, fitted on train only where it has state
        self.tokenizer = TOKENIZERS.build(self.tokenizer_cfg)
        if hasattr(self.tokenizer, "fit"):
            self.tokenizer.fit(Xn[train_idx])

        # label encoders from train only; unseen test labels map to -100 and
        # are ignored by cross-entropy rather than silently shifting indices
        self.label_maps = {}
        for col in self.label_columns:
            vals = sorted(pd.unique(obs[col].astype(str).to_numpy()[train_idx]))
            self.label_maps[col] = {v: i for i, v in enumerate(vals)}
            unseen = set(obs[col].astype(str)) - set(vals)
            if unseen:
                log.info(
                    "%d unseen '%s' value(s) outside train (ignored in loss): %s",
                    len(unseen), col, sorted(unseen)[:4],
                )

        self.adata, self.obs, self.X, self.Xn = adata, obs, X, Xn
        for name, idx in self.splits.items():
            self.datasets[name] = CellDataset(
                counts=X[idx], normalized=Xn[idx], obs=obs.iloc[idx],
                tokenizer=self.tokenizer, vocab=self.vocab,
                label_maps=self.label_maps,
                mask_ratio=self.mask_ratio if name == "train" else self.mask_ratio,
                seed=self.seed, train=(name == "train"),
                return_counts=self.return_counts,
            )
        log.info(
            "datamodule ready: %s | %d genes | splits %s",
            self.provenance, self.vocab.n_genes,
            {k: int(v.size) for k, v in self.splits.items()},
        )
        return self

    # -- accessors -------------------------------------------------------
    @property
    def n_classes(self) -> int:
        return len(self.label_maps.get("cell_type", {}))

    @property
    def n_batches(self) -> int:
        return len(self.label_maps.get("batch_id", {}))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def n_value_bins(self) -> int:
        return max(2, int(self.tokenizer.n_value_bins))

    def loader(self, split: str, shuffle: Optional[bool] = None) -> DataLoader:
        if split not in self.datasets:
            raise KeyError(f"no split '{split}'; have {list(self.datasets)}")
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=(split == "train") if shuffle is None else shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_cells,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def dense_split(self, split: str):
        """Normalized dense matrix + obs for the baseline models."""
        idx = self.splits[split]
        return np.asarray(self.Xn[idx].todense()), self.obs.iloc[idx]
