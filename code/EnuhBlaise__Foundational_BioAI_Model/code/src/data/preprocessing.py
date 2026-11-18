"""Count preprocessing and gene-space harmonization.

Every choice here is a place where information leaks or is destroyed, so each
function states what it assumes and records what it did into ``adata.uns``.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)

__all__ = [
    "REQUIRED_OBS",
    "ensure_obs_columns",
    "is_raw_counts",
    "normalize_counts",
    "select_hvg",
    "harmonize_genes",
    "GeneVocabulary",
]

#: obs columns the evaluation and splitting layers rely on.  A source that
#: cannot supply one must fill it with "unknown" explicitly -- never silently.
REQUIRED_OBS = (
    "donor_id",
    "batch_id",
    "study_id",
    "assay",
    "tissue",
    "cell_type",
)


def ensure_obs_columns(adata, fill: str = "unknown"):
    """Guarantee ``REQUIRED_OBS`` exist; record which were fabricated.

    Downstream, ``adata.uns['fabricated_obs']`` is what lets a splitter refuse
    to do a donor-held-out split on a dataset with no real donor labels
    instead of silently producing a random split and calling it rigorous.
    """
    fabricated = []
    for col in REQUIRED_OBS:
        if col not in adata.obs.columns:
            adata.obs[col] = fill
            fabricated.append(col)
    adata.obs[list(REQUIRED_OBS)] = adata.obs[list(REQUIRED_OBS)].astype(str)
    prev = list(adata.uns.get("fabricated_obs", []))
    adata.uns["fabricated_obs"] = sorted(set(prev) | set(fabricated))
    if fabricated:
        log.warning(
            "obs columns fabricated as '%s': %s -- splits and metrics that "
            "depend on them are not meaningful for this dataset.",
            fill,
            ", ".join(fabricated),
        )
    return adata


def is_raw_counts(X, tol: float = 1e-6, max_check: int = 100_000) -> bool:
    """Heuristic: are these integer counts rather than normalized values?

    Catches the common and costly mistake of pretraining a count likelihood
    (ZINB / Poisson) on log-normalized input.
    """
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    if data.size == 0:
        return True
    sample = data[:max_check]
    if np.any(sample < 0):
        return False
    return bool(np.all(np.abs(sample - np.rint(sample)) < tol))


def normalize_counts(
    X,
    target_sum: float = 1e4,
    log1p: bool = True,
    copy: bool = True,
):
    """Counts-per-``target_sum`` then optional ``log1p``.

    Returns ``(X_norm, library_size)``.  Library size is returned rather than
    discarded because count depth is a confounder that evaluation must be able
    to regress against -- an embedding that encodes depth and nothing else can
    still score well on clustering metrics.
    """
    if sp.issparse(X):
        X = X.tocsr(copy=copy).astype(np.float32)
        lib = np.asarray(X.sum(axis=1)).ravel()
        scale = np.divide(
            target_sum, lib, out=np.zeros_like(lib, dtype=np.float64), where=lib > 0
        )
        X = sp.diags(scale.astype(np.float32)) @ X
        if log1p:
            X.data = np.log1p(X.data)
    else:
        X = np.array(X, dtype=np.float32, copy=copy)
        lib = X.sum(axis=1)
        scale = np.divide(target_sum, lib, out=np.zeros_like(lib), where=lib > 0)
        X = X * scale[:, None]
        if log1p:
            X = np.log1p(X)
    return X, lib


def select_hvg(
    X,
    n_top: int = 2000,
    batch: Optional[np.ndarray] = None,
    min_cells: int = 3,
    flavor: str = "pearson_residuals",
    theta: float = 100.0,
    chunk_size: int = 512,
) -> np.ndarray:
    """Highly variable genes from **raw counts**.

    flavor
        ``"pearson_residuals"`` (default) scores each gene by the variance of
        its analytic Pearson residuals under a negative-binomial null with
        depth as the only covariate (Lause, Berens & Kobak 2021).  A gene
        scores highly only if it varies more than depth and count noise
        predict, whatever its expression level.

        ``"dispersion"`` scores by variance / mean.  Kept as a comparison
        point, not a recommendation: under NB noise variance/mean grows with
        the mean, so this flavor largely returns the most abundant genes.

    When ``batch`` is given, genes are ranked *within* each batch and the
    ranks are averaged, so a gene that differs between batches but is flat
    inside each one is not mistaken for a variable gene.

    Returns sorted integer indices into the gene axis.
    """
    if flavor not in ("pearson_residuals", "dispersion"):
        raise ValueError("flavor must be 'pearson_residuals' or 'dispersion'")

    def _score_dispersion(Xs) -> tuple[np.ndarray, np.ndarray]:
        if sp.issparse(Xs):
            Xs = Xs.tocsc()
            n = Xs.shape[0]
            mean = np.asarray(Xs.mean(axis=0)).ravel()
            sq = np.asarray(Xs.multiply(Xs).mean(axis=0)).ravel()
            nnz = np.diff(Xs.indptr)
        else:
            Xs = np.asarray(Xs)
            n = Xs.shape[0]
            mean = Xs.mean(axis=0)
            sq = (Xs ** 2).mean(axis=0)
            nnz = (Xs > 0).sum(axis=0)
        var = np.maximum(sq - mean ** 2, 0.0) * (n / max(n - 1, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(mean > 0, var / np.maximum(mean, 1e-12), 0.0)
        return score, nnz

    def _score_pearson(Xs) -> tuple[np.ndarray, np.ndarray]:
        Xs = sp.csc_matrix(Xs) if not sp.issparse(Xs) else Xs.tocsc()
        n, g = Xs.shape
        lib = np.asarray(Xs.sum(axis=1)).ravel().astype(np.float64)
        gene_sum = np.asarray(Xs.sum(axis=0)).ravel().astype(np.float64)
        total = max(lib.sum(), 1e-12)
        clip = np.sqrt(n)
        score = np.zeros(g)
        # dense only one block of genes at a time, so memory stays bounded
        for start in range(0, g, chunk_size):
            end = min(start + chunk_size, g)
            block = Xs[:, start:end].toarray().astype(np.float64)
            mu = np.outer(lib, gene_sum[start:end] / total)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (block - mu) / np.sqrt(mu + mu ** 2 / theta)
            r[~np.isfinite(r)] = 0.0
            np.clip(r, -clip, clip, out=r)
            score[start:end] = r.var(axis=0)
        return score, np.diff(Xs.indptr)

    scorer = _score_pearson if flavor == "pearson_residuals" else _score_dispersion

    def _rank_one(Xs) -> np.ndarray:
        score, nnz = scorer(Xs)
        score = np.where(nnz < min_cells, -np.inf, score)
        order = np.argsort(-score, kind="stable")
        ranks = np.empty(order.size, dtype=np.float64)
        ranks[order] = np.arange(order.size)
        return ranks

    n_genes = X.shape[1]
    n_top = int(min(n_top, n_genes))
    if batch is None:
        ranks = _rank_one(X)
    else:
        batch = np.asarray(batch)
        acc, k = np.zeros(n_genes), 0
        for b in np.unique(batch):
            idx = np.flatnonzero(batch == b)
            if idx.size < max(min_cells, 10):
                continue
            acc += _rank_one(X[idx])
            k += 1
        ranks = acc / max(k, 1) if k else _rank_one(X)
    return np.sort(np.argsort(ranks, kind="stable")[:n_top])


class GeneVocabulary:
    """Maps gene identifiers to model token ids.

    Index 0 is PAD, 1 is MASK, 2 is CLS.  Real genes start at 3.  The
    vocabulary is frozen at construction and saved with a checkpoint: a model
    whose gene ordering shifts between pretraining and evaluation is silently
    broken, and this is a common source of irreproducible results.
    """

    PAD, MASK, CLS = 0, 1, 2
    N_SPECIAL = 3

    def __init__(self, genes: Sequence[str]):
        genes = [str(g) for g in genes]
        if len(set(genes)) != len(genes):
            seen, dups = set(), set()
            for g in genes:
                (dups if g in seen else seen).add(g)
            raise ValueError(
                f"duplicate gene identifiers in vocabulary: "
                f"{sorted(dups)[:5]}{'...' if len(dups) > 5 else ''}"
            )
        self.genes = list(genes)
        self._to_id = {g: i + self.N_SPECIAL for i, g in enumerate(self.genes)}

    def __len__(self) -> int:
        return len(self.genes) + self.N_SPECIAL

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    def ids(self, genes: Optional[Iterable[str]] = None) -> np.ndarray:
        if genes is None:
            return np.arange(self.N_SPECIAL, len(self), dtype=np.int64)
        return np.array([self._to_id.get(str(g), -1) for g in genes], dtype=np.int64)

    def contains(self, genes: Iterable[str]) -> np.ndarray:
        return np.array([str(g) in self._to_id for g in genes], dtype=bool)

    def to_dict(self) -> dict:
        return {"genes": self.genes, "n_special": self.N_SPECIAL}

    @classmethod
    def from_dict(cls, d: dict) -> "GeneVocabulary":
        return cls(d["genes"])

    def __repr__(self) -> str:
        return f"GeneVocabulary(n_genes={self.n_genes}, vocab_size={len(self)})"


def harmonize_genes(adata, vocab: GeneVocabulary, var_key: Optional[str] = None):
    """Reindex ``adata`` onto a fixed vocabulary.

    Genes absent from the dataset become structural zeros rather than being
    dropped, so that the same token id always means the same gene across
    datasets.  Returns ``(X_aligned, coverage_fraction)``.
    """
    names = (
        adata.var_names.astype(str).to_numpy()
        if var_key is None
        else adata.var[var_key].astype(str).to_numpy()
    )
    pos = {g: i for i, g in enumerate(names)}
    cols, present = [], []
    for g in vocab.genes:
        j = pos.get(g)
        present.append(j is not None)
        cols.append(j if j is not None else 0)
    present = np.array(present)
    cols = np.array(cols)

    X = adata.X
    Xa = X[:, cols]
    if sp.issparse(Xa):
        Xa = Xa.tocsc()
        for j in np.flatnonzero(~present):
            Xa.data[Xa.indptr[j] : Xa.indptr[j + 1]] = 0
        Xa.eliminate_zeros()
        Xa = Xa.tocsr()
    else:
        Xa = np.asarray(Xa, dtype=np.float32).copy()
        Xa[:, ~present] = 0.0
    coverage = float(present.mean())
    if coverage < 0.5:
        log.warning(
            "only %.1f%% of the model's gene vocabulary is present in this "
            "dataset; embeddings are not comparable to the pretraining corpus.",
            100 * coverage,
        )
    return Xa, coverage
