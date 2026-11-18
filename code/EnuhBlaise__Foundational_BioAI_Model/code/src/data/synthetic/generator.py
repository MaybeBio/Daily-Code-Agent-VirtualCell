"""A simulator that reproduces the statistics that make scRNA-seq hard.

Design goal: a model that scores well here should score well on real data for
the same reasons, and a metric that cannot detect batch effects here will not
detect them on real data either.  The simulator therefore reproduces, with
parameters taken from the ranges reported in the literature:

* **Negative-binomial counts** with a mean-dependent overdispersion.  Real
  UMI data is close to NB; dispersion falls with expression roughly as
  ``phi ~ a + b/mu`` (Svensson 2020; Hafemeister & Satija 2019).
* **Log-normal library sizes** spanning roughly an order of magnitude, the
  dominant technical axis in any real dataset.
* **Zero inflation beyond NB**, used sparingly: for UMI protocols most zeros
  are explained by NB sampling alone (Svensson 2020; Townes 2019), so the
  default ``extra_dropout`` is small and switchable.
* **Batch effects** as per-batch, per-gene multiplicative shifts with a
  gene-specific component -- i.e. not removable by a global scale factor,
  which is what makes batch correction nontrivial.
* **Ambient RNA** contamination: a fraction of each cell's counts drawn from
  a pooled soup profile (CellBender / SoupX phenomenology).
* **Doublets** at a rate proportional to loading, as in droplet protocols.
* **Hierarchical cell types** with shared programs, so that a metric can
  distinguish "separates coarse lineages" from "resolves close subtypes".
* **Donor effects** nested inside batches, so that donor-held-out splitting
  is actually testing something.

Every effect can be switched off individually, which is what makes this
useful for debugging: generate with ``batch_effect=0`` and a batch metric
must read ~0, generate with a large one and it must not.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

log = logging.getLogger(__name__)

__all__ = ["SyntheticConfig", "SyntheticCellSimulator", "make_synthetic"]


@dataclass
class SyntheticConfig:
    """Parameters of the simulator. Defaults sit in realistic ranges."""

    n_cells: int = 4000
    n_genes: int = 2000
    n_cell_types: int = 8
    n_lineages: int = 3          # coarse groups the types nest inside
    n_batches: int = 4
    n_donors: int = 10
    n_studies: int = 2
    n_tissues: int = 2

    # expression program
    mean_log_expression: float = -1.0   # log10 scale of gene means
    sd_log_expression: float = 1.5   # heavy tail -> ~90% zeros, as in 10x
    program_genes_per_type: int = 40    # marker genes per cell type
    program_strength: float = 2.2       # log2 fold-change of markers
    lineage_program_genes: int = 80
    lineage_strength: float = 1.4

    # counting / technical
    library_size_median: float = 5000.0
    library_size_log_sd: float = 0.45
    dispersion_a: float = 0.08          # phi -> a as mu -> inf
    dispersion_b: float = 2.0           # phi ~ a + b/mu
    extra_dropout: float = 0.0          # zero inflation beyond NB

    # confounders
    batch_effect: float = 0.6           # sd of per-gene log shift per batch
    batch_library_effect: float = 0.25  # per-batch depth shift (log sd)
    donor_effect: float = 0.25
    ambient_fraction: float = 0.08      # mean soup contamination
    ambient_sd: float = 0.04
    doublet_rate: float = 0.05
    # make cell-type composition differ by batch, which breaks naive metrics
    composition_shift: float = 0.5

    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_cell_types < self.n_lineages:
            raise ValueError("n_cell_types must be >= n_lineages")
        if self.n_donors < self.n_batches:
            log.warning("n_donors < n_batches: donor and batch will be near-aliased")
        for f in ("ambient_fraction", "extra_dropout", "doublet_rate"):
            v = getattr(self, f)
            if not 0 <= v < 1:
                raise ValueError(f"{f} must be in [0, 1), got {v}")


class SyntheticCellSimulator:
    """Generates an ``AnnData`` of raw counts with known ground truth.

    The ground-truth noise-free log-expression is kept in ``.layers['truth']``
    and the per-cell latent program weights in ``.obsm['latent']``, so an
    embedding can be scored against what it was supposed to recover rather
    than only against cluster labels.
    """

    def __init__(self, config: Optional[SyntheticConfig] = None, **overrides):
        self.cfg = config or SyntheticConfig(**overrides)
        if overrides and config is not None:
            for k, v in overrides.items():
                setattr(self.cfg, k, v)
        self.rng = np.random.default_rng(self.cfg.seed)

    # -- structure -------------------------------------------------------
    def _gene_means(self) -> np.ndarray:
        c = self.cfg
        lg = self.rng.normal(c.mean_log_expression, c.sd_log_expression, c.n_genes)
        return np.power(10.0, lg)

    def _programs(self) -> np.ndarray:
        """``[n_cell_types, n_genes]`` log2 multiplicative program."""
        c = self.cfg
        prog = np.zeros((c.n_cell_types, c.n_genes))
        lineage_of = np.array(
            [t % c.n_lineages for t in range(c.n_cell_types)]
        )
        # shared lineage programs
        for l in range(c.n_lineages):
            genes = self.rng.choice(c.n_genes, c.lineage_program_genes, replace=False)
            w = self.rng.normal(c.lineage_strength, 0.3, genes.size)
            for t in np.flatnonzero(lineage_of == l):
                prog[t, genes] += w
        # type-specific markers
        for t in range(c.n_cell_types):
            genes = self.rng.choice(c.n_genes, c.program_genes_per_type, replace=False)
            prog[t, genes] += self.rng.normal(c.program_strength, 0.4, genes.size)
        self.lineage_of = lineage_of
        return prog

    def _assign_metadata(self) -> pd.DataFrame:
        c, rng = self.cfg, self.rng
        donors = np.array([f"donor_{i:02d}" for i in range(c.n_donors)])
        batches = np.array([f"batch_{i}" for i in range(c.n_batches)])
        # donors nest inside batches; studies nest donors
        donor_batch = {d: batches[i % c.n_batches] for i, d in enumerate(donors)}
        donor_study = {d: f"study_{i % c.n_studies}" for i, d in enumerate(donors)}
        donor_tissue = {d: f"tissue_{i % c.n_tissues}" for i, d in enumerate(donors)}

        cell_donor = rng.choice(donors, c.n_cells)
        cell_batch = np.array([donor_batch[d] for d in cell_donor])

        # batch-dependent composition: different batches over-represent
        # different cell types, so a method that merely mixes batches without
        # respecting biology is penalized.
        base = rng.dirichlet(np.ones(c.n_cell_types) * 3.0)
        comp = {}
        for b in batches:
            tilt = rng.dirichlet(np.ones(c.n_cell_types))
            p = (1 - c.composition_shift) * base + c.composition_shift * tilt
            comp[b] = p / p.sum()
        cell_type_idx = np.array(
            [rng.choice(c.n_cell_types, p=comp[b]) for b in cell_batch]
        )
        return pd.DataFrame(
            {
                "donor_id": cell_donor,
                "batch_id": cell_batch,
                "study_id": [donor_study[d] for d in cell_donor],
                "tissue": [donor_tissue[d] for d in cell_donor],
                "assay": [
                    "10x_3p_v3" if donor_study[d] == "study_0" else "10x_3p_v2"
                    for d in cell_donor
                ],
                "_type_idx": cell_type_idx,
            }
        )

    # -- generation ------------------------------------------------------
    def generate(self):
        import anndata as ad

        c, rng = self.cfg, self.rng
        base_mu = self._gene_means()
        prog = self._programs()
        obs = self._assign_metadata()
        n, g = c.n_cells, c.n_genes

        type_names = np.array([f"celltype_{i:02d}" for i in range(c.n_cell_types)])
        obs["cell_type"] = type_names[obs["_type_idx"].to_numpy()]
        obs["lineage"] = np.array([f"lineage_{l}" for l in self.lineage_of])[
            obs["_type_idx"].to_numpy()
        ]

        # per-gene, per-batch and per-donor log shifts (non-uniform across
        # genes: a single global scale factor cannot remove them)
        batches = np.unique(obs["batch_id"])
        donors = np.unique(obs["donor_id"])
        batch_shift = {
            b: rng.normal(0, c.batch_effect, g) for b in batches
        }
        donor_shift = {d: rng.normal(0, c.donor_effect, g) for d in donors}
        batch_depth = {b: rng.normal(0, c.batch_library_effect) for b in batches}

        # latent program weights with within-type variation
        latent = rng.normal(0, 1, (n, c.n_cell_types)) * 0.25
        latent[np.arange(n), obs["_type_idx"].to_numpy()] += 1.0

        log2_mu = np.log2(base_mu)[None, :] + latent @ prog
        bshift = np.stack([batch_shift[b] for b in obs["batch_id"]])
        dshift = np.stack([donor_shift[d] for d in obs["donor_id"]])
        log2_mu_tech = log2_mu + bshift + dshift

        truth = log2_mu.astype(np.float32)  # biology only, no technical terms
        mu = np.power(2.0, log2_mu_tech)
        mu /= mu.sum(axis=1, keepdims=True)

        # library sizes, log-normal and batch-shifted
        depth_shift = np.array([batch_depth[b] for b in obs["batch_id"]])
        lib = np.exp(
            np.log(c.library_size_median)
            + rng.normal(0, c.library_size_log_sd, n)
            + depth_shift
        )
        lam = mu * lib[:, None]

        # ambient soup: pooled profile, mixed in per cell
        soup = mu.mean(axis=0)
        amb = np.clip(rng.normal(c.ambient_fraction, c.ambient_sd, n), 0, 0.9)
        lam = (1 - amb)[:, None] * lam + (amb * lib)[:, None] * soup[None, :]

        # NB sampling with mean-dependent dispersion phi ~ a + b/mu
        phi = c.dispersion_a + c.dispersion_b / np.maximum(lam, 1e-8)
        r = 1.0 / np.maximum(phi, 1e-8)
        p = r / (r + lam)
        counts = rng.negative_binomial(np.clip(r, 1e-4, 1e8), np.clip(p, 1e-8, 1 - 1e-8))
        counts = counts.astype(np.float32)

        if c.extra_dropout > 0:
            counts *= rng.random(counts.shape) >= c.extra_dropout

        # doublets: sum two random cells, label them
        n_doub = int(round(c.doublet_rate * n))
        is_doublet = np.zeros(n, dtype=bool)
        if n_doub:
            idx = rng.choice(n, n_doub, replace=False)
            partner = rng.choice(n, n_doub, replace=True)
            counts[idx] = counts[idx] + counts[partner]
            is_doublet[idx] = True

        X = sp.csr_matrix(counts)
        X.eliminate_zeros()

        var = pd.DataFrame(
            {"gene_mean": base_mu, "is_marker": (np.abs(prog).sum(0) > 0)},
            index=[f"GENE{i:05d}" for i in range(g)],
        )
        obs = obs.drop(columns=["_type_idx"])
        obs["is_doublet"] = is_doublet
        obs["observed_library_size"] = np.asarray(X.sum(axis=1)).ravel()
        obs.index = [f"cell_{i:06d}" for i in range(n)]

        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.layers["truth"] = truth
        adata.obsm["latent"] = latent.astype(np.float32)
        adata.uns["synthetic"] = True
        adata.uns["synthetic_config"] = asdict(c)
        adata.uns["fabricated_obs"] = []
        log.info(
            "synthetic data: %d cells x %d genes, median depth %.0f, "
            "sparsity %.1f%%",
            n, g, float(np.median(obs["observed_library_size"])),
            100 * (1 - X.nnz / (n * g)),
        )
        return adata


def make_synthetic(**kwargs):
    """Convenience: ``make_synthetic(n_cells=2000, batch_effect=0.0)``."""
    return SyntheticCellSimulator(SyntheticConfig(**kwargs)).generate()
