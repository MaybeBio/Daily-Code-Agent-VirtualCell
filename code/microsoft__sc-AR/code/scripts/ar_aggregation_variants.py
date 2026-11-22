"""Alternative AR weight aggregation rules (aggregation ablation).

The default sc-AR rule (scripts/AR_resampling_weight_calculator.py) bins each
latent dimension k into a histogram, smooths and inverts the bin densities and
normalizes them over cells, giving per-dimension weights u*_kn (each column
sums to 1). The weight of cell n is the maximum over dimensions, w_n = max_k
u*_kn, as in the MIT 6.S191 (2019) DB-VAE course code and its 2021
reproduction (https://goodboychan.github.io/python/tensorflow/mit/2021/02/27/Debiasing.html).

Rules in this module (all return weights that sum to 1):

    max       max_k u*_kn (default sc-AR rule)
    mean      mean_k u*_kn
    geomean   exp(mean_k log u*_kn)
    knn       mean Euclidean distance to the k nearest neighbours in the full
              latent space; a rotation-invariant inverse joint-density proxy
    uniform   1/N; resampling with replacement but no density weights

knn uses the mean distance itself, not the k-NN density estimate 1/p ~ r^K,
which in K=64 dimensions would put all mass on a handful of cells.
"""

from __future__ import annotations

import numpy as np
import torch

from scripts.AR_resampling_weight_calculator import calculate_resampling_weights


def _as_numpy(z) -> np.ndarray:
    """Latent matrix (N, K) as a numpy array, keeping its dtype (so the max
    rule gives exactly the same weights as in training)."""
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    return np.asarray(z)


def _normalize(w: np.ndarray) -> np.ndarray:
    return w / np.sum(w)


def _marginal_inverse_weights(z: np.ndarray, bins: int, smoothing_fac: float) -> np.ndarray:
    """Per-dimension normalized inverse smoothed densities u*_kn, shape (N, K).
    Same steps as calculate_resampling_weights, before the max."""
    n_samples, n_dims = z.shape
    per_dim = np.zeros((n_samples, n_dims), dtype=float)

    for i in range(n_dims):
        latent_distribution = z[:, i]
        hist_density, bin_edges = np.histogram(latent_distribution, density=True, bins=bins)
        bin_edges[0] = -float("inf")
        bin_edges[-1] = float("inf")
        bin_idx = np.digitize(latent_distribution, bin_edges)
        hist_smoothed_density = hist_density + smoothing_fac
        hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)
        p = 1.0 / hist_smoothed_density[bin_idx - 1]
        p = p / np.sum(p)
        per_dim[:, i] = p

    return per_dim


def _exp_normalize(log_w: np.ndarray) -> np.ndarray:
    """Normalize weights given in log space without underflow."""
    return _normalize(np.exp(log_w - log_w.max()))


def calculate_resampling_weights_mean(z, bins: int = 100, smoothing_fac: float = 0.0001) -> np.ndarray:
    """Mean over dimensions of the per-dimension inverse-density weights."""
    per_dim = _marginal_inverse_weights(_as_numpy(z), bins, smoothing_fac)
    return _normalize(per_dim.mean(axis=1))


def calculate_resampling_weights_geomean(z, bins: int = 100, smoothing_fac: float = 0.0001) -> np.ndarray:
    """Geometric mean over dimensions of the per-dimension inverse-density weights."""
    per_dim = _marginal_inverse_weights(_as_numpy(z), bins, smoothing_fac)
    return _exp_normalize(np.log(per_dim).mean(axis=1))


def knn_mean_distance(z, k: int = 30, device=None, chunk_size: int = 1024) -> np.ndarray:
    """Mean Euclidean distance of every cell to its k nearest other cells.

    Exact brute-force search in chunks with torch, on the GPU if available
    (float32) or on the CPU (float64). The cell itself is excluded; exact
    duplicates of a cell count as neighbours at distance 0.

    Args:
        z: latent matrix (N, K)
        k (int): number of neighbours
        device: torch device; default cuda if available, else cpu
        chunk_size (int): rows of the distance matrix computed at once

    Returns:
        np.ndarray: mean k-NN distance per cell, shape (N,)"""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    x = torch.as_tensor(_as_numpy(z), dtype=dtype, device=device)
    n = x.shape[0]
    k = min(k, n - 1)
    sq_norm = (x * x).sum(dim=1)
    mean_dist = torch.empty(n, dtype=dtype, device=device)

    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            # squared distances ||a||^2 + ||b||^2 - 2 a.b, clipped at 0 for rounding
            d2 = sq_norm[start:end, None] + sq_norm[None, :] - 2.0 * (x[start:end] @ x.T)
            d2.clamp_(min=0)
            rows = torch.arange(end - start, device=device)
            d2[rows, rows + start] = float("inf")
            nearest = torch.topk(d2, k, dim=1, largest=False).values
            mean_dist[start:end] = nearest.sqrt().mean(dim=1)

    return mean_dist.cpu().numpy().astype(float)


def calculate_resampling_weights_knn(z, k: int = 30, device=None) -> np.ndarray:
    """Weights proportional to the mean distance to the k nearest neighbours
    in the full latent space (large distance = low joint density)."""
    dist = knn_mean_distance(z, k=k, device=device)
    if not np.sum(dist) > 0:
        # all cells identical: no density information
        return np.full(len(dist), 1.0 / len(dist))
    return _normalize(dist)


def calculate_resampling_weights_uniform(z) -> np.ndarray:
    """Uniform weights 1/N."""
    n = _as_numpy(z).shape[0]
    return np.full(n, 1.0 / n)


AGGREGATION_METHODS = ["max", "mean", "geomean", "knn", "uniform"]


def calculate_resampling_weights_with_method(
    z,
    method: str = "max",
    bins: int = 100,
    smoothing_fac: float = 0.0001,
    knn_k: int = 30,
    device=None,
) -> np.ndarray:
    """Resampling weights of every cell for one aggregation rule.

    Args:
        z: latent matrix (N, K), numpy array or torch tensor
        method (str): one of AGGREGATION_METHODS
        bins (int): histogram bins per latent dimension (histogram rules)
        smoothing_fac (float): smoothing added to the bin densities (histogram rules)
        knn_k (int): number of neighbours (knn)
        device: torch device for knn (default cuda if available)

    Returns:
        np.ndarray: weights of shape (N,) that sum to 1"""

    method = method.lower()
    if method == "max":
        return calculate_resampling_weights(_as_numpy(z), bins=bins, smoothing_fac=smoothing_fac)
    if method == "mean":
        return calculate_resampling_weights_mean(z, bins=bins, smoothing_fac=smoothing_fac)
    if method == "geomean":
        return calculate_resampling_weights_geomean(z, bins=bins, smoothing_fac=smoothing_fac)
    if method == "knn":
        return calculate_resampling_weights_knn(z, k=knn_k, device=device)
    if method == "uniform":
        return calculate_resampling_weights_uniform(z)
    raise ValueError(f"Unknown method '{method}'. Choose from {AGGREGATION_METHODS}")


def normalized_ess(w: np.ndarray) -> float:
    """Normalized effective sample size (sum w)^2 / (N sum w^2), in (0, 1];
    1 = uniform weights, 1/N = all mass on one cell."""
    w = np.asarray(w, dtype=float)
    return float(np.sum(w) ** 2 / (len(w) * np.sum(w ** 2)))
