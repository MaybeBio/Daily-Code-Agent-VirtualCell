"""Metrics are validated against embeddings whose answer is known."""
import numpy as np
import pytest

from src.eval.metrics import (
    batch_correction, bio_conservation, evaluate_embedding,
    library_size_leakage, pcr_batch,
)

rng = np.random.default_rng(0)
N = 600
Y = rng.integers(0, 4, N)
B = rng.integers(0, 3, N)
Z_BIO = np.eye(4)[Y] * 6 + rng.normal(0, 1, (N, 4))
Z_BATCH = np.eye(3)[B] * 6 + rng.normal(0, 1, (N, 3))
Z_NOISE = rng.normal(0, 1, (N, 8))


def test_bio_metrics_separate_signal_from_noise():
    assert bio_conservation(Z_BIO, Y)["ari"] > 0.8
    assert bio_conservation(Z_NOISE, Y)["ari"] < 0.1


def test_pcr_detects_batch_structure():
    assert pcr_batch(Z_BATCH, B) > 0.5
    assert pcr_batch(Z_BIO, B) < 0.1


def test_batch_metrics_reward_mixing():
    good = batch_correction(Z_BIO, B, Y)
    bad = batch_correction(Z_BATCH, B, Y)
    assert good["knn_mixing"] > bad["knn_mixing"]
    assert good["kbet"] > bad["kbet"]


def test_noise_scores_well_on_batch_and_badly_on_biology():
    """The reason both families must always be reported together."""
    m = evaluate_embedding(Z_NOISE, Y, B)
    assert m["batch_score"] > 0.8
    assert m["bio_score"] < 0.3


def test_library_size_leakage_detects_depth_encoding():
    lib = rng.lognormal(8, 0.5, N)
    Z_depth = np.log1p(lib)[:, None] + rng.normal(0, 0.01, (N, 3))
    assert library_size_leakage(Z_depth, lib) > 0.9
    assert library_size_leakage(Z_BIO, lib) < 0.3


def test_overall_combines_both_families():
    m = evaluate_embedding(Z_BIO, Y, B)
    assert set(["bio_score", "batch_score", "overall"]) <= set(m)
    assert 0 <= m["overall"] <= 1
