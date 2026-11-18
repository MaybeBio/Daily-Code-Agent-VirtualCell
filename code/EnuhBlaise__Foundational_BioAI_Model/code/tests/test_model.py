import numpy as np
import pytest
import torch

from src.core.types import CellBatch
from src.models import build_model


def _cfg(d=32, **kw):
    c = {
        "stem": {"name": "expression", "d_model": d, "n_genes": 60, "n_value_bins": 8},
        "backbone": {"name": "set_transformer", "d_model": d, "n_layers": 2, "n_heads": 4},
        "heads": {"mlm": {"name": "masked_value", "d_model": d, "n_value_bins": 8}},
    }
    c.update(kw)
    return c


def _batch(B=4, L=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    gid = torch.randint(3, 60, (B, L), generator=g)
    bins = torch.randint(1, 8, (B, L), generator=g)
    pad = torch.ones(B, L, dtype=torch.bool)
    pad[0, 6:] = False
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, :2] = True
    return CellBatch(
        gene_ids=gid, values=bins.float(), value_bins=bins, pad_mask=pad,
        mask_positions=mask,
        targets={"value_bins": bins, "cell_type": torch.randint(0, 3, (B,), generator=g),
                 "batch_id": torch.randint(0, 2, (B,), generator=g)},
    )


def test_forward_and_backward():
    m = build_model(_cfg())
    out = m(_batch())
    assert out.loss.requires_grad
    out.loss.backward()
    assert any(p.grad is not None for p in m.parameters())


def test_embedding_is_permutation_invariant():
    """Genes are a set: reordering tokens must not change the cell embedding."""
    m = build_model(_cfg()).eval()
    b = _batch()
    perm = torch.randperm(b.seq_len)
    b2 = CellBatch(
        gene_ids=b.gene_ids[:, perm], values=b.values[:, perm],
        value_bins=b.value_bins[:, perm], pad_mask=b.pad_mask[:, perm],
    )
    with torch.no_grad():
        assert torch.allclose(m.encode(b), m.encode(b2), atol=1e-5)


def test_padding_does_not_affect_embedding():
    """Appending pad tokens must leave the embedding unchanged."""
    m = build_model(_cfg()).eval()
    b = _batch(B=2, L=6)
    pad_extra = torch.zeros(2, 4, dtype=torch.bool)
    b2 = CellBatch(
        gene_ids=torch.cat([b.gene_ids, torch.zeros(2, 4, dtype=torch.long)], 1),
        values=torch.cat([b.values, torch.zeros(2, 4)], 1),
        value_bins=torch.cat([b.value_bins, torch.zeros(2, 4, dtype=torch.long)], 1),
        pad_mask=torch.cat([b.pad_mask, pad_extra], 1),
    )
    with torch.no_grad():
        assert torch.allclose(m.encode(b), m.encode(b2), atol=1e-5)


def test_dimension_mismatch_is_caught():
    c = _cfg()
    c["backbone"]["d_model"] = 64
    with pytest.raises(ValueError, match="d_model"):
        build_model(c)


def test_auto_placeholder_is_filled_from_context():
    c = _cfg()
    c["stem"]["n_genes"] = "AUTO"
    m = build_model(c, n_genes=77)
    assert m.stem.gene_emb.num_embeddings == 77


def test_auto_placeholder_without_context_raises():
    c = _cfg()
    c["stem"]["n_genes"] = "AUTO"
    with pytest.raises(KeyError):
        build_model(c)


def test_disabled_head_is_not_built():
    c = _cfg()
    c["heads"]["ct"] = {"enabled": False, "name": "cell_type", "d_model": 32, "n_classes": 3}
    m = build_model(c)
    assert "ct" not in m.heads


def test_loss_only_counted_at_masked_positions():
    m = build_model(_cfg()).eval()
    b = _batch()
    b.mask_positions = torch.zeros_like(b.mask_positions)
    assert m(b).loss is None


def test_zinb_head_runs_on_counts():
    c = _cfg()
    c["heads"] = {"zinb": {"name": "zinb", "d_model": 32}}
    m = build_model(c)
    b = _batch()
    b.targets["counts"] = torch.randint(0, 20, b.gene_ids.shape).float()
    out = m(b)
    assert torch.isfinite(out.loss)
