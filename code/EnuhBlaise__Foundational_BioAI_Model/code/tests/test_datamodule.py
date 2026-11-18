import numpy as np
import torch

from src.data.datamodule import collate_cells


def test_setup_produces_disjoint_splits(datamodule):
    idx = np.concatenate(list(datamodule.splits.values()))
    assert np.unique(idx).size == idx.size


def test_leakage_audit_is_clean_by_default(datamodule):
    assert datamodule.leakage.clean


def test_batch_shapes_and_targets(datamodule):
    b = next(iter(datamodule.loader("train")))
    assert b.gene_ids.shape == b.values.shape == b.pad_mask.shape
    assert {"values", "value_bins", "counts", "cell_type", "batch_id"} <= set(b.targets)
    assert b.value_bins.max() <= datamodule.n_value_bins - 1


def test_masking_hits_the_requested_fraction(datamodule):
    b = next(iter(datamodule.loader("train")))
    frac = b.mask_positions.sum().item() / b.pad_mask.sum().item()
    assert 0.05 < frac < 0.35


def test_masked_positions_are_zeroed_in_input_but_kept_in_targets(datamodule):
    b = next(iter(datamodule.loader("train")))
    m = b.mask_positions
    assert (b.value_bins[m] == 0).all()
    assert (b.targets["value_bins"][m] > 0).any()


def test_padding_is_marked_false(datamodule):
    b = next(iter(datamodule.loader("train")))
    assert (b.gene_ids[~b.pad_mask] == 0).all()


def test_vocabulary_is_fixed_across_splits(datamodule):
    a = next(iter(datamodule.loader("train")))
    c = next(iter(datamodule.loader("test")))
    assert a.gene_ids.max() < datamodule.vocab_size
    assert c.gene_ids.max() < datamodule.vocab_size


def test_label_maps_come_from_train_only(datamodule):
    train_types = set(
        datamodule.obs["cell_type"].astype(str).to_numpy()[datamodule.splits["train"]]
    )
    assert set(datamodule.label_maps["cell_type"]) == train_types


def test_training_masks_change_per_epoch_but_eval_masks_do_not(datamodule):
    tr, te = datamodule.datasets["train"], datamodule.datasets["test"]
    try:
        tr.set_epoch(0); te.set_epoch(0)
        a_tr, a_te = tr[0]["mask_positions"].copy(), te[0]["mask_positions"].copy()
        tr.set_epoch(1); te.set_epoch(1)
        assert not np.array_equal(a_tr, tr[0]["mask_positions"])
        assert np.array_equal(a_te, te[0]["mask_positions"])
        tr.set_epoch(0)
        assert np.array_equal(a_tr, tr[0]["mask_positions"])     # reproducible
    finally:
        tr.set_epoch(0); te.set_epoch(0)
