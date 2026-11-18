"""Config composition tests.

These exist because every bug they catch is invisible until a run starts:
a model config that drifts from an experiment override, a component whose
YAML carries a key its constructor does not accept, a group that cannot be
swapped from the CLI. Each of those produced a real failure during
development, and none of them is caught by unit tests of the components.
"""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.models import build_model

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
EXPERIMENTS = [p.stem for p in (Path(CONFIG_DIR) / "experiment").glob("*.yaml")]
MODELS = [p.stem for p in (Path(CONFIG_DIR) / "model").glob("*.yaml")]
TOKENIZERS = [p.stem for p in (Path(CONFIG_DIR) / "data" / "tokenizer").glob("*.yaml")]
SPLITTERS = [p.stem for p in (Path(CONFIG_DIR) / "data" / "splitter").glob("*.yaml")]

CONTEXT = dict(n_genes=500, n_value_bins=52, n_classes=8, n_batches=4)


def _compose(*overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides))
        return OmegaConf.to_container(cfg, resolve=True)


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_every_experiment_composes(exp):
    cfg = _compose(f"+experiment={exp}")
    assert "model" in cfg and "data" in cfg and "train" in cfg


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_every_experiment_builds_a_model(exp):
    cfg = _compose(f"+experiment={exp}")
    model = build_model(cfg["model"], **CONTEXT)
    assert model.n_parameters() > 0


@pytest.mark.parametrize("model_name", MODELS)
def test_every_model_config_builds(model_name):
    cfg = _compose(f"model={model_name}")
    model = build_model(cfg["model"], **CONTEXT)
    assert model.d_model == cfg["model"]["d_model"]


@pytest.mark.parametrize("tok", TOKENIZERS)
def test_every_tokenizer_group_is_swappable(tok):
    from src.core.registry import TOKENIZERS as TOK_REG
    cfg = _compose(f"data/tokenizer={tok}")
    t = TOK_REG.build(cfg["data"]["tokenizer"])
    assert t.n_value_bins >= 1


@pytest.mark.parametrize("sp", SPLITTERS)
def test_every_splitter_group_is_swappable(sp):
    from src.core.registry import SPLITTERS as SP_REG
    cfg = _compose(f"data/splitter={sp}")
    SP_REG.build(cfg["data"]["splitter"])


def test_d_model_propagates_to_every_component():
    cfg = _compose("model.d_model=192")
    model = build_model(cfg["model"], **CONTEXT)
    assert model.stem.d_model == 192
    assert model.backbone.d_model == 192
    assert all(h.d_model == 192 for h in model.heads.values())


def test_disabled_heads_stay_disabled_by_default():
    cfg = _compose("model=small")
    model = build_model(cfg["model"], **CONTEXT)
    assert "cell_type" not in model.heads
    assert "batch_adversarial" not in model.heads


def test_baselines_cannot_be_emptied_from_config():
    """The mandatory-controls guarantee."""
    from src.runners.evaluate import DEFAULT_BASELINES
    cfg = _compose()
    assert len(cfg["eval"]["baselines"]) >= 4
    assert len(DEFAULT_BASELINES) >= 4


def test_leakage_guard_is_on_by_default():
    cfg = _compose()
    assert cfg["data"]["fail_on_leakage"] is True


def test_default_split_is_donor_held_out_not_random():
    cfg = _compose()
    assert cfg["data"]["splitter"]["name"] == "donor_held_out"
