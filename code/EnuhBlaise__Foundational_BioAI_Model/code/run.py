#!/usr/bin/env python3
"""Single entry point.

    python run.py +experiment=smoke_test
    python run.py +experiment=pretrain_synthetic model=mlp_control
    python run.py +experiment=pretrain_census data.source.n_cells=200000
    python run.py --list                      # show registered components
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent))

from src.core.registry import ALL_REGISTRIES, autodiscover  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

log = logging.getLogger(__name__)


def _list_components() -> None:
    autodiscover()
    print("Registered components\n" + "=" * 60)
    for kind, reg in ALL_REGISTRIES.items():
        print(f"\n{kind}:")
        prov = reg.provenance()
        for name in reg:
            print(f"  {name:24s} {prov[name]}")
    print()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    autodiscover()
    seed_everything(int(cfg.seed))
    plain = OmegaConf.to_container(cfg, resolve=True)

    out_dir = Path(plain["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    from src.runners.evaluate import evaluate
    from src.runners.pretrain import pretrain

    model, dm, history = pretrain(plain)
    device = resolve_device(plain.get("execution", {}).get("device", "auto"))
    report = evaluate(model, dm, plain, device,
                      split=plain.get("eval", {}).get("split", "test"))

    text = report.to_text()
    verdict = report.verdict()
    print("\n" + text + "\n\nVERDICT: " + verdict + "\n")
    report.save(out_dir / "report.json")
    (out_dir / "verdict.txt").write_text(verdict + "\n")
    log.info("artifacts written to %s", out_dir)


if __name__ == "__main__":
    if "--list" in sys.argv:
        _list_components()
    else:
        main()
