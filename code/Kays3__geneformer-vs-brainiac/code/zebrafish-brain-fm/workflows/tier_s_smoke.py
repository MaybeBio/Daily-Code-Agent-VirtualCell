"""Tier S smoke test as a Neuro-Workflow graph: SyntheticCells -> EncoderTrain.

Synthetic cells only. It checks that the smallest model trains and gives timing.
It says nothing about zebrafish. Run from the zebrafish-brain-fm folder:

    uv run python workflows/tier_s_smoke.py              # tier S: 10,000 cells, 2,048 genes per cell
    uv run python workflows/tier_s_smoke.py --steps 300  # fewer steps, to fit the time ceiling
    uv run python workflows/tier_s_smoke.py --tiny       # seconds on a laptop CPU
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nodes"))

from neuroworkflow import WorkflowBuilder  # noqa: E402

from encoder_train import EncoderTrainNode  # noqa: E402
from synthetic_cells import SyntheticCellsNode  # noqa: E402

TITLE = "Tier S smoke test on synthetic cells"


def option(argv, flag, default):
    return argv[argv.index(flag) + 1] if flag in argv else default


def build(argv=()):
    """Build the graph without running it."""
    argv = list(argv)
    tiny = "--tiny" in argv
    name = "tier_s_tiny" if tiny else "tier_s_smoke"
    cells, train = SyntheticCellsNode("Cells"), EncoderTrainNode("Train")
    cells.configure(out_path=str(ROOT / "data" / "derived" / f"{name}_cells.pt"))
    train.configure(out_dir=str(ROOT / "runs" / name))
    if tiny:
        cells.configure(n_cells=200, vocab_size=500, seq_len=64)
        train.configure(steps=30, batch_size=16, ceiling_minutes=1.0)
    if "--steps" in argv:
        train.configure(steps=int(option(argv, "--steps", 600)))
    builder = WorkflowBuilder(f"zfbrain_{name}", context={"species": "none (synthetic cells)"})
    builder.add_node(cells)
    builder.add_node(train)
    builder.connect("Cells", "data_path", "Train", "data_path")
    return builder, {"train": train}


def main():
    builder, nodes = build(sys.argv)
    success = builder.build().execute()
    print("\n--- tier S smoke test ---")
    print("workflow success:", success)
    print(json.dumps(nodes["train"].get_output("metrics"), indent=2))


if __name__ == "__main__":
    main()
