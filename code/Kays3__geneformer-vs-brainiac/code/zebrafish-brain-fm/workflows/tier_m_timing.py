"""Tier M timing run as a Neuro-Workflow graph: SyntheticCells -> EncoderTrain.

The tier M model (6 layers, width 256) on synthetic cells. It measures speed and
memory for the GPU plan. It is not the tier M model of the plan, which needs real
cells, and it says nothing about zebrafish. Run from the zebrafish-brain-fm folder:

    uv run python workflows/tier_m_timing.py --steps 12 --ceiling 5 --name calibration
    uv run python workflows/tier_m_timing.py --steps 2500 --ceiling 60
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nodes"))

from neuroworkflow import WorkflowBuilder  # noqa: E402

from encoder_train import EncoderTrainNode  # noqa: E402
from synthetic_cells import SyntheticCellsNode  # noqa: E402

TITLE = "Tier M timing run on synthetic cells"


def option(argv, flag, default):
    return argv[argv.index(flag) + 1] if flag in argv else default


def build(argv=()):
    """Build the graph without running it."""
    argv = list(argv)
    name = "tier_m_" + option(argv, "--name", "timing")
    cells, train = SyntheticCellsNode("Cells"), EncoderTrainNode("Train")
    cells.configure(n_cells=int(option(argv, "--cells", 20000)), out_path=str(ROOT / "data" / "derived" / "tier_m_cells.pt"))
    train.configure(
        layers=6, width=256, heads=4,
        steps=int(option(argv, "--steps", 600)),
        batch_size=int(option(argv, "--batch", 32)),
        ceiling_minutes=float(option(argv, "--ceiling", 60)),
        out_dir=str(ROOT / "runs" / name),
    )
    builder = WorkflowBuilder(f"zfbrain_{name}", context={"species": "none (synthetic cells)"})
    builder.add_node(cells)
    builder.add_node(train)
    builder.connect("Cells", "data_path", "Train", "data_path")
    return builder, {"train": train}


def main():
    builder, nodes = build(sys.argv)
    success = builder.build().execute()
    print("\n--- tier M timing ---")
    print("workflow success:", success)
    print(json.dumps(nodes["train"].get_output("metrics"), indent=2))


if __name__ == "__main__":
    main()
