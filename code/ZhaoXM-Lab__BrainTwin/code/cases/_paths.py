"""Point case scripts at the BrainTwin code and the reserved data directories."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "braintwin"
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "checkpoints"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
