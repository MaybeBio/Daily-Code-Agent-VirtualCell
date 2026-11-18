"""The model's calibration file (``CALIBRATION.json``): temperatures and certified thresholds.

Fitted on the model's own readouts with both option orders (the order given and the reversed order, averaged), which
is how the engine reads every question. Temperatures are per question type and, for ``choice``, per option-count
bucket; a ``max_error`` target (0.05 or 0.01) maps to a confidence threshold certified on held-out data, per type or
per category.
"""
import json
from pathlib import Path

import numpy as np

BUCKETS = (("2-10", 2, 10), ("11-26", 11, 26), ("27-255", 27, 255))
CATEGORY_BY_TYPE = {"choice": "omics choice", "noul": "omics yes/no", "score": "omics score"}


def bucket(k: int) -> str:
    return next(name for name, lo, hi in BUCKETS if lo <= k <= hi)


def scale(p, t: float) -> np.ndarray:
    """Temperature-scaled probabilities."""
    z = np.log(np.clip(np.asarray(p, float), 1e-12, 1)) / t
    z = np.exp(z - z.max())
    return z / z.sum()


class Calibration:
    def __init__(self, path):
        self.path = Path(path)
        c = json.loads(self.path.read_text())
        self.raw = c
        self.orders = int(c.get("orders", 2))
        if self.orders != 2:
            raise SystemExit(f"{self.path} was fitted with {self.orders} option order(s); the engine reads 2")
        self.T_prim = c.get("temperature_per_prim", {})
        self.T_bucket = c.get("temperature_per_bucket", {})
        self.thresholds = c.get("thresholds", {})
        self.categories = c.get("category_thresholds", {})

    def temperature(self, prim: str, k: int) -> float:
        return self.T_bucket.get(prim, {}).get(bucket(k), self.T_prim.get(prim, 1.0))

    def apply(self, prim: str, p) -> np.ndarray:
        return scale(np.asarray(p, float), self.temperature(prim, len(p)))

    def category_names(self) -> list[str]:
        return sorted({c for v in self.categories.values() for c in v})

    def threshold(self, prim: str, max_error: float, category: str | None = None):
        """(threshold, source, measured test error and coverage) for a target error, or None if not certified."""
        key = f"{max_error:g}" if f"{max_error:g}" in self.thresholds else str(max_error)
        cat = (self.categories.get(key) or {}).get(category) if category else None
        if cat and cat.get("confidence") is not None:
            return dict(threshold=float(cat["confidence"]), source=f"category: {category}",
                        test_error=cat.get("test_error"), test_coverage=cat.get("test_coverage"),
                        holds_on_test=cat.get("holding"))
        t = ((self.thresholds.get(key) or {}).get("per_type") or {}).get(prim)
        if t and t.get("confidence") is not None:
            return dict(threshold=float(t["confidence"]), source=f"type: {prim}", test_error=t.get("test_error"),
                        test_coverage=t.get("test_coverage"),
                        holds_on_test=None if t.get("test_error") is None else t["test_error"] <= max_error)
        return None

    def summary(self) -> dict:
        return dict(file=self.path.name, orders=self.orders, temperature_per_prim=self.T_prim,
                    temperature_per_bucket=self.T_bucket, targets=sorted(self.thresholds),
                    categories=self.category_names())
