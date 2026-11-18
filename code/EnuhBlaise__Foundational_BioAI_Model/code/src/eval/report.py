"""Side-by-side comparison of a model against its controls.

The report refuses to present a model score alone. Its output format puts
baselines on the same rows as the model, marks the best value per metric, and
states the split regime, because a number from a random split and a number
from a donor-held-out split are not comparable quantities.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["ComparisonReport"]

_HIGHER_WORSE = {"pcr_batch", "library_size_r2"}

#: compact column headers so a full panel fits in a terminal
_ABBREV = {
    "probe_macro_f1": "probeF1", "probe_accuracy": "probeAcc",
    "knn_macro_f1": "knnF1", "knn_accuracy": "knnAcc",
    "celltype_asw": "ctASW", "knn_mixing": "iLISI",
    "graph_connectivity": "graphConn", "pcr_batch": "PCRbatch",
    "library_size_r2": "libR2", "bio_score": "BIO",
    "batch_score": "BATCH", "overall": "OVERALL",
}


@dataclass
class ComparisonReport:
    split_regime: str
    group_key: Optional[str]
    is_synthetic: bool
    provenance: str
    n_train: int = 0
    n_test: int = 0
    n_test_groups: int = 0
    rows: Dict[str, Dict[str, float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    leakage: Optional[str] = None

    def add(self, name: str, metrics: Dict[str, float]) -> None:
        self.rows[name] = {k: float(v) for k, v in metrics.items()}

    # -- rendering -------------------------------------------------------
    def _columns(self) -> List[str]:
        preferred = [
            "probe_macro_f1", "knn_macro_f1", "ari", "nmi", "celltype_asw",
            "knn_mixing", "kbet", "batch_asw", "graph_connectivity",
            "pcr_batch", "library_size_r2", "bio_score", "batch_score", "overall",
        ]
        present = {
            k for r in self.rows.values() for k in r
            if not k.endswith(("_lo", "_hi", "n_test"))
        }
        cols = [c for c in preferred if c in present]
        cols += sorted(present - set(cols))
        return cols

    def to_text(self, width: int = 10) -> str:
        cols = self._columns()
        name_w = max([12] + [len(n) for n in self.rows])
        head = "model".ljust(name_w) + "".join(
            _ABBREV.get(c, c)[:width].rjust(width + 1) for c in cols
        )
        lines = [head, "-" * len(head)]

        best: Dict[str, float] = {}
        for c in cols:
            vals = [r[c] for r in self.rows.values() if c in r and np.isfinite(r[c])]
            if vals:
                best[c] = min(vals) if c in _HIGHER_WORSE else max(vals)

        for name, r in self.rows.items():
            line = name.ljust(name_w)
            for c in cols:
                v = r.get(c, float("nan"))
                s = "  n/a" if not np.isfinite(v) else f"{v:.3f}"
                if c in best and np.isfinite(v) and abs(v - best[c]) < 1e-9:
                    s += "*"
                else:
                    s += " "
                line += s.rjust(width + 1)
            lines.append(line)

        lines.append("")
        lines.append(f"split regime   : {self.split_regime}" + (
            f" (held-out '{self.group_key}')" if self.group_key else ""))
        lines.append(f"data           : {self.provenance}")
        if self.is_synthetic:
            lines.append(
                "                 *** SYNTHETIC DATA -- these numbers "
                "characterize the simulator, not biology ***"
            )
        lines.append(
            f"cells          : train={self.n_train} test={self.n_test}"
            + (f" over {self.n_test_groups} held-out group(s)" if self.n_test_groups else "")
        )
        if self.n_test_groups and self.n_test_groups < 3:
            lines.append(
                "                 WARNING: fewer than 3 held-out groups; the "
                "cell-level CIs below understate real uncertainty badly."
            )
        lines.append("* = best in column.  pcr_batch and library_size_r2: lower is better.")
        if self.leakage:
            lines.append("")
            lines.append(self.leakage)
        for n in self.notes:
            lines.append(f"note: {n}")
        return "\n".join(lines)

    def verdict(self, model_key: str = "foundation_model",
                metric: str = "probe_macro_f1") -> str:
        """State plainly whether the model beat its controls."""
        if model_key not in self.rows or metric not in self.rows[model_key]:
            return "no verdict: model row or metric missing"
        mv = self.rows[model_key][metric]
        others = {
            k: v[metric] for k, v in self.rows.items()
            if k != model_key and metric in v and np.isfinite(v[metric])
        }
        if not others:
            return "no verdict: no baselines were run"
        best_name = max(others, key=others.get)
        bv = others[best_name]
        lo = self.rows[model_key].get("probe_macro_f1_lo")
        hi_b = None
        delta = mv - bv
        if delta > 0 and lo is not None and np.isfinite(lo) and lo > bv:
            return (
                f"model beats best baseline ({best_name}) on {metric}: "
                f"{mv:.3f} vs {bv:.3f} (+{delta:.3f}); model CI excludes it."
            )
        if delta > 0:
            return (
                f"model is nominally ahead of {best_name} on {metric} "
                f"({mv:.3f} vs {bv:.3f}, +{delta:.3f}) but the interval "
                "overlaps -- not a demonstrated improvement."
            )
        return (
            f"model does NOT beat {best_name} on {metric}: "
            f"{mv:.3f} vs {bv:.3f} ({delta:+.3f}). "
            "This is the common outcome in published benchmarks at this scale."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split_regime": self.split_regime, "group_key": self.group_key,
            "is_synthetic": self.is_synthetic, "provenance": self.provenance,
            "n_train": self.n_train, "n_test": self.n_test,
            "n_test_groups": self.n_test_groups, "rows": self.rows,
            "notes": self.notes, "leakage": self.leakage,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        p.with_suffix(".txt").write_text(self.to_text())
        return p
