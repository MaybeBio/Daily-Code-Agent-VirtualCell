#!/usr/bin/env python3
"""Turn FreeSurfer stats into the structured sample MRIParsingSession reads.

Two inputs produce the same object:

- a subject directory that contains stats/aseg.stats, lh.aparc.stats, rh.aparc.stats
- the wide tables already used for ADNI (aseg_features.csv and lh/rh aparc tables)

Regional names use underscores, matching volume.regional.Left_Hippocampus.Volume.
FreeSurfer's later name Left-Thalamus-Proper is stored as Left_Thalamus, which is
the key already used by the ADNI tables. Cortical region names are taken from the
parcellation file, so Desikan aparc.stats and DKT tables both fit.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

# FreeSurfer renamed Thalamus to Thalamus-Proper. The existing interface uses Thalamus.
STRUCTURE_ALIASES = {
    "Left-Thalamus-Proper": "Left-Thalamus",
    "Right-Thalamus-Proper": "Right-Thalamus",
}

ASEG_ROW_METRICS = {
    "NVoxels": "NVoxels",
    "Volume_mm3": "Volume",
    "normMean": "normMean",
    "normStdDev": "normStdDev",
    "normMin": "normMin",
    "normMax": "normMax",
    "normRange": "normRange",
}

SURFACE_ROW_METRICS = (
    "NumVert",
    "SurfArea",
    "GrayVol",
    "ThickAvg",
    "ThickStd",
    "MeanCurv",
    "GausCurv",
    "FoldInd",
    "CurvInd",
)


def region_key(struct_name: str) -> str:
    canonical = STRUCTURE_ALIASES.get(struct_name, struct_name)
    return canonical.replace("-", "_")


def _as_float(value: Any) -> float:
    if value is None or value == "":
        raise ValueError("empty FreeSurfer value")
    return float(value)


def _parse_stats(path: Path) -> tuple[list[tuple[str, float]], list[dict[str, str]]]:
    measures: list[tuple[str, float]] = []
    rows: list[dict[str, str]] = []
    headers: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("# Measure"):
            parts = [part.strip() for part in line[len("# Measure") :].split(",")]
            if len(parts) >= 3:
                measures.append((parts[1], _as_float(parts[-2])))
            continue
        if line.startswith("# ColHeaders"):
            headers = line.split()[2:]
            continue
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if headers and len(parts) >= len(headers):
            rows.append(dict(zip(headers, parts[: len(headers)])))
        elif len(parts) >= 10:
            names = ("StructName",) + SURFACE_ROW_METRICS
            rows.append(dict(zip(names, parts[: len(names)])))
    return measures, rows


def _stats_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "stats" / "aseg.stats").is_file():
        return path / "stats"
    if (path / "aseg.stats").is_file():
        return path
    raise FileNotFoundError(f"No aseg.stats under {path}")


def _empty_surface_region() -> dict[str, dict]:
    return {"left_hemisphere": {}, "right_hemisphere": {}}


def subject_from_stats(subject_dir: str | Path) -> dict[str, Any]:
    """Read one FreeSurfer subject and return the MRI session object."""
    stats = _stats_dir(Path(subject_dir))
    required = ("aseg.stats", "lh.aparc.stats", "rh.aparc.stats")
    missing = [name for name in required if not (stats / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{stats} is missing {', '.join(missing)}")

    aseg_measures, aseg_rows = _parse_stats(stats / "aseg.stats")
    volume_regional: dict[str, dict[str, float]] = {}
    for row in aseg_rows:
        struct = row.get("StructName")
        if not struct:
            continue
        metrics = {}
        for source, dest in ASEG_ROW_METRICS.items():
            if source in row:
                metrics[dest] = _as_float(row[source])
        volume_regional[region_key(struct)] = metrics

    surface_global: dict[str, dict[str, float]] = {}
    surface_regional: dict[str, dict[str, dict[str, float]]] = {}
    for side, filename in (("left_hemisphere", "lh.aparc.stats"), ("right_hemisphere", "rh.aparc.stats")):
        measures, rows = _parse_stats(stats / filename)
        for short_name, value in measures:
            surface_global.setdefault(short_name, {})[side] = value
        for row in rows:
            struct = row.get("StructName")
            if not struct:
                continue
            bucket = surface_regional.setdefault(region_key(struct), _empty_surface_region())
            for metric in SURFACE_ROW_METRICS:
                if metric in row:
                    bucket[side][metric] = _as_float(row[metric])

    return {
        "volume": {
            "global": {short_name: value for short_name, value in aseg_measures},
            "regional": volume_regional,
        },
        "surface": {
            "global": surface_global,
            "regional": surface_regional,
        },
    }


def _split_aseg_column(column: str) -> tuple[str, str] | None:
    if column.endswith("_Volume_mm3"):
        return column[: -len("_Volume_mm3")], "Volume"
    for metric in ("NVoxels", "normMean", "normStdDev", "normMin", "normMax", "normRange"):
        suffix = f"_{metric}"
        if column.endswith(suffix):
            return column[: -len(suffix)], metric
    return None


def _split_aparc_column(column: str) -> tuple[str, str] | None:
    for metric in SURFACE_ROW_METRICS:
        suffix = f"_{metric}"
        if column.endswith(suffix):
            return column[: -len(suffix)], metric
    return None


def subject_from_feature_row(
    aseg_row: Mapping[str, Any],
    lh_row: Mapping[str, Any],
    rh_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply process_ADNI.ipynb to one already-wide subject row."""
    aseg_columns = list(aseg_row)
    volume_global = {}
    for column in aseg_columns[3:22]:
        volume_global[column] = _as_float(aseg_row[column])
    volume_regional: dict[str, dict[str, float]] = {}
    for column in aseg_columns[22:]:
        parsed = _split_aseg_column(column)
        if parsed is None:
            continue
        region, metric = parsed
        volume_regional.setdefault(region_key(region), {})[metric] = _as_float(aseg_row[column])

    lh_columns = list(lh_row)
    surface_global: dict[str, dict[str, float]] = {}
    for column in lh_columns[2:12]:
        surface_global[column] = {
            "left_hemisphere": _as_float(lh_row[column]),
            "right_hemisphere": _as_float(rh_row[column]),
        }
    surface_regional: dict[str, dict[str, dict[str, float]]] = {}
    for column in lh_columns[12:]:
        parsed = _split_aparc_column(column)
        if parsed is None:
            continue
        region, metric = parsed
        bucket = surface_regional.setdefault(region_key(region), _empty_surface_region())
        bucket["left_hemisphere"][metric] = _as_float(lh_row[column])
        if column in rh_row:
            bucket["right_hemisphere"][metric] = _as_float(rh_row[column])
    return {
        "volume": {"global": volume_global, "regional": volume_regional},
        "surface": {"global": surface_global, "regional": surface_regional},
    }


def _read_csv_row(path: Path, subject_id: str) -> dict[str, str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        wanted = {subject_id, str(subject_id)}
        id_column = "Subject_ID" if reader.fieldnames and "Subject_ID" in reader.fieldnames else "ID"
        for row in reader:
            if str(row.get(id_column, "")) in wanted:
                return row
    raise KeyError(f"{subject_id} is not in {path}")


def subject_from_feature_csvs(
    aseg_csv: str | Path,
    lh_csv: str | Path,
    rh_csv: str | Path,
    subject_id: str,
) -> dict[str, Any]:
    return subject_from_feature_row(
        _read_csv_row(Path(aseg_csv), subject_id),
        _read_csv_row(Path(lh_csv), subject_id),
        _read_csv_row(Path(rh_csv), subject_id),
    )


def iter_stats_subjects(dataset_root: str | Path) -> Iterable[tuple[str, dict[str, Any]]]:
    root = Path(dataset_root).expanduser().resolve()
    for subject in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            _stats_dir(subject)
        except FileNotFoundError:
            continue
        yield subject.name, subject_from_stats(subject)


def open_mri_session(subject_dir: str | Path, config: Any = None):
    """Build an MRIParsingSession from a FreeSurfer subject directory."""
    from mri_session import MRIParsingSession

    sample = subject_from_stats(subject_dir)
    if config is None:
        return MRIParsingSession(sample)
    return MRIParsingSession(sample, config=config)
