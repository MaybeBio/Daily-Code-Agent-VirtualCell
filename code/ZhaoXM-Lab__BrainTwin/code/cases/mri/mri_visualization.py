"""
Highlight selected MRI features on ggseg atlases.

Uses python-ggseg API:
https://github.com/ggsegverse/python-ggseg

- Cortical ROIs  -> ggseg.plot_dk  (Desikan-Killiany)
- Subcortical ROIs -> ggseg.plot_aseg (FreeSurfer aseg)

Regions are shown as binary highlights (same color), not continuous feature values.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")

import ggseg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# FreeSurfer aseg name in structured JSON -> ggseg aseg key
ASEG_NAME_MAP: dict[str, str] = {
    "Left_Lateral_Ventricle": "Left-Lateral-Ventricle",
    "Right_Lateral_Ventricle": "Right-Lateral-Ventricle",
    "Left_Inf_Lat_Vent": "Left-Inf-Lat-Vent",
    "Right_Inf_Lat_Vent": "Right-Inf-Lat-Vent",
    "3rd_Ventricle": "3rd-Ventricle",
    "4th_Ventricle": "4th-Ventricle",
    "5th_Ventricle": "5th-Ventricle",
    "Left_Thalamus": "Left-Thalamus",
    "Right_Thalamus": "Right-Thalamus",
    "Left_Caudate": "Left-Caudate",
    "Right_Caudate": "Right-Caudate",
    "Left_Putamen": "Left-Putamen",
    "Right_Putamen": "Right-Putamen",
    "Left_Pallidum": "Left-Pallidum",
    "Right_Pallidum": "Right-Pallidum",
    "Left_Hippocampus": "Left-Hippocampus",
    "Right_Hippocampus": "Right-Hippocampus",
    "Left_Amygdala": "Left-Amygdala",
    "Right_Amygdala": "Right-Amygdala",
    "Left_Accumbens_area": "Left-Accumbens-area",
    "Right_Accumbens_area": "Right-Accumbens-area",
    "Left_VentralDC": "Left-VentralDC",
    "Right_VentralDC": "Right-VentralDC",
    "Brain_Stem": "Brain-Stem",
    "CC_Posterior": "CC_Posterior",
    "CC_Mid_Posterior": "CC_Mid_Posterior",
    "CC_Central": "CC_Central",
    "CC_Mid_Anterior": "CC_Mid_Anterior",
    "CC_Anterior": "CC_Anterior",
}

# Bilateral JSON fields merged into single ggseg aseg region
ASEG_BILATERAL_MERGE: dict[str, tuple[str, ...]] = {
    "Cerebellum-Cortex": ("Left_Cerebellum_Cortex", "Right_Cerebellum_Cortex"),
    "Cerebellum-White-Matter": (
        "Left_Cerebellum_White_Matter",
        "Right_Cerebellum_White_Matter",
    ),
}

# When subcortical focus is selected, optionally surface related cortex (and vice versa)
RELATED_DK_FOR_ASEG: dict[str, list[str]] = {
    "Left-Hippocampus": ["parahippocampal_left", "entorhinal_left"],
    "Right-Hippocampus": ["parahippocampal_right", "entorhinal_right"],
    "Left-Amygdala": ["entorhinal_left", "fusiform_left"],
    "Right-Amygdala": ["entorhinal_right", "fusiform_right"],
    "Left-Thalamus": ["posteriorcingulate_left", "precuneus_left"],
    "Right-Thalamus": ["posteriorcingulate_right", "precuneus_right"],
}

RELATED_ASEG_FOR_DK: dict[str, list[str]] = {
    "entorhinal_left": ["Left-Hippocampus"],
    "entorhinal_right": ["Right-Hippocampus"],
    "parahippocampal_left": ["Left-Hippocampus"],
    "parahippocampal_right": ["Right-Hippocampus"],
    "fusiform_left": ["Left-Amygdala"],
    "fusiform_right": ["Right-Amygdala"],
}

CORTICAL_FIGSIZE = (10, 10)
SUBCORTICAL_FIGSIZE = (12, 5)
HIGHLIGHT_VALUE = 1.0


def _ggseg_data_dir(atlas: str) -> str:
    return os.path.join(os.path.dirname(ggseg.__file__), "data", atlas)


def _load_valid_keys(atlas: str) -> set[str]:
    directory = _ggseg_data_dir(atlas)
    if not os.path.isdir(directory):
        return set()
    return {
        name
        for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name))
        and name not in {"Coronal", "Sagittal", "NA", "NA_left", "NA_right"}
        and not name.startswith("lateral_")
        and not name.startswith("medial_")
    }


VALID_ASEG_KEYS = _load_valid_keys("aseg")
VALID_DK_KEYS = _load_valid_keys("dk")


def _format_region_label(name: str) -> str:
    return name.replace("_", " ")


def _extract_numeric(sample_info: dict[str, Any], feature_path: str) -> float | None:
    try:
        node: Any = sample_info
        parts = feature_path.split(".")
        for part in parts:
            node = node[part]
        if isinstance(node, dict):
            metric = parts[-1]
            if metric in node and isinstance(node[metric], (int, float)):
                return float(node[metric])
            for value in node.values():
                if isinstance(value, (int, float)):
                    return float(value)
            return None
        if isinstance(node, (int, float)):
            return float(node)
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _resolve_aseg_key(region_name: str) -> str | None:
    if region_name in ASEG_BILATERAL_MERGE:
        return region_name if region_name in VALID_ASEG_KEYS else None

    mapped = ASEG_NAME_MAP.get(region_name)
    if mapped and mapped in VALID_ASEG_KEYS:
        return mapped

    candidate = region_name.replace("_", "-")
    if candidate in VALID_ASEG_KEYS:
        return candidate
    return None


def _resolve_dk_key(region: str, hemi_token: str) -> str | None:
    side = "left" if hemi_token.startswith("left") else "right"
    dk_key = f"{region}_{side}"
    return dk_key if dk_key in VALID_DK_KEYS else None


def _merge_bilateral_aseg_values(
    sample_info: dict[str, Any],
    ggseg_key: str,
    source_regions: tuple[str, ...],
) -> float | None:
    values: list[float] = []
    for region in source_regions:
        value = _extract_numeric(
            sample_info, f"volume.regional.{region}.Volume"
        )
        if value is not None:
            values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def _collect_feature_mappings(
    selected_features: list[str],
    sample_info: dict[str, Any] | None,
) -> tuple[
    dict[str, float],
    dict[str, float],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    dk_data: dict[str, float] = {}
    aseg_data: dict[str, float] = {}
    cortical_meta: list[dict[str, Any]] = []
    subcortical_meta: list[dict[str, Any]] = []
    unmapped: list[str] = []

    for feature in selected_features:
        parts = feature.split(".")
        value = _extract_numeric(sample_info, feature) if sample_info else None

        if len(parts) >= 4 and parts[0] == "volume" and parts[1] == "regional":
            region = parts[2]
            ggseg_key = _resolve_aseg_key(region)
            if ggseg_key:
                aseg_data[ggseg_key] = HIGHLIGHT_VALUE
                subcortical_meta.append(
                    {
                        "label": _format_region_label(region),
                        "feature": feature,
                        "value": value,
                        "related": False,
                    }
                )
            else:
                merged = False
                for ggseg_key, sources in ASEG_BILATERAL_MERGE.items():
                    if region in sources and ggseg_key in VALID_ASEG_KEYS:
                        merged_value = (
                            _merge_bilateral_aseg_values(sample_info or {}, ggseg_key, sources)
                            if sample_info
                            else None
                        )
                        plot_value = HIGHLIGHT_VALUE
                        aseg_data[ggseg_key] = plot_value
                        subcortical_meta.append(
                            {
                                "label": _format_region_label(ggseg_key.replace("-", " ")),
                                "feature": feature,
                                "value": merged_value,
                                "related": False,
                            }
                        )
                        merged = True
                        break
                if not merged:
                    unmapped.append(feature)
            continue

        if len(parts) >= 5 and parts[0] == "surface" and parts[1] == "regional":
            region = parts[2]
            dk_key = _resolve_dk_key(region, parts[3])
            if dk_key:
                dk_data[dk_key] = HIGHLIGHT_VALUE
                side = "left" if parts[3].startswith("left") else "right"
                cortical_meta.append(
                    {
                        "label": f"{_format_region_label(region)} ({side})",
                        "feature": feature,
                        "value": value,
                        "related": False,
                    }
                )
            else:
                unmapped.append(feature)
            continue

        unmapped.append(feature)

    if sample_info:
        _add_related_regions(
            dk_data,
            aseg_data,
            cortical_meta,
            subcortical_meta,
            sample_info,
        )

    return dk_data, aseg_data, cortical_meta, subcortical_meta, unmapped


def _add_related_regions(
    dk_data: dict[str, float],
    aseg_data: dict[str, float],
    cortical_meta: list[dict[str, Any]],
    subcortical_meta: list[dict[str, Any]],
    sample_info: dict[str, Any],
) -> None:
    """Bridge cortical/subcortical views when the selected features span only one atlas."""

    for aseg_key, dk_keys in RELATED_DK_FOR_ASEG.items():
        if aseg_key not in aseg_data:
            continue
        for dk_key in dk_keys:
            if dk_key in dk_data or dk_key not in VALID_DK_KEYS:
                continue
            region, side = dk_key.rsplit("_", 1)
            hemi = "left_hemisphere" if side == "left" else "right_hemisphere"
            metric = "ThickAvg"
            feature = f"surface.regional.{region}.{hemi}.{metric}"
            value = _extract_numeric(sample_info, feature)
            if value is None:
                continue
            dk_data[dk_key] = HIGHLIGHT_VALUE
            cortical_meta.append(
                {
                    "label": f"{_format_region_label(region)} ({side}, related)",
                    "feature": feature,
                    "value": value,
                    "related": True,
                }
            )

    for dk_key, aseg_keys in RELATED_ASEG_FOR_DK.items():
        if dk_key not in dk_data:
            continue
        for aseg_key in aseg_keys:
            if aseg_key in aseg_data or aseg_key not in VALID_ASEG_KEYS:
                continue
            inverse = {v: k for k, v in ASEG_NAME_MAP.items()}
            json_region = inverse.get(aseg_key)
            if not json_region:
                continue
            feature = f"volume.regional.{json_region}.Volume"
            value = _extract_numeric(sample_info, feature)
            if value is None:
                continue
            aseg_data[aseg_key] = HIGHLIGHT_VALUE
            subcortical_meta.append(
                {
                    "label": f"{_format_region_label(json_region)} (related)",
                    "feature": feature,
                    "value": value,
                    "related": True,
                }
            )


def _figure_to_data_url(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, facecolor="white")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def _plot_ggseg(
    plot_fn,
    data: dict[str, float],
    *,
    title: str,
    figsize: tuple[float, float],
) -> str:
    if not data:
        raise ValueError("No plottable regions")

    plot_fn(
        data,
        cmap="Reds",
        background="white",
        edgecolor="#666666",
        bordercolor="#dddddd",
        figsize=figsize,
        vminmax=[0, HIGHLIGHT_VALUE],
        title=title,
        ylabel="",
        fontsize=12,
    )
    fig = plt.gcf()
    if len(fig.axes) > 1:
        for ax in list(fig.axes[1:]):
            ax.remove()
    return _figure_to_data_url(fig)


def render_highlight_images(
    dk_data: dict[str, float],
    aseg_data: dict[str, float],
) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []

    if dk_data:
        images.append(
            {
                "type": "dk",
                "label": "Cortical regions (Desikan-Killiany atlas)",
                "data_url": _plot_ggseg(
                    ggseg.plot_dk,
                    dk_data,
                    title="Cortical regions of interest",
                    figsize=CORTICAL_FIGSIZE,
                ),
            }
        )

    if aseg_data:
        images.append(
            {
                "type": "aseg",
                "label": "Subcortical regions (FreeSurfer aseg)",
                "data_url": _plot_ggseg(
                    ggseg.plot_aseg,
                    aseg_data,
                    title="Subcortical regions of interest",
                    figsize=SUBCORTICAL_FIGSIZE,
                ),
            }
        )

    return images


def build_visualization(
    selected_features: list[str],
    sample_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dk_data, aseg_data, cortical_meta, subcortical_meta, unmapped = _collect_feature_mappings(
        selected_features,
        sample_info,
    )
    images = render_highlight_images(dk_data, aseg_data)

    cortical_labels = [item["label"] for item in cortical_meta]
    subcortical_labels = [item["label"] for item in subcortical_meta]

    return {
        "regions": cortical_labels + subcortical_labels,
        "cortical_regions": cortical_labels,
        "subcortical_regions": subcortical_labels,
        "cortical_details": cortical_meta,
        "subcortical_details": subcortical_meta,
        "unmapped_features": unmapped,
        "images": images,
    }
