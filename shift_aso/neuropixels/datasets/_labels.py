"""Scoring labels for both Neuropixels cases.

Label contract (public): every native GT ``st``/``cl`` pair within the first
300 s carries ``unit_id``, ``is_good`` (unit <= 599), ``is_background``
(600 <= unit <= 1199), ``geometry_um``, ``local_to_original_AP``,
``sample_index`` and ``estimated_local_channel``. Scoring groups are explicit.

Reference semantics (accepted, do not change):
  * nodrift: the packaged frozen ``_unit_to_main_channel.json`` map
    (scorer-only; original 30-60 s extraction support).
  * meddrift: its OWN dynamic GT template reference (10 s blocks). The
    nodrift static map must be refused for meddrift, and the dynamic template
    builder must be refused for nodrift.
  * GT is exposed to scoring references only (including TEST); it never feeds
    detector CAL/DEV inputs or settings. The ``drift`` field of the params
    NPZ is never read (it is a waveform-bank index, not micrometers).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._registry import CaseSpec

LABEL_FIELDS = (
    "sample_index",
    "unit_id",
    "estimated_local_channel",
    "is_good",
    "is_background",
    "scoring_group",
    "geometry_um",
    "local_to_original_AP",
)


class LabelContractError(RuntimeError):
    """Raised when GT labels or reference channels violate the contract."""


def load_gt_times_units(params_npz, *, stop: int, n_units: int):
    """Load native ``st``/``cl``, validate, and restrict to the first ``stop`` samples."""
    if "st" not in params_npz or "cl" not in params_npz:
        raise LabelContractError("params NPZ lacks native 'st'/'cl' arrays")
    t = np.asarray(params_npz["st"], np.int64).ravel()
    u = np.asarray(params_npz["cl"], np.int64).ravel()
    if len(t) != len(u):
        raise LabelContractError("st and cl have different lengths")
    if len(t) == 0:
        raise LabelContractError("params NPZ has no GT spikes")
    if np.any(np.diff(t) < 0):
        raise LabelContractError("native st must be sorted")
    if t.min() < 0:
        raise LabelContractError("native st contains negative sample indices")
    if u.min() < 0 or u.max() >= n_units:
        raise LabelContractError(f"native cl must lie in [0, {n_units})")
    keep = t < stop
    if not np.any(keep):
        raise LabelContractError("no GT spikes within the benchmark duration")
    return t[keep], u[keep]


def frozen_unit_main_channels(n_active_channels: int):
    """Return (mapping, source_note) of the packaged frozen unit map (nodrift only)."""
    path = Path(__file__).resolve().parents[1] / "_unit_to_main_channel.json"
    data = json.loads(path.read_text())
    mapping = np.asarray(data["unit_to_main_channel"], np.int64)
    if mapping.ndim != 1 or mapping.size != 1200:
        raise LabelContractError(f"frozen unit map must have exactly 1200 entries, got {mapping.shape}")
    if mapping.min() < 0 or mapping.max() >= n_active_channels:
        raise LabelContractError(
            f"frozen unit map references local channels [0, {n_active_channels}); got max {mapping.max()}"
        )
    return mapping, data.get("source")


def reference_kind_for_case(case: str) -> str:
    if case == "nodrift":
        return "frozen_unit_map"
    if case == "meddrift":
        return "gt_template_10s_blocks"
    raise LabelContractError(f"unknown case {case!r}")


def static_map_channels(case: str, n_active_channels: int):
    """Frozen static reference channels; refused for meddrift (case-bound references)."""
    if case != "nodrift":
        raise LabelContractError(
            "the frozen static unit map is the nodrift reference only; meddrift requires "
            "its own dynamic GT template reference (gt_template_10s_blocks)"
        )
    return frozen_unit_main_channels(n_active_channels)


def template_reference_allowed(case: str) -> None:
    """Guard the dynamic template builder; refused for nodrift."""
    if case != "meddrift":
        raise LabelContractError(
            "the dynamic GT template reference is the meddrift reference only; nodrift uses "
            "the packaged frozen_unit_map"
        )


def unit_group_masks(units, good_unit_ids, background_unit_ids):
    units = np.asarray(units, np.int64)
    is_good = (units >= good_unit_ids[0]) & (units <= good_unit_ids[1])
    is_background = (units >= background_unit_ids[0]) & (units <= background_unit_ids[1])
    if np.any(~(is_good | is_background)):
        raise LabelContractError("unit ids outside both scoring groups")
    return is_good, is_background


def build_labels(*, sample_index, unit_id, estimated_local_channel, geometry_um,
                 local_to_original_AP, good_unit_ids, background_unit_ids):
    """Assemble the public scoring label set shared by both cases."""
    t = np.asarray(sample_index, np.int64)
    u = np.asarray(unit_id, np.int64)
    channels = np.asarray(estimated_local_channel, np.int64)
    geometry = np.asarray(geometry_um, np.float64)
    ap_ids = np.asarray(local_to_original_AP, np.int64)
    if not (len(t) == len(u) == len(channels)):
        raise LabelContractError("label arrays must have equal length")
    if geometry.ndim != 2 or geometry.shape[1] != 2 or ap_ids.shape != geometry.shape[:1]:
        raise LabelContractError("geometry_um/local_to_original_AP disagree with the AP map")
    if channels.min() < -1 or channels.max() >= len(ap_ids):
        raise LabelContractError("estimated_local_channel must be -1 (unknown) or an active channel index")
    is_good, is_background = unit_group_masks(u, good_unit_ids, background_unit_ids)
    group = np.where(is_good, 0, 1).astype(np.int8)
    return {
        "sample_index": t,
        "unit_id": u,
        "estimated_local_channel": channels,
        "is_good": is_good,
        "is_background": is_background,
        "scoring_group": group,
        "geometry_um": geometry,
        "local_to_original_AP": ap_ids,
    }


def labels_metadata(spec: CaseSpec, reference_kind: str, provenance: dict) -> dict:
    """Explicit scoring-group and provenance metadata written next to labels."""
    return {
        "schema": "shift_aso_np_labels_v1",
        "case": spec.case,
        "duration_s": spec.duration_s,
        "fs": spec.fs,
        "stop_sample": spec.stop_sample,
        "scoring_groups": {
            "good": {"unit_ids": list(spec.good_unit_ids), "group_id": 0},
            "background": {"unit_ids": list(spec.background_unit_ids), "group_id": 1},
        },
        "reference_kind": reference_kind,
        "label_exposure": (
            "All GT times/IDs (including TEST) are used only to build scoring references; "
            "they are never detector CAL/DEV inputs or settings."
        ),
        "provenance": provenance,
    }
