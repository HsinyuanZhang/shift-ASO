"""Parent-owned dataset registry loading and validation (read-only).

The registry ``shift_aso/datasets_registry.json`` is created and owned by the
parent. This module validates it against the public benchmark contract and
the locked production reference constants; it never rewrites it. Any mismatch
is a reported blocker, not something this package silently adapts to.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ._template_reference import PRODUCTION_REFERENCE

REGISTRY_SCHEMA = "shift_aso_dataset_registry_v1"

# Public benchmark contract (ARCHITECTURE.md); cross-checked, not invented.
CONTRACT_ACTIVE_CHANNELS = 383
CONTRACT_FS_HZ = 30000
CONTRACT_DURATION_S = 300
CONTRACT_GOOD_UNITS = (0, 599)
CONTRACT_BACKGROUND_UNITS = (600, 1199)
CONTRACT_CASES = ("nodrift", "meddrift")


class RegistryError(RuntimeError):
    """Raised when the registry violates the public dataset contract."""


def default_registry_path() -> Path:
    """Return the packaged registry path (``shift_aso/datasets_registry.json``)."""
    return Path(__file__).resolve().parents[2] / "datasets_registry.json"


def load_registry(path=None) -> dict:
    """Load and structurally validate the parent dataset registry."""
    source = Path(path) if path is not None else default_registry_path()
    if not source.is_file():
        raise RegistryError(f"dataset registry missing: {source}")
    try:
        registry = json.loads(source.read_text())
    except (OSError, ValueError) as exc:
        raise RegistryError(f"dataset registry unreadable: {source}: {exc}") from exc
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise RegistryError(f"unsupported registry schema {registry.get('schema')!r}; expected {REGISTRY_SCHEMA!r}")
    section = registry.get("neuropixels")
    if not isinstance(section, dict):
        raise RegistryError("registry lacks a 'neuropixels' section")
    for key in ("fs", "active_channels", "duration_s", "duration_definition",
                "good_unit_ids", "background_unit_ids", "cases"):
        if key not in section:
            raise RegistryError(f"registry neuropixels section lacks {key!r}")
    if section["fs"] != CONTRACT_FS_HZ:
        raise RegistryError(f"registry fs {section['fs']} != benchmark contract {CONTRACT_FS_HZ}")
    if section["active_channels"] != CONTRACT_ACTIVE_CHANNELS:
        raise RegistryError(f"registry active_channels {section['active_channels']} != benchmark contract {CONTRACT_ACTIVE_CHANNELS}")
    if section["duration_s"] != CONTRACT_DURATION_S:
        raise RegistryError(f"registry duration_s {section['duration_s']} != benchmark contract {CONTRACT_DURATION_S}")
    _check_units(section["good_unit_ids"], CONTRACT_GOOD_UNITS, "good_unit_ids")
    _check_units(section["background_unit_ids"], CONTRACT_BACKGROUND_UNITS, "background_unit_ids")
    cases = section["cases"]
    if set(cases) != set(CONTRACT_CASES):
        raise RegistryError(f"registry cases {sorted(cases)} != benchmark contract {sorted(CONTRACT_CASES)}")
    for case in CONTRACT_CASES:
        entry = cases[case]
        if not isinstance(entry, dict):
            raise RegistryError(f"registry case {case!r} is not an object")
        for key in ("archive", "bytes", "file_id", "reference_kind"):
            if key not in entry:
                raise RegistryError(f"registry case {case!r} lacks {key!r}")
        if not isinstance(entry["archive"], str) or not entry["archive"].endswith(".zip"):
            raise RegistryError(f"registry case {case!r} archive must be a .zip filename")
        if not isinstance(entry["bytes"], int) or entry["bytes"] <= 0:
            raise RegistryError(f"registry case {case!r} bytes must be a positive integer")
        if "computed_md5" in entry and entry["computed_md5"] is not None and not isinstance(entry["computed_md5"], str):
            raise RegistryError(f"registry case {case!r} computed_md5 must be a string or null")
    if cases["nodrift"]["reference_kind"] != "frozen_unit_map":
        raise RegistryError("registry nodrift reference_kind must be 'frozen_unit_map'")
    _check_template_case(cases["meddrift"])
    _cross_check_production(section)
    return registry


def _check_units(value, expected, name):
    if (not isinstance(value, list) or len(value) != 2
            or not all(isinstance(x, int) for x in value) or tuple(value) != expected):
        raise RegistryError(f"registry {name} {value!r} != benchmark contract {list(expected)}")


def _check_template_case(entry):
    if entry.get("reference_kind") != "gt_template_10s_blocks":
        raise RegistryError("registry meddrift reference_kind must be 'gt_template_10s_blocks'")
    for key in ("reference_local_channels", "reference_max_snippets_per_parity",
                "reference_min_snippets", "reference_window_samples"):
        if key not in entry:
            raise RegistryError(f"registry meddrift case lacks {key!r}")


def _cross_check_production(section):
    """Lock the registry's meddrift reference parameters to the ported source constants."""
    entry = section["cases"]["meddrift"]
    p = PRODUCTION_REFERENCE
    expected = {
        "reference_local_channels": p["local_channels"],
        "reference_max_snippets_per_parity": p["max_snippets"],
        "reference_min_snippets": p["min_snippets"],
        "reference_window_samples": [p["window_left"], p["window_right"] - 1],
    }
    for key, want in expected.items():
        got = entry.get(key)
        if got != want:
            raise RegistryError(
                f"registry meddrift {key}={got!r} disagrees with the accepted reference "
                f"source constant {want!r}; report this mismatch instead of adapting"
            )
    if section["fs"] * section["duration_s"] != p["stop"]:
        raise RegistryError("registry fs*duration_s disagrees with the production reference stop")


def n_units(section: dict) -> int:
    """Number of simulated units (1200) implied by the unit id groups."""
    return int(section["background_unit_ids"][1]) + 1


@dataclass(frozen=True)
class CaseSpec:
    """Validated per-case view of the registry (case identity + provenance)."""
    case: str
    archive: str
    bytes: int
    registry_md5: str | None
    file_id: int
    reference_kind: str
    fs: int
    active_channels: int
    duration_s: int
    duration_definition: str
    good_unit_ids: tuple
    background_unit_ids: tuple
    article_id: int | None = None
    reference_params: dict = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return int(self.background_unit_ids[1]) + 1

    @property
    def stop_sample(self) -> int:
        return int(self.fs) * int(self.duration_s)

    @classmethod
    def from_registry(cls, registry: dict, case: str) -> "CaseSpec":
        if case not in CONTRACT_CASES:
            raise RegistryError(f"unknown neuropixels case {case!r}; expected one of {CONTRACT_CASES}")
        section = registry["neuropixels"]
        entry = section["cases"][case]
        params = {}
        if entry["reference_kind"] == "gt_template_10s_blocks":
            params = {
                "reference_local_channels": entry["reference_local_channels"],
                "reference_max_snippets_per_parity": entry["reference_max_snippets_per_parity"],
                "reference_min_snippets": entry["reference_min_snippets"],
                "reference_window_samples": entry["reference_window_samples"],
            }
        return cls(
            case=case,
            archive=entry["archive"],
            bytes=int(entry["bytes"]),
            registry_md5=entry.get("computed_md5"),
            file_id=int(entry["file_id"]),
            reference_kind=entry["reference_kind"],
            fs=int(section["fs"]),
            active_channels=int(section["active_channels"]),
            duration_s=int(section["duration_s"]),
            duration_definition=str(section["duration_definition"]),
            good_unit_ids=tuple(section["good_unit_ids"]),
            background_unit_ids=tuple(section["background_unit_ids"]),
            article_id=section.get("article_id"),
            reference_params=params,
        )
