"""Explicit reference preparation (``--build-reference``; user-run only).

This module authorizes nothing on its own: the author of this package never
runs it against real Neuropixels archives, and the default CLI inspection
never reaches it. When a future user runs it explicitly it:

  * validates the case archive (metadata + contracts) first;
  * computes the archive MD5 (nodrift: must match the registry checksum);
  * meddrift: runs the ported accepted template builder with the locked
    production constants over two full stream passes;
  * nodrift: uses the packaged frozen unit map (no recording decode);
  * writes outputs create-once, then commits the receipt LAST with source
    version/hash/provenance/env and unknown/fallback diagnostics;
  * on failure writes ``_build_failure.json`` and keeps every partial output.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ._archive import CaseArchive, discover_members, inspect_case, open_params_npz
from ._cbin import ZipCbinStream, parse_ch_header
from ._labels import (
    build_labels,
    labels_metadata,
    load_gt_times_units,
    static_map_channels,
    template_reference_allowed,
)
from ._meta import active_geometry
from ._registry import CaseSpec, default_registry_path, load_registry
from ._template_reference import (
    PRODUCTION_REFERENCE,
    SOURCE_NAME,
    SOURCE_SHA256,
    build_template_reference,
)

RECEIPT_SCHEMA = "shift_aso_np_reference_receipt_v1"
_MODULE_FILES = (
    "__init__.py", "__main__.py", "_archive.py", "_build.py", "_cbin.py",
    "_labels.py", "_meta.py", "_registry.py", "_template_reference.py",
)


class BuildContractError(RuntimeError):
    """Raised when explicit reference preparation violates its contract."""


def _dump_json(path: Path, value: dict) -> None:
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        return str(v)

    path.write_text(json.dumps(value, indent=2, default=convert) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_hashes() -> dict:
    base = Path(__file__).resolve().parent
    hashes = {name: _sha256_file(base / name) for name in _MODULE_FILES if (base / name).is_file()}
    hashes["neuropixels_package_init.py"] = _sha256_file(base.parent / "__init__.py")
    return hashes


def _environment() -> dict:
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "thread_env": {name: os.environ.get(name) for name in thread_vars},
        "utc": datetime.now(timezone.utc).isoformat(),
    }


def reference_build(case: str, data_dir, output, *, registry_path=None) -> dict:
    """Run explicit reference preparation. Never invoked by package import or default CLI."""
    if case not in ("nodrift", "meddrift"):
        raise BuildContractError(f"unknown case {case!r}")
    registry = load_registry(registry_path)
    spec = CaseSpec.from_registry(registry, case)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)  # create-once; second run must fail
    try:
        report = inspect_case(case, data_dir, spec=spec)
        archive = CaseArchive(spec, data_dir)
        md5 = archive.md5_hex()
        md5_ok = None if spec.registry_md5 is None else md5 == spec.registry_md5
        if spec.registry_md5 is not None and md5 != spec.registry_md5:
            md5_ok = False
            raise BuildContractError(
                f"archive MD5 {md5} does not match registry checksum {spec.registry_md5}; "
                "case identity mismatch"
            )
        with archive.open_zip() as zip_file:
            members = discover_members(zip_file)
            ch_text = zip_file.read(members["ch"]).decode("utf-8")
            meta_text = zip_file.read(members["meta"]).decode("utf-8")
        ap_ids, geometry = active_geometry(meta_text)
        stop = spec.stop_sample
        if stop != PRODUCTION_REFERENCE["stop"]:
            raise BuildContractError("registry stop disagrees with locked production constants")

        outputs = {}
        if case == "meddrift":
            template_reference_allowed(case)
            header = parse_ch_header(ch_text)
            stream = ZipCbinStream(archive.path, members["cbin"], header, ap_ids)
            with open_params_npz(archive, members) as params:
                t, u = load_gt_times_units(params, stop=stop, n_units=spec.n_units)
            arrays, diagnostics = build_template_reference(
                stream.stream_factory(stop), t, u, geometry, ap_ids, **PRODUCTION_REFERENCE
            )
            labels = build_labels(
                sample_index=arrays["sample_index"],
                unit_id=arrays["unit_id"],
                estimated_local_channel=arrays["estimated_local_channel"],
                geometry_um=arrays["geometry_um"],
                local_to_original_AP=arrays["local_to_original_AP"],
                good_unit_ids=spec.good_unit_ids,
                background_unit_ids=spec.background_unit_ids,
            )
            np.savez_compressed(out / "template_dynamic_reference.npz", **arrays)
            np.savez_compressed(out / "labels.npz", **labels)
            _dump_json(out / "template_dynamic_reference_diagnostic.json", diagnostics)
            _dump_json(out / "labels_metadata.json", labels_metadata(spec, "gt_template_10s_blocks", {
                "accepted_source": {"name": SOURCE_NAME, "sha256": SOURCE_SHA256},
                "production_constants": dict(PRODUCTION_REFERENCE),
                "diagnostics": {
                    "fallback_fraction": diagnostics["fallback_fraction"],
                    "unknown_event_fraction": diagnostics["unknown_event_fraction"],
                    "unavailable_fraction": diagnostics["unavailable_fraction"],
                },
            }))
            outputs = {
                "template_dynamic_reference.npz": None,
                "labels.npz": None,
                "template_dynamic_reference_diagnostic.json": None,
                "labels_metadata.json": None,
            }
            reference_kind = "gt_template_10s_blocks"
        else:
            mapping, map_source = static_map_channels(case, spec.active_channels)
            with open_params_npz(archive, members) as params:
                t, u = load_gt_times_units(params, stop=stop, n_units=spec.n_units)
            labels = build_labels(
                sample_index=t,
                unit_id=u,
                estimated_local_channel=mapping[u],
                geometry_um=geometry,
                local_to_original_AP=ap_ids,
                good_unit_ids=spec.good_unit_ids,
                background_unit_ids=spec.background_unit_ids,
            )
            np.savez_compressed(out / "labels.npz", **labels)
            _dump_json(out / "labels_metadata.json", labels_metadata(spec, "frozen_unit_map", {
                "packaged_map": "_unit_to_main_channel.json",
                "map_source": map_source,
                "support": "original 30-60 s reference extraction; scorer-only",
            }))
            outputs = {"labels.npz": None, "labels_metadata.json": None}
            diagnostics = {
                "status": "complete",
                "kind": "frozen_unit_map",
                "map_source": map_source,
                "support": "original 30-60 s reference extraction; scorer-only",
                "recording_decoded": False,
            }
            reference_kind = "frozen_unit_map"

        for name in outputs:
            outputs[name] = {
                "sha256": _sha256_file(out / name),
                "bytes": (out / name).stat().st_size,
            }
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "complete",
            "generated_by": "python -m shift_aso.neuropixels.datasets --build-reference",
            "case": case,
            "reference_kind": reference_kind,
            "accepted_source": ({
                "name": SOURCE_NAME,
                "sha256": SOURCE_SHA256,
                "note": "algorithm ported verbatim; see datasets/README.md",
            } if case == "meddrift" else None),
            "production_constants": dict(PRODUCTION_REFERENCE) if case == "meddrift" else None,
            "static_map_sha256": (_sha256_file(Path(__file__).resolve().parents[1]
                                               / "_unit_to_main_channel.json")
                                   if case == "nodrift" else None),
            "registry": {
                "path": str(registry_path) if registry_path else str(Path(__file__).resolve().parents[2] / "datasets_registry.json"),
                "case_archive": spec.archive,
                "sha256": _sha256_file(Path(registry_path) if registry_path
                                        else default_registry_path()),
            },
            "archive": {
                "path": str(archive.path),
                "bytes": archive.stat_bytes,
                "md5": md5,
                "registry_md5": spec.registry_md5,
                "md5_match": md5_ok,
                "md5_status": "verified" if md5_ok else "observed_only_no_golden_checksum",
                "members": dict(members),
            },
            "modules": _module_hashes(),
            "environment": _environment(),
            "inspection": report,
            "reference_diagnostics": diagnostics,
            "outputs": outputs,
            "label_exposure": (
                "All GT times/IDs (including TEST) are used only to build scoring references; "
                "they are never detector CAL/DEV inputs or settings."
            ),
        }
        _dump_json(out / "reference_receipt.json", receipt)  # committed last
        return receipt
    except Exception as exc:  # keep partials; never clean up a failed run
        failure = {
            "schema": "shift_aso_np_reference_failure_v1",
            "status": "failed",
            "case": case,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "environment": _environment(),
            "partials_kept": sorted(p.name for p in out.iterdir()),
        }
        _dump_json(out / "_build_failure.json", failure)
        raise
