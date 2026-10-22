"""Case-bound ZIP archive discovery and metadata inspection.

Generic suffix-based nested member discovery (no hardcoded ``sim.imec0.ap.*``
member name), case identity enforced through the registry archive filename,
entry-size and member-identity validation, and a metadata-only inspection
that reads the ``.ap.ch`` and ``.ap.meta`` members but never decodes the
recording and never reads the GT ``params`` NPZ.
"""
from __future__ import annotations

import hashlib
import zipfile
from contextlib import contextmanager
from pathlib import Path

from ._cbin import CbinContractError, N_RECORDED, parse_ch_header
from ._meta import N_AP, MetaContractError, active_geometry, sample_rate_hz
from ._registry import CaseSpec, RegistryError

# Role suffixes of the four required members inside a case ZIP. Nested
# directory prefixes (e.g. ``sim_med_drift/sim.imec0.ap.cbin``) are allowed;
# the suffixes are disjoint so each member matches exactly one role.
ROLE_SUFFIXES = {
    "cbin": ".ap.cbin",
    "ch": ".ap.ch",
    "meta": ".ap.meta",
    "params": ".ap_params.npz",
}
INSPECTION_SCHEMA = "shift_aso_np_datasets_inspection_v1"


class ArchiveContractError(RuntimeError):
    """Raised when the case archive violates the registry/format contract."""


def discover_members(zip_file) -> dict:
    """Map each role to its unique ZIP member name by suffix match."""
    names = [info.filename for info in zip_file.infolist() if not info.is_dir()]
    found = {role: [name for name in names if name.endswith(suffix)]
             for role, suffix in ROLE_SUFFIXES.items()}
    members = {}
    for role, matches in found.items():
        if len(matches) != 1:
            raise ArchiveContractError(
                f"expected exactly one member matching role {role!r} "
                f"(suffix {ROLE_SUFFIXES[role]!r}), found {matches}"
            )
        members[role] = matches[0]
    return members


class CaseArchive:
    """A case-bound view of ``<data-dir>/<registry archive filename>``."""

    def __init__(self, spec: CaseSpec, data_dir):
        self.spec = spec
        root = Path(data_dir)
        self.path = root / spec.archive
        if not self.path.is_file():
            raise ArchiveContractError(
                f"case {spec.case!r} expects archive {spec.archive!r} in {root}; file not found"
            )
        self.stat_bytes = self.path.stat().st_size
        if self.stat_bytes != spec.bytes:
            raise ArchiveContractError(
                f"archive {self.path.name} has {self.stat_bytes} bytes; registry declares "
                f"{spec.bytes} bytes (case identity mismatch)"
            )

    def open_zip(self) -> zipfile.ZipFile:
        try:
            return zipfile.ZipFile(self.path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise ArchiveContractError(f"archive is not a readable ZIP file: {exc}") from exc

    def read_members(self, members: dict) -> dict:
        with self.open_zip() as archive:
            return {role: archive.read(name) for role, name in members.items()}

    def md5_hex(self, chunk_bytes: int = 8 << 20) -> str:
        digest = hashlib.md5()
        with open(self.path, "rb") as handle:
            while True:
                block = handle.read(chunk_bytes)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()


def inspect_case(case: str, data_dir, *, registry_path=None, spec: CaseSpec | None = None) -> dict:
    """Metadata-only validation of one case archive. Never decodes the recording.

    Reads the ZIP central directory plus the ``.ap.ch`` and ``.ap.meta``
    members only. The ``params`` NPZ (native GT ``st``/``cl``) may be loaded
    only by explicit reference preparation, never here. Produces no output
    files.
    """
    if spec is None:
        from ._registry import load_registry
        spec = CaseSpec.from_registry(load_registry(registry_path), case)
    archive = CaseArchive(spec, data_dir)
    report = {
        "schema": INSPECTION_SCHEMA,
        "status": "valid",
        "case": spec.case,
        "case_identity": {
            "archive": spec.archive,
            "path": str(archive.path),
            "article_id": spec.article_id,
            "file_id": spec.file_id,
        },
        "archive": {
            "bytes": archive.stat_bytes,
            "registry_bytes": spec.bytes,
            "bytes_match": archive.stat_bytes == spec.bytes,
            "registry_md5": spec.registry_md5,
            "md5_computed": False,
            "md5_note": "archive MD5 is computed only by explicit --build-reference",
        },
        "writes": "none",
        "recording_decoded": False,
        "params_member_read": False,
    }
    with archive.open_zip() as zip_file:
        members = discover_members(zip_file)
        sizes = {info.filename: info.file_size for info in zip_file.infolist()}
        ch_text = zip_file.read(members["ch"]).decode("utf-8")
        meta_text = zip_file.read(members["meta"]).decode("utf-8")
    report["members"] = dict(members)

    header = parse_ch_header(ch_text)
    required_first_samples = spec.stop_sample
    entry_size_match = sizes[members["cbin"]] == header.total_compressed_bytes
    if not entry_size_match:
        raise CbinContractError(
            f"cbin member {members['cbin']!r} has {sizes[members['cbin']]} uncompressed bytes; "
            f".ch offsets declare {header.total_compressed_bytes} (member identity mismatch)"
        )
    report["cbin"] = {
        "member": members["cbin"],
        "n_channels": header.n_channels,
        "n_chunks": header.n_chunks,
        "total_samples": header.total_samples,
        "compressed_bytes": header.total_compressed_bytes,
        "entry_size_matches_offsets": True,
        "recorded_channels_expected": N_RECORDED,
        "coverage": {
            "required_first_samples": required_first_samples,
            "available_samples": header.total_samples,
            "ok": header.total_samples >= required_first_samples,
            "definition": spec.duration_definition,
        },
    }
    if header.total_samples < required_first_samples:
        raise CbinContractError(
            f"recording covers {header.total_samples} samples; the benchmark requires coverage "
            f"of the first {required_first_samples} samples ({spec.duration_s} s at {spec.fs} Hz; "
            "upstream recordings may be longer)"
        )

    active, geometry = active_geometry(meta_text)
    fs = sample_rate_hz(meta_text)
    if len(active) != spec.active_channels:
        raise MetaContractError(
            f"meta declares {len(active)} active AP channels; registry requires {spec.active_channels}"
        )
    if fs != spec.fs:
        raise MetaContractError(f"meta sample rate {fs} Hz != registry {spec.fs} Hz")
    report["meta"] = {
        "member": members["meta"],
        "ap_entries": N_AP,
        "active_channels": len(active),
        "registry_active_channels": spec.active_channels,
        "sample_rate_hz": fs,
        "registry_sample_rate_hz": spec.fs,
        "geometry_um_shape": list(geometry.shape),
    }

    if spec.reference_kind == "frozen_unit_map":
        from ._labels import frozen_unit_main_channels
        mapping, source_note = frozen_unit_main_channels(spec.active_channels)
        report["reference"] = {
            "kind": "frozen_unit_map",
            "packaged_map_units": int(len(mapping)),
            "map_max_local_channel": int(mapping.max()),
            "map_source": source_note,
            "support": "original 30-60 s reference extraction; scorer-only",
        }
    else:
        report["reference"] = {
            "kind": "gt_template_10s_blocks",
            "registry_params": dict(spec.reference_params),
            "registry_params_match_production": True,
        }
    report["duration"] = {"duration_s": spec.duration_s, "definition": spec.duration_definition}
    report["unit_groups"] = {
        "good_unit_ids": list(spec.good_unit_ids),
        "background_unit_ids": list(spec.background_unit_ids),
        "n_units": spec.n_units,
    }
    return report


@contextmanager
def open_params_npz(archive: CaseArchive, members: dict):
    """Open the GT params NPZ member (explicit reference preparation only).

    The yielded ``NpzFile`` is only valid inside the ``with`` block; native
    ``st``/``cl`` loading must happen there.
    """
    import numpy as np
    with archive.open_zip() as zip_file:
        with zip_file.open(members["params"]) as handle:
            yield np.load(handle, allow_pickle=False)
