"""Registry, archive discovery, native decode, and CLI metadata-inspection tests.

All archives are synthetic and tiny. The real packaged registry is read
read-only for structural cross-checks; synthetic registries (patched sizes)
drive the CLI paths. No real NP recording, GT, or results are ever read.
"""
import json
import zipfile

import numpy as np
import pytest

import shift_aso.neuropixels.datasets as datasets
from shift_aso.neuropixels.datasets import (
    ArchiveContractError,
    CbinContractError,
    CaseSpec,
    MetaContractError,
    RegistryError,
    ZipCbinStream,
    active_geometry,
    default_registry_path,
    discover_members,
    inspect_case,
    load_registry,
    parse_ch_header,
    sample_rate_hz,
)
from shift_aso.neuropixels.datasets.__main__ import main

from . import _synthetic as syn


# ---------------- registry ----------------

def test_packaged_registry_loads_and_cross_checks_production():
    registry = load_registry()
    spec_med = CaseSpec.from_registry(registry, "meddrift")
    spec_no = CaseSpec.from_registry(registry, "nodrift")
    assert spec_med.reference_kind == "gt_template_10s_blocks"
    assert spec_no.reference_kind == "frozen_unit_map"
    assert spec_med.reference_params == {
        "reference_local_channels": datasets.PRODUCTION_REFERENCE["local_channels"],
        "reference_max_snippets_per_parity": datasets.PRODUCTION_REFERENCE["max_snippets"],
        "reference_min_snippets": datasets.PRODUCTION_REFERENCE["min_snippets"],
        "reference_window_samples": [datasets.PRODUCTION_REFERENCE["window_left"],
                                     datasets.PRODUCTION_REFERENCE["window_right"] - 1],
    }
    assert spec_med.stop_sample == 9_000_000
    assert spec_med.n_units == 1200
    assert spec_no.archive == "sim_no_drift.zip" and spec_med.archive == "sim_med_drift.zip"
    with pytest.raises(RegistryError):
        CaseSpec.from_registry(registry, "highdrift")


def test_registry_rejects_schema_drift(tmp_path):
    payload = syn.synthetic_registry_payload(no_bytes=1)
    payload["schema"] = "something_else"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RegistryError):
        load_registry(path)


# ---------------- meta ----------------

def test_meta_geometry_matches_native_formula():
    active, g = active_geometry(syn.make_meta_text())
    assert len(active) == 383 and active[200 - 1] == 199 and active[200] == 201
    # independently re-derived expectations for the first few channels
    assert g[0].tolist() == [0 * 32 + 27, 0 * 20]
    assert g[1].tolist() == [1 * 32 + 27, 0 * 20]
    assert g[2].tolist() == [0 * 32 + 11, 1 * 20]
    assert g[3].tolist() == [1 * 32 + 11, 1 * 20]
    assert sample_rate_hz(syn.make_meta_text()) == 30000


def test_meta_contract_errors():
    with pytest.raises(MetaContractError):
        active_geometry("~imSampRate=30000\n")  # no shank map
    with pytest.raises(MetaContractError):
        active_geometry(syn.make_meta_text(n_ap=100))  # fewer than 384 AP entries
    with pytest.raises(MetaContractError):
        sample_rate_hz("~snsShankMap=(0:0:0:1)\n")  # no sample rate


# ---------------- member discovery ----------------

def test_nested_member_discovery(tmp_path):
    z = syn.make_archive(tmp_path / "sim_med_drift.zip", total_samples=12, chunk_len=5,
                         prefix="sim_med_drift/")
    with zipfile.ZipFile(z) as archive:
        members = discover_members(archive)
    assert members["cbin"] == "sim_med_drift/sim.imec0.ap.cbin"
    assert members["ch"].endswith("sim.imec0.ap.ch")
    assert members["meta"].endswith("sim.imec0.ap.meta")
    assert members["params"].endswith("sim.imec0.ap_params.npz")


def test_discovery_rejects_duplicate_and_missing_roles(tmp_path):
    z = syn.make_archive(tmp_path / "base.zip", total_samples=12, chunk_len=5)
    dup = tmp_path / "dup.zip"
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(dup, "w") as dst:
        for name in src.namelist():
            dst.writestr(name, src.read(name))
        dst.writestr("second_rec/sim.imec0.ap.cbin", b"")
    with zipfile.ZipFile(dup) as archive:
        with pytest.raises(ArchiveContractError) as exc:
            discover_members(archive)
        assert "second_rec/sim.imec0.ap.cbin" in str(exc.value)
    missing = tmp_path / "missing.zip"
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(missing, "w") as dst:
        for name in src.namelist():
            if name.endswith(".ap.meta"):
                continue
            dst.writestr(name, src.read(name))
    with zipfile.ZipFile(missing) as archive:
        with pytest.raises(ArchiveContractError):
            discover_members(archive)


# ---------------- ch header ----------------

def test_ch_header_contract():
    good = {"algorithm": "zlib", "do_time_diff": True, "n_channels": 385,
            "chunk_bounds": [0, 5, 10], "chunk_offsets": [0, 10, 20]}
    assert parse_ch_header(json.dumps(good)).total_samples == 10
    for patch in ({"algorithm": "blosc"}, {"do_time_diff": False}, {"n_channels": 384},
                  {"chunk_bounds": [1, 5, 10]}, {"chunk_offsets": [5, 10, 20]},
                  {"chunk_bounds": [0, 5, 5]}):
        bad = dict(good); bad.update(patch)
        with pytest.raises(CbinContractError):
            parse_ch_header(json.dumps(bad))


# ---------------- native decode ----------------

def _write_decode_zip(path, raw, chunk_len):
    bounds = syn.chunk_spans(len(raw), chunk_len)
    offsets = [0]
    blobs = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        blob = syn.encode_chunk(raw[s:e])
        blobs.append(blob)
        offsets.append(offsets[-1] + len(blob))
    header = {"algorithm": "zlib", "do_time_diff": True, "n_channels": 385,
              "chunk_bounds": bounds, "chunk_offsets": offsets}
    meta = syn.make_meta_text()
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("nested/sim.imec0.ap.ch", json.dumps(header))
        z.writestr("nested/sim.imec0.ap.meta", meta)
        z.writestr("nested/sim.imec0.ap.cbin", b"".join(blobs), compress_type=zipfile.ZIP_STORED)
    return header, meta


def test_cbin_decode_roundtrip_fortran_order_and_truncation(tmp_path):
    total = 12
    s = np.arange(total, dtype=np.int64)
    c = np.arange(385, dtype=np.int64)
    raw = (((s[:, None] * 7 + c[None, :] * 13) % 101) - 50).astype(np.int16)
    path = tmp_path / "dec.zip"
    header, meta = _write_decode_zip(path, raw, 5)
    active, _ = active_geometry(meta)
    stream = ZipCbinStream(path, "nested/sim.imec0.ap.cbin",
                           parse_ch_header(json.dumps(header)), active)
    pieces = [(start, chunk) for start, chunk in stream.chunks(total)]
    assert [start for start, _ in pieces] == [0, 5, 10]
    assert [len(chunk) for _, chunk in pieces] == [5, 5, 2]
    decoded = np.concatenate([chunk for _, chunk in pieces], axis=0)
    expected = raw[:, :384][:, active].astype(np.float32)
    assert decoded.dtype == np.float32
    assert np.array_equal(decoded, expected)  # catches Fortran-order mistakes
    # truncation mid-chunk at stop=9: second chunk yields only 4 samples
    pieces = list(stream.chunks(9))
    assert [start for start, _ in pieces] == [0, 5]
    assert pieces[1][1].shape == (4, len(active))


def test_cbin_stream_is_reiterable_across_passes(tmp_path):
    total = 10
    raw = np.zeros((total, 385), np.int16)
    raw[3, 10] = -777
    path = tmp_path / "re.zip"
    header, meta = _write_decode_zip(path, raw, 10)
    active, _ = active_geometry(meta)
    stream = ZipCbinStream(path, "nested/sim.imec0.ap.cbin",
                           parse_ch_header(json.dumps(header)), active)
    first = [(s, x.copy()) for s, x in stream.chunks(total)]
    second = [(s, x.copy()) for s, x in stream.chunks(total)]
    assert len(first) == len(second)
    for (s1, x1), (s2, x2) in zip(first, second):
        assert s1 == s2 and np.array_equal(x1, x2)  # multiple passes identical


def test_entry_size_member_identity_mismatch_detected(tmp_path):
    raw = np.zeros((10, 385), np.int16)
    path = tmp_path / "sim_med_drift.zip"
    header, meta = _write_decode_zip(path, raw, 10)
    header["chunk_offsets"][-1] += 7  # declared size no longer matches the member
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("sim.imec0.ap.ch", json.dumps(header))
        z.writestr("sim.imec0.ap.meta", meta)
        blob = syn.encode_chunk(raw)
        z.writestr("sim.imec0.ap.cbin", blob, compress_type=zipfile.ZIP_STORED)
        z.writestr("sim.imec0.ap_params.npz", b"")
    registry = syn.write_registry(tmp_path / "registry.json",
                                  no_bytes=1, med_bytes=path.stat().st_size)
    spec = CaseSpec.from_registry(load_registry(registry), "meddrift")
    with pytest.raises(CbinContractError):
        inspect_case("meddrift", tmp_path, spec=spec)


# ---------------- CLI: default metadata inspection ----------------

_REAL_ZIPFILE = zipfile.ZipFile
MEMBER_READS = []


class _RecordingZip:
    """ZipFile wrapper recording which member data is ever read/opened."""

    def __init__(self, *args, **kwargs):
        self._zip = _REAL_ZIPFILE(*args, **kwargs)

    def read(self, name, *args, **kwargs):
        MEMBER_READS.append(name)
        return self._zip.read(name, *args, **kwargs)

    def open(self, name, *args, **kwargs):
        MEMBER_READS.append(name)
        return self._zip.open(name, *args, **kwargs)

    def infolist(self):
        return self._zip.infolist()

    def namelist(self):
        return self._zip.namelist()

    def __enter__(self):
        self._zip.__enter__()
        return self

    def __exit__(self, *exc):
        return self._zip.__exit__(*exc)


def test_cli_default_is_metadata_only_no_np_load_no_writes(tmp_path, monkeypatch, capsys):
    st, cl = syn.full_st_cl()
    z = syn.make_archive(tmp_path / "sim_med_drift.zip", total_samples=syn.FULL_TOTAL,
                         chunk_len=syn.FULL_CHUNK, prefix="sim_med_drift/",
                         spikes=syn.full_spikes(), params_st=st, params_cl=cl)
    registry = syn.write_registry(tmp_path / "registry.json", no_bytes=1,
                                  med_bytes=z.stat().st_size)
    monkeypatch.setattr(datasets._registry, "default_registry_path", lambda: registry)
    MEMBER_READS.clear()
    monkeypatch.setattr(zipfile, "ZipFile", _RecordingZip)
    outdir = tmp_path / "should_not_exist"
    try:
        code = main(["--case", "meddrift", "--data-dir", str(tmp_path), "--output", str(outdir)])
    finally:
        monkeypatch.setattr(zipfile, "ZipFile", _REAL_ZIPFILE)
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "valid"
    assert report["recording_decoded"] is False and report["params_member_read"] is False
    assert report["writes"] == "none"
    assert report["members"]["cbin"] == "sim_med_drift/sim.imec0.ap.cbin"
    assert report["cbin"]["coverage"]["ok"] is True
    assert report["cbin"]["total_samples"] == syn.FULL_TOTAL
    assert report["meta"]["active_channels"] == 383
    assert report["reference"]["registry_params_match_production"] is True
    assert report["output_ignored"]
    assert not outdir.exists()  # metadata inspection writes nothing
    # Only the .ch and .meta member data was ever read: no recording decode,
    # no params (GT) load, despite full-coverage members being present.
    assert set(MEMBER_READS) <= {"sim_med_drift/sim.imec0.ap.ch", "sim_med_drift/sim.imec0.ap.meta"}


def test_cli_default_reports_invalid_short_recording(tmp_path, monkeypatch, capsys):
    z = syn.make_archive(tmp_path / "sim_med_drift.zip", total_samples=12, chunk_len=5)
    registry = syn.write_registry(tmp_path / "registry.json", no_bytes=1,
                                  med_bytes=z.stat().st_size)
    monkeypatch.setattr(datasets._registry, "default_registry_path", lambda: registry)
    code = main(["--case", "meddrift", "--data-dir", str(tmp_path)])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "invalid"
    assert "coverage" in payload["error"] or "samples" in payload["error"]


def test_cli_case_identity_uses_registry_archive_name(tmp_path, monkeypatch, capsys):
    z = syn.make_archive(tmp_path / "sim_med_drift.zip", total_samples=12, chunk_len=5)
    registry = syn.write_registry(tmp_path / "registry.json", no_bytes=1,
                                  med_bytes=z.stat().st_size)
    monkeypatch.setattr(datasets._registry, "default_registry_path", lambda: registry)
    code = main(["--case", "nodrift", "--data-dir", str(tmp_path)])  # wrong archive for case
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert "sim_no_drift.zip" in payload["error"]


def test_cli_rejects_build_reference_without_output(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--case", "meddrift", "--data-dir", str(tmp_path), "--build-reference"])
    assert exc.value.code == 2
