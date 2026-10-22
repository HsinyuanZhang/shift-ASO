"""Labels, case-bound reference semantics, and explicit build tests (synthetic).

The full-coverage synthetic archives exercise the real build orchestration
with locked production constants (9,000,000 samples, 30 blocks, 1200 units)
without touching any real Neuropixels recording: create-once outputs,
receipts committed last with source/module hashes, preserved failure
partials, static-map refusal for meddrift, and scorer-only GT exposure.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import shift_aso.neuropixels.datasets as datasets
from shift_aso.neuropixels.datasets import (
    LABEL_FIELDS,
    LabelContractError,
    build_labels,
    frozen_unit_main_channels,
    labels_metadata,
    load_gt_times_units,
    reference_build,
    static_map_channels,
    template_reference_allowed,
    unit_group_masks,
)
from shift_aso.neuropixels.datasets._registry import CaseSpec, load_registry

from . import _synthetic as syn

GOOD = (0, 599)
BACKGROUND = (600, 1199)


@pytest.fixture(scope="module")
def med_env(tmp_path_factory):
    """Full-coverage synthetic meddrift archive + matching synthetic registry."""
    root = tmp_path_factory.mktemp("med")
    st, cl = syn.full_st_cl()
    z = syn.make_archive(root / "sim_med_drift.zip", total_samples=syn.FULL_TOTAL,
                         chunk_len=syn.FULL_CHUNK, prefix="sim_med_drift/",
                         spikes=syn.full_spikes(), params_st=st, params_cl=cl,
                         params_extra={"drift": np.zeros((150, 383), np.int64)})
    registry = syn.write_registry(root / "registry.json", no_bytes=1,
                                  med_bytes=z.stat().st_size)
    return root, z, registry, st, cl


# ---------------- labels ----------------

def test_gt_loader_validates_sorted_range_and_duration():
    import io
    buf = io.BytesIO()
    np.savez(buf, st=np.array([5, 3], np.int64), cl=np.array([0, 0], np.int64))
    buf.seek(0)
    with np.load(buf, allow_pickle=False) as npz:
        with pytest.raises(LabelContractError):
            load_gt_times_units(npz, stop=100, n_units=1200)
    buf = io.BytesIO()
    np.savez(buf, st=np.array([3, 5], np.int64), cl=np.array([0, 1200], np.int64))
    buf.seek(0)
    with np.load(buf, allow_pickle=False) as npz:
        with pytest.raises(LabelContractError):
            load_gt_times_units(npz, stop=100, n_units=1200)
    buf = io.BytesIO()
    np.savez(buf, st=np.array([150, 200], np.int64), cl=np.array([0, 1], np.int64))
    buf.seek(0)
    with np.load(buf, allow_pickle=False) as npz:
        with pytest.raises(LabelContractError):  # nothing inside benchmark duration
            load_gt_times_units(npz, stop=100, n_units=1200)
    buf = io.BytesIO()
    np.savez(buf, st=np.array([10, 20, 30], np.int64), cl=np.array([0, 600, 1199], np.int64),
             drift=np.zeros((3, 3), np.int64))
    buf.seek(0)
    with np.load(buf, allow_pickle=False) as npz:
        t, u = load_gt_times_units(npz, stop=25, n_units=1200)
    assert t.tolist() == [10, 20] and u.tolist() == [0, 600]  # drift key never read


def test_frozen_unit_map_is_packaged_and_local():
    mapping, source = frozen_unit_main_channels(383)
    assert mapping.shape == (1200,)
    assert mapping.min() >= 0 and mapping.max() < 383
    assert "30-60 s" in source


def test_static_map_refused_for_meddrift_and_template_refused_for_nodrift():
    with pytest.raises(LabelContractError) as exc:
        static_map_channels("meddrift", 383)
    assert "meddrift requires" in str(exc.value)
    with pytest.raises(LabelContractError) as exc:
        template_reference_allowed("nodrift")
    assert "frozen_unit_map" in str(exc.value)


def test_build_labels_fields_and_groups():
    ap = np.array([0, 1, 2], np.int64)
    g = np.array([[0.0, 0.0], [0.0, 20.0], [0.0, 40.0]])
    labels = build_labels(sample_index=np.array([10, 20, 30]), unit_id=np.array([0, 599, 600]),
                          estimated_local_channel=np.array([0, 1, -1]), geometry_um=g,
                          local_to_original_AP=ap, good_unit_ids=GOOD,
                          background_unit_ids=BACKGROUND)
    assert tuple(sorted(labels)) == tuple(sorted(LABEL_FIELDS))
    assert labels["is_good"].tolist() == [True, True, False]
    assert labels["is_background"].tolist() == [False, False, True]
    assert labels["scoring_group"].tolist() == [0, 0, 1]
    assert np.array_equal(labels["local_to_original_AP"], ap)
    is_good, is_background = unit_group_masks(np.array([0, 599, 600, 1199]), GOOD, BACKGROUND)
    assert is_good.tolist() == [True, True, False, False]
    assert is_background.tolist() == [False, False, True, True]
    with pytest.raises(LabelContractError):
        build_labels(sample_index=np.array([10]), unit_id=np.array([0]),
                     estimated_local_channel=np.array([3]), geometry_um=g,
                     local_to_original_AP=ap, good_unit_ids=GOOD,
                     background_unit_ids=BACKGROUND)  # channel index out of range


def test_labels_metadata_exposes_groups_and_exposure():
    spec = CaseSpec.from_registry(load_registry(), "meddrift")
    meta = labels_metadata(spec, "gt_template_10s_blocks", {"x": 1})
    assert meta["scoring_groups"]["good"]["unit_ids"] == [0, 599]
    assert meta["scoring_groups"]["background"]["unit_ids"] == [600, 1199]
    assert "never detector CAL/DEV inputs" in meta["label_exposure"]


def test_public_api_has_no_detector_surfaces():
    for name in datasets.__all__:
        low = name.lower()
        assert not any(tok in low for tok in ("cal", "detector", "config", "tuning", "threshold"))


# ---------------- explicit build: meddrift dynamic template ----------------

def test_full_meddrift_build(med_env):
    root, z, registry, st, cl = med_env
    out = root / "out_med"
    receipt = reference_build("meddrift", root, out, registry_path=registry)
    ref = np.load(out / "template_dynamic_reference.npz")
    assert set(ref.files) == {
        "sample_index", "unit_id", "estimated_local_channel", "block_channel",
        "block_valid", "block_fallback", "fallback_distance_blocks",
        "geometry_um", "local_to_original_AP"}
    assert np.array_equal(ref["sample_index"], st) and np.array_equal(ref["unit_id"], cl)
    channels = ref["estimated_local_channel"]
    # active set excludes AP 200 -> local index of original AP 310 is 309
    assert np.all(channels[cl == 0] == 10)
    assert np.all(channels[cl == 1] == 60)
    assert np.all(channels[cl == 650] == 110)
    assert channels[cl == 1199][0] == -1  # single snippet -> unknown, kept
    cen = ref["block_channel"]
    assert cen[1, 1] == cen[0, 1] == 60  # tie fallback to the lower valid block
    assert ref["fallback_distance_blocks"][1, 1] == 1 and ref["block_fallback"][1, 1]
    assert np.all(cen[:, 2] == -1)  # unit 2 never fires -> unknown everywhere
    labels = np.load(out / "labels.npz")
    assert tuple(sorted(labels.files)) == tuple(sorted(LABEL_FIELDS))
    assert labels["is_background"][cl == 650].all() and labels["is_good"][cl == 0].all()
    assert labels["scoring_group"][cl == 1199].tolist() == [1]
    diag = json.loads((out / "template_dynamic_reference_diagnostic.json").read_text())
    assert diag["status"] == "complete"
    assert diag["window_samples"] == [-50, 70]
    assert diag["minimum_snippets"] == 4
    assert diag["maximum_snippets_per_time_parity"] == 16
    assert diag["local_channels"] == 13
    assert diag["no_global_collision_exclusion"] is True
    assert diag["same_unit_overlap_fraction"] > 0  # computed ...
    assert ref["block_valid"][0, 0]  # ... but never applied as exclusion
    assert diag["unknown_event_fraction"] == pytest.approx(1 / 47)
    assert diag["fallback_fraction"] == pytest.approx(35996 / 36000)
    assert diag["fallback_event_fraction"] == pytest.approx(1 / 47)
    # receipt committed last, with source/module/output hashes and env
    assert (out / "reference_receipt.json").exists()
    receipt_mtime = (out / "reference_receipt.json").stat().st_mtime
    for name in receipt["outputs"]:
        assert (out / name).stat().st_mtime <= receipt_mtime
        import hashlib
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == receipt["outputs"][name]["sha256"]
    assert receipt["accepted_source"]["sha256"] == datasets.SOURCE_SHA256
    assert receipt["production_constants"] == dict(datasets.PRODUCTION_REFERENCE)
    assert receipt["archive"]["md5_match"] is None
    assert receipt["archive"]["md5_status"] == "observed_only_no_golden_checksum"
    assert receipt["registry"]["sha256"] == hashlib.sha256(registry.read_bytes()).hexdigest()
    assert receipt["archive"]["md5"] and receipt["archive"]["members"]["cbin"].endswith(".ap.cbin")
    assert receipt["environment"]["numpy"]
    assert "never detector CAL/DEV inputs" in receipt["label_exposure"]
    assert receipt["reference_diagnostics"]["unknown_event_fraction"] == pytest.approx(1 / 47)


def test_build_is_create_once(med_env):
    root, z, registry, st, cl = med_env
    out = root / "out_once"
    out.mkdir()  # pre-existing output directory: create-once refuses to touch it
    marker = out / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        reference_build("meddrift", root, out, registry_path=registry)
    assert marker.read_text() == "keep"  # nothing overwritten or cleaned
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]


def test_failed_build_preserves_partials(med_env, tmp_path):
    root, z, registry, st, cl = med_env
    corrupt = syn.patch_ch_member(z, tmp_path / "sim_med_drift.zip", bump_offset_index=1)
    bad_dir = tmp_path / "data"
    bad_dir.mkdir()
    corrupt.rename(bad_dir / "sim_med_drift.zip")
    bad_registry = syn.write_registry(tmp_path / "registry.json", no_bytes=1,
                                      med_bytes=(bad_dir / "sim_med_drift.zip").stat().st_size)
    out = tmp_path / "out_fail"
    with pytest.raises(Exception):
        reference_build("meddrift", bad_dir, out, registry_path=bad_registry)
    failure = json.loads((out / "_build_failure.json").read_text())
    assert failure["status"] == "failed" and failure["case"] == "meddrift"
    assert failure["traceback"]
    assert out.is_dir()  # partials kept, never cleaned up


# ---------------- explicit build: nodrift frozen map ----------------

def test_full_nodrift_build_uses_frozen_map_without_decode(med_env, tmp_path, monkeypatch):
    root, z, registry, st, cl = med_env
    data = tmp_path / "data"
    data.mkdir()
    import shutil
    shutil.copy(z, data / "sim_no_drift.zip")
    import hashlib
    md5 = hashlib.md5((data / "sim_no_drift.zip").read_bytes()).hexdigest()
    no_registry = syn.write_registry(tmp_path / "registry.json", no_bytes=z.stat().st_size,
                                     no_md5=md5, med_bytes=z.stat().st_size)
    def _no_stream(*args, **kwargs):
        raise AssertionError("nodrift reference preparation must not decode the recording")
    monkeypatch.setattr(datasets._build, "ZipCbinStream", _no_stream)
    out = tmp_path / "out_no"
    receipt = reference_build("nodrift", data, out, registry_path=no_registry)
    mapping, _ = frozen_unit_main_channels(383)
    labels = np.load(out / "labels.npz")
    assert np.array_equal(labels["estimated_local_channel"], mapping[cl])
    assert labels["is_good"].tolist() == (cl <= 599).tolist()
    assert receipt["reference_kind"] == "frozen_unit_map"
    assert receipt["archive"]["md5_match"] is True
    assert receipt["archive"]["md5_status"] == "verified"
    assert receipt["accepted_source"] is None
    assert receipt["production_constants"] is None
    assert receipt["static_map_sha256"] == hashlib.sha256(
        (Path(datasets.__file__).resolve().parents[1] / "_unit_to_main_channel.json").read_bytes()).hexdigest()
    assert receipt["registry"]["sha256"] == hashlib.sha256(no_registry.read_bytes()).hexdigest()
    meta = json.loads((out / "labels_metadata.json").read_text())
    assert meta["provenance"]["support"].startswith("original 30-60 s")
    assert not (out / "template_dynamic_reference.npz").exists()


def test_nodrift_md5_case_identity_enforced(med_env, tmp_path):
    root, z, registry, st, cl = med_env
    data = tmp_path / "data2"
    data.mkdir()
    import shutil
    shutil.copy(z, data / "sim_no_drift.zip")
    no_registry = syn.write_registry(tmp_path / "registry.json", no_bytes=z.stat().st_size,
                                     no_md5="0" * 32, med_bytes=z.stat().st_size)
    out = tmp_path / "out_md5"
    with pytest.raises(datasets.BuildContractError) as exc:
        reference_build("nodrift", data, out, registry_path=no_registry)
    assert "MD5" in str(exc.value)
    assert (out / "_build_failure.json").exists()
