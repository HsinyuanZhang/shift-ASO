"""Parent integration regressions. Synthetic inputs only; no real generation."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from shift_aso.mearec.generation import core, native
from shift_aso.mearec.generation import test_generation as mea_fixtures
from shift_aso.neuropixels.datasets._meta import MetaContractError, sample_rate_hz

ROOT = Path(__file__).resolve().parents[1]


def context(tmp_path, dataset="D0"):
    config = core.load_config()
    return native.GenerationContext(config, Path(core.__file__).with_name("config.json"),
                                    dataset, tmp_path / "assets", tmp_path / "out",
                                    1, 20 * (1 << 30), tmp_path / "scratch")


def rewrite_manifest(root, change):
    path = root / "generation_manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))
    core.ledger_update(root, path.name, core.sha256_file(path))


def test_readme_has_upstream_citations_and_separate_software_workflow():
    text = (ROOT / "README.md").read_text()
    for required in ("10.1007/s12021-020-09467-7", "10.1038/s41592-024-02232-7",
                     "sim_no_drift.zip", "sim_med_drift.zip", "5 µVrms", "D0–D4"):
        assert required in text
    assert "python -m shift_aso.mearec dev" in text
    assert "python -m shift_aso.mearec test" in text
    assert "No benchmark scores" in text
    for forbidden in ("results/", "mearec_comparison", "neuropixels_comparison",
                      "expected_method_summary", "| Method | MEArec"):
        assert forbidden not in text
    assert "--metadata-only" in text and "--execute" in text


def test_config_mirrors():
    for source, packaged in (("configs/datasets.json", "shift_aso/datasets_registry.json"),
                              ("configs/mearec_multiseed.json", "shift_aso/mearec/generation/config.json")):
        assert (ROOT / source).read_bytes() == (ROOT / packaged).read_bytes()


def test_external_recipe_must_match_frozen_config(tmp_path):
    recipe = core.load_config()
    recipe["datasets"][1]["position_seed"] += 1
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(recipe))
    with pytest.raises(core.StopGeneration, match="CONFIG_FROZEN"):
        core.load_config(path)


def test_d0_identity_receipt_has_no_search_statistics():
    models = np.array(["model_a", "model_b"])
    locations = np.array([[0., 0., 0.], [25., 0., 0.]])
    rotations = np.zeros((2, 3))
    eaps = np.full((2, 2, 3), -100.)
    receipt = core.selection_receipt(models, locations, rotations, eaps, [0, 1], 7,
                                     {"template_ids": [0, 1]}, {}, "original_reference")
    assert receipt["template_ids"] == [0, 1]
    assert "backtrack_steps" not in receipt


def test_preserved_scratch_is_never_adopted(tmp_path):
    scratch = tmp_path / "scratch"
    native.fresh_scratch(scratch)
    (scratch / "PROGRESS_unit_white.jsonl").write_text('{"chunk": 0}\n')
    with pytest.raises(core.StopGeneration, match="PRESERVED_SCRATCH"):
        native.fresh_scratch(scratch)
    assert (scratch / "PROGRESS_unit_white.jsonl").exists()


def test_execution_binding_checked_before_stage_or_skip(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    guarded = []
    monkeypatch.setattr(native, "asset_guard", lambda ctx: guarded.append(True) or {"asset": "hash"})
    native.bind_execution(ctx)
    native.bind_execution(ctx)
    assert len(guarded) == 2
    ctx.config_sha256 = "different"
    with pytest.raises(core.StopGeneration, match="EXECUTION_BINDING_DRIFT"):
        native.bind_execution(ctx)


def test_unbound_bundle_is_not_adopted(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    ctx.bundle.mkdir(parents=True)
    (ctx.bundle / "partial.npy").write_bytes(b"preserve")
    monkeypatch.setattr(native, "asset_guard", lambda ctx: {})
    with pytest.raises(core.StopGeneration, match="UNBOUND_BUNDLE"):
        native.bind_execution(ctx)
    assert (ctx.bundle / "partial.npy").read_bytes() == b"preserve"


def test_clean_skip_requires_metadata_hash(tmp_path):
    ctx = context(tmp_path)
    ctx.bundle.mkdir(parents=True)
    np.save(ctx.bundle / "clean.npy", np.zeros((2, 2), np.float32))
    core.ledger_update(ctx.bundle, "clean.npy", core.sha256_file(ctx.bundle / "clean.npy"))
    with pytest.raises(core.StopGeneration, match="MISSING_PREREQUISITE"):
        native.stage_clean(ctx)


def test_gate_checks_output_and_scratch_filesystems(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    visited = []
    def free(path):
        visited.append(Path(path))
        return 10**15 if Path(path) == ctx.output_root else 0
    monkeypatch.setattr(core, "free_bytes", free)
    with pytest.raises(core.StopGeneration, match="STOP_BEFORE_WRITE"):
        native.gate_write(ctx, 4096)
    assert visited == [ctx.output_root, ctx.scratch_root]
    ctx.reserve_bytes = 20_000_000_000
    with pytest.raises(core.StopGeneration, match="RESERVE_TOO_SMALL"):
        native.gate_write(ctx, 4096)


def test_geometry_requires_full_unique_square_grid():
    x, y = np.meshgrid(np.arange(16) * 36., np.arange(16) * 36.)
    geometry = np.column_stack((x.ravel(), y.ravel()))
    native._assert_probe_geometry(geometry, core.load_config())
    geometry[1] = geometry[0]
    with pytest.raises(core.StopGeneration, match="GEOMETRY_DUPLICATE"):
        native._assert_probe_geometry(geometry, core.load_config())


@pytest.mark.parametrize("kind", ["dtype", "shape", "metadata", "manifest", "targets", "path", "binding"])
def test_virtual_reader_fail_closed(tmp_path, kind):
    mea_fixtures.TestIterInput()._write_bundle(tmp_path)
    if kind in ("dtype", "shape"):
        values = (np.zeros((6, 2), np.float64) if kind == "dtype"
                  else np.zeros((3, 4), np.float32))
        np.save(tmp_path / "clean.npy", values)
        rewrite_manifest(tmp_path, lambda m: m["provenance"]["files"]["clean"].update(
            sha256=core.sha256_file(tmp_path / "clean.npy")))
    elif kind == "binding":
        (tmp_path / "execution_binding.json").write_text("changed binding")
    elif kind == "metadata":
        (tmp_path / "metadata.npz").write_bytes(b"changed metadata")
    elif kind == "manifest":
        path = tmp_path / "generation_manifest.json"
        path.write_text(path.read_text() + " ")  # no matching ledger commit
    elif kind == "targets":
        rewrite_manifest(tmp_path, lambda m: m.update(target_total_CAL_SNR_dB=[10, 15]))
    else:
        rewrite_manifest(tmp_path, lambda m: m["provenance"]["files"]["clean"].update(file="../clean.npy"))
    with pytest.raises(core.StopGeneration):
        list(core.iter_input(tmp_path, 15))


@pytest.mark.parametrize("stage", ["all", "noise"])
def test_mea_cli_controls_precede_first_numpy_import(tmp_path, stage):
    code = '''
import sys, os, importlib.abc
import shift_aso.mearec, shift_aso.mearec.generation
assert 'numpy' not in sys.modules
class Spy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numpy':
            assert os.environ['OPENBLAS_NUM_THREADS'] == EXPECTED
            assert os.environ['OMP_NUM_THREADS'] == EXPECTED
            assert os.environ['MKL_NUM_THREADS'] == '1'
            assert os.environ['NUMBA_NUM_THREADS'] == '1'
sys.meta_path.insert(0, Spy())
from shift_aso.mearec.generation.cli import main
assert main(ARGS) == 0
'''.replace("EXPECTED", repr("2" if stage == "noise" else "1")).replace(
        "ARGS", repr(["--dataset", "D0", "--output", str(tmp_path / "out"), "--stage", stage]))
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OPENBLAS_NUM_THREADS": "7",
           "OMP_NUM_THREADS": "7", "MKL_NUM_THREADS": "7", "NUMBA_NUM_THREADS": "7"}
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "out").exists()


def test_np_cli_is_lazy_and_single_thread_before_numpy():
    code = '''
import sys, os, importlib.abc
import shift_aso.neuropixels, shift_aso.neuropixels.datasets
assert 'numpy' not in sys.modules
class Spy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numpy':
            assert os.environ['OPENBLAS_NUM_THREADS'] == '1'
            assert os.environ['OMP_NUM_THREADS'] == '1'
sys.meta_path.insert(0, Spy())
import shift_aso.neuropixels.datasets.__main__
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            env={**os.environ, "PYTHONPATH": str(ROOT),
                                 "OPENBLAS_NUM_THREADS": "7", "OMP_NUM_THREADS": "7"},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_native_meta_sample_rate_conventions():
    assert sample_rate_hz("imSampRate=30000\n") == 30000
    assert sample_rate_hz("~imSampRate=30000\n") == 30000
    for text in ("imSampRate=30000.5\n", "imSampRate=nan\n",
                 "imSampRate=30000\n~imSampRate=30000\n"):
        with pytest.raises(MetaContractError):
            sample_rate_hz(text)


def download_module():
    spec = importlib.util.spec_from_file_location("dataset_download", ROOT / "scripts/download_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_case_identity_and_missing_checksum():
    module = download_module()
    registry = json.loads((ROOT / "configs/datasets.json").read_text())
    files = [{"name": entry["archive"], "id": entry["file_id"], "size": entry["bytes"],
              "computed_md5": entry.get("computed_md5")}
             for entry in registry["neuropixels"]["cases"].values()]
    selected = module.selected_files({"files": files}, case="both")
    assert len(selected) == 2
    med = next(e for e in selected if e["name"] == "sim_med_drift.zip")
    assert med["expected_md5"] is None
    files[0]["size"] += 1
    with pytest.raises(RuntimeError, match="identity changed"):
        module.selected_files({"files": files}, case="both")


def test_download_failure_retains_partial_and_refuses_automatic_retry(tmp_path, monkeypatch):
    module = download_module()
    monkeypatch.setattr(module, "urlopen", lambda *a, **kw: io.BytesIO(b"tiny"))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**15))
    entry = {"name": "synthetic.zip", "size": 4, "expected_md5": "0" * 32,
             "download_url": "https://example.invalid/synthetic"}
    with pytest.raises(RuntimeError, match="mismatch"):
        module.download(entry, tmp_path)
    assert (tmp_path / "synthetic.zip.part").read_bytes() == b"tiny"
    with pytest.raises(RuntimeError, match="previous partial"):
        module.download(entry, tmp_path)
    assert not (tmp_path / "synthetic.zip").exists()


def test_download_default_is_metadata_only(tmp_path, monkeypatch):
    module = download_module()
    monkeypatch.setattr(module, "article_metadata", lambda: {"id": 25298815})
    monkeypatch.setattr(module, "selected_files", lambda *args: [])
    monkeypatch.setattr(module, "download", lambda *args: pytest.fail("unexpected download"))
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["download_data.py", "--output", str(output)])
    module.main()
    assert not output.exists()
