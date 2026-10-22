"""Tiny synthetic and mocked tests for the public MEA generation package.

These tests exercise the portable pure core plus the driver/guard logic on
synthetic fixtures only. They do NOT run native MEArec production generation,
selection over the real bank, real trains, or real noise draws; native
execution paths are implemented but unexecuted on this host (MEArec is not
installed here), which is an explicitly reported limitation.

Run (one CPU, one thread, own cache scope):

    cd <repo>/shift_aso/mearec/generation
    PYTHONPATH=<repo> taskset -c 0 python -m unittest test_generation -v
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

from shift_aso.mearec.generation import (  # noqa: E402
    EXC_MODELS, STATUS_COMPLETE, STAGE_ORDER, StopGeneration, assert_selection,
    build_candidates, cal_moments, cal_snr_db, component_bytes, create_once,
    independent_chunk_rng, iter_input, load_config, mixed_input_block_f32,
    mixed_noise_block_f32, native_amp, plan_dataset, read_generation_manifest,
    resolve_assets_dir, resolve_thread_env, sample_dataset, sample_unit_times,
    seconds_to_samples, seconds_to_samples_is_exact_roundtrip, select_positions,
    solve_scale, spatial_chunk_seeds, spatial_covariance, total_sigma,
    unit_split_sigma, validate_config, validate_schedule, verified_skip,
    write_receipt, SplitAccumulator)
from shift_aso.mearec.generation import core, native  # noqa: E402


def scratch_dir(name: str) -> Path:
    root = HERE / ".test_tmp"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{name}_", dir=str(root)))


# ---------------------------------------------------------------------------
# Tiny synthetic template bank helpers
# ---------------------------------------------------------------------------
def tiny_bank():
    """7 instances, two models, crafted strict amplitudes and distances.

    Bank layout: m1 = instances 0..3, m2 = instances 4..6.
    amp(i) = 60 + 30*i (all strictly inside (50, 300) except where noted).
    Locations: a 3D grid with >= 25 um spacing except instance 2 and 5,
    which sit 10 um apart to force a real min-dist conflict.
    """
    celltypes = np.array(["m1"] * 4 + ["m2"] * 3)
    locations = np.array([
        [0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0], [300.0, 0.0, 0.0],
        [0.0, 100.0, 0.0], [200.0, 0.0, 10.0], [0.0, 300.0, 0.0]])
    rotations = np.zeros((7, 3))
    eaps = []
    for i in range(7):
        eap = np.zeros((4, 3), np.float32)
        eap[1, 0] = -(60.0 + 30.0 * i)  # np.min(eap) = -(60+30i) -> amp = 60+30i
        eaps.append(eap)
    return celltypes, locations, rotations, np.asarray(eaps)


def brute_force_selection(celltypes, locations, eaps, original_ids, seed,
                          min_dist=25.0, min_amp=50.0, max_amp=300.0):
    """Unpruned reference: seeded per-unit permutation, first feasible DFS."""
    cand = build_candidates(celltypes, locations, eaps, original_ids, min_amp, max_amp)
    rng = np.random.RandomState(seed)
    order = [[int(i) for i in rng.permutation(c)] for c in cand]
    chosen = []
    n = len(original_ids)

    def dfs(u):
        if u == n:
            return True
        for idx in order[u]:
            if any(float(np.linalg.norm(locations[idx] - locations[p])) < min_dist
                   for p in chosen):
                continue
            chosen.append(idx)
            if dfs(u + 1):
                return True
            chosen.pop()
        return False

    return list(chosen) if dfs(0) else None


class TestConfigScope(unittest.TestCase):
    def test_default_config_validates_fixed_public_scope(self):
        config = load_config()
        self.assertEqual(config["independent_floor_uVrms"], 5.0)
        self.assertEqual(config["target_CAL_SNR_dB"], [10, 15, 20])
        self.assertEqual([d["dataset_id"] for d in config["datasets"]],
                         ["D0", "D1", "D2", "D3", "D4"])
        self.assertEqual(config["units"], 64)
        self.assertEqual(sum(config["per_unit_counts"]), config["total_GT"])
        self.assertEqual(config["total_GT"], 176248)
        self.assertEqual(len(config["original_template_ids"]), 64)

    def test_out_of_scope_config_rejected(self):
        config = load_config()
        for mutate in (
            lambda c: c.__setitem__("independent_floor_uVrms", 10.0),
            lambda c: c.__setitem__("target_CAL_SNR_dB", [10, 25, 40]),
            lambda c: c.__setitem__("datasets", c["datasets"][:2]),
        ):
            bad = json.loads(json.dumps(config))
            mutate(bad)
            with self.assertRaises(StopGeneration):
                validate_config(bad)


class TestPublicAssetMetadata(unittest.TestCase):
    """Metadata inspection of the read-only public assets (no generation)."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.assets = resolve_assets_dir(cls.config)

    def test_trains_npz_schema_and_total(self):
        with np.load(self.assets / "spiketrains.npz") as archive:
            keys = set(archive.files)
            unit_keys = {f"unit_{u:02d}_times_s" for u in range(64)}
            self.assertEqual(keys, unit_keys | {"firing_rates_hz"})
            self.assertEqual(len(archive["firing_rates_hz"]), 64)
            total = sum(len(archive[k]) for k in sorted(unit_keys))
        self.assertEqual(total, 176248)  # firing_rates_hz is metadata, not spikes

    def test_declared_asset_hashes_present(self):
        for key in ("templates_sha256", "spiketrains_sha256",
                    "recording_info_sha256", "annotations_sha256"):
            self.assertEqual(len(self.config["assets"][key]), 64)

    def test_plan_dataset_covers_all_datasets_and_stages(self):
        out = scratch_dir("plan")
        try:
            for dataset in ("D0", "D1", "D2", "D3", "D4"):
                plan = plan_dataset(self.config, dataset, self.assets, stage="all",
                                    output_root=str(out))
                self.assertEqual(plan["dataset_id"], dataset)
                self.assertEqual(plan["bundle"], str(out / dataset))
                self.assertEqual([s["stage"] for s in plan["stages"]], STAGE_ORDER)
                self.assertEqual(plan["profile"]["independent_floor_uVrms"], 5.0)
                self.assertEqual(plan["profile"]["target_CAL_SNR_dB"], [10, 15, 20])
                self.assertTrue(all(f["exists"] for f in plan["assets"]["files"].values()))
        finally:
            shutil.rmtree(out, ignore_errors=True)


class TestPositionSelection(unittest.TestCase):
    def test_native_amp_metric_is_min_based(self):
        eap = np.zeros((5, 4)); eap[2, 1] = -123.0; eap[0, 3] = 50.0
        self.assertEqual(native_amp(eap), 123.0)

    def test_candidates_same_model_strict_amp_exclude_old(self):
        celltypes, locations, _, eaps = tiny_bank()
        original = [0, 1, 4]
        cand = build_candidates(celltypes, locations, eaps, original)
        self.assertEqual(cand[0], [1, 2, 3])   # m1 minus old instance 0
        self.assertEqual(cand[1], [0, 2, 3])
        self.assertEqual(cand[2], [5, 6])      # m2 minus old instance 4
        # strict window: boundary amps are excluded
        eaps2 = eaps.copy()
        eaps2[3, 1, 0] = -50.0   # amp exactly 50 -> excluded (strict)
        cand2 = build_candidates(celltypes, locations, eaps2, original)
        self.assertNotIn(3, cand2[0])

    def test_select_matches_unpruned_first_feasible_reference(self):
        celltypes, locations, _, eaps = tiny_bank()
        original = [0, 1, 4]
        for seed in (20261201, 20261202, 20261203, 20261204, 7):
            got = select_positions(celltypes, locations, eaps, original, seed)
            want = brute_force_selection(celltypes, locations, eaps, original, seed)
            self.assertIsNotNone(want, seed)
            self.assertEqual(got["template_ids"], want, seed)

    def test_select_infeasible_stops(self):
        celltypes, locations, _, eaps = tiny_bank()
        eaps = eaps.copy()
        # Instances 0, 1, 2 fall outside the strict amp window, so units 0 and
        # 1 (both m1) must share instance 3: unique-instance conflict.
        eaps[0, 1, 0] = -10.0
        eaps[1, 1, 0] = -10.0
        eaps[2, 1, 0] = -10.0
        original = [0, 1, 4]
        with self.assertRaises(StopGeneration) as cm:
            select_positions(celltypes, locations, eaps, original, 20261201)
        self.assertIn("INFEASIBLE", str(cm.exception))
        # zero-candidate unit is reported directly
        eaps2 = eaps.copy()
        eaps2[5, 1, 0] = -10.0
        eaps2[6, 1, 0] = -10.0
        with self.assertRaises(StopGeneration) as cm:
            select_positions(celltypes, locations, eaps2, original, 20261201)
        self.assertIn("INFEASIBLE", str(cm.exception))

    def test_assert_selection_catches_violations(self):
        celltypes, locations, _, eaps = tiny_bank()
        original = [0, 1, 4]
        sel = select_positions(celltypes, locations, eaps, original, 20261201)
        checks = assert_selection(celltypes, locations, eaps, original, sel,
                                  n_exc=0, n_inh=3)
        self.assertGreaterEqual(checks["min_pairwise_dist_um"], 25.0)
        broken = {"template_ids": [0, 0, 5]}   # model binding/duplicate broken
        with self.assertRaises(StopGeneration):
            assert_selection(celltypes, locations, eaps, original, broken,
                             n_exc=0, n_inh=3)
        # original instance must be rejected for new realizations
        with self.assertRaises(StopGeneration):
            assert_selection(celltypes, locations, eaps, [3, 1, 4],
                             {"template_ids": [3, 2, 5]}, n_exc=0, n_inh=3)
        # ... but allowed (identity) for D0-style reference layouts
        assert_selection(celltypes, locations, eaps, [3, 1, 4],
                         {"template_ids": [3, 1, 4]}, n_exc=0, n_inh=3,
                         allow_original=True)

    def test_exc_model_binding_counts(self):
        self.assertEqual(len(EXC_MODELS), 4)


class TestCountConditionedSampler(unittest.TestCase):
    NI = [5, 40, 120]
    T = 100000

    def test_oracles_exact_counts_isi_range_determinism(self):
        times = sample_dataset(self.NI, self.T, 20261211)
        report = validate_schedule(times, self.NI, self.T, 40)
        self.assertTrue(all(report[k] for k in
                            ("per_unit_count_ok", "sorted_ok", "min_isi_ok",
                             "in_range_ok")))
        self.assertGreaterEqual(report["min_isi_samples"], 40)
        again = sample_dataset(self.NI, self.T, 20261211)
        self.assertTrue(all(np.array_equal(times[u], again[u]) for u in times))
        other = sample_dataset(self.NI, self.T, 20261212)
        self.assertFalse(np.array_equal(times[1], other[1]))

    def test_compressed_space_infeasible_stops(self):
        with self.assertRaises(StopGeneration):
            sample_unit_times(10, 100, np.random.default_rng(0), 40)

    def test_validator_detects_broken_schedule(self):
        bad = {0: np.array([0, 10, 500], np.int64), 1: np.empty(0, np.int64),
               2: np.empty(0, np.int64)}
        report = validate_schedule(bad, self.NI, self.T, 40)
        self.assertFalse(report["per_unit_count_ok"])
        self.assertFalse(report["min_isi_ok"])

    def test_seconds_roundtrip_exact(self):
        samples = np.arange(0, 1000) * 601
        self.assertTrue(seconds_to_samples_is_exact_roundtrip(samples, 20000))
        t = np.array([0.0, 0.00005, 299.99995])
        self.assertEqual(seconds_to_samples(t, 20000).tolist(), [0, 1, 5999999])


class TestCovarianceAndSeeds(unittest.TestCase):
    def test_covariance_15_over_distance_unit_diagonal(self):
        geometry = np.array([[0.0, 0.0], [36.0, 0.0], [0.0, 36.0]])
        cov, eigmin = spatial_covariance(geometry)
        self.assertTrue(np.allclose(np.diag(cov), 1.0))
        expected = 15.0 / 36.0
        self.assertTrue(np.allclose(cov[0, 1], expected))
        self.assertTrue(np.allclose(cov[1, 2], 15.0 / (36.0 * np.sqrt(2))))
        self.assertGreaterEqual(eigmin, 0.0)

    def test_spatial_chunk_seeds_randomstate_rule(self):
        got = spatial_chunk_seeds(20261018, 60)
        rng = np.random.RandomState(20261018)
        want = [int(rng.randint(1000)) for _ in range(60)]
        self.assertEqual(got, want)

    def test_independent_chunk_rng_pcg64_rule(self):
        want = np.random.default_rng(np.random.SeedSequence([20261118, 7]))
        got = independent_chunk_rng(20261118, 7)
        self.assertTrue(np.array_equal(
            got.standard_normal(5), want.standard_normal(5)))


class TestCalibrationSolver(unittest.TestCase):
    def setUp(self):
        self.amplitude = np.array([100.0, 200.0, 50.0])
        self.main = np.array([0, 1, 0], np.int32)

    def test_moments_and_total_sigma_formula(self):
        # spatial constant 3 -> sp_var 0; unit constant 1 -> floor 5 -> var 0;
        # cross 0. Use varying synthetic input instead:
        rng = np.random.default_rng(1)
        sp, un = rng.normal(size=(200, 2)), rng.normal(size=(200, 2))
        m = cal_moments(sp, un, 5.0)
        self.assertTrue(np.allclose(m["floor_var"], (un * 5.0).var(axis=0)))
        self.assertTrue(np.allclose(m["sp_var"], sp.var(axis=0)))
        sigma = total_sigma(0.0, m)
        self.assertTrue(np.allclose(sigma, np.sqrt(m["floor_var"])))

    def test_snr_is_linear_median(self):
        m = {"sp_var": np.full(2, 1.0), "floor_var": np.full(2, 4.0),
             "cross": np.zeros(2)}
        # sigma(0) = 2 -> ratios [50, 100, 25] -> median 50 -> 33.9794 dB
        self.assertAlmostEqual(cal_snr_db(0.0, m, self.amplitude, self.main),
                               20 * np.log10(50.0), places=9)

    def test_solve_scale_reachable_and_f32_readback(self):
        m = {"sp_var": np.full(2, 1.0), "floor_var": np.full(2, 4.0),
             "cross": np.zeros(2)}
        result = solve_scale(20.0, m, self.amplitude, self.main)
        # median ratio at scale 0 is 100/2 -> target 20 dB needs sigma 10:
        # sqrt(4 + s^2) = 10 -> s = sqrt(96).
        self.assertAlmostEqual(result["root"], float(np.sqrt(96.0)), places=9)
        self.assertEqual(result["scale_f32"], float(np.float32(result["root"])))
        self.assertGreaterEqual(result["scale_f32"], 0.0)
        readback = cal_snr_db(result["scale_f32"], m, self.amplitude, self.main)
        self.assertLess(abs(readback - 20.0), 1e-4)

    def test_solve_scale_unreachable_stops(self):
        m = {"sp_var": np.full(2, 1.0), "floor_var": np.full(2, 10000.0),
             "cross": np.zeros(2)}
        with self.assertRaises(StopGeneration) as cm:
            solve_scale(10.0, m, self.amplitude, self.main)
        self.assertIn("UNREACHABLE_TARGET_SNR", str(cm.exception))


class TestFloat32Association(unittest.TestCase):
    def test_mixed_input_association_order_is_exact(self):
        # Values chosen so float32 association order is observable.
        clean = np.full((1, 2), 16777216.0, np.float32)
        spatial = np.ones((1, 2), np.float32)
        unit = np.full((1, 2), np.float32(0.1))
        got = mixed_input_block_f32(clean, spatial, unit, 1.0)
        self.assertEqual(got.dtype, np.float32)
        step1 = np.asarray(spatial) * np.float32(1.0)
        step2 = np.asarray(clean) + step1
        step3 = np.asarray(unit) * np.float32(5.0)
        want = step2 + step3
        self.assertTrue(np.array_equal(got, want))
        # and NOT equal to the once-rounded float64 association
        once = np.float32(np.asarray(clean, np.float64)
                          + np.asarray(spatial, np.float64) * 1.0
                          + np.asarray(unit, np.float64) * 5.0)
        self.assertEqual(float(got[0, 0]), float(want[0, 0]))
        self.assertEqual(float(got[0, 0]), 16777216.0)  # left-to-right result

    def test_mixed_noise_block(self):
        spatial = np.full((1, 2), 2.0, np.float32)
        unit = np.full((1, 2), np.float32(0.1))
        got = mixed_noise_block_f32(spatial, unit, 1.5)
        want = (spatial * np.float32(1.5) + unit * np.float32(5.0))
        self.assertTrue(np.array_equal(got, want))
        self.assertEqual(got.dtype, np.float32)

    def test_split_accumulator_streamed_equals_full(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(250, 3)).astype(np.float32)
        splits = {"CAL": (0, 100), "DEV": (100, 200), "TEST": (200, 250)}
        acc = SplitAccumulator(splits, 3)
        for first in range(0, len(data), 40):
            acc.add(first, data[first:first + 40])
        for name, (lo, hi) in splits.items():
            self.assertTrue(np.allclose(acc.sigma(name),
                                        data[lo:hi].astype(np.float64).std(axis=0)))

    def test_unit_split_sigma_block_streamed(self):
        rng = np.random.default_rng(4)
        data = rng.normal(size=(300, 2)).astype(np.float32)
        path = HERE / ".test_tmp_sigma.npy"
        np.save(path, data)
        try:
            mm = np.load(path, mmap_mode="r")
            got = unit_split_sigma(mm, 0, 300, channels=2, block=70)
            self.assertTrue(np.allclose(got, data.astype(np.float64).std(axis=0)))
        finally:
            path.unlink()


class TestLedgerCreateOnceAndGates(unittest.TestCase):
    def setUp(self):
        self.bundle = scratch_dir("ledger")
        self.artifact = self.bundle / "component.npy"
        np.save(self.artifact, np.arange(5, dtype=np.float32))

    def tearDown(self):
        shutil.rmtree(self.bundle, ignore_errors=True)

    def test_create_once_never_overwrites(self):
        with self.assertRaises(StopGeneration):
            create_once(self.artifact)
        fresh = self.bundle / "fresh.json"
        create_once(fresh)  # guard only: passes and creates nothing
        self.assertFalse(fresh.exists())

    def test_verified_skip_ok_drift_and_partial(self):
        from shift_aso.mearec.generation import ledger_update, sha256_file
        ledger_update(self.bundle, "component.npy", sha256_file(self.artifact))
        self.assertTrue(verified_skip(self.artifact, self.bundle, "component.npy"))
        # hash drift: STOP, never overwrite or adopt
        np.save(self.artifact, np.arange(9, dtype=np.float32))
        with self.assertRaises(StopGeneration) as cm:
            verified_skip(self.artifact, self.bundle, "component.npy")
        self.assertIn("ARTIFACT_HASH_DRIFT", str(cm.exception))
        # preserved partial without a verifiable hash: STOP
        other = self.bundle / "partial.npy"
        np.save(other, np.ones(3, np.float32))
        with self.assertRaises(StopGeneration) as cm:
            verified_skip(other, self.bundle, "partial.npy")
        self.assertIn("NO_VERIFIABLE_HASH", str(cm.exception))

    def test_write_receipt_commits_after_checks(self):
        payload = {"sha256": "0" * 64, "value": 1}
        path = write_receipt(self.bundle, "stage_receipt.json", payload,
                             artifact_name="component.npy")
        self.assertTrue(path.exists())
        ledger = json.loads((self.bundle / "LEDGER.json").read_text())
        self.assertEqual(ledger["component.npy"], "0" * 64)
        with self.assertRaises(StopGeneration):
            write_receipt(self.bundle, "stage_receipt.json", payload)

    def test_stop_before_write_reserve_math(self):
        original = core.free_bytes

        def fake_free(path):
            return 1000
        core.free_bytes = fake_free
        try:
            self.assertFalse(core.stop_before_write(".", 600, 500)["ok"])
            self.assertTrue(core.stop_before_write(".", 400, 500)["ok"])
        finally:
            core.free_bytes = original
        self.assertEqual(component_bytes(100000, 256), 100000 * 256 * 4)


class TestStageEnvDispatch(unittest.TestCase):
    def test_resolve_thread_env(self):
        self.assertEqual(resolve_thread_env("noise_spatial"),
                         {"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2",
                          "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"})
        for stage in ("select", "train", "clean", "noise_unit", "calibrate"):
            env = resolve_thread_env(stage)
            self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(env["OMP_NUM_THREADS"], "1")
            self.assertEqual(env["MKL_NUM_THREADS"], "1")
            self.assertEqual(env["NUMBA_NUM_THREADS"], "1")

    def test_build_child_argv_carries_bindings(self):
        config = load_config()
        assets = resolve_assets_dir(config)
        from shift_aso.mearec.generation.native import build_child_argv
        ctx = native.GenerationContext(config, HERE / "config.json", "D3", assets,
                                       Path("/tmp/outmea"), 4, 20_000_000_000)
        argv = build_child_argv(ctx, "noise_unit")
        joined = " ".join(argv)
        self.assertIn("-m shift_aso.mearec.generation", joined)
        self.assertIn("--stage noise_unit", joined)
        self.assertIn("--dataset D3", joined)
        self.assertIn("--execute", joined)
        self.assertIn("--assets-dir", joined)
        self.assertIn("--config", joined)
        self.assertIn("--reserve-bytes 20000000000", joined)


class TestDryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = scratch_dir("dryrun")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_module_imports_without_optional_dependencies(self):
        # Fresh interpreter: importing the package and running a dry run must
        # never import the optional MEArec dependency (even when installed).
        env = {**dict(os.environ), "PYTHONPATH": str(REPO),
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "PYTHONDONTWRITEBYTECODE": "1"}
        code = ("import json, sys; "
                "import shift_aso.mearec.generation as g; "
                "imported_after_import = 'MEArec' in sys.modules; "
                "from shift_aso.mearec.generation import cli, core; "
                "import contextlib, io; "
                "buf = io.StringIO(); "
                "ctx = contextlib.redirect_stdout(buf); "
                "\nwith ctx:\n"
                "    rc = cli.main(['--dataset', 'D1', '--output', '/nonexistent-dry'])\n"
                "print(json.dumps({'rc': rc, 'mearec_after_import': imported_after_import, "
                "'mearec_after_dryrun': 'MEArec' in sys.modules}))")
        result = subprocess.run([sys.executable, "-c", code], cwd=str(HERE),
                                env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(report["rc"], 0)
        self.assertFalse(report["mearec_after_import"])
        self.assertFalse(report["mearec_after_dryrun"])

    def test_in_process_dry_run_creates_nothing(self):
        from shift_aso.mearec.generation import cli
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["--dataset", "D2", "--output", str(self.tmp / "out")])
        self.assertEqual(rc, 0)
        plan = json.loads(buf.getvalue())
        self.assertEqual(plan["dataset_id"], "D2")
        self.assertTrue(plan["dry_run"])
        self.assertIn("position_seed", plan["seeds"])
        self.assertEqual([s["stage"] for s in plan["stages"]], STAGE_ORDER)
        self.assertFalse((self.tmp / "out").exists())

    def test_subprocess_module_dry_run(self):
        env = {**dict(os.environ), "PYTHONPATH": str(REPO),
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run(
            [sys.executable, "-m", "shift_aso.mearec.generation",
             "--dataset", "D0", "--output", str(self.tmp / "o2"),
             "--stage", "calibrate"],
            cwd=str(HERE), env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["dataset_id"], "D0")
        self.assertEqual([s["stage"] for s in plan["stages"]], ["calibrate"])
        self.assertFalse((self.tmp / "o2").exists())


class TestIterInput(unittest.TestCase):
    CHANNELS = 2
    SAMPLES = 6

    def _write_bundle(self, tmp):
        rng = np.random.default_rng(11)
        clean = rng.normal(size=(self.SAMPLES, self.CHANNELS)).astype(np.float32)
        spatial = rng.normal(size=(self.SAMPLES, self.CHANNELS)).astype(np.float32)
        unit = rng.normal(size=(self.SAMPLES, self.CHANNELS)).astype(np.float32)
        np.save(tmp / "clean.npy", clean)
        np.save(tmp / "spatial_noise.npy", spatial)
        np.save(tmp / "noise_unit.npy", unit)
        manifest = {
            "status": STATUS_COMPLETE,
            "target_total_CAL_SNR_dB": [10, 15, 20],
            "spatial_scale": {"10": 1.5, "15": 0.75, "20": 0.25},
            "floor_sigma_uV": 5.0,
            "samples": self.SAMPLES,
            "channels": self.CHANNELS,
            "provenance": {"files": {
                name: {"file": f"{name}.npy", "sha256": core.sha256_file(tmp / f"{name}.npy")}
                for name in ("clean", "spatial_noise", "noise_unit")}},
        }
        (tmp / "selection.json").write_text("{}\n")
        np.savez(tmp / "trains.npz", synthetic=np.zeros(1))
        np.savez(tmp / "metadata.npz", synthetic=np.zeros(1))
        for key, filename in (("selection", "selection.json"), ("trains", "trains.npz"),
                              ("metadata", "metadata.npz")):
            manifest["provenance"]["files"][key] = {
                "file": filename, "sha256": core.sha256_file(tmp / filename)}
        binding = tmp / "execution_binding.json"
        binding.write_text("{}")
        manifest["provenance"]["execution_binding_sha256"] = core.sha256_file(binding)
        path = tmp / "generation_manifest.json"
        path.write_text(json.dumps(manifest))
        core.ledger_update(tmp, path.name, core.sha256_file(path))
        return clean, spatial, unit

    def setUp(self):
        self.tmp = scratch_dir("iter")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_iter_input_yields_exact_association_blocks(self):
        clean, spatial, unit = self._write_bundle(self.tmp)
        blocks = list(iter_input(self.tmp, 15, block_samples=4))
        self.assertEqual([first for first, _ in blocks], [0, 4])
        want0 = mixed_input_block_f32(clean[0:4], spatial[0:4], unit[0:4], 0.75)
        self.assertTrue(np.array_equal(blocks[0][1], want0))
        self.assertEqual(blocks[0][1].dtype, np.float32)

    def test_iter_input_rejects_unknown_snr_and_bad_count(self):
        self._write_bundle(self.tmp)
        with self.assertRaises(ValueError):
            list(iter_input(self.tmp, 25))
        with self.assertRaises(ValueError):
            list(iter_input(self.tmp, 10, count=999))
        with self.assertRaises(ValueError):
            list(iter_input(self.tmp, 10, block_samples=0))

    def test_iter_input_requires_complete_manifest(self):
        self._write_bundle(self.tmp)
        manifest = read_generation_manifest(self.tmp)
        manifest["status"] = "RUNNING"
        (self.tmp / "generation_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(StopGeneration):
            list(iter_input(self.tmp, 10))

    def test_iter_input_stops_on_source_hash_drift(self):
        self._write_bundle(self.tmp)
        drifted = np.zeros((self.SAMPLES, self.CHANNELS), np.float32)
        np.save(self.tmp / "clean.npy", drifted)
        with self.assertRaises(StopGeneration) as cm:
            list(iter_input(self.tmp, 10))
        self.assertIn("SOURCE_HASH_DRIFT", str(cm.exception))


class TestNativeGuardsWithoutMEArec(unittest.TestCase):
    """Driver/guard logic testable without the optional MEArec dependency."""

    def setUp(self):
        self.tmp = scratch_dir("guards")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_require_mearec_lazy_optional_dependency(self):
        try:
            module = native.require_mearec()
            self.assertTrue(hasattr(module, "gen_recordings"))
        except StopGeneration as stop:
            self.assertIn("MEAREC_UNAVAILABLE", str(stop))  # honest stop when absent

    def test_journal_stale_scratch_stops(self):
        scratch = self.tmp / "scratch"
        scratch.mkdir()
        params = {"dataset": "D1", "n": 10}
        native.journal(scratch, "unit_white", params)
        with self.assertRaises(StopGeneration) as cm:
            native.journal(scratch, "unit_white", {"dataset": "D1", "n": 11})
        self.assertIn("STALE_SCRATCH_STOP", str(cm.exception))
        # identical params journal idempotently
        native.journal(scratch, "unit_white", params)
        # changed params on a fresh stage name journal fine
        native.journal(scratch, "unit_filter", params)

    def test_done_chunks_resume_bookkeeping(self):
        scratch = self.tmp / "scratch2"
        scratch.mkdir()
        params = {"dataset": "D1"}
        native.journal(scratch, "unit_white", params)
        self.assertEqual(native.done_chunks(scratch, "unit_white", params), set())
        native.record_chunk(scratch, "unit_white", 3)
        native.record_chunk(scratch, "unit_white", 7)
        self.assertEqual(native.done_chunks(scratch, "unit_white", params), {3, 7})
        # differing params invalidate progress (no adoption)
        self.assertEqual(native.done_chunks(scratch, "unit_white", {"dataset": "D2"}),
                         set())

    def test_missing_prerequisite_stops(self):
        config = load_config()
        ctx = native.GenerationContext(config, HERE / "config.json", "D1",
                                       resolve_assets_dir(config), self.tmp / "out",
                                       4, 20_000_000_000)
        with self.assertRaises(StopGeneration) as cm:
            ctx.require("trains.npz", stage="clean")
        self.assertIn("MISSING_PREREQUISITE", str(cm.exception))

    def test_build_params_like_original(self):
        info = {"recordings": {"duration": 60.0, "chunk_duration": 5, "filter": True},
                "spiketrains": {"duration": 60.0},
                "seeds": {"spiketrains": 1, "templates": 2, "recordings": 3}}
        params = native.build_params_like_original(info, 300.0, 20261007)
        self.assertEqual(params["recordings"]["duration"], 300.0)
        self.assertEqual(params["recordings"]["chunk_duration"], 5)
        self.assertEqual(params["spiketrains"]["duration"], 300.0)
        self.assertEqual(set(params["seeds"].values()), {20261007})
        # source info untouched
        self.assertEqual(info["recordings"]["duration"], 60.0)
        self.assertEqual(info["seeds"]["spiketrains"], 1)

    def test_public_trains_loader_ignores_metadata_key(self):
        config = load_config()
        ctx = native.GenerationContext(config, HERE / "config.json", "D0",
                                       resolve_assets_dir(config), self.tmp / "out",
                                       4, 20_000_000_000)
        trains, extra = native.load_public_trains_seconds(ctx)
        self.assertEqual(len(trains), 64)
        self.assertEqual(extra, ["firing_rates_hz"])   # metadata only, not unit 65
        self.assertEqual(sum(len(t) for t in trains.values()), 176248)


if __name__ == "__main__":
    unittest.main()
