"""Native execution stages of the public MEArec multistage generator.

Every stage is fully implemented against the native MEArec 1.9.3 chunk API
(lazy imports; MEArec/neo/quantities/h5py are optional execution-time
dependencies). Nothing in this module runs at import time, and importing it
never imports MEArec.

Failure policy everywhere: deterministic hard stop. Partial outputs are
preserved, never adopted, never retried, and constraints are never relaxed
or redrawn. Stage artifacts are create-once, hash-bound in the bundle
ledger, and receipts are committed last, after all quality assertions pass.
"""
from __future__ import annotations

import copy as _copy
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import core
from .core import StopGeneration

__all__ = ["GenerationContext", "build_context", "asset_guard", "journal",
           "done_chunks", "record_chunk", "run_stage", "stage_select",
           "stage_train", "stage_clean", "stage_noise_spatial",
           "stage_noise_unit", "stage_noise", "stage_calibrate",
           "source_hashes", "env_receipt", "gate_write"]


class GenerationContext:
    """Everything a stage needs; built by the CLI, read-only afterwards."""

    def __init__(self, config, config_path, dataset_id, assets_dir, output_root,
                 workers, reserve_bytes, scratch_dir=None):
        self.config = config
        self.config_path = Path(config_path)
        self.config_sha256 = core.sha256_file(self.config_path)
        self.dataset_id = dataset_id
        self.entry = core.dataset_entry(config, dataset_id)
        self.assets_dir = Path(assets_dir)
        self.assets = core.asset_paths(config, self.assets_dir)
        self.output_root = Path(output_root)
        self.bundle = self.output_root / dataset_id
        self.workers = int(workers)
        self.reserve_bytes = int(reserve_bytes)
        self.scratch_root = (Path(scratch_dir) if scratch_dir is not None
                             else self.output_root / ".scratch")
        # Fixed frozen profile
        self.fs = int(config["fs"])
        self.channels = int(config["channels"])
        self.samples = int(config["samples"])
        self.duration_s = float(config["duration_s"])
        self.block = int(config["filter"]["chunk_samples"])
        self.n_chunks = self.samples // self.block

    def require(self, *names, stage=None):
        """Ledger/hash-verified prerequisite artifacts from earlier stages."""
        for name in names:
            path = self.bundle / name
            if not path.exists():
                raise StopGeneration(
                    f"MISSING_PREREQUISITE: {path} required by stage "
                    f"{stage or 'unknown'}; run the earlier stage first")
            if not core.verified_skip(path, self.bundle, name):
                raise StopGeneration(f"PREREQUISITE_UNVERIFIED: {path}")


def build_context(args) -> GenerationContext:
    config = core.load_config(args.config)
    assets_dir = core.resolve_assets_dir(config, args.assets_dir)
    config_path = (Path(args.config) if args.config
                   else Path(__file__).with_name("config.json"))
    return GenerationContext(config, config_path, args.dataset, assets_dir,
                             args.output, args.workers, args.reserve_bytes,
                             args.scratch_dir)


def source_hashes() -> dict:
    """Hashes of this package's own sources, recorded into every receipt."""
    here = Path(__file__).resolve().parent
    paths = {name: here / name for name in
             ("core.py", "native.py", "cli.py", "__main__.py", "__init__.py", "config.json")}
    paths["mearec_package_init.py"] = here.parent / "__init__.py"
    return {name: core.sha256_file(path) for name, path in paths.items()}


def env_receipt() -> dict:
    """Environment + thread provenance (no cross-platform bit identity claim)."""
    versions = {"python": "{}.{}.{}".format(*sys.version_info[:3]),
                "numpy": np.__version__}
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except Exception:
        versions["scipy"] = None
    try:
        import MEArec
        versions["MEArec"] = MEArec.__version__
    except Exception:
        versions["MEArec"] = None
    return {
        "versions": versions,
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMBA_NUM_THREADS")},
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "platform": sys.platform,
    }


def require_mearec():
    """Lazy optional-dependency import of native MEArec."""
    try:
        import MEArec  # noqa: F401
        import MEArec.generators.recgensteps  # noqa: F401
    except Exception as exc:
        raise StopGeneration(
            f"MEAREC_UNAVAILABLE: native generation requires the optional MEArec "
            f"dependency (pip install 'MEArec==1.9.3'); got {exc!r}") from exc
    import MEArec
    return MEArec


def asset_guard(ctx: GenerationContext) -> dict:
    """Execute-time guard: all four public assets must match declared sha256."""
    checks = {}
    for name, key in (("templates", "templates_sha256"), ("spiketrains", "spiketrains_sha256"),
                      ("recording_info", "recording_info_sha256"),
                      ("annotations", "annotations_sha256")):
        path = ctx.assets[name]
        actual = core.sha256_file(path)
        expected = ctx.config["assets"][key]
        checks[name] = {"ok": actual == expected, "sha256": actual}
        if actual != expected:
            raise StopGeneration(f"ASSET_MISMATCH {name}: {actual} != {expected}")
    return checks


def gate_write(ctx: GenerationContext, needed: int) -> dict:
    """>= 20 GiB reserve gate before any large write."""
    if ctx.reserve_bytes < 20 * (1 << 30):
        raise StopGeneration("RESERVE_TOO_SMALL: at least 20 GiB is required")
    ctx.output_root.mkdir(parents=True, exist_ok=True)
    ctx.scratch_root.mkdir(parents=True, exist_ok=True)
    gates = {name: core.stop_before_write(path, needed, ctx.reserve_bytes)
             for name, path in (("output", ctx.output_root), ("scratch", ctx.scratch_root))}
    if not all(gate["ok"] for gate in gates.values()):
        raise StopGeneration(f"STOP_BEFORE_WRITE: {gates}")
    return gates


# ---------------------------------------------------------------------------
# Scratch journaling (diagnostics only; execution always requires fresh scratch)
# ---------------------------------------------------------------------------
def journal(scratch: Path, stage: str, params: dict) -> Path:
    """Never overwrite a differing META; stale scratch is a hard stop."""
    meta = scratch / f"META_{stage}.json"
    current = dict(sorted(params.items()))
    if meta.exists():
        existing = json.loads(meta.read_text(encoding="utf-8"))
        if existing != current:
            raise StopGeneration(
                f"STALE_SCRATCH_STOP: {meta} params differ; preserve scratch and "
                "investigate (no adoption, no overwrite)")
        return meta
    meta.write_text(json.dumps(current, sort_keys=True), encoding="utf-8")
    return meta


def done_chunks(scratch: Path, stage: str, params: dict) -> set:
    meta = scratch / f"META_{stage}.json"
    try:
        if not meta.exists() or json.loads(meta.read_text(encoding="utf-8")) != \
                dict(sorted(params.items())):
            return set()
    except Exception:
        return set()
    progress = scratch / f"PROGRESS_{stage}.jsonl"
    out = set()
    if progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.add(json.loads(line)["chunk"])
    return out


def record_chunk(scratch: Path, stage: str, chunk: int) -> None:
    with (scratch / f"PROGRESS_{stage}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"chunk": int(chunk)}) + "\n")


def _run_pool(jobs, worker, workers, on_done):
    """Deterministic in-order chunk dispatch; max 2 spatial workers."""
    if not jobs:
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk in pool.map(worker, jobs):
            on_done(chunk)


# ---------------------------------------------------------------------------
# Native chunk workers (verbatim MEArec chunk-function bindings)
# ---------------------------------------------------------------------------
def gen_spatial_chunk(args):
    """chunk_distance_correlated_noise_ per 100000-sample chunk -> raw memmap."""
    chunk, covariance, seeds, raw_path, fs, channels = args
    start, stop = chunk * 100000, (chunk + 1) * 100000
    import quantities as pq
    from MEArec.generators.recgensteps import chunk_distance_correlated_noise_
    result = chunk_distance_correlated_noise_(
        chunk, start, stop, fs * pq.Hz, None, 1.0, covariance, channels,
        False, 300, 2, 1, np.float32, seeds,
    )["additive_noise"]
    target = np.lib.format.open_memmap(raw_path, mode="r+")
    target[start:stop] = result
    target.flush()
    return chunk


def _chunk_filter(source_or_block, chunk, lo, hi, fs, dtype=np.float32):
    """One original chunk_apply_filter_ call: order 3, 300-6000 Hz filtfilt,
    pad 3*int(fs/300)=198 samples, float32."""
    import quantities as pq
    from MEArec.generators.recgensteps import chunk_apply_filter_
    return chunk_apply_filter_(
        chunk, lo, hi, fs * pq.Hz, None, source_or_block, 3 * int(fs / 300),
        np.array([300, 6000]) * pq.Hz, 3, "filtfilt", dtype)["filtered_chunk"]


def filter_component(source, destination, fs, block):
    """Chunked filter of a whole component; exactly once per component."""
    target = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32,
                                       shape=source.shape)
    for chunk, start in enumerate(range(0, len(source), block)):
        stop = min(len(source), start + block)
        target[start:stop] = _chunk_filter(source, chunk, start, stop, fs)
        if chunk % 10 == 0:
            print(json.dumps({"filter": Path(destination).name, "chunk": chunk}),
                  flush=True)
    target.flush()
    return target


def gen_white_chunk(args):
    """PCG64 SeedSequence([seed, chunk]) white draw -> raw memmap."""
    seed, chunk, raw_path, channels, block, samples = args
    rng = core.independent_chunk_rng(seed, chunk)
    noise = np.lib.format.open_memmap(raw_path, mode="r+")
    lo, hi = chunk * block, min((chunk + 1) * block, samples)
    noise[lo:hi] = rng.standard_normal((hi - lo, channels), dtype=np.float32)
    return chunk


def filter_unit_chunk(args):
    raw_path, out_path, chunk, fs, block, samples = args
    lo, hi = chunk * block, min((chunk + 1) * block, samples)
    raw = np.load(raw_path, mmap_mode="r")
    result = _chunk_filter(raw, chunk, lo, hi, fs)
    target = np.lib.format.open_memmap(out_path, mode="r+")
    target[lo:hi] = result
    return chunk


def normalize_unit_chunk(args):
    out_path, sigma, chunk, block, samples = args
    lo, hi = chunk * block, min((chunk + 1) * block, samples)
    target = np.lib.format.open_memmap(out_path, mode="r+")
    target[lo:hi] /= sigma.astype(np.float32)
    return chunk


# ---------------------------------------------------------------------------
# Stage: select
# ---------------------------------------------------------------------------
def _load_bank(ctx):
    import h5py
    require_mearec()  # bank consumers run only in native stages
    with h5py.File(ctx.assets["templates"], "r") as f:
        celltypes = np.array([c.decode() for c in f["celltypes"][:]])
        locations, rotations, eaps = f["locations"][:], f["rotations"][:], f["templates"][:]
    return celltypes, locations, rotations, eaps


def stage_select(ctx: GenerationContext) -> dict:
    """D0: identity reference layout. D1-D4: strict same-model selection."""
    out = ctx.bundle / "selection.json"
    if core.verified_skip(out, ctx.bundle, "selection.json"):
        return {"skipped_verified": True}
    ctx.bundle.mkdir(parents=True, exist_ok=True)
    original_ids = [int(i) for i in ctx.config["original_template_ids"]]
    celltypes, locations, rotations, eaps = _load_bank(ctx)
    if ctx.entry["kind"] == "original_reference":
        selection = {"template_ids": list(original_ids)}
        checks = core.assert_selection(celltypes, locations, eaps, original_ids,
                                       selection, ctx.config["minimum_distance_um"],
                                       ctx.config["template_amplitude_strict_uV"][0],
                                       ctx.config["template_amplitude_strict_uV"][1],
                                       ctx.config["exc_units"], ctx.config["inh_units"],
                                       allow_original=True)
        receipt = core.selection_receipt(celltypes, locations, rotations, eaps,
                                         original_ids, ctx.entry["position_seed"],
                                         selection, checks,
                                         kind="original_reference")
        receipt["note"] = ("D0 keeps the original public asset schedule and the "
                           "original per-unit bank instances (identity layout)")
    else:
        selection = core.select_positions(
            celltypes, locations, eaps, original_ids, ctx.entry["position_seed"],
            ctx.config["minimum_distance_um"],
            ctx.config["template_amplitude_strict_uV"][0],
            ctx.config["template_amplitude_strict_uV"][1])
        checks = core.assert_selection(celltypes, locations, eaps, original_ids,
                                       selection, ctx.config["minimum_distance_um"],
                                       ctx.config["template_amplitude_strict_uV"][0],
                                       ctx.config["template_amplitude_strict_uV"][1])
        receipt = core.selection_receipt(celltypes, locations, rotations, eaps,
                                         original_ids, ctx.entry["position_seed"],
                                         selection, checks)
    # Cross-dataset layout identity is forbidden (sharing instances is not).
    for other in sorted(ctx.output_root.glob("*/selection.json")):
        if other.parent == ctx.bundle:
            continue
        other_ids = json.loads(other.read_text(encoding="utf-8"))["template_ids"]
        if other_ids == receipt["template_ids"]:
            raise StopGeneration(f"LAYOUT_IDENTICAL_TO {other.parent.name}")
    receipt["env"] = env_receipt()
    receipt["sources"] = source_hashes()
    receipt["config_sha256"] = ctx.config_sha256
    core.create_once(out)
    out.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    core.ledger_update(ctx.bundle, "selection.json", core.sha256_file(out))
    return {"done": True, "checks": checks}


# ---------------------------------------------------------------------------
# Stage: train
# ---------------------------------------------------------------------------
def load_public_trains_seconds(ctx) -> dict:
    """The 64 public asset train fields, verbatim seconds.

    firing_rates_hz in the public NPZ is metadata only; it is NOT a spike
    train and never counted (65 keys total: 64 units + 1 metadata vector).
    """
    with np.load(ctx.assets["spiketrains"]) as archive:
        keys = set(archive.files)
        trains = {}
        for unit in range(ctx.config["units"]):
            key = f"unit_{unit:02d}_times_s"
            if key not in keys:
                raise StopGeneration(f"PUBLIC_TRAINS_MISSING: {key}")
            trains[unit] = np.asarray(archive[key], np.float64)
        extra = keys - {f"unit_{u:02d}_times_s" for u in range(ctx.config["units"])}
    return trains, sorted(extra)


def stage_train(ctx: GenerationContext) -> dict:
    """D0: verbatim public schedule. D1-D4: count-conditioned R=40 schedules."""
    out = ctx.bundle / "trains.npz"
    if core.verified_skip(out, ctx.bundle, "trains.npz"):
        return {"skipped_verified": True}
    ctx.require("selection.json", stage="train")
    core.create_once(out)
    counts = [int(c) for c in ctx.config["per_unit_counts"]]
    refractory = int(ctx.config["refractory_samples"])
    if ctx.entry["kind"] == "original_reference":
        trains, extra_keys = load_public_trains_seconds(ctx)
        if extra_keys != ["firing_rates_hz"]:
            raise StopGeneration(f"PUBLIC_TRAINS_UNEXPECTED_KEYS: {extra_keys}")
        seconds = trains
        schedule_note = ("original public asset schedule, verbatim seconds; "
                         "firing_rates_hz is metadata and is not counted")
    else:
        times = core.sample_dataset(counts, ctx.samples, ctx.entry["spiketrain_seed"],
                                    refractory)
        report = core.validate_schedule(times, counts, ctx.samples, refractory)
        if not all(report[k] for k in ("per_unit_count_ok", "sorted_ok",
                                       "min_isi_ok", "in_range_ok")):
            raise StopGeneration(f"TRAIN_SCHEDULE_INVALID: {report}")
        seconds = {u: np.asarray(times[u], np.float64) / ctx.fs for u in times}
        schedule_note = "count-conditioned compressed-slot schedule (R=40, no clip/pad)"
    per_unit = {str(u): int(len(seconds[u])) for u in seconds}
    total = int(sum(per_unit.values()))
    # Exact counts, never trimmed or padded to hide a failure.
    if [per_unit[str(u)] for u in range(len(counts))] != counts:
        raise StopGeneration("TRAIN_COUNT_MISMATCH: per-unit counts differ from config")
    if total != ctx.config["total_GT"]:
        raise StopGeneration(f"TRAIN_TOTAL_MISMATCH: {total} != {ctx.config['total_GT']}")
    lo, hi = 0.0, ctx.duration_s
    for u in range(len(counts)):
        t = seconds[u]
        if len(t) and (t[0] < lo or t[-1] >= hi):
            raise StopGeneration(f"TRAIN_BOUNDS: unit {u} outside [0, {hi}) s")
    np.savez(out, **{f"unit_{u:02d}": seconds[u] for u in sorted(seconds)})
    receipt = {
        "kind": ctx.entry["kind"], "spiketrain_seed": ctx.entry["spiketrain_seed"],
        "schedule": schedule_note,
        "total": total, "per_unit": per_unit,
        "refractory_samples": refractory,
        "validation": (report if ctx.entry["kind"] != "original_reference" else
                       {"source": "public asset spiketrains.npz",
                        "per_unit_count_ok": True, "metadata_keys_ignored": extra_keys}),
        "env": env_receipt(), "sources": source_hashes(),
        "config_sha256": ctx.config_sha256,
        "note": ("trains.npz stores unit_00..unit_63 float64 seconds; D0 values are "
                 "bit-identical to the public asset fields"),
    }
    core.write_receipt(ctx.bundle, "trains_receipt.json", receipt)
    core.ledger_update(ctx.bundle, "trains.npz", core.sha256_file(out))
    return {"done": True, "total": total}


# ---------------------------------------------------------------------------
# Stage: clean
# ---------------------------------------------------------------------------
def build_params_like_original(info: dict, duration_s: float, convolution_seed: int) -> dict:
    """Original build_params: deep-copied public recording params, all seeds =
    the shared convolution control seed."""
    params = _copy.deepcopy(info)
    params["recordings"]["duration"] = duration_s
    params["recordings"]["chunk_duration"] = 5
    params["spiketrains"]["duration"] = duration_s
    params["seeds"] = {key: convolution_seed for key in params["seeds"]}
    return params


def _assert_probe_geometry(geometry: np.ndarray, config: dict) -> None:
    if geometry.shape != (config["channels"], 2):
        raise StopGeneration(f"GEOMETRY_SHAPE: {geometry.shape} != (256, 2)")
    if not np.all(np.isfinite(geometry)):
        raise StopGeneration("GEOMETRY_NON_FINITE")
    if len(np.unique(geometry, axis=0)) != config["channels"]:
        raise StopGeneration("GEOMETRY_DUPLICATE_CONTACTS")
    for axis in range(2):
        coords = np.unique(geometry[:, axis])
        if len(coords) != 16 or not np.allclose(np.diff(coords), config["pitch_um"]):
            raise StopGeneration("GEOMETRY_GRID: expected 16x16 contacts at 36 um pitch")


def stage_clean(ctx: GenerationContext) -> dict:
    """Native convolution + the single original chunk filter pass + metadata."""
    out_clean = ctx.bundle / "clean.npy"
    out_meta = ctx.bundle / "metadata.npz"
    if core.verified_skip(out_clean, ctx.bundle, "clean.npy"):
        ctx.require("metadata.npz", stage="clean")
        return {"skipped_verified": True}
    ctx.require("selection.json", "trains.npz", stage="clean")
    started = time.time()
    mr = require_mearec()
    import h5py
    import neo
    import quantities as pq
    from MEArec.tools import filter_analog_signals
    asset_guard(ctx)
    core.create_once(out_clean)
    core.create_once(out_meta)
    info = json.loads(Path(ctx.assets["recording_info"]).read_text(encoding="utf-8"))
    annotations = json.loads(Path(ctx.assets["annotations"]).read_text(encoding="utf-8"))
    params = build_params_like_original(info, ctx.duration_s,
                                        int(ctx.config["convolution_seed"]))
    params["recordings"].update(noise_level=0, noise_mode="distance-correlated",
                                noise_half_distance=30, filter=False,
                                noise_color=False, extract_waveforms=False)
    trains = np.load(ctx.bundle / "trains.npz")
    if len(trains.files) != ctx.config["units"]:
        raise StopGeneration(f"TRAINS_SCHEMA: {len(trains.files)} units")
    spiketrains = []
    for unit in range(ctx.config["units"]):
        values = np.asarray(trains[f"unit_{unit:02d}"], np.float64)
        st = neo.SpikeTrain(values * pq.s, t_start=0.0 * pq.s,
                            t_stop=ctx.duration_s * pq.s)
        st.annotations = _copy.deepcopy(annotations[unit])
        spiketrains.append(st)
    tempgen = mr.load_templates(str(ctx.assets["templates"]), verbose=False)
    spgen = mr.SpikeTrainGenerator(spiketrains=spiketrains,
                                   params=params["spiketrains"], verbose=False)
    scratch = ctx.scratch_root / f"{ctx.dataset_id}_clean"
    gate_write(ctx, core.component_bytes(ctx.samples, ctx.channels) * 2)
    fresh_scratch(scratch)
    ids = np.asarray(json.loads((ctx.bundle / "selection.json")
                                .read_text(encoding="utf-8"))["template_ids"], dtype=int)
    recgen = mr.gen_recordings(tempgen=tempgen, spgen=spgen, params=params,
                               template_ids=ids, tmp_mode="memmap",
                               tmp_folder=str(scratch), n_jobs=2, verbose=True)
    # Native postconditions BEFORE any receipt is written.
    if np.shape(recgen.recordings) != (ctx.samples, ctx.channels):
        raise StopGeneration("CLEAN_RECORDING_SHAPE: native output differs from fixed profile")
    with h5py.File(ctx.assets["templates"], "r") as f:
        bank_eaps = f["templates"][:][ids]
        bank_locs = f["locations"][:][ids]
        bank_rots = f["rotations"][:][ids]
        bank_ct = np.array([c.decode() for c in f["celltypes"][:]])[ids]
    original_ids = np.asarray(ctx.config["original_template_ids"], int)
    with h5py.File(ctx.assets["templates"], "r") as f:
        orig_ct = np.array([c.decode() for c in f["celltypes"][:]])[original_ids]
    if not np.array_equal(np.asarray(recgen.original_templates, np.float32),
                          bank_eaps.astype(np.float32)):
        raise StopGeneration("CLEAN_TEMPLATES_MISMATCH: recgen != selected bank instances")
    if not np.allclose(np.asarray(recgen.template_locations), bank_locs, atol=0, rtol=0):
        raise StopGeneration("CLEAN_LOCATIONS_MISMATCH")
    if not np.allclose(np.asarray(recgen.template_rotations), bank_rots, atol=0, rtol=0):
        raise StopGeneration("CLEAN_ROTATIONS_MISMATCH")
    if not np.array_equal(np.asarray(recgen.template_celltypes), bank_ct):
        raise StopGeneration("CLEAN_CELLTYPES_MISMATCH")
    if not np.array_equal(bank_ct, orig_ct):
        raise StopGeneration("CLEAN_MODEL_BINDING: per-unit model binding broken")
    if not (len(bank_ct) == ctx.config["units"]
            and sum(c in core.EXC_MODELS for c in bank_ct) == ctx.config["exc_units"]):
        raise StopGeneration("CLEAN_EI_COUNTS")
    geometry = np.asarray(recgen.channel_positions, np.float64)[:, 1:3]
    _assert_probe_geometry(geometry, ctx.config)
    tr_receipt = json.loads((ctx.bundle / "trains_receipt.json").read_text(encoding="utf-8"))
    # Exact original chunk filter, exactly once, straight into the artifact.
    target = np.lib.format.open_memmap(out_clean, mode="w+", dtype=np.float32,
                                       shape=(ctx.samples, ctx.channels))
    for chunk, start in enumerate(range(0, len(recgen.recordings), ctx.block)):
        stop = min(len(recgen.recordings), start + ctx.block)
        target[start:stop] = _chunk_filter(recgen.recordings, chunk, start, stop, ctx.fs)
    target.flush()
    original_templates = np.asarray(recgen.original_templates, np.float32)
    gt_main = np.argmax(np.max(np.abs(original_templates), axis=2), axis=1).astype(np.int32)
    nominal = np.asarray(recgen.templates, np.float64).mean(axis=1)
    filtered = np.empty_like(nominal)
    for unit in range(len(nominal)):
        filtered[unit] = filter_analog_signals(
            nominal[unit].T, np.array([300, 6000]) * pq.Hz, ctx.fs * pq.Hz,
            filter_type="bandpass", mode="filtfilt", order=3).T
    snr_main = np.argmax(np.max(np.abs(filtered), axis=2), axis=1).astype(np.int32)
    amplitudes = np.max(np.abs(filtered[np.arange(len(filtered)), snr_main]), axis=1)
    times_l, units_l = [], []
    for unit, train in enumerate(recgen.spiketrains):
        t = np.rint(np.asarray(train.times.rescale(pq.s).magnitude)
                    * ctx.fs).astype(np.int64)
        keep = (t >= 0) & (t < ctx.samples)
        times_l.append(t[keep])
        units_l.append(np.full(int(keep.sum()), unit, np.int32))
    times_l, units_l = np.concatenate(times_l), np.concatenate(units_l)
    order = np.argsort(times_l, kind="stable")
    gt_times, gt_units = times_l[order], units_l[order]
    # GT postconditions vs the training receipt: no trim, no pad, no relabel.
    if int(len(gt_times)) != tr_receipt["total"]:
        raise StopGeneration("GT_TOTAL_MISMATCH: clean GT != train total")
    got_counts = {str(u): int((gt_units == u).sum()) for u in range(ctx.config["units"])}
    if got_counts != {k: int(v) for k, v in tr_receipt["per_unit"].items()}:
        raise StopGeneration("GT_PER_UNIT_MISMATCH")
    if int(len(gt_times)) != ctx.config["total_GT"]:
        raise StopGeneration(f"GT_TOTAL_FIXED: {len(gt_times)} != 176248")
    for unit in range(ctx.config["units"]):
        unit_times = gt_times[gt_units == unit]
        if len(unit_times) > 1:
            if int(np.diff(unit_times).min()) < int(ctx.config["refractory_samples"]):
                raise StopGeneration(f"GT_MIN_ISI: unit {unit}")
    if int(gt_times.min()) < 0 or int(gt_times.max()) >= ctx.samples:
        raise StopGeneration("GT_BOUNDS")
    np.savez_compressed(out_meta, geometry=geometry, gt_times=gt_times,
                        gt_channels=gt_main[gt_units], gt_units=gt_units,
                        gt_main=gt_main, snr_main=snr_main,
                        template_amplitudes_uV=amplitudes,
                        filtered_templates=filtered,
                        original_templates=original_templates,
                        template_ids=ids,
                        template_locations=np.asarray(recgen.template_locations, np.float64),
                        template_rotations=np.asarray(recgen.template_rotations, np.float64),
                        template_celltypes=np.asarray(recgen.template_celltypes))
    receipt = {
        "clean_sha256": core.sha256_file(out_clean),
        "metadata_sha256": core.sha256_file(out_meta),
        "duration_s": ctx.duration_s, "gt_total": int(len(gt_times)),
        "gt_per_unit": got_counts, "n_units": ctx.config["units"],
        "convolution_seed": ctx.config["convolution_seed"],
        "filter": dict(ctx.config["filter"]),
        "filter_passes": 1,
        "mearec": mr.__version__,
        "postconditions": ("bank-instance/model-binding/geometry/EI/GT asserts "
                           "passed before receipt"),
        "env": env_receipt(), "sources": source_hashes(),
        "config_sha256": ctx.config_sha256,
        "elapsed_s": round(time.time() - started, 1),
    }
    core.write_receipt(ctx.bundle, "clean_receipt.json", receipt)
    core.ledger_update(ctx.bundle, "clean.npy", receipt["clean_sha256"])
    core.ledger_update(ctx.bundle, "metadata.npz", receipt["metadata_sha256"])
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    return {"done": True, "gt_total": receipt["gt_total"]}


# ---------------------------------------------------------------------------
# Stage: noise (spatial + unit)
# ---------------------------------------------------------------------------
def stage_noise_spatial(ctx: GenerationContext) -> dict:
    """Native MEArec distance-correlated chunks; RandomState chunk seeds."""
    out_sp = ctx.bundle / "spatial_noise.npy"
    if core.verified_skip(out_sp, ctx.bundle, "spatial_noise.npy"):
        return {"skipped_verified": True}
    ctx.require("metadata.npz", stage="noise_spatial")
    require_mearec()
    asset_guard(ctx)
    core.create_once(out_sp)
    gate_write(ctx, core.component_bytes(ctx.samples, ctx.channels) * 2)
    scratch = ctx.scratch_root / f"{ctx.dataset_id}_spatial"
    fresh_scratch(scratch)
    meta = np.load(ctx.bundle / "metadata.npz", allow_pickle=False)
    covariance, eigmin = core.spatial_covariance(meta["geometry"])
    seeds = core.spatial_chunk_seeds(ctx.entry["spatial_master"], ctx.n_chunks)
    params = {"dataset": ctx.dataset_id, "spatial_master": ctx.entry["spatial_master"],
              "n_samples": ctx.samples}
    journal(scratch, "spatial_raw", params)
    raw = scratch / "noise_raw.npy"
    if not raw.exists():
        mm = np.lib.format.open_memmap(raw, mode="w+", dtype=np.float32,
                                       shape=(ctx.samples, ctx.channels))
        mm.flush()
        del mm
    todo = [c for c in range(ctx.n_chunks)
            if c not in done_chunks(scratch, "spatial_raw", params)]
    workers = min(ctx.workers, 2)  # BLAS2 x 2 = 4 total compute threads
    _run_pool([(c, covariance, seeds, str(raw), ctx.fs, ctx.channels) for c in todo],
              gen_spatial_chunk, workers,
              lambda c: record_chunk(scratch, "spatial_raw", c))
    filter_component(np.load(raw, mmap_mode="r"), out_sp, ctx.fs, ctx.block)
    receipt = {
        "sha256": core.sha256_file(out_sp), "bytes": out_sp.stat().st_size,
        "spatial_master": ctx.entry["spatial_master"],
        "chunk_seed_rule": "RandomState(master).randint(1000) per 100000-sample chunk",
        "n_chunks": ctx.n_chunks,
        "covariance_min_eig": eigmin,
        "covariance_rule": "15 um / pairwise distance, unit diagonal",
        "filter": dict(ctx.config["filter"]),
        "workers": workers,
        "env": env_receipt(), "sources": source_hashes(),
        "config_sha256": ctx.config_sha256,
    }
    core.write_receipt(ctx.bundle, "spatial_receipt.json", receipt)
    core.ledger_update(ctx.bundle, "spatial_noise.npy", receipt["sha256"])
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    return {"done": True}


def stage_noise_unit(ctx: GenerationContext) -> dict:
    """Independent PCG64 noise: white -> filter -> CAL-sigma normalize."""
    out_un = ctx.bundle / "noise_unit.npy"
    if core.verified_skip(out_un, ctx.bundle, "noise_unit.npy"):
        return {"skipped_verified": True}
    require_mearec()
    asset_guard(ctx)
    core.create_once(out_un)
    gate_write(ctx, core.component_bytes(ctx.samples, ctx.channels) * 2)
    scratch = ctx.scratch_root / f"{ctx.dataset_id}_unit"
    fresh_scratch(scratch)
    params = {"dataset": ctx.dataset_id, "independent": ctx.entry["independent"],
              "n_samples": ctx.samples}
    for st in ("unit_white", "unit_filter", "unit_normalize"):
        journal(scratch, st, params)
    wraw = scratch / "white.npy"
    if not wraw.exists():
        mm = np.lib.format.open_memmap(wraw, mode="w+", dtype=np.float32,
                                       shape=(ctx.samples, ctx.channels))
        mm.flush()
        del mm
    mm = np.lib.format.open_memmap(out_un, mode="w+", dtype=np.float32,
                                   shape=(ctx.samples, ctx.channels))
    mm.flush()
    del mm
    workers = min(ctx.workers, 4)  # LIBS1 processes; 4 total compute threads
    todo = [c for c in range(ctx.n_chunks)
            if c not in done_chunks(scratch, "unit_white", params)]
    _run_pool([(ctx.entry["independent"], c, str(wraw), ctx.channels, ctx.block,
                ctx.samples) for c in todo],
              gen_white_chunk, workers,
              lambda c: record_chunk(scratch, "unit_white", c))
    todo = [c for c in range(ctx.n_chunks)
            if c not in done_chunks(scratch, "unit_filter", params)]
    _run_pool([(str(wraw), str(out_un), c, ctx.fs, ctx.block, ctx.samples) for c in todo],
              filter_unit_chunk, workers,
              lambda c: record_chunk(scratch, "unit_filter", c))
    cal_frames = int(ctx.config["CAL_frames"])
    noise = np.load(out_un, mmap_mode="r")
    sigma = core.unit_split_sigma(noise, 0, min(cal_frames, ctx.samples),
                                  ctx.channels, ctx.block)
    del noise
    if not np.all(sigma > 0):
        raise StopGeneration("UNIT_SIGMA_NONPOSITIVE; STOP")
    todo = [c for c in range(ctx.n_chunks)
            if c not in done_chunks(scratch, "unit_normalize", params)]
    _run_pool([(str(out_un), sigma, c, ctx.block, ctx.samples) for c in todo],
              normalize_unit_chunk, workers,
              lambda c: record_chunk(scratch, "unit_normalize", c))
    noise = np.load(out_un, mmap_mode="r")
    err = float(np.max(np.abs(
        core.unit_split_sigma(noise, 0, min(cal_frames, ctx.samples),
                              ctx.channels, ctx.block) - 1.0)))
    del noise
    if err >= 1e-5:
        raise StopGeneration(f"CAL_UNIT_SIGMA_ERROR {err} >= 1e-5; STOP, no renormalization")
    receipt = {
        "sha256": core.sha256_file(out_un), "bytes": out_un.stat().st_size,
        "independent_seed": ctx.entry["independent"],
        "rng": "PCG64 SeedSequence([seed, chunk]) per 100000-sample chunk",
        "normalization": (f"per-channel CAL(0-{cal_frames // ctx.fs} s) sigma, "
                          "full-duration float32 division"),
        "cal_unit_sigma_max_abs_error": err,
        "filter": dict(ctx.config["filter"]),
        "workers": workers,
        "env": env_receipt(), "sources": source_hashes(),
        "config_sha256": ctx.config_sha256,
    }
    core.write_receipt(ctx.bundle, "unit_receipt.json", receipt)
    core.ledger_update(ctx.bundle, "noise_unit.npy", receipt["sha256"])
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    return {"done": True, "cal_unit_sigma_max_abs_error": err}


def build_child_argv(ctx: GenerationContext, stage: str, module=None) -> list:
    """Fresh per-stage subprocess argv with the correct env (pure; tested)."""
    argv = [sys.executable, "-m",
            module or "shift_aso.mearec.generation",
            "--dataset", ctx.dataset_id, "--output", str(ctx.output_root),
            "--assets-dir", str(ctx.assets_dir), "--config", str(ctx.config_path),
            "--stage", stage, "--workers", str(ctx.workers),
            "--reserve-bytes", str(ctx.reserve_bytes), "--execute"]
    if ctx.scratch_root != ctx.output_root / ".scratch":
        argv += ["--scratch-dir", str(ctx.scratch_root)]
    return argv


def run_child(ctx: GenerationContext, stage: str) -> None:
    """Run one internal stage as a fresh subprocess (env set pre-NumPy there)."""
    env = {**dict(os.environ), **core.resolve_thread_env(stage)}
    result = subprocess.run(build_child_argv(ctx, stage), env=env)
    if result.returncode != 0:
        raise StopGeneration(f"CHILD_STAGE_FAILED: {stage} rc={result.returncode}")


def stage_noise(ctx: GenerationContext) -> dict:
    """Public noise stage: spatial here, unit in a fresh LIBS1 subprocess."""
    spatial = stage_noise_spatial(ctx)
    run_child(ctx, "noise_unit")
    return {"done": True, "spatial": spatial}


# ---------------------------------------------------------------------------
# Stage: calibrate
# ---------------------------------------------------------------------------
def stage_calibrate(ctx: GenerationContext) -> dict:
    """Total CAL SNR 10/15/20 dB scalars with actual component covariance."""
    out = ctx.bundle / "generation_manifest.json"
    if core.verified_skip(out, ctx.bundle, "generation_manifest.json"):
        return {"skipped_verified": True}
    ctx.require("metadata.npz", "spatial_noise.npy", "noise_unit.npy",
                "clean.npy", "selection.json", "trains.npz", stage="calibrate")
    core.create_once(out)
    meta = np.load(ctx.bundle / "metadata.npz", allow_pickle=False)
    amplitude, main_channel = meta["template_amplitudes_uV"], meta["snr_main"]
    spatial = np.load(ctx.bundle / "spatial_noise.npy", mmap_mode="r")
    unit = np.load(ctx.bundle / "noise_unit.npy", mmap_mode="r")
    cal_n = min(int(ctx.config["CAL_frames"]), len(spatial))
    moments = core.cal_moments(spatial[:cal_n], unit[:cal_n],
                               float(ctx.config["independent_floor_uVrms"]))
    if not (np.all(np.isfinite(moments["sp_var"])) and np.all(moments["sp_var"] > 0)
            and np.all(np.isfinite(moments["floor_var"]))
            and np.all(moments["floor_var"] > 0)):
        raise StopGeneration("CAL_MOMENTS_NON_FINITE_OR_NONPOSITIVE; STOP")
    ceiling = core.cal_snr_db(0.0, moments, amplitude, main_channel)
    if ceiling <= max(ctx.config["target_CAL_SNR_dB"]):
        raise StopGeneration(
            f"FLOOR_ONLY_CEILING_TOO_LOW: {ceiling} dB <= 20 dB; STOP")
    scales = {str(t): core.solve_scale(float(t), moments, amplitude,
                                       main_channel)["scale_f32"]
              for t in ctx.config["target_CAL_SNR_dB"]}
    splits = {"CAL": (0, int(ctx.config["CAL_frames"])),
              "DEV": tuple(int(v) for v in ctx.config["DEV_bounds"]),
              "TEST": tuple(int(v) for v in ctx.config["TEST_bounds"])}
    accums = {str(t): core.SplitAccumulator(splits, ctx.channels)
              for t in ctx.config["target_CAL_SNR_dB"]}
    floor = float(ctx.config["independent_floor_uVrms"])
    for first in range(0, len(spatial), ctx.block):
        last = min(first + ctx.block, len(spatial))
        sp = np.asarray(spatial[first:last])
        un = np.asarray(unit[first:last])
        for t in ctx.config["target_CAL_SNR_dB"]:
            accums[str(t)].add(first, core.mixed_noise_block_f32(sp, un, scales[str(t)],
                                                                 floor))
    summaries = {str(t): accums[str(t)].summaries(amplitude, main_channel)
                 for t in ctx.config["target_CAL_SNR_dB"]}
    for t in ctx.config["target_CAL_SNR_dB"]:
        error = abs(summaries[str(t)]["CAL"]["measured_total_SNR_dB"] - float(t))
        if error >= 1e-4:
            raise StopGeneration(
                f"CAL_VERIFICATION_FAILED: |{error}| >= 1e-4 dB at target {t}")
    files = {}
    for name, filename in (("selection", "selection.json"), ("trains", "trains.npz"),
                           ("clean", "clean.npy"), ("metadata", "metadata.npz"),
                           ("spatial_noise", "spatial_noise.npy"),
                           ("noise_unit", "noise_unit.npy")):
        files[name] = {"file": filename, "sha256": core.sha256_file(ctx.bundle / filename),
                       "bytes": (ctx.bundle / filename).stat().st_size}
    manifest = {
        "status": core.STATUS_COMPLETE,
        "schema": ctx.config["schema"],
        "dataset_id": ctx.dataset_id,
        "kind": ctx.entry["kind"],
        "target_total_CAL_SNR_dB": [int(t) for t in ctx.config["target_CAL_SNR_dB"]],
        "spatial_scale": scales,
        "floor_sigma_uV": floor,
        "floor_only_CAL_SNR_dB": ceiling,
        "fs": ctx.fs, "channels": ctx.channels, "samples": int(len(spatial)),
        "split_seconds": {"CAL": [0, cal_n / ctx.fs],
                          "DEV": [splits["DEV"][0] / ctx.fs, splits["DEV"][1] / ctx.fs],
                          "TEST": [splits["TEST"][0] / ctx.fs, splits["TEST"][1] / ctx.fs]},
        "split_samples": {k: list(v) for k, v in splits.items()},
        "summaries": summaries,
        "input_construction": ("float32(clean + float32(spatial_scale)*spatial + "
                               "float32(5)*independent_unit)"),
        "SNR_definition": ("20log10(median_u(A_u / sigma_total_CAL[snr_main_u])); "
                           "linear median of peak amplitude over total noise sigma"),
        "calibration": ("spatial scalar solved on CAL with actual component covariance, "
                        "brentq xtol 1e-12, rounded to float32 and verified from the "
                        "actual float32 mixed noise"),
        "seeds": {k: ctx.entry[k] for k in
                  ("position_seed", "spiketrain_seed", "spatial_master", "independent")},
        "convolution_seed": ctx.config["convolution_seed"],
        "template_ids": [int(i) for i in meta["template_ids"]],
        "gt_total": int(len(meta["gt_times"])),
        "virtual_mixtures": ("no duplicate full raw waveforms by SNR; use "
                             "iter_input(dataset_root, snr) for the exact association"),
        "provenance": {"files": files,
                       "execution_binding_sha256": core.sha256_file(ctx.bundle / "execution_binding.json"),
                       "receipts": {name: {"file": f"{name}_receipt.json"}
                                    for name in ("trains", "clean", "spatial", "unit")},
                       "config": {"file": "config.json", "sha256": ctx.config_sha256},
                       "sources": source_hashes(),
                       "env": env_receipt(),
                       "limitations": list(ctx.config["limitations"]),
                       "bit_identity": ("no cross-platform bitwise identity guarantee; "
                                        "CPU/BLAS can change noise realizations")},
        "encoder_note": ("dataset-only release: no LC encoding stage; quantization is "
                         "downstream detector behavior, not part of this bundle"),
    }
    out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    core.ledger_update(ctx.bundle, "generation_manifest.json", core.sha256_file(out))
    return {"done": True, "spatial_scale": scales, "ceiling_db": ceiling}


_STAGE_FUNCTIONS = {"select": stage_select, "train": stage_train, "clean": stage_clean,
                    "noise_spatial": stage_noise_spatial, "noise_unit": stage_noise_unit,
                    "noise": stage_noise, "calibrate": stage_calibrate}


def fresh_scratch(path):
    """Preserve failed scratch; never resume/adopt unverified chunk progress."""
    path = Path(path)
    if path.exists():
        raise StopGeneration(f"PRESERVED_SCRATCH: {path}; use a new output/scratch root")
    path.mkdir(parents=True, exist_ok=False)


def bind_execution(ctx):
    """Bind every stage, including skips, to one source/config/asset closure."""
    if ctx.workers < 1 or ctx.workers > 4:
        raise StopGeneration("WORKERS: expected 1..4")
    if ctx.reserve_bytes < 20 * (1 << 30):
        raise StopGeneration("RESERVE_TOO_SMALL: at least 20 GiB is required")
    checks = asset_guard(ctx)
    binding = {"dataset_id": ctx.dataset_id, "config_sha256": ctx.config_sha256,
               "sources": source_hashes(), "assets": checks}
    path = ctx.bundle / "execution_binding.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != binding:
            raise StopGeneration("EXECUTION_BINDING_DRIFT: source/config/assets changed")
    else:
        if ctx.bundle.exists() and any(ctx.bundle.iterdir()):
            raise StopGeneration("UNBOUND_BUNDLE: preserve existing files; use a new output root")
        ctx.bundle.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(binding, indent=1) + "\n", encoding="utf-8")


def run_stage(ctx: GenerationContext, stage: str) -> dict:
    """Execute one stage only after validating the immutable execution closure."""
    if stage not in _STAGE_FUNCTIONS:
        raise StopGeneration(f"UNKNOWN_STAGE: {stage}")
    bind_execution(ctx)
    return _STAGE_FUNCTIONS[stage](ctx)
