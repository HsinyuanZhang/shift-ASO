"""Pure portable core of the public MEArec multistage raw-data generator.

Scientific scope is fixed and narrow: 256-channel SqMEA-16-36 recordings at
20 kHz, 300 s, 64 units / 176248 spikes, an independent 5 uVrms noise floor,
and total CAL SNR targets 10/15/20 dB for datasets D0-D4 only. No other
profiles, studies, or parameter sweeps are defined or exposed by this module.

Every numerical routine here is a semantics-identical port of the frozen
production semantics (strict same-model template selection, count-conditioned
R=40 compressed-slot sampling, 15 um/d spatial covariance with unit diagonal,
PCG64/RandomState chunk seeding, total-CAL-SNR brentq solver with float32
scale rounding, and the exact float32 mixed-input association order). The
module imports only NumPy and the standard library; MEArec is an optional
execution-time dependency used only by :mod:`shift_aso.mearec.generation.native`.

Importing this module never parses the CLI, runs a stage, or creates output.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import numpy as np

__all__ = [
    "StopGeneration",
    "STATUS_COMPLETE",
    "STAGE_ORDER",
    "PUBLIC_STAGES",
    "EXC_MODELS",
    "load_config",
    "validate_config",
    "dataset_entry",
    "package_root",
    "resolve_assets_dir",
    "asset_paths",
    "asset_plan",
    "plan_dataset",
    "native_amp",
    "build_candidates",
    "select_positions",
    "assert_selection",
    "selection_receipt",
    "sample_unit_times",
    "sample_dataset",
    "validate_schedule",
    "seconds_to_samples",
    "seconds_to_samples_is_exact_roundtrip",
    "spatial_covariance",
    "spatial_chunk_seeds",
    "independent_chunk_rng",
    "cal_moments",
    "total_sigma",
    "cal_snr_db",
    "solve_scale",
    "mixed_noise_block_f32",
    "mixed_input_block_f32",
    "unit_split_sigma",
    "SplitAccumulator",
    "sha256_file",
    "sha256_text",
    "component_bytes",
    "free_bytes",
    "stop_before_write",
    "ledger_path",
    "ledger_read",
    "ledger_update",
    "create_once",
    "verified_skip",
    "read_generation_manifest",
    "iter_input",
    "resolve_thread_env",
    "LIBS1",
    "SPATIAL_ENV",
]

#: Manifest status of a fully generated and calibrated dataset bundle.
STATUS_COMPLETE = "PASS_TOTAL_SNR_FLOOR5_MULTISEED"

#: Internal execution order. ``noise_spatial`` and ``noise_unit`` are the two
#: halves of the public ``noise`` stage; each runs as a fresh subprocess so
#: its thread environment is fixed before the first NumPy import.
STAGE_ORDER = ["select", "train", "clean", "noise_spatial", "noise_unit", "calibrate"]

#: Stage names accepted on the public CLI.
PUBLIC_STAGES = ["all", "select", "train", "clean", "noise", "calibrate"]

#: The four excitatory L5 PC models of the fixed 51/13 E/I binding.
EXC_MODELS = ("L5_STPC_cADpyr232_1", "L5_TTPC1_cADpyr232_1",
              "L5_TTPC2_cADpyr232_1", "L5_UTPC_cADpyr232_1")

#: Thread environments (set before the first NumPy import by the CLI).
LIBS1 = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
         "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}
SPATIAL_ENV = {**LIBS1, "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"}

BUNDLE_FILES = ("selection.json", "trains.npz", "clean.npy", "metadata.npz",
                "spatial_noise.npy", "noise_unit.npy", "generation_manifest.json")


class StopGeneration(RuntimeError):
    """Deterministic hard stop. Never retried, relaxed, or adopted."""


def resolve_thread_env(stage: str) -> dict:
    """Pure mapping of an internal stage to its required thread environment."""
    return dict(SPATIAL_ENV) if stage == "noise_spatial" else dict(LIBS1)


# ---------------------------------------------------------------------------
# Configuration (parent-owned, read-only) and dry-run planning
# ---------------------------------------------------------------------------
def validate_config(config: dict) -> dict:
    """Validate the frozen public schema; reject anything outside the scope."""
    if config.get("schema") != "shift_aso_mearec_multiseed_floor5_v1":
        raise StopGeneration("CONFIG_SCHEMA_UNSUPPORTED: " + repr(config.get("schema")))
    if float(config["independent_floor_uVrms"]) != 5.0:
        raise StopGeneration("CONFIG_SCOPE: independent floor must be 5 uVrms")
    if list(config["target_CAL_SNR_dB"]) != [10, 15, 20]:
        raise StopGeneration("CONFIG_SCOPE: targets must be exactly [10, 15, 20] dB")
    ids = [d["dataset_id"] for d in config["datasets"]]
    if ids != ["D0", "D1", "D2", "D3", "D4"]:
        raise StopGeneration("CONFIG_SCOPE: datasets must be exactly D0..D4")
    for entry in config["datasets"]:
        for key in ("position_seed", "spiketrain_seed", "spatial_master", "independent"):
            if not isinstance(entry.get(key), int):
                raise StopGeneration(f"CONFIG_SEEDS: {entry.get('dataset_id')} missing {key}")
    if (config["units"] != 64 or config["channels"] != 256 or config["fs"] != 20000
            or config["samples"] != 6000000 or config["duration_s"] != 300
            or config["exc_units"] != 51 or config["inh_units"] != 13):
        raise StopGeneration("CONFIG_SCOPE: fixed 64-unit/256-ch/20 kHz/300 s geometry violated")
    original = config["original_template_ids"]
    counts = config["per_unit_counts"]
    if len(original) != 64 or len(set(original)) != 64:
        raise StopGeneration("CONFIG_SCOPE: original_template_ids must be 64 unique ids")
    if len(counts) != 64 or sum(counts) != config["total_GT"] or config["total_GT"] != 176248:
        raise StopGeneration("CONFIG_SCOPE: per-unit counts must sum to 176248")
    if config["filter"]["order"] != 3 or list(config["filter"]["cutoff_hz"]) != [300, 6000] \
            or config["filter"]["mode"] != "filtfilt" or config["filter"]["chunk_samples"] != 100000:
        raise StopGeneration("CONFIG_SCOPE: original chunk filter spec violated")
    return config


def load_config(path=None) -> dict:
    """Load and validate the parent-owned frozen config (default: package copy)."""
    packaged = Path(__file__).with_name("config.json")
    source = Path(path) if path is not None else packaged
    config = json.loads(source.read_text(encoding="utf-8"))
    validate_config(config)
    if path is not None and config != json.loads(packaged.read_text(encoding="utf-8")):
        raise StopGeneration("CONFIG_FROZEN: external config must match the packaged public recipe")
    return config


def dataset_entry(config: dict, dataset_id: str) -> dict:
    for entry in config["datasets"]:
        if entry["dataset_id"] == dataset_id:
            return entry
    raise StopGeneration(f"UNKNOWN_DATASET: {dataset_id}")


def package_root() -> Path:
    """Repository root derived from this file (package lives three levels deep)."""
    return Path(__file__).resolve().parents[3]


def resolve_assets_dir(config: dict, assets_dir=None) -> Path:
    """Resolve the public asset directory: explicit arg, package root, then cwd."""
    rel = Path(config["assets"]["directory"])
    if assets_dir is not None:
        return Path(assets_dir).resolve()
    for candidate in (package_root(), Path.cwd()):
        resolved = (candidate / rel).resolve()
        if (resolved / config["assets"]["templates_file"]).exists():
            return resolved
    raise StopGeneration(
        "ASSETS_DIR_UNRESOLVED: pass --assets-dir pointing at the directory that "
        f"contains {rel}/ (templates.h5, spiketrains.npz, recording_params.json, "
        "spiketrain_annotations.json)")


def asset_paths(config: dict, assets_dir: Path) -> dict:
    names = {"templates": "templates_file", "spiketrains": "spiketrains_file",
             "recording_info": "recording_info_file", "annotations": "annotations_file"}
    return {name: Path(assets_dir) / config["assets"][key] for name, key in names.items()}


def asset_plan(config: dict, assets_dir: Path) -> dict:
    """Metadata-only asset description (existence + size + declared sha256).

    Dry-run never hashes asset contents; execution re-verifies every declared
    sha256 before doing work.
    """
    files = {}
    for name, key in (("templates", "templates_file"), ("spiketrains", "spiketrains_file"),
                      ("recording_info", "recording_info_file"),
                      ("annotations", "annotations_file")):
        path = Path(assets_dir) / config["assets"][key]
        files[name] = {
            "file": config["assets"][key],
            "path": str(path),
            "exists": bool(path.exists()),
            "bytes": int(path.stat().st_size) if path.exists() else None,
            "sha256_declared": config["assets"][key.replace("_file", "_sha256")],
        }
    return {"directory": str(Path(assets_dir)), "files": files}


_STAGE_DESCRIPTIONS = {
    "select": ("Strict same-model/strict-amplitude/min-dist bank selection "
               "(D0: identity reference layout). Output selection.json."),
    "train": ("D0: original public asset schedule verbatim; D1-D4: count-conditioned "
              "R=40 compressed-slot schedules. Output trains.npz."),
    "clean": ("Native MEArec convolution (noise_level=0, filter=False) with the selected "
              "bank instances, one original chunk filter pass. Outputs clean.npy, metadata.npz."),
    "noise_spatial": ("Native MEArec distance-correlated noise chunks, RandomState chunk "
                      "seeds, chunked order-3 filtfilt. Output spatial_noise.npy."),
    "noise_unit": ("Independent PCG64 SeedSequence([seed, chunk]) noise, filtered, "
                   "per-channel CAL(0-6 s) sigma normalized. Output noise_unit.npy."),
    "calibrate": ("Total CAL SNR 10/15/20 dB scalar solver with actual component "
                  "covariance, float32 scales, mixed-noise verification. Output "
                  "generation_manifest.json."),
}


def plan_dataset(config: dict, dataset_id: str, assets_dir: Path,
                 stage: str = "all", output_root=None) -> dict:
    """Dry-run plan built from config + asset metadata only."""
    entry = dataset_entry(config, dataset_id)
    stages = STAGE_ORDER if stage == "all" else (
        ["noise_spatial", "noise_unit"] if stage == "noise" else [stage])
    output_root = Path(output_root) if output_root is not None else None
    plan = {
        "dry_run": True,
        "note": ("plan only: no output creation, no MEArec import, no generation; "
                 "execution requires --execute"),
        "schema": config["schema"],
        "dataset_id": dataset_id,
        "kind": entry["kind"],
        "seeds": {k: entry[k] for k in
                  ("position_seed", "spiketrain_seed", "spatial_master", "independent")},
        "convolution_seed": config["convolution_seed"],
        "profile": {
            "probe": config["probe"], "pitch_um": config["pitch_um"],
            "channels": config["channels"], "fs": config["fs"],
            "samples": config["samples"], "duration_s": config["duration_s"],
            "units": config["units"], "exc_units": config["exc_units"],
            "inh_units": config["inh_units"], "total_GT": config["total_GT"],
            "independent_floor_uVrms": config["independent_floor_uVrms"],
            "target_CAL_SNR_dB": list(config["target_CAL_SNR_dB"]),
            "template_amplitude_strict_uV": list(config["template_amplitude_strict_uV"]),
            "minimum_distance_um": config["minimum_distance_um"],
            "refractory_samples": config["refractory_samples"],
            "filter": dict(config["filter"]),
            "splits_samples": {"CAL": [0, config["CAL_frames"]],
                               "DEV": list(config["DEV_bounds"]),
                               "TEST": list(config["TEST_bounds"])},
        },
        "assets": asset_plan(config, assets_dir),
        "stages": [{"stage": name,
                    "thread_env": resolve_thread_env(name),
                    "description": _STAGE_DESCRIPTIONS[name],
                    "fresh_subprocess": True}
                   for name in stages],
        "output_root": str(output_root) if output_root is not None else None,
        "bundle": str((output_root / dataset_id) if output_root is not None else ""),
        "bundle_files": list(BUNDLE_FILES),
        "limitations": list(config["limitations"]),
        "encoder_note": ("no encoding stage in this dataset release; quantization is a "
                         "downstream detector concern, deliberately not part of the bundle"),
    }
    return plan


# ---------------------------------------------------------------------------
# Strict same-model position selection (native MEArec semantics)
# ---------------------------------------------------------------------------
MAX_BACKTRACK_STEPS = 10_000_000


def native_amp(eap: np.ndarray) -> float:
    """Native select_templates amplitude metric (verbatim)."""
    return float(np.max(np.abs(np.min(eap))))


def build_candidates(celltypes, locations, eaps, original_ids,
                     min_amp=50.0, max_amp=300.0):
    """Per-unit candidates: same model, != original instance, STRICT amp window.

    Boundary limits are None in the original construction, so the native
    is_position_within_boundaries imposes no extra spatial restriction.
    """
    n_units = len(original_ids)
    cand = []
    for u in range(n_units):
        model = celltypes[original_ids[u]]
        same_model = np.flatnonzero(celltypes == model)
        keep = []
        for idx in same_model:
            idx = int(idx)
            if idx == int(original_ids[u]):
                continue
            amp = native_amp(eaps[idx])
            if min_amp < amp < max_amp:  # STRICT, native semantics
                keep.append(idx)
        cand.append(keep)
    return cand


def select_positions(celltypes, locations, eaps, original_ids, position_seed,
                     min_dist=25.0, min_amp=50.0, max_amp=300.0):
    """Deterministic same-model selection: seeded per-unit permutation order,
    first feasible solution via backtracking with necessary-condition pruning
    (empty domain, per-model capacity) and failed-state memoization.

    The pruning and memoization are exact: they only discard provably failed
    states, so the returned solution equals the unpruned first-feasible
    search. On infeasibility this raises; there is no fallback, relaxation,
    or redraw.
    """
    assert min_dist > 0, "positive distance also enforces unique instances"
    n = len(original_ids)
    cand = build_candidates(celltypes, locations, eaps, original_ids, min_amp, max_amp)
    empty = [u for u in range(n) if not cand[u]]
    if empty:
        raise StopGeneration(
            f"INFEASIBLE: units with zero amp-window candidates: {empty}")
    rng = np.random.RandomState(position_seed)
    order = [[int(i) for i in rng.permutation(c)] for c in cand]
    masks = [sum(1 << int(i) for i in c) for c in cand]
    # Exact original scalar norm/predicate, precomputed once.
    conflicts = [sum(1 << b for b in range(len(locations))
                     if float(np.linalg.norm(locations[a] - locations[b])) < min_dist)
                 for a in range(len(locations))]
    chosen = [-1] * n
    steps, pruned = [0], [0]
    dead = set()

    def backtrack(u, blocked):
        if u == n:
            return True
        key = (u, blocked)
        if key in dead:
            return False
        steps[0] += 1
        if steps[0] > MAX_BACKTRACK_STEPS:
            raise StopGeneration(
                "SEARCH_BUDGET_EXHAUSTED: no infeasibility proof; STOP")
        unions, counts = {}, {}
        for v in range(u, n):
            available = masks[v] & ~blocked
            if not available:
                pruned[0] += 1
                dead.add(key)
                return False
            model = celltypes[original_ids[v]]
            unions[model] = unions.get(model, 0) | available
            counts[model] = counts.get(model, 0) + 1
        if any(bin(unions[k]).count("1") < counts[k] for k in counts):
            pruned[0] += 1
            dead.add(key)
            return False
        for idx in order[u]:
            if blocked & (1 << idx):
                continue
            chosen[u] = idx
            if backtrack(u + 1, blocked | conflicts[idx]):
                return True
        chosen[u] = -1
        dead.add(key)
        return False

    if not backtrack(0, 0):
        raise StopGeneration("INFEASIBLE: no legal selection under strict constraints")
    return {"template_ids": [int(c) for c in chosen],
            "backtrack_steps": int(steps[0]), "forward_pruned": int(pruned[0]),
            "memo_failed": len(dead), "candidate_sizes": [len(c) for c in cand]}


def assert_selection(celltypes, locations, eaps, original_ids, selection,
                     min_dist=25.0, min_amp=50.0, max_amp=300.0,
                     n_exc=51, n_inh=13, allow_original=False):
    """Post-selection assertions; any violation is a hard stop."""
    ids = selection["template_ids"] if isinstance(selection, dict) else list(selection)
    n_units = len(original_ids)
    if len(ids) != n_units:
        raise StopGeneration(f"SELECTION_SIZE: {len(ids)} != {n_units}")
    if len(set(ids)) != n_units:
        raise StopGeneration("SELECTION_DUPLICATE: duplicate template ids")
    for u, idx in enumerate(ids):
        if celltypes[idx] != celltypes[original_ids[u]]:
            raise StopGeneration(f"SELECTION_MODEL_BINDING: unit {u}")
        if not allow_original and idx == int(original_ids[u]):
            raise StopGeneration(f"SELECTION_ORIGINAL_INSTANCE: unit {u}")
    P = locations[ids]
    if len(np.unique(np.round(P, 6), axis=0)) != n_units:
        raise StopGeneration("SELECTION_DUPLICATE_POSITION")
    d = np.linalg.norm(P[:, None] - P[None, :], axis=2)
    np.fill_diagonal(d, np.inf)
    mind = float(d.min())
    if mind < min_dist:
        raise StopGeneration(f"SELECTION_MIN_DIST: {mind} < {min_dist}")
    amps = [native_amp(eaps[i]) for i in ids]
    if not all(min_amp < a < max_amp for a in amps):
        raise StopGeneration("SELECTION_AMP_WINDOW")
    n_e = sum(1 for i in ids if celltypes[i] in EXC_MODELS)
    if n_e != n_exc or n_units - n_e != n_inh:
        raise StopGeneration(f"SELECTION_EI_COUNTS: exc={n_e}")
    return {"min_pairwise_dist_um": mind,
            "amp_native_min": min(amps), "amp_native_max": max(amps),
            "n_exc": n_e, "n_inh": n_units - n_e}


def selection_receipt(celltypes, locations, rotations, eaps, original_ids,
                      position_seed, selection, checks, kind="new_realization"):
    ids = (selection["template_ids"] if isinstance(selection, dict)
           else [int(i) for i in selection])
    receipt = {
        "kind": kind,
        "position_seed": position_seed,
        "template_ids": ids,
        "original_template_ids": [int(i) for i in original_ids],
        "per_unit": [{"unit": u, "old_id": int(original_ids[u]), "new_id": ids[u],
                      "model": str(celltypes[ids[u]]),
                      "amp_native_uV": native_amp(eaps[ids[u]]),
                      "position_um": [round(float(x), 3) for x in locations[ids[u]]],
                      "rotation": [round(float(x), 6) for x in rotations[ids[u]]]}
                     for u in range(len(original_ids))],
        "checks": checks,
        "policy": ("strict native constraints; STOP on infeasibility; no fallback/"
                   "relaxation/redraw; no outcome-based selection"),
    }
    if isinstance(selection, dict):
        for key in ("backtrack_steps", "forward_pruned", "memo_failed", "candidate_sizes"):
            if key in selection:  # D0 identity layout performs no search.
                receipt[key] = selection[key]
    return receipt


# ---------------------------------------------------------------------------
# Count-conditioned spike-time sampler (R samples refractory)
# ---------------------------------------------------------------------------
def sample_unit_times(n_i: int, total_samples: int, rng: np.random.Generator,
                      refractory_samples: int = 40) -> np.ndarray:
    """N_i unique slots uniform over the compressed space, sorted, +i*(R-1)."""
    if n_i == 0:
        return np.empty(0, np.int64)
    m = total_samples - (n_i - 1) * (refractory_samples - 1)
    if m < n_i:
        raise StopGeneration(
            f"COMPRESSED_SPACE_INFEASIBLE: M={m} < N_i={n_i}")
    slots = np.sort(rng.choice(m, size=n_i, replace=False))
    return (slots + np.arange(n_i, dtype=np.int64) * (refractory_samples - 1)).astype(np.int64)


def sample_dataset(per_unit_Ni, total_samples: int, spiketrain_seed: int,
                   refractory_samples: int = 40) -> dict:
    """One master PCG64 stream seeded SeedSequence([seed, 0]); units drawn in
    original order 0..63. Identical construction for every dataset id."""
    rng = np.random.default_rng(np.random.SeedSequence([spiketrain_seed, 0]))
    return {unit: sample_unit_times(int(per_unit_Ni[unit]), total_samples, rng,
                                    refractory_samples)
            for unit in range(len(per_unit_Ni))}


def validate_schedule(times: dict, per_unit_Ni, total_samples: int,
                      refractory_samples: int = 40) -> dict:
    """Oracle report: exact counts, sorted, min ISI, range. No clipping."""
    report = {"per_unit_count_ok": True, "sorted_ok": True, "min_isi_ok": True,
              "in_range_ok": True, "min_isi_samples": None}
    worst = None
    for unit, t in times.items():
        if len(t) != per_unit_Ni[unit]:
            report["per_unit_count_ok"] = False
        if not np.all(np.diff(t) > 0):
            report["sorted_ok"] = False
        if len(t) > 1:
            isi = int(np.min(np.diff(t)))
            worst = isi if worst is None else min(worst, isi)
            if isi < refractory_samples:
                report["min_isi_ok"] = False
        if len(t) and (t[0] < 0 or t[-1] >= total_samples):
            report["in_range_ok"] = False
    report["min_isi_samples"] = worst
    return report


def seconds_to_samples(times_s: np.ndarray, fs: int) -> np.ndarray:
    """Original GT rounding: np.rint(t * fs) as int64."""
    return np.rint(np.asarray(times_s, np.float64) * fs).astype(np.int64)


def seconds_to_samples_is_exact_roundtrip(samples: np.ndarray, fs: int) -> bool:
    """rint((s / fs) * fs) == s for the schedule magnitudes involved."""
    back = seconds_to_samples(np.asarray(samples, np.float64) / fs, fs)
    return bool(np.array_equal(back, samples))


# ---------------------------------------------------------------------------
# Spatial covariance and chunk seed lineages
# ---------------------------------------------------------------------------
def spatial_covariance(geometry: np.ndarray, nearest_covariance: float = 15.0):
    """cov[i,j] = 15/|x_i - x_j| for i != j, diagonal exactly 1.

    Returns (covariance, min eigenvalue); stops if not PSD.
    """
    channels = len(geometry)
    distance = np.linalg.norm(geometry[:, None] - geometry[None, :], axis=2)
    covariance = np.divide(nearest_covariance, distance, out=np.eye(channels),
                           where=distance > 0)
    eigmin = float(np.linalg.eigvalsh(covariance).min())
    if eigmin < 0:
        raise StopGeneration("COVARIANCE_NOT_PSD: distance covariance eigmin < 0")
    return covariance, eigmin


def spatial_chunk_seeds(master_seed: int, n_chunks: int) -> list:
    """Original rule: RandomState(master).randint(1000), drawn n_chunks times."""
    rng = np.random.RandomState(master_seed)
    return [int(rng.randint(1000)) for _ in range(n_chunks)]


def independent_chunk_rng(seed: int, chunk: int) -> np.random.Generator:
    """Original rule: PCG64 via SeedSequence([seed, chunk]) per chunk."""
    return np.random.default_rng(np.random.SeedSequence([seed, chunk]))


# ---------------------------------------------------------------------------
# Total-CAL-SNR calibration (median LINEAR amplitude/sigma, brentq, f32 scale)
# ---------------------------------------------------------------------------
def cal_moments(spatial_cal_f64: np.ndarray, unit_cal_f64: np.ndarray,
                floor_sigma: float = 5.0) -> dict:
    """CAL float64 moments of the actual mixed components (floor = unit*floor)."""
    cal_sp = np.asarray(spatial_cal_f64, np.float64)
    cal_floor = np.asarray(unit_cal_f64, np.float64) * float(floor_sigma)
    sp_var, floor_var = cal_sp.var(axis=0), cal_floor.var(axis=0)
    cross = np.mean((cal_sp - cal_sp.mean(0)) * (cal_floor - cal_floor.mean(0)), axis=0)
    return {"sp_var": sp_var, "floor_var": floor_var, "cross": cross}


def total_sigma(scale: float, moments: dict) -> np.ndarray:
    m = moments
    return np.sqrt(m["floor_var"] + scale * scale * m["sp_var"] + 2.0 * scale * m["cross"])


def cal_snr_db(scale: float, moments: dict, amplitude: np.ndarray,
               main_channel: np.ndarray) -> float:
    """20*log10(median(amplitude / sigma[main_channel])) - linear median."""
    sigma = total_sigma(scale, moments)
    return float(20 * np.log10(np.median(amplitude / sigma[main_channel])))


def solve_scale(target_db: float, moments: dict, amplitude: np.ndarray,
                main_channel: np.ndarray) -> dict:
    """brentq on [0, upper*2^k] with xtol 1e-12, root rounded to float32.

    Raises StopGeneration when the target is unreachable (the floor-only
    ceiling already sits at or below the target) or non-finite.
    """
    from scipy.optimize import brentq

    def snr(g):
        return cal_snr_db(g, moments, amplitude, main_channel) - target_db

    if not np.isfinite(snr(0.0)):
        raise StopGeneration("CAL_NON_FINITE: floor-only SNR not finite; STOP")
    if snr(0.0) <= 0:
        raise StopGeneration(
            f"UNREACHABLE_TARGET_SNR: floor-only CAL ceiling {snr(0.0) + target_db:.6f} dB "
            f"<= target {target_db} dB; STOP")
    upper = 1.0
    for _ in range(200):
        if snr(upper) <= 0:
            break
        upper *= 2
    else:
        raise StopGeneration("UNREACHABLE_TARGET_SNR: no sign change within doubling budget")
    root = brentq(snr, 0.0, upper, xtol=1e-12)
    if not np.isfinite(root) or root < 0:
        raise StopGeneration("CAL_ROOT_INVALID: non-finite or negative scale; STOP")
    scale_f32 = float(np.float32(root))
    if not np.isfinite(scale_f32) or scale_f32 < 0:
        raise StopGeneration("CAL_SCALE_F32_INVALID: non-finite or negative; STOP")
    return {"target_db": target_db, "root": float(root), "scale_f32": scale_f32}


# ---------------------------------------------------------------------------
# Exact float32 association of the mixed input
# ---------------------------------------------------------------------------
def mixed_noise_block_f32(spatial_block: np.ndarray, unit_block: np.ndarray,
                          scale, floor: float = 5.0) -> np.ndarray:
    """float32(spatial) * float32(scale) + float32(unit) * float32(floor)."""
    return (np.asarray(spatial_block) * np.float32(scale)
            + np.asarray(unit_block) * np.float32(floor))


def mixed_input_block_f32(clean_block: np.ndarray, spatial_block: np.ndarray,
                          unit_block: np.ndarray, scale, floor: float = 5.0) -> np.ndarray:
    """Exact association order: clean + spatial*float32(scale) + unit*float32(floor)."""
    return (np.asarray(clean_block) + np.asarray(spatial_block) * np.float32(scale)
            + np.asarray(unit_block) * np.float32(floor))


# ---------------------------------------------------------------------------
# Streamed statistics (float64 accumulation over float32 sources)
# ---------------------------------------------------------------------------
def unit_split_sigma(noise_mmap, lo_sample: int, hi_sample: int,
                     channels: int = 256, block: int = 100000) -> np.ndarray:
    """Per-channel sigma over [lo, hi) with float64 block accumulation."""
    count, total, square = 0, np.zeros(channels), np.zeros(channels)
    for start in range(lo_sample, hi_sample, block):
        chunk = np.asarray(noise_mmap[start:min(start + block, hi_sample)], np.float64)
        count += len(chunk)
        total += chunk.sum(axis=0)
        square += np.einsum("ij,ij->j", chunk, chunk)
    mean = total / count
    return np.sqrt(np.maximum(0, square / count - mean * mean))


class SplitAccumulator:
    """sum/sumsq per channel per split over streamed float64 blocks."""

    def __init__(self, splits: dict, channels: int = 256):
        self.splits = {name: (int(lo), int(hi)) for name, (lo, hi) in splits.items()}
        self.state = {s: [0, np.zeros(channels), np.zeros(channels)] for s in self.splits}

    def add(self, block_start: int, block: np.ndarray):
        block = np.asarray(block, np.float64)
        for name, (lo, hi) in self.splits.items():
            left, right = max(block_start, lo), min(block_start + len(block), hi)
            if right <= left:
                continue
            part = block[left - block_start:right - block_start]
            acc = self.state[name]
            acc[0] += len(part)
            acc[1] += part.sum(0)
            acc[2] += np.einsum("ij,ij->j", part, part)

    def sigma(self, split: str) -> np.ndarray:
        count, total, square = self.state[split]
        mean = total / count
        return np.sqrt(np.maximum(0.0, square / count - mean * mean))

    def summaries(self, amplitude: np.ndarray, main_channel: np.ndarray) -> dict:
        out = {}
        for split in self.splits:
            sigma = self.sigma(split)
            ratio = amplitude / sigma[main_channel]
            out[split] = {"measured_total_SNR_dB": float(20 * np.log10(np.median(ratio))),
                          "total_noise_sigma_median_uV": float(np.median(sigma)),
                          "sigma_uV_min": float(sigma.min()),
                          "sigma_uV_max": float(sigma.max())}
        return out


# ---------------------------------------------------------------------------
# Hashing, storage gates, ledger, create-once artifacts
# ---------------------------------------------------------------------------
def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for blk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(blk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def component_bytes(samples: int, channels: int = 256) -> int:
    return int(samples) * int(channels) * 4  # float32


def free_bytes(path) -> int:
    st = os.statvfs(str(path))
    return st.f_bavail * st.f_frsize


def stop_before_write(target_dir, needed: int, reserve: int) -> dict:
    """Abort-safe gate: never start a large write unless free >= needed + reserve."""
    free = free_bytes(target_dir)
    ok = free >= needed + reserve
    return {"ok": ok, "free_bytes": free, "needed_bytes": int(needed),
            "reserve_bytes": int(reserve), "min_required_bytes": int(needed) + int(reserve)}


def ledger_path(bundle) -> Path:
    return Path(bundle) / "LEDGER.json"


def ledger_read(bundle) -> dict:
    p = ledger_path(bundle)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def ledger_update(bundle, name: str, sha: str) -> None:
    p = ledger_path(bundle)
    led = ledger_read(bundle)
    led[name] = sha
    p.write_text(json.dumps(led, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def create_once(path) -> None:
    """Artifacts are created exactly once; existing files are never overwritten."""
    if Path(path).exists():
        raise StopGeneration(f"CREATE_ONCE: {path} already exists")


def verified_skip(artifact, bundle, name: str, receipt=None, receipt_key: str = None) -> bool:
    """Hash-verified completed-stage skip (never receipt-existence only).

    Existing artifact without a verifiable expected hash is a preserved
    partial: STOP, never adopt. Hash drift: STOP, never overwrite.
    """
    artifact = Path(artifact)
    if not artifact.exists():
        return False
    expected = ledger_read(bundle).get(name)
    if expected is None and receipt is not None and Path(receipt).exists():
        expected = json.loads(Path(receipt).read_text(encoding="utf-8")).get(receipt_key)
    if expected is None:
        raise StopGeneration(
            f"NO_VERIFIABLE_HASH: {artifact} exists but ledger/receipt lack '{name}' "
            "(preserved partial; no adoption)")
    actual = sha256_file(artifact)
    if actual != expected:
        raise StopGeneration(
            f"ARTIFACT_HASH_DRIFT: {artifact} {actual} != {expected}; STOP")
    return True


def write_receipt(bundle, filename: str, payload: dict, artifact_name: str = None) -> Path:
    """Write a create-once JSON receipt committed last; bind artifact hash."""
    path = Path(bundle) / filename
    create_once(path)
    path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    if artifact_name is not None:
        ledger_update(bundle, artifact_name, payload.get("sha256")
                      or payload.get(artifact_name))
    return path


# ---------------------------------------------------------------------------
# Public reader: virtual mixtures with exact float32 association
# ---------------------------------------------------------------------------
def read_generation_manifest(dataset_root) -> dict:
    dataset_root = Path(dataset_root)
    path = dataset_root / "generation_manifest.json"
    if not verified_skip(path, dataset_root, "generation_manifest.json"):
        raise StopGeneration("MANIFEST_MISSING")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != STATUS_COMPLETE:
        raise StopGeneration(
            f"MANIFEST_INCOMPLETE: status {manifest.get('status')!r} != {STATUS_COMPLETE!r}")
    return manifest


def iter_input(dataset_root, snr, count=None, block_samples=100000):
    """Yield ``(first, float32 block)`` virtual mixtures for a finished bundle.

    Requires a complete, declared generation manifest and re-verifies the
    source file hashes recorded in its provenance before the first yield.
    Association is EXACT: ``clean + spatial*float32(scale) + unit*float32(5)``
    (no duplicate full-raw waveform files exist per SNR).
    """
    dataset_root = Path(dataset_root)
    manifest = read_generation_manifest(dataset_root)
    targets = list(manifest["target_total_CAL_SNR_dB"])
    if targets != [10, 15, 20]:
        raise StopGeneration("MANIFEST_SCOPE: targets must be [10, 15, 20]")
    if snr not in targets:
        raise ValueError(f"snr must be one of {targets}, got {snr}")
    floor = float(manifest["floor_sigma_uV"])
    if floor != 5.0:
        raise StopGeneration(f"MANIFEST_SCOPE: floor {floor} != 5.0 uVrms")
    scale = float(manifest["spatial_scale"][str(snr)])
    if not np.isfinite(scale) or scale < 0:
        raise StopGeneration(f"MANIFEST_SCALE_INVALID for snr {snr}")
    samples = int(manifest["samples"])
    if count is None:
        count = samples
    if not 0 <= count <= samples or block_samples < 1:
        raise ValueError("invalid input length or block size")
    binding_sha = manifest.get("provenance", {}).get("execution_binding_sha256")
    if not binding_sha or sha256_file(dataset_root / "execution_binding.json") != binding_sha:
        raise StopGeneration("MANIFEST_UNBOUND: missing or changed execution binding")
    provenance = manifest.get("provenance", {}).get("files", {})
    for key, filename in (("selection", "selection.json"), ("trains", "trains.npz"),
                          ("metadata", "metadata.npz")):
        entry = provenance.get(key, {})
        if entry.get("file") != filename or "sha256" not in entry:
            raise StopGeneration(f"MANIFEST_UNBOUND: provenance lacks {key}")
        if sha256_file(dataset_root / filename) != entry["sha256"]:
            raise StopGeneration(f"SOURCE_HASH_DRIFT: {filename}")
    sources = {}
    for key in ("clean", "spatial_noise", "noise_unit"):
        entry = provenance.get(key)
        if not entry or "sha256" not in entry:
            raise StopGeneration(f"MANIFEST_UNBOUND: provenance lacks {key} sha256")
        if entry["file"] != key + ".npy":
            raise StopGeneration(f"MANIFEST_PATH: unexpected filename for {key}")
        path = dataset_root / entry["file"]
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise StopGeneration(
                f"SOURCE_HASH_DRIFT: {path} {actual} != {entry['sha256']}")
        source = np.load(path, mmap_mode="r", allow_pickle=False)
        if source.dtype != np.dtype("float32") or source.shape != (samples, int(manifest["channels"])):
            raise StopGeneration(f"SOURCE_FORMAT: {key} must be float32 with declared shape")
        sources[key] = source
    scale_f32 = np.float32(scale)
    for first in range(0, count, block_samples):
        last = min(first + block_samples, count)
        yield first, mixed_input_block_f32(sources["clean"][first:last],
                                           sources["spatial_noise"][first:last],
                                           sources["noise_unit"][first:last],
                                           scale_f32, np.float32(floor))
