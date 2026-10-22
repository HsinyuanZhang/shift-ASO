"""Fixed three-method MEArec software replay; it contains no parameter search."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .data import FS, augment, load_recording
from .frontend import FULL_SCALE_UV
from .feature import neo_ema
from .detect import baseline_nms
from shift_aso.neuropixels._matching import prepare_gt_index, spatial_pairs_fast
from shift_aso.neuropixels import _kernels as kernels

CAL, DEV, TEST, GUARD = 6 * FS, (6 * FS, 20 * FS), (20 * FS, 300 * FS), 40


def _score(events, gt, geometry, lo, hi):
    event = events[(events["sample_index"] >= lo) & (events["sample_index"] + GUARD < hi)]
    truth = (gt["times"] >= lo) & (gt["times"] < hi)
    index = prepare_gt_index(gt["times"][truth], gt["channels"][truth], geometry, 54.)
    tp = len(spatial_pairs_fast(event["sample_index"], event["channel_index"], index, 20)); fp = len(event) - tp; fn = int(truth.sum()) - tp
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "detections": int(len(event)), "labels": int(truth.sum()), "A": tp / (tp + fp + fn), "precision": tp / len(event) if len(event) else 0., "recall": tp / int(truth.sum()), "F1": 2 * tp / (2 * tp + fp + fn), "FP_per_s": fp / ((hi - lo) / FS)}


def _si(q, geometry):
    median = np.median(q[:CAL], axis=0); mad = np.median(np.abs(q[:CAL].astype(np.int32) - median.astype(np.int32)), axis=0)
    h = np.floor(4.5 * mad / .6744897501960817); return baseline_nms(q, h, False, geometry, TEST[0], len(q)), h


def _neo(q, geometry, case, config):
    parameters = config["methods"]["NEO10_EMA"][case]; rn, rd = parameters["rho"]; c = parameters["coefficient"]
    activity = neo_ema(q, rn, rd); h = np.floor(c * np.mean(np.abs(activity[1:CAL-1]), axis=0)); return baseline_nms(activity, h, True, geometry, TEST[0], len(q)), h


def _aso(q, geometry, case, config):
    parameters = config["methods"]["LC6_Shift_ASO"][case]
    offsets = np.zeros(q.shape[1], np.int16)
    activity, _, _ = kernels.feature_range(q, offsets, parameters["rho"], np.zeros(q.shape[1], np.int16), np.zeros(q.shape[1], np.float64))
    med = np.median(activity[:CAL], axis=0); mad = np.median(np.abs(activity[:CAL] - med), axis=0)
    spread = mad if parameters["threshold_family"] == "mad" else np.quantile(activity[:CAL], .75, axis=0) - med
    h = np.maximum(1., med + parameters["K"] * np.maximum(spread, .125))
    b, a, left, right, coh, radius = parameters["window"]
    d = np.linalg.norm(geometry[:, None] - geometry[None, :], axis=2)
    near_list = [np.flatnonzero((d[c] <= 54) & (d[c] > 0)).astype(np.int32) for c in range(len(geometry))]
    ring_list = [np.flatnonzero((d[c] <= 108) & (d[c] > 54)).astype(np.int32) for c in range(len(geometry))]
    def pad(groups):
        width = max(1, max(map(len, groups))); values = np.zeros((len(groups), width), np.int32); count = np.zeros(len(groups), np.int32)
        for channel, group in enumerate(groups): values[channel, :len(group)] = group; count[channel] = len(group)
        return values, count
    ni, nn = pad(near_list); ri, rn = pad(ring_list)
    pp, ch = kernels.peak_indices_range(activity, .3 * h, b, a, left, right, 0, 0, len(q) - 8)
    pool = kernels.extrema_range(q.astype(np.float64), activity, pp, ch, ni, nn, ri, rn, b, a, coh, 0, kernels.descriptor_dtype(ni.shape[1]))
    pool = pool[np.lexsort((pool["p"], pool["ch"], pool["t"]))]
    c = pool["ch"]; hh = h[c]; quality = pool["tv"] <= 2 * pool["ptp"] + 1
    original = quality & (pool["e"] >= hh) & (pool["ptp"] >= parameters["pstrong"])
    veto = original & (pool["e"] < 2 * hh) & (pool["refs"] >= 2) & (pool["ring"] >= 2) & ((pool["ptp"] - pool["ring"]) < 2)
    phase = pool["phase"] & ((pool["refs"] < 2) | (pool["prom"] >= parameters["prominence_floor"]))
    strong = original & ~veto & phase
    potential = quality & (pool["ptp"] >= 2) & (pool["refs"] >= 2) & (pool["e"] >= parameters["alpha"] * hh) & ~original & phase & ((pool["ptp"] - pool["ring"]) >= 1)
    ratio = np.zeros_like(pool["emax"])
    for j in range(ratio.shape[1]): ratio[:, j] = np.where(j < nn[c], pool["emax"][:, j] / h[ni[c, j]], 0.)
    weak = potential & ((np.max(ratio, axis=1) >= 1.) if parameters["weak_support"] == "integral_one" else (np.partition(ratio, -2, axis=1)[:, -2] >= .5))
    sk, wk = kernels.nms_sorted(pool, strong, weak, d <= 54, radius)
    return kernels.events_from(pool, sk, wk, {"b": b, "a": a, "right": right, "coh": coh}, {"nms": radius}), h


def evaluate(recording: str, case: str = "base", methods: tuple[str, ...] = ("si10_mad", "neo10_ema", "lc6_shift_aso"), stop_s: int = 300, config_path: str | None = None):
    if stop_s not in (60, 300): raise ValueError("stop_s must be 60 or 300")
    path = Path(config_path) if config_path else Path(__file__).with_name("config.json")
    config = json.loads(path.read_text())
    raw, geometry, gt = load_recording(recording, stop_s * FS); raw = augment(raw, case); stop = stop_s * FS
    center = np.median(raw[:CAL], axis=0).astype(np.float32)
    q10 = np.clip(np.rint((raw - center) / np.float32(FULL_SCALE_UV / 511.)), -511, 511).astype(np.int16)
    q6, _, _ = kernels.encode_range(raw, np.full(raw.shape[1], config["frontend"]["full_scale_uV"] / 31.), np.zeros(raw.shape[1], np.int16), np.zeros(raw.shape[1], np.int16))
    runners = {"si10_mad": lambda: _si(q10, geometry), "neo10_ema": lambda: _neo(q10, geometry, case, config), "lc6_shift_aso": lambda: _aso(q6, geometry, case, config)}
    results = {}
    for name in methods:
        events, threshold = runners[name]()
        results[name] = {"threshold": threshold, "events": events, "metrics": _score(events, gt, geometry, TEST[0], stop)}
    return results
