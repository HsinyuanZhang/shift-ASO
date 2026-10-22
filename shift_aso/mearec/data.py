"""MEArec HDF5 input, ground truth, and deterministic noise conditions."""
from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 20_000
CASES = ("base", "independent_gaussian_10uVrms", "common_mode_gaussian_10uVrms")


def load_recording(path: str | Path, stop: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load time-major samples plus main-channel ground truth from MEArec HDF5."""
    with h5py.File(path, "r") as handle:
        x = np.asarray(handle["recordings"][:stop], np.float32)
        geometry = np.asarray(handle["channel_positions"], np.float64)[:, 1:3]
        main, times, units = [], [], []
        for unit in range(len(handle["spiketrains"])):
            template = np.asarray(handle["original_templates"][unit], np.float32)
            main.append(int(np.argmax(np.max(np.abs(template), axis=1))))
            t = np.rint(np.asarray(handle[f"spiketrains/{unit}/times"], np.float64) * FS).astype(np.int64)
            t = t[t < len(x)]
            times.append(t); units.append(np.full(len(t), unit, np.int32))
    t = np.concatenate(times); u = np.concatenate(units); order = np.argsort(t, kind="stable")
    return x, geometry, {"times": t[order], "channels": np.asarray(main, np.int32)[u[order]]}


def _noise(seed: object, shape: tuple[int, int]) -> np.ndarray:
    z = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    sos = butter(3, [300, 6000], btype="bandpass", fs=FS, output="sos")
    z = sosfiltfilt(sos, z, axis=0).astype(np.float32)
    return z * np.float32(10.0 / np.sqrt(np.mean(z * z)))


def augment(x: np.ndarray, case: str) -> np.ndarray:
    if case not in CASES: raise ValueError(f"unknown case: {case}")
    if case == "base": return np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != 256: raise ValueError("expected (samples, 256) recording")
    y = np.asarray(x, np.float32).copy(); prefix = min(60 * FS, len(y)); cols = 256 if case.startswith("independent") else 1
    seeds = {"independent_gaussian_10uVrms": 2026100801, "common_mode_gaussian_10uVrms": 2026100802}
    # Prefix normalization is always defined over its full 60 s draw, even for
    # a short replay input.
    full_prefix = _noise(seeds[case], (60 * FS, cols))
    y[:prefix] += full_prefix[:prefix]
    if len(y) > prefix:
        case_id = 1 if case.startswith("independent") else 2
        y[prefix:] += _noise(np.random.SeedSequence([20261013, 300, case_id]), (len(y) - prefix, cols))
    return y
