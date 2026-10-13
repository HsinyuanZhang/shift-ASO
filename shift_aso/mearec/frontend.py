"""Exact signed six-bit, bounded-jump level-crossing front end."""
from __future__ import annotations
import numpy as np

FULL_SCALE_UV = 688.6685485839844


def lc6(x: np.ndarray, full_scale: float = FULL_SCALE_UV, jump_limit: int = 8) -> np.ndarray:
    """Encode samples with truncation toward zero and signed saturation [-31,31]."""
    wave = np.asarray(x, np.float32)
    if wave.ndim != 2: raise ValueError("x must be time-major")
    out = np.empty(wave.shape, np.int8); q = np.zeros(wave.shape[1], np.int32); delta = full_scale / 31.0
    for t, row in enumerate(wave):
        requested = np.trunc((row - q * delta) / delta).astype(np.int32)
        q = np.clip(q + np.clip(requested, -jump_limit, jump_limit), -31, 31)
        out[t] = q
    return out
