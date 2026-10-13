"""Fixed software feature transforms."""
from __future__ import annotations
import numpy as np


def shift_activity(q: np.ndarray, rho: float = .875) -> np.ndarray:
    q = np.asarray(q, np.int16); out = np.empty(q.shape, np.float64); state = np.zeros(q.shape[1], np.float64); prev = np.zeros(q.shape[1], np.int16)
    lut = np.asarray((0, 0, 1, 1, 2, 2, 3, 3, 3), np.int32)
    for t, row in enumerate(q):
        d = row - prev; product = row.astype(np.int32) * d.astype(np.int32)
        weight = np.where(product > 0, np.abs(row).astype(np.int64) << lut[np.minimum(np.abs(d), 8)], 0)
        state = rho * state + weight; out[t] = state; prev = row
    return out


def neo_ema(q: np.ndarray, rho_num: int, rho_den: int) -> np.ndarray:
    q = np.asarray(q, np.int32); state = np.zeros(q.shape[1], np.int64); out = np.zeros(q.shape, np.int32)
    for t in range(1, len(q) - 1):
        psi = q[t] * q[t] - q[t - 1] * q[t + 1]
        numerator = rho_num * state + (rho_den - rho_num) * (psi.astype(np.int64) << 8)
        quotient = numerator // rho_den; remainder = numerator - quotient * rho_den
        state = quotient + ((2 * remainder > rho_den) | ((2 * remainder == rho_den) & ((quotient & 1) != 0)))
        out[t] = state
    return out
