"""Signed float64 NEO/EMA, including continuous-state block processing."""
from __future__ import annotations

import numpy as np
from numba import njit
from .grid import LAGS


@njit(cache=True)
def _activity(values, offset, start, stop, total, lag, rho, state):
    output = np.empty((stop - start, values.shape[1]), np.float64)
    last = state.copy()
    for sample in range(start, stop):
        for channel in range(values.shape[1]):
            drive = 0.0
            if lag <= sample < total - lag:
                x = np.float64(values[sample - offset, channel])
                left = np.float64(values[sample - lag - offset, channel])
                right = np.float64(values[sample + lag - offset, channel])
                drive = x * x - left * right
            last[channel] = rho * last[channel] + drive
            output[sample - start, channel] = last[channel]
    return output, last


def activity(values, lag, rho=0.0, *, buffer_start=0, start=0, stop=None,
             total_frames=None, state=None):
    """Compute [start,stop) with TRUE raw halos and an explicit preceding state.

    Global indices are zero-origin. Never reset state at a chunk or split.
    Missing true raw halos are an error, not implicit zero padding.
    """
    raw = np.asarray(values)
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError('input must be a real (samples, channels) array')
    if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)) or lag not in LAGS:
        raise ValueError(f'lag must be one of {LAGS}')
    if not np.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError('rho must be finite and in [0,1)')
    total = len(raw) if total_frames is None else int(total_frames)
    stop = buffer_start + len(raw) if stop is None else int(stop)
    if not 0 <= buffer_start <= start <= stop <= total:
        raise ValueError('invalid global block bounds')
    first, last = max(start, lag), min(stop, total - lag)
    if first < last and (first - lag < buffer_start or last - 1 + lag >= buffer_start + len(raw)):
        raise ValueError('missing true input halo')
    previous = np.zeros(raw.shape[1], np.float64) if state is None else np.asarray(state, np.float64)
    if previous.shape != (raw.shape[1],) or not np.all(np.isfinite(previous)):
        raise ValueError('invalid preceding EMA state')
    if state is None and start > 0:
        raise ValueError('nonzero start requires the preceding EMA state')
    if not np.all(np.isfinite(raw)):
        raise ValueError('input must be finite')
    return _activity(np.ascontiguousarray(raw), int(buffer_start), int(start), stop,
                     total, int(lag), float(rho), previous)


def centered_neo(values, lag):
    """Signed centered NEO; invalid recording-edge centers have zero drive."""
    return activity(values, lag)[0]


def ema(energy, rho):
    """Unnormalized signed EMA from zero state; empty input is supported."""
    values = np.asarray(energy, np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError('energy must be a finite two-dimensional array')
    if not np.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError('rho must be finite and in [0,1)')
    output = np.empty_like(values)
    state = np.zeros(values.shape[1], np.float64)
    for sample in range(len(values)):
        state = float(rho) * state + values[sample]
        output[sample] = state
    return output


def cal_stat(energy, cal, lag):
    """Per-channel mean(abs(unmasked E[lag:cal-lag])); no epsilon/floor."""
    values = np.asarray(energy)
    if not lag < cal - lag <= len(values):
        raise ValueError('CAL is too short for this lag')
    result = np.mean(np.abs(values[lag:cal-lag]), axis=0, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite CAL statistic')
    return result
