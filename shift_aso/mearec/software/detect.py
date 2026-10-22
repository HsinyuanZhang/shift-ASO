"""Spatial locally exclusive NMS, post-H ties/polarity, then timestamp retiming.

Comparisons follow SpikeInterface's asymmetric past/future convention; see
THIRD_PARTY_NOTICES.md. Neighbor energy is NEVER summed into the feature.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from .grid import LAGS, NMS_RADIUS_UM, NMS_SAMPLES, WINDOW_AFTER, WINDOW_BEFORE
from ...neuropixels._matching import resolve_quantized_ties

EVENT = np.dtype([('sample_index', '<i8'), ('channel_index', '<i4'),
                  ('feature_peak', '<i8'), ('amplitude', '<f8'),
                  ('availability_sample', '<i8')])


def neighbors(geometry, radius_um=NMS_RADIUS_UM):
    coordinates = np.asarray(geometry, np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not len(coordinates) or not np.all(np.isfinite(coordinates)):
        raise ValueError('geometry must be finite, nonempty (channels,2)')
    if not np.isfinite(radius_um) or radius_um < 0:
        raise ValueError('invalid neighborhood radius')
    distances = np.sum((coordinates[:, None] - coordinates[None, :]) ** 2, axis=2)
    return [np.flatnonzero(distances[c] <= radius_um * radius_um) for c in range(len(coordinates))]


def neighborhood(geometry):
    groups = neighbors(geometry)
    counts = np.asarray([len(group) for group in groups], np.int32)
    ids = np.full((len(groups), int(counts.max())), -1, np.int32)
    for channel, group in enumerate(groups):
        ids[channel, :len(group)] = group
    return ids, counts


@njit(cache=True)
def _nms(values, threshold, ids, counts, sweep):
    """H gates output only; all unthresholded spatial/temporal values compete."""
    n, nc = values.shape
    times = np.empty(n * nc, np.int64)
    channels = np.empty(n * nc, np.int32)
    amplitudes = np.empty(n * nc, np.float64)
    count = 0
    for t in range(sweep, n - sweep):
        for c in range(nc):
            value = values[t, c]
            if value <= 0 or value <= threshold[c]:
                continue
            keep = True
            for j in range(counts[c]):
                other = ids[c, j]
                if other != c and value < values[t, other]:
                    keep = False
                    break
                for distance in range(1, sweep + 1):
                    if value <= values[t-distance, other] or value < values[t+distance, other]:
                        keep = False
                        break
                if not keep:
                    break
            if keep:
                times[count], channels[count], amplitudes[count] = t, c, value
                count += 1
    return times[:count], channels[:count], amplitudes[:count]


def make_events(times, channels, amplitudes, lag):
    out = np.empty(len(times), EVENT)
    out['sample_index'] = times
    out['feature_peak'] = times
    out['channel_index'] = channels
    out['amplitude'] = amplitudes
    out['availability_sample'] = times + int(lag) + NMS_SAMPLES
    return out


def peak_pool(feature, geometry, lag=1, chunk_samples=30000):
    """Extract positive NMS candidates in bounded windows with true E halos."""
    if isinstance(lag,(bool,np.bool_)) or not isinstance(lag,(int,np.integer)) or lag not in LAGS:
        raise ValueError('invalid public lag')
    energy = np.asarray(feature, np.float64)
    if energy.ndim != 2 or not np.all(np.isfinite(energy)):
        raise ValueError('feature must be finite and two-dimensional')
    ids, counts = neighborhood(geometry)
    if energy.shape[1] != len(ids) or chunk_samples < 1:
        raise ValueError('geometry/channel mismatch or invalid chunk size')
    pieces, sweep = [], NMS_SAMPLES
    for first in range(0, len(energy), chunk_samples):
        stop = min(first + chunk_samples, len(energy))
        left, right = max(0, first-sweep), min(len(energy), stop+sweep)
        times, channels, amplitudes = _nms(np.ascontiguousarray(energy[left:right]),
                                         np.zeros(len(ids)), ids, counts, sweep)
        times = times + left
        keep = (times >= first) & (times < stop) & (times >= lag+sweep) & (times < len(energy)-lag-sweep)
        pieces.append(make_events(times[keep], channels[keep], amplitudes[keep], lag))
    return np.concatenate(pieces) if pieces else np.empty(0, EVENT)


@njit(cache=True)
def trough_times(raw, peaks, channels, before, after):
    out = np.empty(len(peaks), np.int64)
    for i in range(len(peaks)):
        sample, channel = peaks[i], channels[i]
        first, stop = max(0, sample-before), min(len(raw), sample+after+1)
        best = first
        for t in range(first+1, stop):
            if raw[t, channel] < raw[best, channel]:
                best = t
        out[i] = best
    return out


def window_minimum(signal, sample, channel):
    return int(trough_times(np.asarray(signal), np.array([sample]), np.array([channel]),
                            WINDOW_BEFORE, WINDOW_AFTER)[0])


def activate(pool, signal, threshold, geometry, *, negative_center=True,
             timing='window_minimum'):
    """Required order: H, equal-amplitude ties, center polarity, retiming.

    In particular a positive-center tie winner can suppress a negative-center
    neighbor before the polarity test. No second NMS follows retiming.
    """
    raw = np.asarray(signal)
    levels = np.asarray(threshold, np.float64)
    if raw.ndim != 2 or levels.shape != (raw.shape[1],) or len(geometry) != raw.shape[1]:
        raise ValueError('input/threshold/geometry shape mismatch')
    if not np.all(np.isfinite(levels)) or np.any(levels < 0):
        raise ValueError('threshold must be finite and nonnegative')
    active = pool[pool['amplitude'] > levels[pool['channel_index']]]
    active = resolve_quantized_ties(active, geometry, NMS_RADIUS_UM)
    if negative_center:
        active = active[raw[active['feature_peak'], active['channel_index']] < 0]
    if timing == 'window_minimum':
        active = active.copy()
        active['sample_index'] = trough_times(raw, active['feature_peak'], active['channel_index'],
                                             WINDOW_BEFORE, WINDOW_AFTER)
    elif timing != 'energy_peak':
        raise ValueError('unknown timestamp policy')
    return active


def eligible(events, lo, hi, guard=40):
    """Filter outputs, not GT: keep full GT denominators in [lo,hi)."""
    return events[(events['sample_index'] >= lo) & (events['sample_index'] + guard < hi)
                  & (events['availability_sample'] < hi)]


def detect(signal, feature, threshold, geometry, *, lag=1, negative_center=True,
           timing='window_minimum', bounds=None, guard=40):
    """Batch convenience API. Use streaming replay for full-size recordings."""
    raw, energy = np.asarray(signal), np.asarray(feature)
    if raw.shape != energy.shape or raw.ndim != 2 or not np.all(np.isfinite(raw)):
        raise ValueError('signal and feature must have equal finite 2-D shapes')
    pool = peak_pool(energy, geometry, lag)
    active = activate(pool, raw, threshold, geometry, negative_center=negative_center, timing=timing)
    lo, hi = (0, len(raw)) if bounds is None else bounds
    if not 0 <= lo <= hi <= len(raw) or guard < 0:
        raise ValueError('invalid output bounds or guard')
    return eligible(active, lo, hi, guard)
