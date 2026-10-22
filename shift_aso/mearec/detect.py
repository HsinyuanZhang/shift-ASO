"""Peak extraction and spatial non-maximum suppression used by public replay."""
from __future__ import annotations
import numpy as np


def neighbors(geometry: np.ndarray, radius_um: float = 54.) -> list[np.ndarray]:
    d = geometry[:, None] - geometry[None, :]
    return [np.flatnonzero(np.einsum("ij,ij->i", d[c], d[c]) <= radius_um * radius_um) for c in range(len(geometry))]


def local_maxima(signal: np.ndarray, threshold: np.ndarray, positive: bool, left: int = 8, right: int = 8, start: int = 0) -> np.ndarray:
    rows = []
    for channel in range(signal.shape[1]):
        u = signal[:, channel] if positive else -signal[:, channel]; h = threshold[channel]
        for t in range(left, len(u) - right):
            if u[t] >= h and np.all(u[t] > u[t-left:t]) and np.all(u[t] >= u[t+1:t+right+1]): rows.append((t + start, channel, float(u[t])))
    return np.asarray(rows, dtype=[("sample_index", "<i8"), ("channel_index", "<i4"), ("amplitude", "<f8")])


def nms(events: np.ndarray, geometry: np.ndarray, exclude: int = 8) -> np.ndarray:
    if not len(events): return events
    near = neighbors(geometry); keep = np.ones(len(events), bool)
    by_channel = []
    for channel in range(len(geometry)):
        ix = np.flatnonzero(events["channel_index"] == channel)
        by_channel.append(ix[np.argsort(events["sample_index"][ix], kind="stable")])
    order = np.lexsort((events["channel_index"], -events["amplitude"]))
    for i in order:
        if not keep[i]: continue
        time = events["sample_index"][i]
        for channel in near[events["channel_index"][i]]:
            ix = by_channel[int(channel)]; samples = events["sample_index"][ix]
            lo = np.searchsorted(samples, time - exclude, side="left")
            hi = np.searchsorted(samples, time + exclude, side="right")
            victims = ix[lo:hi]; keep[victims[victims != i]] = False
    return events[keep]


def baseline_nms(feature: np.ndarray, threshold: np.ndarray, positive: bool, geometry: np.ndarray, lo: int, hi: int, guard: int = 40, exclude: int = 8) -> np.ndarray:
    """Fixed baseline10 locally-exclusive peak rule, including asymmetric ties."""
    near = neighbors(geometry); times = []; chans = []; amps = []
    for first in range(exclude, len(feature) - exclude, 10000):
        last = min(len(feature) - exclude, first + 10000); center = feature[first:last]
        past = feature[first-exclude:last-exclude].copy(); future = feature[first+1:last+1].copy()
        for offset in range(1, exclude):
            a = feature[first-exclude+offset:last-exclude+offset]; b = feature[first+1+offset:last+1+offset]
            past = np.maximum(past, a) if positive else np.minimum(past, a)
            future = np.maximum(future, b) if positive else np.minimum(future, b)
        def cross(values):
            out = np.full(values.shape, -np.inf if positive else np.inf, dtype=np.result_type(values.dtype, np.float32))
            for channel, ids in enumerate(near):
                if len(ids): out[:, channel] = values[:, ids].max(1) if positive else values[:, ids].min(1)
            return out
        other, before, after = cross(center), cross(past), cross(future)
        if positive: keep = (center > threshold) & (center >= other) & (center > np.maximum(past, before)) & (center >= np.maximum(future, after))
        else: keep = (center < -threshold) & (center <= other) & (center < np.minimum(past, before)) & (center <= np.minimum(future, after))
        rows, cols = np.nonzero(keep & ((np.arange(first, last) >= lo) & (np.arange(first, last) < hi - guard))[:, None])
        times.append((first + rows).astype(np.int64)); chans.append(cols.astype(np.int32)); amps.append(center[rows, cols])
    return np.rec.fromarrays([np.concatenate(times), np.concatenate(chans), np.concatenate(amps)], names="sample_index,channel_index,amplitude")
