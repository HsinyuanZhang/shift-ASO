"""Bounded-memory detector replay from sample zero, with no chunk/split reset."""
from __future__ import annotations

import numpy as np
from .api import normalize
from .features import activity
from .detect import (EVENT, _nms, neighborhood, make_events, trough_times,
                     eligible, resolve_quantized_ties)
from .grid import NMS_SAMPLES, NMS_RADIUS_UM, WINDOW_BEFORE, WINDOW_AFTER


class _Detector:
    def __init__(self, cfg, threshold, geometry, total, bounds, guard):
        self.cfg = normalize(cfg)
        self.lag, self.rho = self.cfg['lag'], self.cfg['rho']
        self.geometry = np.asarray(geometry, np.float64)
        self.ids, self.counts = neighborhood(geometry)
        self.H = np.asarray(threshold, np.float64)
        if self.H.shape != (len(geometry),) or not np.all(np.isfinite(self.H)) or np.any(self.H < 0):
            raise ValueError('invalid frozen per-channel H')
        self.total, self.bounds, self.guard = total, bounds, guard
        self.state = np.zeros(len(geometry), np.float64)
        self.raw, self.raw_start = None, 0
        self.energy, self.energy_start = None, self.lag
        self.energy_next, self.final_next = self.lag, self.lag + NMS_SAMPLES
        self.outputs = []

    def push(self, start, values):
        block = np.asarray(values)
        end = self.raw_start + (0 if self.raw is None else len(self.raw))
        if start != end or block.ndim != 2 or block.shape[1] != len(self.geometry):
            raise ValueError('blocks must start at zero and be contiguous with matching channels')
        self.raw = block if self.raw is None else np.concatenate((self.raw, block))
        limit = min(self.raw_start + len(self.raw) - self.lag, self.total - self.lag)
        if limit > self.energy_next:
            first = max(0, self.energy_next-self.lag-self.raw_start)
            last = min(len(self.raw), limit+self.lag-self.raw_start)
            piece, self.state = activity(self.raw[first:last], self.lag, self.rho,
                buffer_start=self.raw_start+first, start=self.energy_next,
                stop=limit, total_frames=self.total, state=self.state)
            self.energy = piece if self.energy is None else np.concatenate((self.energy, piece))
            self.energy_next = limit
        self._finalize()
        self._trim()

    def _finalize(self):
        stop = min(self.energy_next-NMS_SAMPLES, self.total-self.lag-NMS_SAMPLES)
        if stop <= self.final_next:
            return
        left, right = self.final_next-NMS_SAMPLES, stop+NMS_SAMPLES
        if left < self.energy_start:
            raise RuntimeError('lost true feature halo')
        values = np.ascontiguousarray(self.energy[left-self.energy_start:right-self.energy_start])
        times, channels, amplitudes = _nms(values, self.H, self.ids, self.counts, NMS_SAMPLES)
        events = make_events(times+left, channels, amplitudes, self.lag)
        events = resolve_quantized_ties(events, self.geometry, NMS_RADIUS_UM)
        if len(events):
            events = events[self.raw[events['feature_peak']-self.raw_start, events['channel_index']] < 0]
        if len(events):
            lo = max(0, int(events['feature_peak'].min())-WINDOW_BEFORE)
            hi = min(self.total, int(events['feature_peak'].max())+WINDOW_AFTER+1)
            if lo < self.raw_start or hi > self.raw_start+len(self.raw):
                raise RuntimeError('lost true retiming halo')
            raw = self.raw[lo-self.raw_start:hi-self.raw_start]
            events = events.copy()
            events['sample_index'] = trough_times(raw, events['feature_peak']-lo,
                                                   events['channel_index'], WINDOW_BEFORE, WINDOW_AFTER)+lo
            events = eligible(events, *self.bounds, self.guard)
            if len(events):
                self.outputs.append(events)
        self.final_next = stop

    def _trim(self):
        need = max(0, min(self.final_next-max(WINDOW_BEFORE,NMS_SAMPLES), self.energy_next-self.lag))
        if need > self.raw_start:
            self.raw = self.raw[need-self.raw_start:]
            self.raw_start = need
        need = max(self.energy_start, self.final_next-NMS_SAMPLES)
        if self.energy is not None and need > self.energy_start:
            self.energy = self.energy[need-self.energy_start:]
            self.energy_start = need

    def result(self):
        return np.concatenate(self.outputs) if self.outputs else np.empty(0, EVENT)


def replay(signal, geometry, configurations, thresholds, *, bounds=None, guard=40,
           block_samples=30000):
    """Replay all frozen variants in ONE input pass, returning events per name.

    Input may be a read-only NPY memmap. Recording/feature matrices are not
    materialized in full; output events themselves necessarily scale with count.
    Every variant is warmed from zero, even when bounds begin at TEST.
    """
    raw = np.asarray(signal)
    bounds = (0,len(raw)) if bounds is None else tuple(bounds)
    if raw.ndim != 2 or not 0 <= bounds[0] <= bounds[1] <= len(raw) or block_samples < 1 or guard < 0:
        raise ValueError('invalid signal, bounds, block size, or guard')
    if set(configurations) != set(thresholds):
        raise ValueError('configuration and threshold names differ')
    variants = {name:_Detector(cfg, thresholds[name], geometry, len(raw), bounds, guard)
                for name,cfg in configurations.items()}
    for first in range(0,len(raw),block_samples):
        block = raw[first:first+block_samples]
        if not np.all(np.isfinite(block)):
            raise ValueError('nonfinite input block')
        for detector in variants.values():
            detector.push(first, block)
    return {name:detector.result() for name,detector in variants.items()}
