"""Port of the accepted meddrift GT-label-driven template reference builder.

Source of record (algorithm ported verbatim; parent-copied SOURCE ONLY):
    reference_sources/np_drift_template_reference_v1.py
    SHA-256 d312d459616ddba090cec6a098f9b47efe169086335b645a9db6a5ed7bffccee

The older ``np_drift_reference_v1.py`` trajectory projection is NOT ported and
must not be called; it is a different, superseded reference.

Semantics preserved from the accepted source (do not "fix" silently):
  * 10 s blocks (30 blocks over the first 300 s), 1200 units.
  * Rough per-(block, unit) channel = argmin of the mean of raw GT-aligned
    samples (negative trough); 13 nearest local channels by geometry.
  * Windows [-50, +70] inclusive (121 samples).
  * 16 uniformly spaced ranks per time parity (rank = per-(block, unit)
    occurrence index; parity = rank & 1).
  * Chunk-boundary semantics: a snippet is stored only when its whole window
    fits inside one yielded chunk; boundary spikes that fail the window check
    still consume a rank increment once per visiting chunk (the accepted
    source relies on this exact behaviour).
  * Pointwise median over combined snippets, most negative sample per local
    channel, argmin channel; validity requires >= 4 combined snippets.
  * Odd/even parity templates require >= 2 snippets each (diagnostic only).
  * Nearest-valid-10s-block fallback (ties resolve to the lower block index),
    unknown cells stay -1 with fallback distance -1.
  * Same-unit overlap statistic is computed but NOT applied as an exclusion.
  * The ``drift`` field of the params NPZ is a 2 s discrete waveform-bank
    index, not a physical position; this builder never reads it and never
    projects it to micrometers.
  * All GT times/IDs (including TEST) feed scorer-only references; they are
    never detector CAL/DEV inputs or settings.

This module is pure: it consumes a caller-supplied reiterable stream factory
and parameters. Production constants are locked below and the CLI build path
uses exactly them.
"""
from __future__ import annotations

import numpy as np

SOURCE_SHA256 = "d312d459616ddba090cec6a098f9b47efe169086335b645a9db6a5ed7bffccee"
SOURCE_NAME = "np_drift_template_reference_v1.py"

LABEL_EXPOSURE = (
    "All GT times/IDs make scorer-only references; never CAL/DEV detector "
    "inputs or settings. Unknown reference channels remain -1 and are "
    "excluded from the spatial index while retained in the FN denominator. "
    "NPZ drift is a 2s discrete waveform-bank index, not a confirmed "
    "physical position."
)

# Production constants locked to the accepted source. Explicit reference
# preparation (``--build-reference``) always uses exactly these values.
PRODUCTION_REFERENCE = {
    "stop": 9_000_000,       # STOP = 300 s * 30 kHz (first-300s benchmark)
    "blk": 300_000,          # BLK = 10 s block
    "n_units": 1200,         # NU
    "window_left": -50,      # L (window is [L, R) -> reported [-50, 70])
    "window_right": 71,      # R (exclusive bound, matches R - L = 121 samples)
    "max_snippets": 16,      # K uniformly sampled ranks per time parity
    "local_channels": 13,    # NL nearest local channels
    "min_snippets": 4,       # combined-snippet validity threshold
}


class ReferenceContractError(RuntimeError):
    """Raised when reference construction violates the public contract."""


def _check_chunk(raw, n_channels, start):
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != n_channels:
        raise ReferenceContractError(
            f"stream chunk at start={start} has shape {raw.shape}, expected "
            f"(*, {n_channels}) columns matching the active AP geometry"
        )
    if start < 0:
        raise ReferenceContractError("stream chunk start must be >= 0")
    return raw


def build_template_reference(
    stream_factory,
    times,
    units,
    geometry,
    ap_ids,
    *,
    stop,
    blk,
    n_units,
    window_left,
    window_right,
    max_snippets,
    local_channels,
    min_snippets,
):
    """Build the fixed 10 s GT-label-driven template reference (pure port).

    Parameters
    ----------
    stream_factory:
        Zero-argument callable returning a fresh iterable of
        ``(start, raw)`` chunks; ``raw`` must have one column per active AP
        channel (matching ``geometry`` rows) and cover ``[start,
        start + len(raw))``. Called once per pass (two passes), mirroring the
        accepted source's two ``ZipCbinStream.chunks(STOP)`` iterations.
    times, units:
        Sorted GT sample indices and unit ids, already restricted to
        ``0 <= t < stop`` (the accepted source filtered ``t < STOP`` and
        relied on sorted native ``st``).
    geometry:
        (n_active, 2) float geometry in micrometers, active channels only.
    ap_ids:
        Original AP channel indices of the active channels.

    Returns
    -------
    (arrays, diagnostics): ``arrays`` holds the exact NPZ field set of the
    accepted source; ``diagnostics`` holds the exact diagnostic JSON keys.
    """
    L = int(window_left); R = int(window_right); K = int(max_snippets)
    NL = int(local_channels)
    t = np.asarray(times, np.int64)
    u = np.asarray(units, np.int64)
    g = np.asarray(geometry, np.float64)
    ids = np.asarray(ap_ids, np.int64)
    if g.ndim != 2 or g.shape[1] != 2:
        raise ReferenceContractError("geometry must be (n_active, 2)")
    n = int(g.shape[0])
    if ids.shape != (n,):
        raise ReferenceContractError("ap_ids must have one entry per geometry row")
    if len(t) != len(u):
        raise ReferenceContractError("times and units must have equal length")
    if len(t) == 0:
        raise ReferenceContractError("GT label stream is empty; reference construction requires spikes")
    if np.any(np.diff(t) < 0):
        raise ReferenceContractError("GT times must be sorted (native st order)")
    if t.min() < 0 or t.max() >= int(stop):
        raise ReferenceContractError(f"GT times must lie in [0, {int(stop)})")
    if u.min() < 0 or u.max() >= int(n_units):
        raise ReferenceContractError(f"GT units must lie in [0, {int(n_units)})")
    stop = int(stop); blk = int(blk); n_units = int(n_units)
    NB = stop // blk
    if NB * blk != stop:
        raise ReferenceContractError("stop must be an integer number of blocks")

    # First pass forms a fixed compact local bank. It does not choose
    # detector settings. (source pass 1)
    sm = np.zeros((NB, n_units, n), np.float64)
    ct = np.zeros((NB, n_units), np.int32)
    rank = np.zeros((NB, n_units), np.int32)
    for st, raw in stream_factory():
        raw = _check_chunk(raw, n, st)
        lo = np.searchsorted(t, st, 'left'); hi = np.searchsorted(t, st + len(raw), 'left')
        tt, uu = t[lo:hi], u[lo:hi]
        for b in np.unique(tt // blk):
            z = (tt // blk) == b
            np.add.at(sm[b], uu[z], raw[tt[z] - st]); np.add.at(ct[b], uu[z], 1)

    rough = np.argmin(sm / np.maximum(ct[:, :, None], 1), axis=2).astype(np.int16)
    local = np.empty((NB, n_units, NL), np.int16)
    for b in range(NB):
        for v in range(n_units):
            local[b, v] = np.argsort(((g - g[rough[b, v]]) ** 2).sum(1))[:NL]

    # Two earliest time-parity snippets per cell, then a pointwise median
    # template. (source pass 2; boundary/rank semantics preserved verbatim)
    sn = np.zeros((NB, n_units, 2, K, R - L, NL), np.int16)
    used = np.zeros((NB, n_units, 2), np.int8)
    order = np.lexsort((t, u)); ov = np.zeros(len(t), bool)
    d = np.diff(t[order]); same = u[order][1:] == u[order][:-1]
    bad = np.flatnonzero(same & (d < R - L)); ov[order[bad]] = True; ov[order[bad + 1]] = True
    for st, raw in stream_factory():
        raw = _check_chunk(raw, n, st)
        lo = np.searchsorted(t, st - R, 'left'); hi = np.searchsorted(t, st + len(raw) - L, 'left')
        for i in range(lo, hi):
            b = int(t[i] // blk); v = int(u[i]); par = int(rank[b, v] & 1)
            rp = int(rank[b, v] // 2); rank[b, v] += 1; k = int(used[b, v, par]); beg = int(t[i] + L - st)
            total_par = (int(ct[b, v]) + 1 - par) // 2
            targets = np.floor((np.arange(K) + .5) * total_par / K).astype(np.int32)
            if k < K and np.any(targets == rp) and beg >= 0 and beg + R - L <= len(raw):
                sn[b, v, par, k] = raw[beg:beg + R - L][:, local[b, v]]; used[b, v, par] += 1

    cen = np.full((NB, n_units), -1, np.int16); od = cen.copy(); ev = cen.copy()
    valid = np.zeros((NB, n_units), bool)
    for b in range(NB):
        for v in range(n_units):
            no, ne = map(int, used[b, v]); w = np.concatenate((sn[b, v, 0, :no], sn[b, v, 1, :ne]))
            if no + ne >= min_snippets:
                cen[b, v] = local[b, v, np.argmin(np.median(w, axis=0).min(0))]; valid[b, v] = True
            if no >= 2: od[b, v] = local[b, v, np.argmin(np.median(sn[b, v, 0, :no], axis=0).min(0))]
            if ne >= 2: ev[b, v] = local[b, v, np.argmin(np.median(sn[b, v, 1, :ne], axis=0).min(0))]
    glob = np.argmin(sm.sum(0) / np.maximum(ct.sum(0)[:, None], 1), axis=1).astype(np.int16)
    fb = np.zeros_like(valid); dist = np.zeros_like(cen)
    for v in range(n_units):
        good = np.flatnonzero(valid[:, v])
        for b in np.flatnonzero(~valid[:, v]):
            if len(good):
                h = good[np.argmin(np.abs(good - b))]; cen[b, v] = cen[h, v]; dist[b, v] = abs(h - b)
            else:
                cen[b, v] = -1; dist[b, v] = -1
            fb[b, v] = True
    both = (od >= 0) & (ev >= 0); dd = np.linalg.norm(g[od[both]] - g[ev[both]], axis=1)
    channels = cen[(t // blk).astype(int), u]

    arrays = {
        "sample_index": t,
        "unit_id": u,
        "estimated_local_channel": channels,
        "block_channel": cen,
        "block_valid": valid,
        "block_fallback": fb,
        "fallback_distance_blocks": dist,
        "geometry_um": g,
        "local_to_original_AP": ids,
    }
    event_fallback = float(np.mean(fb[(t // blk).astype(int), u]))
    event_unknown = float(np.mean(channels < 0))
    diagnostics = {
        'status': 'complete',
        'kind': 'GT-label-driven 10 s uniformly sampled pointwise-median spatial template reference',
        'window_samples': [L, R - 1],
        'minimum_snippets': min_snippets,
        'maximum_snippets_per_time_parity': K,
        'local_channels': NL,
        'no_global_collision_exclusion': True,
        'same_unit_overlap_fraction': float(ov.mean()),
        'fallback_fraction': float(fb.mean()),
        'fallback_event_fraction': event_fallback,
        'unknown_event_fraction': event_unknown,
        'unavailable_fraction': float(np.mean(cen < 0)),
        'fallback_distance_blocks_p95': float(np.quantile(dist[fb & (dist >= 0)], .95)) if np.any(fb & (dist >= 0)) else None,
        'odd_even': {
            'eligible_cells': int(both.sum()),
            'same_channel_fraction': float(np.mean(od[both] == ev[both])) if np.any(both) else None,
            'median_spatial_delta_um': float(np.median(dd)) if len(dd) else None,
            'p95_spatial_delta_um': float(np.quantile(dd, .95)) if len(dd) else None,
            'within40um_fraction': float(np.mean(dd <= 40)) if len(dd) else None,
        },
        'label_exposure': LABEL_EXPOSURE,
    }
    # ``glob`` is computed by the accepted source but not exported; it is
    # retained here (unused) so the port stays line-for-line faithful.
    del glob
    return arrays, diagnostics
