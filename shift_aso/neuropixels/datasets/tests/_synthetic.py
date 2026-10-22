"""Synthetic tiny Neuropixels-like archive builders (tests only; no real NP data).

Everything here is generated in-memory: a synthetic 384-entry shank map with
one dead AP channel (383 active), time-differenced Fortran-order zlib CBIN
chunks with 385 recorded channels, a params NPZ with native st/cl, and a
registry payload patched from the real packaged registry (structure only).
"""
from __future__ import annotations

import io
import json
import zipfile
import zlib
from pathlib import Path

import numpy as np

N_AP = 384
N_REC = 385
FS = 30000
DEAD_AP = 200  # single dead AP channel -> 383 active channels
FULL_TOTAL = 9_000_000
FULL_CHUNK = 300_000


def make_meta_text(fs=FS, dead=(DEAD_AP,), n_ap=N_AP):
    rows = []
    for i in range(n_ap):
        col, row = i % 2, i // 2
        used = 0 if i in dead else 1
        rows.append(f"({i}:{col}:{row}:{used})")
    return "~imSampRate=%d\n~snsShankMap=%s\n~snsSaveVarsRaw=1\n" % (fs, "".join(rows))


def encode_chunk(raw: np.ndarray) -> bytes:
    """Inverse of the native decode: time-difference, Fortran order, zlib."""
    raw = np.ascontiguousarray(raw, dtype=np.int16)
    diff = np.diff(raw, axis=0, prepend=np.zeros((1, raw.shape[1]), np.int16))
    return zlib.compress(diff.tobytes(order="F"), 1)


def chunk_spans(total: int, chunk_len: int):
    bounds = list(range(0, total + 1, chunk_len))
    if bounds[-1] != total:
        bounds.append(total)
    return bounds


def make_archive(zip_path, *, total_samples, chunk_len, spikes=(), prefix="", fs=FS,
                 n_channels=N_REC, algorithm="zlib", do_time_diff=True, ch_patch=None,
                 params_st=(), params_cl=(), params_extra=None, bump_offset_index=None):
    """Write a synthetic case ZIP with nested member names under ``prefix``.

    ``spikes`` is an iterable of ``(sample, ap_channel)``; each spike plants a
    -1000 trough (with -500/-700 shoulders) at that AP channel. Identical
    empty chunks share one cached compressed blob.
    """
    bounds = chunk_spans(total_samples, chunk_len)
    zero_cache = {}
    blobs = []
    offsets = [0]
    cursor = 0
    for s, e in zip(bounds[:-1], bounds[1:]):
        n = e - s
        local = [(t, c) for (t, c) in spikes if s <= t < e]
        if local:
            raw = np.zeros((n, n_channels), np.int16)
            for t, c in local:
                lt = t - s
                raw[lt, c] = -1000
                if lt >= 1:
                    raw[lt - 1, c] = -500
                if lt + 1 < n:
                    raw[lt + 1, c] = -700
            blob = encode_chunk(raw)
        else:
            if n not in zero_cache:
                zero_cache[n] = encode_chunk(np.zeros((n, n_channels), np.int16))
            blob = zero_cache[n]
        blobs.append(blob)
        cursor += len(blob)
        offsets.append(cursor)
    if bump_offset_index is not None:
        offsets[bump_offset_index] += 500  # monotone but wrong: fails during decode
    header = {
        "algorithm": algorithm,
        "do_time_diff": do_time_diff,
        "n_channels": n_channels,
        "chunk_bounds": bounds,
        "chunk_offsets": offsets,
    }
    if ch_patch:
        header.update(ch_patch)
    buf = io.BytesIO()
    np.savez(buf, st=np.asarray(params_st, np.int64), cl=np.asarray(params_cl, np.int64),
             **(params_extra or {}))
    p = prefix or ""
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(f"{p}sim.imec0.ap.ch", json.dumps(header))
        z.writestr(f"{p}sim.imec0.ap.meta", make_meta_text(fs=fs))
        z.writestr(f"{p}sim.imec0.ap.cbin", b"".join(blobs), compress_type=zipfile.ZIP_STORED)
        z.writestr(f"{p}sim.imec0.ap_params.npz", buf.getvalue())
    return Path(zip_path)


def synthetic_registry_payload(*, no_bytes, no_md5=None, med_bytes=None, med_md5=None):
    """Patch the real packaged registry's case sizes/checksums (structure kept intact)."""
    from shift_aso.neuropixels.datasets import default_registry_path
    registry = json.loads(json.dumps(json.loads(default_registry_path().read_text())))
    cases = registry["neuropixels"]["cases"]
    cases["nodrift"]["bytes"] = int(no_bytes)
    if no_md5 is not None:
        cases["nodrift"]["computed_md5"] = no_md5
    if med_bytes is not None:
        cases["meddrift"]["bytes"] = int(med_bytes)
    if med_md5 is not None:
        cases["meddrift"]["computed_md5"] = med_md5
    return registry


def write_registry(path, **kwargs) -> Path:
    payload = synthetic_registry_payload(**kwargs)
    Path(path).write_text(json.dumps(payload, indent=2))
    return Path(path)


# ---- full-coverage meddrift-style fixture (production 300 s @ 30 kHz) ----

def full_spikes():
    spikes = []
    for i in range(10):
        spikes.append((150000 + i * 10, 60))    # unit 1, block 0
    spikes.append((180000, 310))                # unit 1199, block 0, single spike
    for i in range(20):
        spikes.append((200000 + i * 100, 10))   # unit 0, block 0 (gaps 100 < 121)
    for i in range(10):
        spikes.append((650000 + i * 10, 60))    # unit 1, block 2
    for i in range(6):
        spikes.append((950000 + i * 5000, 110))  # unit 650 (background), block 3
    return spikes


def full_st_cl():
    entries = ([(150000 + i * 10, 1) for i in range(10)]
               + [(180000, 1199)]
               + [(200000 + i * 100, 0) for i in range(20)]
               + [(650000 + i * 10, 1) for i in range(10)]
               + [(950000 + i * 5000, 650) for i in range(6)])
    entries.sort()
    st = np.array([e[0] for e in entries], np.int64)
    cl = np.array([e[1] for e in entries], np.int64)
    return st, cl


def patch_ch_member(src_zip, dst_zip, *, bump_offset_index):
    """Copy a synthetic archive, bumping one interior chunk offset (decode-time failure)."""
    with zipfile.ZipFile(src_zip) as z:
        members = {name: z.read(name) for name in z.namelist()}
    ch_name = next(name for name in members if name.endswith(".ap.ch"))
    header = json.loads(members[ch_name])
    header["chunk_offsets"][bump_offset_index] += 500
    members[ch_name] = json.dumps(header)
    with zipfile.ZipFile(dst_zip, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return Path(dst_zip)
