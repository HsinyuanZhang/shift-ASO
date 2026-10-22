"""Native mtscomp CBIN decode semantics with generic (discovered) member names.

Decode chain (identical to the accepted runtime ``ZipCbinStream`` and the
legacy public ``stream_cbin``): zlib-compressed int16 blobs, per-chunk time
differencing undone with a Fortran-order reshape and an int32 cumulative sum
cast back to int16, first 384 columns are AP channels, masked to the active
channel list, yielded as float32 with exact integer values.

Differences from the legacy detector reader (which stays untouched):
  * the ZIP member name is a constructor argument, not the hardcoded
    ``sim.imec0.ap.cbin``;
  * chunks are yielded per native ``chunk_bounds`` segment, truncated at the
    requested stop sample, exactly like the accepted reference runtime; the
    legacy detector reader re-blocks into 1 s (FS-sample) pieces for its
    encoder loop and is a different consumer;
  * the stream is reiterable (one fresh ZIP handle per ``chunks()`` pass) and
    closes its handles deterministically.
"""
from __future__ import annotations

import json
import zipfile
import zlib

import numpy as np

from ._meta import N_AP

N_RECORDED = 385  # 384 AP channels + 1 sync/reference channel


class CbinContractError(RuntimeError):
    """Raised when a ``.ap.ch`` header or CBIN payload violates the native contract."""


class CbinHeader:
    """Validated view of an ``.ap.ch`` JSON header."""

    __slots__ = ("bounds", "offsets", "n_channels", "raw")

    def __init__(self, header: dict):
        self.raw = dict(header)
        if header.get("algorithm") != "zlib" or not header.get("do_time_diff"):
            raise CbinContractError("expected zlib time-differenced .ch")
        bounds = np.asarray(header["chunk_bounds"], np.int64)
        offsets = np.asarray(header["chunk_offsets"], np.int64)
        n_channels = int(header["n_channels"])
        if n_channels != N_RECORDED:
            raise CbinContractError(f"expected {N_RECORDED} recorded channels, got {n_channels}")
        if bounds.ndim != 1 or offsets.ndim != 1 or len(offsets) != len(bounds) or len(bounds) < 2:
            raise CbinContractError("chunk_bounds/chunk_offsets must be aligned 1-D arrays with >= 2 entries")
        if int(bounds[0]) != 0 or np.any(np.diff(bounds) <= 0):
            raise CbinContractError("chunk_bounds must start at 0 and strictly increase (contiguous tiling)")
        if int(offsets[0]) != 0 or np.any(np.diff(offsets) <= 0):
            raise CbinContractError("chunk_offsets must start at 0 and strictly increase")
        self.bounds = bounds
        self.offsets = offsets
        self.n_channels = n_channels

    @property
    def total_samples(self) -> int:
        return int(self.bounds[-1])

    @property
    def total_compressed_bytes(self) -> int:
        return int(self.offsets[-1])

    @property
    def n_chunks(self) -> int:
        return int(len(self.bounds) - 1)


def parse_ch_header(text: str) -> CbinHeader:
    """Parse and validate an ``.ap.ch`` header from its JSON text."""
    try:
        header = json.loads(text)
    except ValueError as exc:
        raise CbinContractError(f".ch header is not valid JSON: {exc}") from exc
    return CbinHeader(header)


class ZipCbinStream:
    """Reiterable forward-only ZIP-cbin reader yielding complete native chunks.

    ``chunks(stop_sample)`` yields ``(start, active_masked_float32)`` per
    decompressed ``chunk_bounds`` segment, truncated at ``stop_sample``; each
    call performs one sequential read pass of the ZIP member and closes its
    handles when the pass ends.
    """

    def __init__(self, zip_path, cbin_member: str, header: CbinHeader, ap_ids):
        self.zip_path = str(zip_path)
        self.cbin_member = str(cbin_member)
        self.header = header
        self.ap_ids = np.asarray(ap_ids, np.int64)
        if int(header.n_channels) != N_RECORDED:
            raise CbinContractError("header channel count already validated; got unexpected value")
        if self.ap_ids.size == 0 or self.ap_ids.max() >= N_AP or self.ap_ids.min() < 0:
            raise CbinContractError("ap_ids must be a non-empty subset of the 384 AP channels")

    def __enter__(self) -> "ZipCbinStream":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        """Handles are opened and closed per pass; nothing is held between passes."""

    def chunks(self, stop_sample: int):
        """Yield ``(start, raw)`` for each native chunk up to ``stop_sample``."""
        stop_sample = int(stop_sample)
        header = self.header
        with zipfile.ZipFile(self.zip_path) as archive, archive.open(self.cbin_member) as handle:
            cursor = 0
            for chunk in range(header.n_chunks):
                start, end = int(header.bounds[chunk]), int(header.bounds[chunk + 1])
                if start >= stop_sample:
                    break
                target = int(header.offsets[chunk + 1])
                if target < cursor:
                    raise CbinContractError('non-monotonic compressed offsets')
                blob = handle.read(target - cursor)
                cursor = target
                raw = np.frombuffer(zlib.decompress(blob), dtype=np.int16)
                expected = (end - start) * header.n_channels
                if raw.size != expected:
                    raise CbinContractError(f'chunk {chunk} decoded {raw.size}, expected {expected}')
                full = np.cumsum(raw.reshape((end - start, header.n_channels), order='F'), axis=0,
                                 dtype=np.int32).astype(np.int16)
                use = min(end, stop_sample) - start
                yield start, full[:use, :N_AP][:, self.ap_ids].astype(np.float32, copy=False)

    def stream_factory(self, stop_sample: int):
        """Return a zero-argument callable for one ``chunks(stop)`` pass."""
        stop_sample = int(stop_sample)

        def factory():
            return self.chunks(stop_sample)

        return factory
