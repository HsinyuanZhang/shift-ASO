"""Native Open Ephys ``.ap.meta`` parsing (geometry and sample rate).

Port of the public geometry contract: 384 AP shank-map entries, active
channels flagged per entry, staggered NP1.0 geometry (x = bank column * 32
+ 27/11 um, y = row * 20 um). The parser is independent of the legacy
detector module; the legacy ``shift_aso.neuropixels.core`` stays untouched.
"""
from __future__ import annotations

import re

import numpy as np

N_AP = 384
SNS_SHANK_MAP = "~snsShankMap="
SNS_SAMPLE_RATE = "~imSampRate="


class MetaContractError(RuntimeError):
    """Raised when an ``.ap.meta`` member violates the native format contract."""


def parse_shank_map_rows(text: str):
    line = next((x for x in text.splitlines() if x.startswith(SNS_SHANK_MAP)), None)
    if line is None:
        raise MetaContractError("meta lacks ~snsShankMap")
    rows = re.findall(r"\((\d+):(\d+):(\d+):(\d+)\)", line)
    if len(rows) < N_AP:
        raise MetaContractError(f"meta has fewer than {N_AP} AP map entries")
    return rows[:N_AP]


def active_geometry(meta_text: str):
    """Return (active_original_ap_ids, geometry_um) for the active AP channels."""
    rows = parse_shank_map_rows(meta_text)
    active = np.asarray([i for i, (_, _, _, used) in enumerate(rows) if int(used)], np.int64)
    g = np.asarray(
        [
            [
                int(rows[i][1]) * 32.0 + (27.0 if int(rows[i][2]) % 2 == 0 else 11.0),
                int(rows[i][2]) * 20.0,
            ]
            for i in active
        ],
        np.float64,
    )
    return active, g


def sample_rate_hz(meta_text: str) -> int:
    lines = [x for x in meta_text.splitlines()
             if x.startswith((SNS_SAMPLE_RATE, "imSampRate="))]
    if len(lines) != 1:
        raise MetaContractError("meta must have one imSampRate (optional leading ~)")
    try:
        rate = float(lines[0].split("=", 1)[1].strip())
        if not np.isfinite(rate) or rate <= 0 or not rate.is_integer():
            raise ValueError("sample rate must be a positive integer")
        return int(rate)
    except ValueError as exc:
        raise MetaContractError(f"meta imSampRate is not a valid integer: {lines[0]!r}") from exc
