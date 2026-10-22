"""Explicit method configuration, CAL derivation, and small-array convenience API."""
from __future__ import annotations

import numpy as np
from .detect import detect
from .features import activity, cal_stat
from .grid import EMA_RHOS, LAGS


def validate(method, lag, rho, k):
    """Accept ANY finite nonnegative K, including local-refinement factors."""
    if method not in {'plain', 'ema'}:
        raise ValueError('method must be plain or ema')
    if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)) or lag not in LAGS:
        raise ValueError(f'lag must be one of {LAGS}')
    if not np.isfinite(rho) or (method == 'plain' and rho != 0) or (method == 'ema' and rho not in EMA_RHOS):
        raise ValueError('rho does not match the selected method')
    if isinstance(k, (bool, np.bool_)) or not np.isfinite(k) or k < 0:
        raise ValueError('K must be finite and nonnegative')
    return {'method': method, 'lag': int(lag), 'rho': float(rho), 'K': float(k),
            'policy': 'center_only', 'timing': 'window_minimum'}


def normalize(configuration):
    """Reject unsupported polarity/time policies rather than silently replacing them."""
    cfg = validate(configuration['method'], configuration['lag'], configuration['rho'], configuration['K'])
    for field in ('policy', 'timing'):
        if configuration.get(field, cfg[field]) != cfg[field]:
            raise ValueError('unsupported current-method '+field+'; legacy replay is separate')
    return cfg


def calibrate(signal, configuration, cal_samples):
    """Derive an unlabeled CAL scale without reading samples after CAL."""
    cfg = normalize(configuration)
    if not 0 < cal_samples <= len(signal):
        raise ValueError('invalid CAL length')
    energy, _ = activity(signal[:cal_samples], cfg['lag'], cfg['rho'])
    return cal_stat(energy, cal_samples, cfg['lag'])


def run(signal, geometry, threshold=None, method='plain', lag=1, rho=0., k=1.,
        *, cal_samples=120000, bounds=None, guard=40):
    """Small-array batch convenience API.

    ``threshold`` is a supplied per-channel CAL BASE, not final H. If omitted,
    derive it from unlabeled [0,cal_samples). Final H=K*CAL_base, strict >H.
    This path allocates the full feature; use ``replay`` for large arrays.
    """
    cfg = validate(method, lag, rho, k)
    feature, _ = activity(signal, cfg['lag'], cfg['rho'])
    base = cal_stat(feature, cal_samples, cfg['lag']) if threshold is None else np.asarray(threshold, np.float64)
    return detect(signal, feature, base * cfg['K'], geometry, lag=cfg['lag'], bounds=bounds, guard=guard)
