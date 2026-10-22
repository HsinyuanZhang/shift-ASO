# MEArec D0–D4: clean recording, noise, and total-CAL-SNR preparation

## Scope and reproducibility boundary

Only **5 µVrms independent noise floor**, **10/15/20 dB total CAL SNR**, and
**D0–D4** are defined by `configs/mearec_multiseed.json` (installed mirror:
`shift_aso/mearec/generation/config.json`). Each recording is 6,000,000 × 256
float32 samples in µV: 300 s at 20 kHz, SqMEA-16-36, 64 units (51 E / 13 I),
and 176,248 GT spikes including overlaps.

This portable implementation follows the executed multi-realization method.
Its full native production path has **not been rerun for this release**.
Unit/synthetic tests are not a claim of verified full-recording reproduction.
There is no cross-platform waveform-bit-identity promise: native NumPy/BLAS
and CPU effects are recorded as limitations, not hidden by changing seeds.

## 1. Bank selection and schedules

The public MEArec 1.9.3 bank has SHA-256
`53d608a9a7c729a674b55ea98a8cdc65075bc0fef432fc819898859f62f0b662`.
D0 uses the original 64 bank IDs and public `unit_00_times_s` through
`unit_63_times_s` arrays verbatim. The extra `firing_rates_hz` vector is
metadata, not a 65th spike train.

D1–D4 keep each unit's original cell-model identity and exact spike count,
but select a different instance of that model. Native amplitude must satisfy
**50 < A < 300 µV**, and pairwise soma distance must be **≥25 µm**. Position,
orientation, and the template waveform change jointly; this is selection
from a finite simulated bank, not a new continuous-position NEURON run.
Seeded per-unit candidate permutations and first-feasible backtracking are
fixed. Exact pruning does not reorder feasible candidates. Infeasibility or
a search-budget stop is an error: no fallback, redraw, relaxation, or
outcome-based layout choice.

For each new schedule, let `T=6000000`, per-unit count `N_i`, and `R=40`
samples (2 ms). Draw `N_i` unique slots without replacement from
`M = T - (N_i-1)*(R-1)`, sort, then add `rank*(R-1)`. Units are visited in
original order using one PCG64 stream `SeedSequence([train_seed, 0])`.
This preserves counts and the minimum ISI; it is **count-conditioned**, not
an unconditional Poisson process. Sample times are stored as float64 seconds
and GT is recovered by `rint(seconds*fs)`.

| Dataset | Position seed | Train seed | Spatial master | Independent seed |
|---|---:|---:|---:|---:|
| D0 | 20261007 (identity) | 20261007 (saved schedule) | 20261007 | 20261017 |
| D1 | 20261201 | 20261211 | 20261018 | 20261118 |
| D2 | 20261202 | 20261212 | 20261019 | 20261119 |
| D3 | 20261203 | 20261213 | 20261020 | 20261120 |
| D4 | 20261204 | 20261214 | 20261021 | 20261121 |

All share convolution control seed `20261007`, the bank, model assignments,
and counts. Thus five realizations are not five independent populations.

## 2. Clean convolution and noise components

Native convolution uses the selected bank instances and supplied schedule,
with `noise_level=0` and `filter=False`. Selection/model bindings, actual
native positions/rotations/templates, E/I counts, geometry, and GT counts/range
are checked before a clean receipt. The original order-3 300–6000 Hz
`filtfilt` chunk filter is applied **once**, in 100,000-sample chunks with
198 samples of context padding.

Spatial noise uses covariance diagonal 1 and off-diagonal `15 µm / distance`.
Sixty native chunk seeds are drawn from
`RandomState(spatial_master).randint(1000)` and passed to the native MEArec
distance-correlated noise function. Independent noise uses a fresh
`PCG64(SeedSequence([independent_seed, chunk_index]))` per chunk. Both
components receive the same original chunk filtering. The independent
component is normalized by each channel's **CAL 0–6 s** sigma, and that same
normalization is held for the full duration; it is not renormalized per split.

## 3. CAL-only SNR and virtual mixtures

For each target, solve a nonnegative spatial scale using the actual CAL
component variances **and covariance**. Unit amplitude is derived from the
filtered nominal template on its SNR main channel. The definition is

```text
SNR_dB = 20*log10(median_u(A_u / sigma_total_CAL[snr_main_u]))
sigma_total^2 = scale^2*var(spatial) + var(5*unit) + 2*scale*cov(spatial,5*unit)
```

This is the **median of linear ratios**, not the median of log ratios. The
denominator is pure noise, not clean signal or quantization error. A
floor-only ceiling at/below a target causes a hard stop. Brent root solving
uses `xtol=1e-12`; scale is rounded to float32 and the actual float32 mixed
noise is checked to <1e-4 dB CAL error. DEV/TEST noise statistics are descriptive,
not used to choose scale. Splits are CAL 0–6, DEV 6–20, TEST 20–300 s.

The exact input association is:

```python
(clean + spatial * np.float32(scale)) + unit * np.float32(5.0)
```

No duplicate full waveform is stored for every SNR. Read a completed bundle:

```python
from shift_aso.mearec.generation import iter_input
for first, block_uV in iter_input("data/mearec_multiseed/D0", 15):
    pass  # consume float32 blocks; no encoding/detector is implied
```

The reader requires a complete manifest and verifies source hashes and declared
array formats before yielding. GT and geometry are in `metadata.npz`.

## 4. CLI and artifacts

Run from a source checkout containing `assets/mearec/`, or pass its directory
explicitly (the directory containing `templates.h5`, not its parent):

```bash
python -m pip install -e '.[data]'
# Default plan only: no output creation, optional imports, or generation.
python -m shift_aso.mearec.generation --dataset D0 --output data/mearec_multiseed
# Explicit heavy execution for a user wishing to generate the dataset.
python -m shift_aso.mearec.generation --dataset D0 --output data/mearec_multiseed \
  --assets-dir assets/mearec --execute
# Repeat explicitly for D1, D2, D3, D4 as needed.
```

`--stage all|select|train|clean|noise|calibrate` defaults to `all`.
`all` uses fresh stage processes. Spatial noise has BLAS/OMP=2 and at most
two workers; other library thread controls are 1 (process pools ≤4).
A minimum **20 GiB reserve** is enforced on output and scratch filesystems
before large writes; plan for tens of GiB per dataset plus scratch. Installed
wheels do not include the large template assets; supply `--assets-dir` from
the source checkout.

`<output>/DN/` contains `selection.json`, `trains.npz`, `clean.npy`,
`metadata.npz`, `spatial_noise.npy`, `noise_unit.npy`, and
`generation_manifest.json`, plus execution binding, ledger, and stage receipts.
Source/config/assets, seeds, versions, thread settings, hashes and split noise
statistics are recorded. Native bank/GT checks and float32 CAL readback occur
before successful receipts. Successful artifacts may be skipped only after
hash verification under the same execution binding. Failed partial artifacts
and scratch are retained and **never automatically adopted or resumed**;
inspect them and choose a new output/scratch root for an explicit retry.

## Legacy separation

`scripts/generate_mearec.py`, `configs/mearec_generation.json`, and the legacy
HDF5 replay describe the original input recipe, not the D0–D4 preparation
method. The generation package does not encode Q, detect events, or select
thresholds. The separate [current software package](mearec.md) operates on
supplied FP/frozen LC6 arrays. No observed experiment records, event arrays,
logs, or scores are stored in this repository.
