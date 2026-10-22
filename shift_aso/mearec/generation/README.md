# MEArec multistage raw-data generation (public scope)

This package generates and reads the public MEArec raw-data condition used by
the shift-ASO benchmarks: **256-channel SqMEA-16-36, 20 kHz, 300 s, 64 units,
176248 spikes, independent 5 uVrms noise floor, total CAL SNR targets 10/15/20
dB, datasets D0-D4 only**. The portable implementation has no private
experiment paths or imports and is split into a pure core
(`core.py`) plus a native MEArec execution path (`native.py`, `cli.py`).

The 74 MB public template bank and spike-train assets under `assets/mearec/`
are read-only inputs; their sha256 values are pinned in the frozen,
`config.json` next to this README. Full native generation has not been rerun
for this portable release; synthetic validation is not production verification.

## CLI

```bash
python -m shift_aso.mearec.generation --dataset D0|D1|D2|D3|D4 --output PATH \
    [--assets-dir PATH] [--stage all|select|train|clean|noise|calibrate] \
    [--execute] [--config PATH] [--workers N] [--reserve-bytes N] \
    [--scratch-dir PATH]
```

- **Without `--execute` (default)**: prints a dry-run plan built from config
  and asset metadata only — dataset id, seeds, profile, asset paths/sizes,
  per-stage thread environments. No output directory is created, nothing is
  generated, and MEArec is never imported.
- **With `--execute`**: runs the requested stage. `--stage all` dispatches
  every stage as a **fresh subprocess** whose thread environment is fixed
  before the first NumPy import (spatial stage: `OPENBLAS/OMP=2` with at most
  2 workers = 4 compute threads total; every other stage: all BLAS/OMP/MKL/
  NUMBA = 1). The public `noise` stage runs the spatial half in-process and
  the unit half in a fresh single-thread child process (`noise_spatial` and
  `noise_unit` are the two internal child stages).
- MEArec, neo, quantities, and h5py are **optional** execution-time
  dependencies (install with `pip install -e ".[data]"`, MEArec 1.9.3);
  importing this package or dry-running the CLI never imports them.

### Datasets

| id | kind | layout | spike schedule | noise seeds |
|---|---|---|---|---|
| D0 | original reference | original per-unit bank instances (identity) | original public asset `unit_XX_times_s` schedule, verbatim | original lineage (spatial master 20261007, independent 20261017) |
| D1-D4 | new realizations | strict same-model selection | count-conditioned R=40 schedules preserving per-unit N_i | per-dataset preregistered seeds in `config.json` |

D0's `spiketrains.npz` asset contains 65 keys: the 64 `unit_00_times_s` ..
`unit_63_times_s` fields (exactly 176248 spikes) plus `firing_rates_hz`,
which is **metadata** — it is not a spike train and is never counted.

## Stages and output bundle

Each dataset writes a bundle `<output>/<DN>/`:

```
selection.json            strict same-model layout receipt (or D0 identity)
trains.npz                unit_00..unit_63 float64 seconds
clean.npy                 6000000 x 256 float32 uV (single chunk-filter pass)
metadata.npz              geometry(256,2), gt_times, gt_units, gt_channels,
                          gt_main, snr_main, template_amplitudes_uV,
                          filtered/original templates, template_ids,
                          template_locations/rotations/celltypes
spatial_noise.npy         6000000 x 256 float32 uV filtered distance noise
noise_unit.npy            6000000 x 256 float32 uV CAL-normalized floor
generation_manifest.json  scales, SNR verification, provenance, receipts
execution_binding.json    immutable source/config/asset closure for all stages
LEDGER.json               artifact -> sha256 bindings
*_receipt.json            per-stage commit receipts (written last)
```

Failure policy: every stage is **create-once** and **hash-verified**. An
existing artifact without a verifiable hash is a preserved partial (hard
stop, no adoption); a hash mismatch is drift (hard stop, no overwrite);
infeasible selection, unreachable SNR targets, non-finite moments, CAL
verification error >= 1e-4 dB, unit-sigma error >= 1e-5, missing
prerequisites, or < 20 GiB free reserve all stop deterministically. Partial
outputs and failed scratch are preserved, never automatically resumed or
adopted; counts are never trimmed or padded to hide a failure;
constraints and seeds are never relaxed or redrawn.

Scientific semantics (ported from the frozen production pipeline):

- **select** — per unit: same cell model, a bank instance different from the
  unit's original instance, native amplitude metric
  `max(|min(eap)|)` strictly inside (50, 300) uV, pairwise 3D distance
  >= 25 um; seeded per-unit permutation with first-feasible backtracking
  (pruning and memoization are exact and change no feasible branch). D0 keeps
  the original identity layout. Layouts must not be identical across datasets.
- **train** — D1-D4: N_i unique slots uniform over the compressed space
  `M = T - (N_i-1)*(R-1)`, sorted, `+ i*(R-1)`, R = 40 samples, one PCG64
  stream `SeedSequence([spiketrain_seed, 0])`, units drawn in order.
- **clean** — native MEArec convolution with the selected bank instances,
  `noise_level=0, filter=False` (plus the original distance-correlated noise
  parameter block and all seeds = the shared convolution control seed
  20261007), then the **original chunk filter exactly once** after
  convolution (order 3, 300-6000 Hz filtfilt, 198-sample pad, 100000-sample
  chunks). Bank-instance, model-binding, geometry, E/I, GT count/ISI/bounds
  assertions run before the receipt is committed.
- **noise_spatial** — native `chunk_distance_correlated_noise_` chunks,
  covariance `15 um / pairwise distance` with unit diagonal, chunk seeds
  `RandomState(master).randint(1000)` drawn 60 times, then the same chunked
  filter.
- **noise_unit** — white PCG64 `SeedSequence([seed, chunk])` per chunk, the
  same chunked filter, then per-channel CAL(0-6 s) sigma normalization of the
  full 300 s record.
- **calibrate** — CAL moments with the actual component covariance, spatial
  scalar per target from `brentq(xtol=1e-12)` on the **median linear**
  amplitude/sigma SNR, rounded to a nonnegative float32, verified from the
  actual float32 mixed noise to < 1e-4 dB.

## Reading data: `iter_input`

```python
from shift_aso.mearec.generation import iter_input

for first, block in iter_input("output_root/D1", snr=15, block_samples=100000):
    ...  # float32 (block_samples, 256) uV
```

Mixtures are **virtual**: no duplicate full raw waveforms exist per SNR.
`iter_input` requires a complete `generation_manifest.json`, re-verifies the
manifest ledger, execution binding, metadata/selection/train and component
hashes, plus declared array dtype/shape, before the first yield,
and reproduces the exact association order
`clean + spatial*float32(scale) + unit*float32(5)`.

## Exported pure functions

`native_amp`, `build_candidates`, `select_positions`, `assert_selection`,
`selection_receipt`, `sample_unit_times`, `sample_dataset`,
`validate_schedule`, `seconds_to_samples`, `spatial_covariance`,
`spatial_chunk_seeds`, `independent_chunk_rng`, `cal_moments`, `total_sigma`,
`cal_snr_db`, `solve_scale`, `mixed_noise_block_f32`,
`mixed_input_block_f32`, `unit_split_sigma`, `SplitAccumulator`,
`sha256_file`, `stop_before_write`, `component_bytes`, `load_config`,
`validate_config`, `plan_dataset`, `resolve_assets_dir`, `create_once`,
`verified_skip`, `ledger_update`, `write_receipt`, `iter_input`,
`read_generation_manifest`, `resolve_thread_env`, `StopGeneration`.

## No encoding stage

This is a **dataset-only** release. The shift-ASO level-crossing ADC encoder
and detector are downstream consumers of `iter_input`; they are deliberately
not part of the bundle or the stage list (there is no stub stage). This keeps
raw-data provenance independent of detector-side quantization changes.

## Testing

```bash
cd shift_aso/mearec/generation
PYTHONPATH=<repo> taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 python -m unittest test_generation -v
```

The suite is tiny-synthetic/mocked only: same-model strict-amp/min-dist
selection with an unpruned first-feasible parity reference, R=40
count-conditioned sampler oracles, covariance 15/d/diag-1, linear-median SNR
and brentq solver with float32 readback and unreachable-target stop, exact
float32 association order, streamed split statistics, create-once/ledger/
hash-drift/partial-preservation guards, stage env dispatch and child argv,
dry-run plans (in-process, subprocess `python -m`, and fresh-interpreter
optional-import hygiene), and synthetic-bundle `iter_input` including hash
drift.

## Limitations

- Native production generation is **implemented but not executed** in this
  authoring wave: no 300 s generation, no real selection/train/noise draws,
  no smoke run. Native fidelity is ported from the frozen reference sources;
  full native execution needs separate validation before claiming reproduction.
- No cross-platform bitwise identity guarantee: CPU/BLAS can change noise
  realizations. Receipts record versions, thread env, seeds, and payload/file
  hashes.
- Finite shared native template bank; count-conditioned (not unconditional
  Poisson) schedules; position and orientation change jointly; the
  convolution control seed is shared — see `limitations` in `config.json`.
