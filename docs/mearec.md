# MEArec software: plain NEO and post-NEO EMA

The current implementation lives in `shift_aso/mearec/software/`. It takes
caller-supplied FP arrays or frozen LC6 arrays. It does not download recordings,
regenerate noise, encode Q, or bundle experiment results.

## Two explicit stages

```bash
# Complete every case's DEV selection, then write one global freeze barrier.
python -m shift_aso.mearec dev --manifest inputs.json --output ../runs/dev

# Verify inputs/source/runtime, replay frozen DEV exactly, then evaluate TEST.
python -m shift_aso.mearec test --dev-run ../runs/dev --output ../runs/test
```

`inputs.json` is a caller-created manifest. Paths are relative to that manifest:

```json
{
  "cases": [
    {
      "id": "recording_fp",
      "front": "FP",
      "signal": "inputs/signal.npy",
      "geometry": "inputs/geometry.npy",
      "labels": "inputs/labels.npz"
    },
    {
      "id": "recording_lc6",
      "front": "LC6",
      "signal": "inputs/frozen_q.npy",
      "geometry": "inputs/geometry.npy",
      "labels": "inputs/labels.npz"
    }
  ]
}
```

- Signal: `(samples, channels)` NPY, float32/float64 for FP or int8 for LC6.
  LC6 is supplied, not regenerated: range31 and actual emitted slew8 are checked,
  carrying the real preceding Q across blocks and using Q[-1]=0 at origin.
- Geometry: `(channels,2)` NPY in micrometers.
- Labels: NPZ with equal-length integer `times` (zero-origin sample indices) and
  `channels` (local channels). Unknown/out-of-range channels remain GT rows, not
  dropped observations. Overlapping spikes are retained.
- Sampling rate:20kHz. Default CAL `[0,120000)`, DEV `[120000,400000)`, TEST
  `[400000,N)`. Each case may explicitly override `cal_samples` and `dev_stop`
  for a different supplied interval; no automatic cropping or resampling occurs.

The manifest can contain all desired recordings/inputs. TEST is unavailable
until **every** listed DEV case has succeeded and `COMPLETE.json` was written
last. There is no hidden retry, overwrite, resume, or automatic stage switch.
A failed run preserves its partial directory; inspect it and choose a new path.

## Method and selection rules

Signed float64 centered NEO:

```text
psi[n,c] = x[n,c]^2 - x[n-L,c] * x[n+L,c]
E[n,c] = rho * E[n-1,c] + psi[n,c]
```

Incomplete recording-edge centers have zero drive. There is no rectification,
normalization, neighbor-energy addition, or split-state reset. Plain uses rho0;
EMA searches only0.5,0.875,0.9375. Both search lags1,2,4,6,8 and use the SAME
fixed negative-center polarity and window-minimum timing.

CAL base is per-channel `mean(abs(E[L:CAL-L]))`; H=K*base, strict amplitude>H,
without a floor or epsilon. Coarse K is0,1,2,3,4,6,8,12,16,24,32. Each of20
groups adds up to3 new local points in the true coarse-winner bracket
(geometric for positive lower endpoint, otherwise linear). A still-winning
upper endpoint is explicitly doubled with3 interior points plus the endpoint,
until it loses or reaches the finite positive-stat no-output bound. Unique
stable IDs are preserved. Any finite nonnegative refined K can be replayed.

Selection maximizes exact rational A=TP/(TP+FP+FN), then lower FP, then lower
stable ID. Strict optimistic pruning is GROUP-local, never family-global, so it
cannot change the true group winner or refinement bracket. Pruned metrics stay
unknown. Ordered time/channel aliases can reuse scores without asserting that
feature amplitudes are equal. Seed-first evaluation changes order, not IDs.

## Detection and timing

Spatial locally exclusive NMS uses8samples and54um. Each candidate must be
positive, at least every same-time neighbor, strictly above all past values,
and at least all future values within the window. Comparisons use unthresholded
energy; a peak that fails its own H or polarity can still suppress a neighbor.
After H, exact same-time/equal-amplitude ties use channel-ascending direct
neighbors, then the raw feature-center must be negative. Only then is time
moved to the earliest raw minimum in[-12,+8]. There is no second NMS.

Events contain `sample_index`, `channel_index`, `feature_peak`, `amplitude`,
and `availability_sample=feature_peak+L+8`. Eligibility requires sample>=lo,
sample+40<hi, availability<hi, and complete global feature/NMS support. GT in
[lo,hi) stays in the denominator, including unknown-channel/end-guard labels.
Matching maximizes one-to-one pair count within20samples and54um, then minimizes
absolute timing error. Pair indices point to original full GT rows.

## Replaying without a search

For a small array use `software.run(...)`. An explicitly supplied `threshold`
argument is a CAL **base**; K is applied once. Without it, base is derived from
unlabeled CAL. For large NPY memmaps use `software.replay(...)`, supplying
configurations and already-frozen final H vectors. It warms every variant from
sample zero in one sequential pass, with bounded raw/feature halos and state.
Output events, unlike recording buffers, necessarily scale with event count.

## Validation and public/private boundary

DEV writes local choices, candidate statuses, H/CAL/events/pairs and hash
bindings. TEST first re-derives CAL and exactly replays DEV H/events/pairs/metrics.
Only then does it stream TEST using the frozen choices. Source, dependency
versions and input hashes are checked; unknown/failing results are never adopted.
These generated files are LOCAL outputs, not repository contents.

Tests use synthetic signals, including nonempty multichannel/tie/boundary and
complete two-stage cases. Synthetic equality checks are not a claim that this
portable release has re-scored the private full-recording experiments. No
benchmark table, selected winner file, or observed performance number is
published. See the dataset construction instructions in [mearec_data.md](mearec_data.md).

## Separate legacy HDF5 interface

Old SI10/NEO10/Shift-ASO HDF5 code is retained explicitly as legacy, not silently
relabeled as the current methods:

```bash
python -m shift_aso.mearec legacy --help
```

It uses the original HDF5 recipe and API. Its old result tables are not bundled.
It is NOT the current FP/LC6 small-grid NEO/EMA workflow above.
