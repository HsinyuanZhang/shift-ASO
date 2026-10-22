# Legacy no-drift label and score contract

> This page specifies the legacy all-1200-unit scoring policy, without observed diagnostics.
> New prepared labels retain explicit good/background groups, but do not change
> this scorer. Medium-drift requires its own reference; see
> [case-bound preparation and group-policy boundary](neuropixels_data.md).
> This is not a good-600/background-ignore or medium-drift evaluation policy.

This note records what a published Neuropixels detection must match, and what
an unmatched detection means. It describes the scoring contract only. This
repository does not publish counts, scores, waveforms, or experiment records.

## Source arrays

`public_labels` in `shift_aso/neuropixels/core.py` reads the Kilosort4
`sim_no_drift.zip` member `sim.imec0.ap_params.npz`.

| Field | Role |
| --- | --- |
| `st` | Spike time, integer sample index at 30 kHz. This is the time that scoring uses. |
| `cl` | Simulator unit id. Scoring does not compare unit ids. |

The channel paired with each spike is not stored in that NPZ. The code looks
up `cl` in `shift_aso/neuropixels/_unit_to_main_channel.json`, a frozen vector
of 1200 local AP channels. Its source string says the map was estimated from
raw CBIN over 30–60 s: zero-phase Butterworth order 3, 300–6000 Hz, at most 80
spikes per unit, window `[-20, +40]` samples, and the channel of maximum
absolute template amplitude. That window overlaps TEST (20–300 s). The map is
a scoring reference, not the simulator electrode and not the negative-trough
channel a detector reports.

If `evaluation_labels.npz` is present, `load_labels` uses it instead. It must
contain `sample_index`, `estimated_local_channel`, `geometry_um`, and
`local_to_original_AP`, and the geometry must match the AP map. Those channels
are the same kind of estimate.

Electrode positions come from `~snsShankMap` in `sim.imec0.ap.meta`
(`geometry` in `core.py`). Disabled contacts are dropped. The active set has
383 channels. Horizontal position is `column * 32 + (27 if even else 11)` µm
and vertical position is `row * 20` µm.

## What counts as a match

`canonical_score` keeps a detection only when
`sample_index >= lo` and `sample_index + 60 < hi`. Every label with
`st >= lo` and `st < hi` stays in the denominator, including labels in the
last 60 samples. The 60-sample cut is an output guard, not a label filter.

`spatial_pairs` in `shift_aso/neuropixels/_matching.py` then builds one
pairing. A pair is allowed when both of these hold:

- `|detection_sample - st| <= 30` (1 ms at 30 kHz, inclusive);
- Euclidean distance between the detection channel and the estimated label
  channel is `<= 54` µm (inclusive).

The assignment first maximizes the number of pairs, then minimizes the total
absolute sample error. One label can credit only one detection, and one
detection can take only one label. Unit id, waveform shape, and amplitude are
not matched.

Detection neighborhoods are separate from this score radius. `canonical_geometry_sets`
uses other electrodes at `0 < d <= 40` µm as neighbors and `40 < d <= 80` µm
as the ring.

True positives are the pairs. False positives are detections left unpaired.
False negatives are labels left unpaired.

```text
A = TP / (TP + FP + FN)
precision = TP / (TP + FP)
recall = TP / labels
```

`A` is lower than precision because missed labels stay in the denominator.
A second detection of a spike that already has a pair is a false positive even
though that spike is in `st`.

## What an unmatched detection is

An unmatched detection may still have a simulator spike nearby. The one-to-one
rule can reject it because another detection already took that label, because
the electrode is outside the scoring radius, or because the time is outside
the scoring tolerance. Only a detection with no label in that neighborhood is
a candidate for an event absent from the simulator record. This page states
the contract without publishing observed counts or scores.

## Code map

| Step | Location |
| --- | --- |
| Read `st` and `cl` | `public_labels` in `shift_aso/neuropixels/core.py` |
| Estimated channel | `shift_aso/neuropixels/_unit_to_main_channel.json` |
| Optional external labels | `load_labels` |
| Electrode map | `geometry` |
| 40 µm / 80 µm neighborhoods | `canonical_geometry_sets` |
| 60-sample guard, 54 µm, ±30 samples, `A` | `canonical_score` |
| One-to-one assignment | `spatial_pairs` in `shift_aso/neuropixels/_matching.py` |
