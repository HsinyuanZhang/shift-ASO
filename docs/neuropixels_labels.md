# Neuropixels label and score contract

This note records what a published Neuropixels detection must match, and what
an unmatched detection means. The numbers in
`results/neuropixels_comparison.csv` use this contract. They are not a count of
biological false alarms.

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

For `LC6_ShiftASO` on TEST 20–300 s the table is TP 1,417,149, FP 312,624,
FN 550,755, and 1,967,904 labels. Then

```text
A = 1417149 / (1417149 + 312624 + 550755) = 0.6214
precision = 0.8193
recall = 0.7201
```

`A` is lower than precision because missed labels stay in the denominator.
A second detection of a spike that already has a pair is a false positive even
though that spike is in `st`.

## What an unmatched detection is

On the no-drift DEV window (6–20 s), the soft-neighbor output has 10,210
unmatched detections. Measured against the same `st` and estimated channels:

| Nearby label | Count | Share of unmatched |
| --- | ---: | ---: |
| Within ±30 samples and 54 µm, but not the assigned detection | 4,239 | 41.5% |
| Within ±30 samples and 54–108 µm | 3,263 | 32.0% |
| Within 30–60 samples and 54 µm | 763 | 7.5% |
| None of the above | 1,945 | 19.0% |

The first three rows have a simulator spike in the file. The scorer refused
the pair because another detection took the label, the electrode is outside
54 µm, or the time is outside ±1 ms. Only the last row has no label in that
neighborhood.

Raw AP samples around 200 detections from each row, read from the decoded
int16 matrix, show the same split. Depth is the negative excursion from the
CAL median in units of that channel's CAL MAD / 0.6745. The deepest other
electrode within 40 µm is the neighbor.

| Group | Center depth | Deepest neighbor | Neighbor / center | Neighbor at least 4 |
| --- | ---: | ---: | ---: | ---: |
| Matched detections | 5.28 | 4.11 | 0.84 | 54% |
| Unmatched, label inside 54 µm and ±1 ms | 3.59 | 4.01 | 1.05 | 50% |
| Unmatched, 54–108 µm | 3.57 | 3.39 | 0.93 | 26% |
| Unmatched, 1–2 ms | 3.16 | 2.87 | 0.92 | 8% |
| Unmatched, no nearby label | 3.13 | 2.84 | 0.93 | 4% |

A label-backed unmatched detection has a multi-channel waveform. Its neighbor
is often deeper than the reported channel, which is what a one-to-one rule
turns into a false positive. The 1,945 detections with no nearby label are
shallower and almost never have a 4-sigma aligned neighbor. They are the only
group that looks like an event the simulator did not record.

A separate draw of about 400 DEV labels from the same matrix gives a center
depth of 4.6 and a deepest 40 µm neighbor of 4.1 (ratio 0.89). Channels at
least 300 µm away, and the same neighbors 100 ms later, sit near 2.6. The
spike is on several nearby electrodes. It is not a single-channel event and
not a probe-wide common mode. Requiring neighbor energy therefore does not
separate these label-backed duplicates from matched spikes: both have the footprint.

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
