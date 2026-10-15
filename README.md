# shift-ASO

Software benchmarks for Shift-ASO spike detection on Quiroga, MEArec, and the
Kilosort4 no-drift Neuropixels simulation.

```bash
python -m pip install -e .
python -m shift_aso.quiroga --data-dir data/quiroga --output outputs/quiroga --workers 8
python -m shift_aso.mearec --recording data/mearec.h5 --output outputs/mearec
python -m shift_aso.neuropixels --data-dir data/neuropixels --output outputs/neuropixels
```

See [data preparation](docs/data.md), [MEArec](docs/mearec.md), and
[Neuropixels](docs/neuropixels.md) for input formats and settings. MEArec and
Neuropixels each run three methods: **SI10 + MAD**, **NEO10 + EMA**, and
**LC6 Shift-ASO with neighbor assistance**. SI and NEO use signed 10-bit inputs.
Shift-ASO uses six-bit level-crossing codes with a maximum update of eight codes
per visit.

## Shift-ASO

For the LC code change `j[n]` and current code `q[n]`, the feature is
`a[n] = max(q[n] * j_hat[n], 0)`, followed by
`E[n] = rho * E[n-1] + a[n]`. The magnitude map is
`1,2,3,4,5,6,7,8 -> 1,2,2,4,4,8,8,8`, with the sign preserved. It applies only
to the feature; the ADC trajectory uses the original update. Products with
`j_hat` can use shifts. Local extrema, neighbor evidence, and peak exclusion
select the final events.

Parameters are supplied in the package JSON files, with readable copies under
`configs/`. Each dataset has its own settings. MEArec uses one fixed range for
all channels; Neuropixels also uses one uniform range. The commands use fixed
settings and do not search parameters.

## Evaluation

| Dataset | CAL (s) | DEV (s) | TEST (s) |
|---|---:|---:|---:|
| Quiroga | 0–6 | 6.002–18 | 18.002–59.998 |
| MEArec | 0–6 | 6–20 | 20–300 |
| Neuropixels | 0–6 | 6–20 | 20–300 |

Accuracy is `A = TP / (TP + FP + FN)`. Multichannel scores use one-to-one
matching within ±1 ms and 54 µm, including overlapping ground-truth spikes.
Neuropixels reference channels are estimates from labeled waveforms used only
for scoring. Detection uses a 40 µm neighborhood on Neuropixels and 54 µm on
MEArec. A Neuropixels detection matches simulator time `st` and the estimated
main channel of unit `cl`. It does not match unit identity or waveform shape.
Unmatched detections can still sit on a labeled spike that already has a pair.
See [Neuropixels labels](docs/neuropixels_labels.md).

| Method | MEArec base A (%) | MEArec independent noise A (%) | MEArec common noise A (%) | Neuropixels A (%) |
|---|---:|---:|---:|---:|
| SI10 + MAD | 91.47 | 76.85 | 75.54 | 58.79 |
| NEO10 + EMA | 67.65 | 42.48 | 41.27 | 56.27 |
| LC6 Shift-ASO | 87.82 | 73.77 | 81.38 | 62.14 |

Counts, recall, precision, and false positives per second are in
[MEArec results](results/mearec_comparison.csv) and
[Neuropixels results](results/neuropixels_comparison.csv). The MEArec generation
recipe recreates the data preparation; its waveform is not bit-identical to the
stored benchmark, so regenerated scores can differ.

Quiroga retains the single-channel comparison. Shift-ASO LC4/5/6 uses `k=0.5`
and `H = k * Q99(E_CAL)`. Its timing diagnostic shifts selected peaks by
−23 samples, uses ±1 ms matching and ±1 ms overlap exclusion, and localizes
peaks on the filtered waveform. Results are means across 16 recordings, with
lowest-record results in [results/quiroga](results/quiroga).
The bundled Quiroga LC kernel requires `g++` on first use.

## Code layout

```text
shift_aso/quiroga/       single-channel input, detection, and scoring
shift_aso/mearec/        HDF5 input, noise, quantization, detection, and scoring
shift_aso/neuropixels/   CBIN input, quantization, detection, and scoring
configs/                dataset and method settings
scripts/                data acquisition and generation
assets/mearec/          templates and spike trains
results/                comparison tables
```
