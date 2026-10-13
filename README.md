# shift-ASO

Fixed-parameter reference replays for the shift-ASO detector on the Quiroga and Neuropixels simulations. These are software replays. Quiroga builds its bundled LC kernel with `g++` on first use.

```bash
python -m pip install -e .
python -m shift_aso.quiroga --data-dir /path/to/quiroga_mat --output results/quiroga --workers 8
python -m shift_aso.neuropixels --data-dir /path/to/neuropixels --output results/neuropixels
```

Quiroga Shift-ASO uses a 4/5/6-bit residual LC encoder with `J=8`,
`a[n] = max(q[n] * j_hat[n], 0)`, and `E[n] = 0.875 E[n-1] + a[n]`.
The step map acts on `abs(j)`: `1,2,3,4,5,6,7,8 -> 1,2,2,4,4,8,8,8`, while
preserving the original sign. LC4, LC5, and LC6 use
`k=0.5`; each recording uses `H = k * Q99(E_CAL)`. The Quiroga command has no
threshold grid. Its primary timing diagnostic corrects selected peaks by -23
samples and matches within ±1 ms after ±1 ms overlap exclusion. Peak localization
uses the filtered waveform, so it is not a fully hardware-mapped timestamp path.

The Neuropixels replay uses the bank8 LC6/J8 fixed-MAD coefficient `K=12.9` in
different score units. It is software only and scores against frozen estimated
reference channels.

`--workers` controls Quiroga record-level workers. The Neuropixels command reads
the first 300 s of the 30 kHz, 383-active-channel no-drift simulation. Parameters
are frozen in this release.

| Replay | CAL (s) | DEV (s) | TEST (s) |
|---|---:|---:|---:|
| Quiroga | 0--6 | 6.002--18 | 18.002--59.998 |
| Neuropixels | 0--10 | 10--20 | 20--300 |

See [docs/data.md](docs/data.md) for public data sources and download metadata.
Comparison files are `results/quiroga/expected_method_summary.csv` and
`results/neuropixels_comparison.csv`.
