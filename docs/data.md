# Data preparation

## Quiroga: native conventions

Place the sixteen original `C_*.mat` recordings in one directory. Obtain the
Quiroga simulations from [Wave_clus](https://github.com/csn-le/wave_clus).
Required fields are `data`, `spike_times`, `spike_class`, and
`samplingInterval`. Sampling rate is derived from the native sampling interval
(in milliseconds), rather than imposing a multichannel recording's rate.
Native MATLAB spike indices are converted to zero-origin indices once by the
reader; class identity and overlapping spikes are not rewritten in the MAT
files. Timing correction, tolerance, and overlap masks belong to evaluation,
not dataset construction. The current diagnostic uses a −23-sample shift;
this does not change source GT.

## MEArec: public D0–D4 method

The current preparation contract is 256 contacts on a 16 × 16 array at 36 µm
pitch, 64 units (51 excitatory / 13 inhibitory), 20 kHz, and 300 s. It includes
only an independent **5 µVrms floor** and total CAL SNR **10/15/20 dB**.

The public `assets/mearec/` bank and original trains are inputs, not freshly
simulated neurons. D0 keeps the reference layout/trains; D1–D4 use seeded
same-model bank selection and fixed-count, refractory-conditioned schedules.
See [the complete method and staged CLI](mearec_data.md).

## Neuropixels: two separate cases

Source: [Kilosort4 Figshare article 25298815](https://figshare.com/articles/dataset/25298815).

| Case | Archive | File ID | Bytes | Computed MD5 expectation |
|---|---|---:|---:|---|
| nodrift | `sim_no_drift.zip` | 44729857 | 43,246,418,580 | `c90bed07267f70c5a3142678e2ed4b4a` |
| meddrift | `sim_med_drift.zip` | 44729869 | 44,873,416,729 | not supplied; no golden MD5 asserted |

```bash
# Metadata only: no download and no output files.
python scripts/download_data.py --case both --metadata-only
# Explicit large download, for users who do not already have the archives.
python scripts/download_data.py --case both --download --output data/neuropixels
# Local metadata inspection only; requires the corresponding archive.
python -m shift_aso.neuropixels.datasets --case meddrift --data-dir data/neuropixels
```

Each archive contains exactly one suffix match for `.ap.cbin`, `.ap.ch`,
`.ap.meta`, and `.ap_params.npz`; nested ZIP prefixes are supported. The
benchmark takes the first 300 s at 30 kHz and requires coverage through that
interval; upstream recordings may be longer. There are 385 recorded columns,
384 AP map entries, and 383 active AP contacts. Native `st`/`cl` labels stay in
sample-index/unit-ID form.

The two cases must not share a scoring-reference map. No-drift uses the frozen
30–60 s label-derived static map. Medium-drift uses a separate GT-aligned,
10 s block template reference, retaining unknown channels as −1. Both are
scorer-only, with label exposure (including TEST) explicitly disclosed.
See [NP preparation and reference policy](neuropixels_data.md).

Readable configuration files under `configs/` are mirrored into installed
package JSON resources. Tests require byte-identical mirrors. Metadata-only
commands do not decode recordings, load NP GT, generate MEArec recordings, or
create result bundles. Heavy operations require separate explicit flags.
