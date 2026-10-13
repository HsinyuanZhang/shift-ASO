# Data

## Quiroga

Place the sixteen native `C_*.mat` recordings in one directory. Required fields
are `data`, `spike_times`, `spike_class`, and `samplingInterval`. The upstream
project is [Wave_clus](https://github.com/csn-le/wave_clus). Obtain the original
Quiroga simulated recordings and supply their directory to the command.

## MEArec

The benchmark uses a static 16 × 16 array at 36 µm pitch, 64 units, 20 kHz
sampling, and 300 s duration. Template assets and spike trains are under
`assets/mearec/`. Install the generation dependencies and follow
[the MEArec preparation instructions](mearec_data.md).

The base recording includes native noise. Two further conditions add either
independent per-channel Gaussian noise or a shared Gaussian trace, band-limited
to 300–6000 Hz and normalized to 10 µV RMS.

## Neuropixels

Download `sim_no_drift.zip` from the
[Kilosort4 Figshare dataset](https://figshare.com/articles/dataset/25298815).
File `44729857` is 43,246,418,580 bytes with MD5
`c90bed07267f70c5a3142678e2ed4b4a`.

```bash
python scripts/download_data.py --metadata-only
python scripts/download_data.py --output data/neuropixels
```

The ZIP must contain `sim.imec0.ap.cbin`, `.ch`, `.meta`, and
`sim.imec0.ap_params.npz`. The replay reads the compressed recording directly
from the ZIP and obtains spike times and unit IDs from the parameters.

There are 383 active channels at 30 kHz. Scoring uses the bundled unit-to-main-
channel map, estimated from 30–60 s labeled waveforms. This map is used only for
scoring, not detector calibration. See [the Neuropixels protocol](neuropixels.md).
