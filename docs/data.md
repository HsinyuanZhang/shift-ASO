# Data

## Quiroga

Place the sixteen native `C_*.mat` recordings directly in one directory and pass
that directory to `shift_aso.quiroga`. The required fields are `data`,
`spike_times`, `spike_class`, and `samplingInterval`. Wave_clus is the public
upstream project: <https://github.com/csn-le/wave_clus>. It does not distribute a
versioned archive of these sixteen MAT files, so obtain them from the original
Quiroga source or an archive and record the source URL and hashes with the run.

## Neuropixels

Download `sim_no_drift.zip` from the Kilosort4 Figshare article:
<https://api.figshare.com/v2/articles/25298815>. It is Figshare file `44729857`,
43,246,418,580 bytes, MD5 `c90bed07267f70c5a3142678e2ed4b4a`:
<https://ndownloader.figshare.com/files/44729857>.

```bash
python scripts/download_data.py --metadata-only
python scripts/download_data.py --output downloads
```

Place the ZIP in the Neuropixels data directory. It must contain
`sim.imec0.ap.cbin`, `.ch`, `.meta`, and `sim.imec0.ap_params.npz`. The package
reads spike times and unit IDs from the parameters. Its scoring channels are
frozen estimates in the repository configuration, derived from 30--60 s labeled
waveforms solely for scoring and not detector calibration.
