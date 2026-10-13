# MEArec software replay

This replay evaluates three fixed software methods on a 20 kHz, 256-channel MEArec recording. Calibration is 0--6 s, development is 6--20 s, and reported scores use 20--300 s. Ground truth matching is one-to-one with a 1 ms time tolerance and a 54 um channel radius.

The methods are SI10+MAD, NEO10+EMA, and LC6 Shift-ASO with neighbor assistance enabled. Their coefficients and per-condition settings are in `configs/mearec.json`. The result table contains these methods in `results/mearec_comparison.csv`.

## Input and run

Obtain or create a compatible MEArec HDF5 file as described in `docs/mearec_data.md`. It must contain `recordings`, `channel_positions`, `original_templates`, and per-unit `spiketrains/<unit>/times` datasets. The supplied spike-train schedule retains the original 0--60 s times and adds a 60--300 s continuation. The waveform is regenerated.

Run a fixed replay:

```bash
python -m shift_aso.mearec --recording recording_static_64u_300s.h5 --case base --stop-s 300 --output replay_base
```

`--stop-s 60` is useful for a short replay. The three noise conditions are deterministic: `base`, `independent_gaussian_10uVrms`, and `common_mode_gaussian_10uVrms`. The latter two use band-limited 10 uVrms Gaussian augmentation, with separate 0--60 s and 60--300 s draws.

The code is a software reference. The comparison table describes the original stored recording. Public generation uses the supplied templates and trains, but regenerated waveforms and scores can differ; see the data preparation instructions.
