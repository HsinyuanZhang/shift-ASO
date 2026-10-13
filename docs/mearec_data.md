# MEArec data generation

This package contains the public scientific inputs for the 300 s MEArec condition: the complete static template generator, recording parameters and seeds, 64 unit annotations, and the fixed 0 to 300 s spike trains. The recording is 20 kHz on a 256-channel 16x16 square MEA with 36 um pitch. It has 64 static units and native MEArec uncorrelated 10 uVrms noise.

Install the pinned dependency, verify the assets, and generate a recording:

```bash
python -m pip install -e '.[data]'
python scripts/generate_mearec.py --dry-run
python scripts/generate_mearec.py --duration-s 300 --output data/mearec.h5
```

For the corresponding 60 s construction, use `--duration-s 60`. This uses the original MEArec 1.9.3 default-parameter path plus the published overrides and seed `20261007`. The 300 s command uses the complete template generator, saved recording parameters, and fixed supplied spike trains. It retains the published 0 to 60 s spike trains and uses the supplied 60 to 300 s continuation; it does not repeat the first 60 s train.

`configs/mearec_generation.json` contains asset SHA-256 values and the full scientific recipe. The external independent and common-mode augmentations use a third-order 300 to 6000 Hz Butterworth `sosfiltfilt` filter and separately normalize the two segments to 10 uVrms. Their continuation seed sequences are `[20261013, 300, 1]` and `[20261013, 300, 2]`.

The stored benchmark waveform is not distributed. Public generation produces the same template pool, parameters, seeds, and fixed ground-truth train schedule, but its waveform values can differ from the benchmark. It must not be described as array-identical to the benchmark recording, whose SHA-256 is `646c1c4838133b5ffae8adcf3d67a653b2c9f295feb9652d973f12d174b5584f`.
