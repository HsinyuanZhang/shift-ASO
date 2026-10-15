# Neuropixels public replay

The public replay contains three fixed methods: `SI10_MAD`, `NEO10_EMA`, and
`LC6_ShiftASO`. Their complete parameters are in
`shift_aso/neuropixels/config.json`; `configs/neuropixels.json` is the same
human-readable copy. Use the package file by default, or provide `--config`.

Download the Kilosort4 `sim_no_drift.zip` archive from Figshare article
25298815, file 44729857. Its published size is 43,246,418,580 bytes and MD5 is
`c90bed07267f70c5a3142678e2ed4b4a`.

```bash
python -m shift_aso.neuropixels --data-dir /path/to/data --output replay
```

The stream reads the compressed CBIN member directly from the ZIP. CAL is 0--6 s,
DEV is 6--20 s, and TEST is 20--300 s. The reported intervals are 0--20,
20--60, 60--300, and 20--300 s. Detection neighbours use 40 um; scoring uses
the frozen estimated unit-to-main-channel map, 54 um, plus/minus 1 ms (30
samples), and a 60-sample right guard. The map is an estimated scoring map only.

Spike times are `st` in `sim.imec0.ap_params.npz`. The paired channel is the
estimated main channel of unit `cl`, not a simulator electrode. One label
credits one detection. A second detection of the same spike inside that window
is a false positive. The label contract, the original functions, and the
no-drift DEV split of unmatched detections are in
[Neuropixels labels](neuropixels_labels.md).

`LC6_ShiftASO` uses one uniform symmetric six-bit range, full scale
2827.6876831054688, rho 0.875, K=12, J=8, and the step map
1/2/2/4/4/8/8/8 only in feature formation. ADC state updates remain limited by
J=8. Its neighbour path is enabled.
