# shift-ASO

Dataset resources for Quiroga, MEArec and Kilosort4 simulations, plus portable
MEArec plain-NEO/EMA software methods. No benchmark scores or experiment
records are stored in this repository.

## Datasets

| Dataset | Recording format | Sampling rate | Preparation scope |
|---|---|---|---|
| Quiroga | 16 original single-channel `C_*.mat` files | Read from each file's `samplingInterval` | Preserve native signals, spike times, and classes |
| MEArec | 256-channel, 64-unit simulations | 20 kHz | D0–D4, 300 s each; independent 5 µVrms noise floor; total CAL SNR 10/15/20 dB |
| Kilosort4 simulations | `sim_no_drift.zip` and `sim_med_drift.zip`; 383 active AP channels | 30 kHz | Native compressed recording and labels; first 300 s |

## Installation

```bash
python -m pip install -e '.[data]'
```

The large Neuropixels archives are downloaded separately. MEArec template and
spike-train inputs are provided under `assets/mearec/` in the source checkout;
they are not included in the Python wheel.

## Quiroga recordings

Obtain the original Quiroga simulated recordings [1] from the
[Wave_clus project](https://github.com/csn-le/wave_clus), and place the sixteen
`C_*.mat` files in `data/quiroga/`.

The native fields are `data`, `spike_times`, `spike_class`, and
`samplingInterval` (milliseconds). Derive the sampling rate from the file,
retain spike classes and overlapping spikes, and convert MATLAB's one-based
spike indices to zero-origin indices only once when reading. Do not rewrite
the source MAT files. See [data formats](docs/data.md).

## MEArec dataset construction

The simulations use [MEArec](https://github.com/SpikeInterface/MEArec) [2].
The electrode array is a static 16 × 16 grid at 36 µm pitch, with 64 units
(51 excitatory and 13 inhibitory) and 176,248 spikes per 300 s realization.

### Realizations D0–D4

- **D0** retains the supplied reference template instances and spike trains.
- **D1–D4** retain the same per-unit cell-model assignments and spike counts,
  but select different instances from the native template bank. Selection
  requires native amplitude **50 < A < 300 µV** and pairwise soma distance
  **≥25 µm**. Position, orientation, and template waveform change together.
- New spike times are drawn with fixed per-unit counts and a **40-sample
  minimum ISI**. These are count-conditioned schedules, not unconditional
  Poisson draws.

The realizations share a finite template bank, cell-model assignments, spike
counts, and convolution control seed; they are not independent biological
populations. Seeds, asset hashes, and construction settings are recorded in
[`configs/mearec_multiseed.json`](configs/mearec_multiseed.json).

### Clean signal and noise

Clean MEArec convolution is generated with native noise disabled, followed by
one order-3, 300–6000 Hz zero-phase chunk-filter pass. Spatial noise uses
covariance diagonal 1 and off-diagonal `15 µm / distance`; the independent
component is normalized by each channel's standard deviation over **CAL
0–6 s**, with the same normalization held for the entire recording.

The independent noise floor is **5 µVrms**. For each target **10/15/20 dB**,
the spatial scale is solved using actual CAL component variances and their
covariance. Total CAL SNR is defined as
`20*log10(median(unit amplitude / CAL noise standard deviation))`, with the
median taken over linear ratios. The exact float32 mixture is:

```python
(clean + spatial * np.float32(scale)) + independent_unit * np.float32(5.0)
```

### Preparation commands

```bash
# Metadata-only plan: no output creation or native generation.
python -m shift_aso.mearec.generation --dataset D0 --output data/mearec_multiseed

# Explicit full generation; repeat with D1, D2, D3, or D4 as needed.
python -m shift_aso.mearec.generation --dataset D0 --output data/mearec_multiseed \
  --assets-dir assets/mearec --execute
```

The bundle contains selection and spike-train records, clean signal, spatial
and independent noise components, GT/geometry metadata, and a generation
manifest with hashes and calibrated scales. SNR mixtures are read virtually
rather than saved as duplicate full recordings. See
[the full generation procedure](docs/mearec_data.md) for stage controls,
storage requirements, seed lineages, and artifact formats.

## Kilosort4 Neuropixels datasets

Use the public simulations associated with **Kilosort4** [3], available from
[Figshare article 25298815](https://figshare.com/articles/dataset/25298815).

| Case | Archive | Figshare file ID | Bytes |
|---|---|---:|---:|
| No drift | `sim_no_drift.zip` | 44729857 | 43,246,418,580 |
| Medium drift | `sim_med_drift.zip` | 44729869 | 44,873,416,729 |

The pinned no-drift computed MD5 is
`c90bed07267f70c5a3142678e2ed4b4a`. No golden computed MD5 is asserted for
medium-drift; its observed checksum is recorded during explicit preparation.

```bash
# Inspect upstream metadata only; no download or file creation.
python scripts/download_data.py --case both --metadata-only

# Explicit large download, if the archives are not already available.
python scripts/download_data.py --case both --download --output data/neuropixels

# Inspect a local archive without decoding the recording or loading GT.
python -m shift_aso.neuropixels.datasets --case nodrift --data-dir data/neuropixels
python -m shift_aso.neuropixels.datasets --case meddrift --data-dir data/neuropixels
```

Each archive must have exactly one matching `.ap.cbin`, `.ap.ch`, `.ap.meta`,
and `.ap_params.npz` member; nested ZIP prefixes are supported. The native
recording has 385 columns, including 384 AP channels, of which 383 are active.
The first 300 s are used here; an upstream recording may be longer. Native
`st` sample indices and `cl` unit IDs are preserved.

### Case-specific reference labels

The two cases share an input format, **not a waveform or channel-reference
map**. No-drift uses the supplied static unit-channel map estimated from
30–60 s labeled waveforms. Medium-drift uses its own GT-aligned template
reference in 10 s blocks: 13 nearby channels, 121-sample windows, 16 uniform
snippet ranks per time parity, and at least four combined snippets. Invalid
blocks use the nearest valid block for that unit; unavailable channels remain
**−1**, not channel zero.

These channel references use labeled data, including times in the later part
of the recording, and are annotation estimates rather than native simulator
channel identities. Prepared labels preserve unit IDs and explicit group flags
for units 0–599 and 600–1199. Reference preparation is opt-in:

```bash
python -m shift_aso.neuropixels.datasets --case nodrift --data-dir data/neuropixels \
  --output data/np_reference_nodrift --build-reference
python -m shift_aso.neuropixels.datasets --case meddrift --data-dir data/neuropixels \
  --output data/np_reference_meddrift --build-reference
```

See [NP preparation](docs/neuropixels_data.md) for decoding, reference-label
provenance, memory requirements, and output manifests.

## Reproducibility notes

The portable preparation code has unit and synthetic-fixture tests. Full
native MEArec generation and real Neuropixels reference construction have
**not been rerun for this portable release**. Synthetic tests do not establish
production waveform identity. Source/input hashes, seeds, versions, and
construction settings are recorded during explicit execution; CPU/BLAS
variation can change noise realizations. Existing unknown files and failed
partials are preserved rather than silently overwritten or resumed.

## MEArec software methods

The current plain NEO and post-NEO EMA methods are separate from data
construction. Both accept caller-supplied FP or frozen LC6 arrays, use the
same spatial peak-exclusion and matching rules, and keep sample-zero state
across chunks and splits.

```bash
# Select on DEV for every manifest case, then freeze all choices.
python -m shift_aso.mearec dev --manifest inputs.json --output ../runs/dev
# Exact DEV replay, then frozen TEST; no reselection.
python -m shift_aso.mearec test --dev-run ../runs/dev --output ../runs/test
```

See [input formats, method definitions and module layout](docs/mearec.md).
The small search uses lags 1/2/4/6/8 and three nonzero EMA coefficients.
Generated local choices, events and scores must not be committed. The older
HDF5 methods are explicitly accessed with `python -m shift_aso.mearec legacy`;
they are not silently substituted for the current implementation.

## References

1. Quiroga, R. Q., Nadasdy, Z., and Ben-Shaul, Y. (2004).
   **Unsupervised Spike Detection and Sorting with Wavelets and Superparamagnetic Clustering.**
   *Neural Computation*, 16, 1661–1687.
   [doi:10.1162/089976604774201631](https://doi.org/10.1162/089976604774201631).
2. Buccino, A. P., and Einevoll, G. T. (2021; first published online 2020).
   **MEArec: A Fast and Customizable Testbench Simulator for Ground-truth Extracellular Spiking Activity.**
   *Neuroinformatics*, 19, 185–204.
   [doi:10.1007/s12021-020-09467-7](https://doi.org/10.1007/s12021-020-09467-7).
3. Pachitariu, M., Sridhar, S., Pennington, J., and Stringer, C. (2024).
   **Spike sorting with Kilosort4.** *Nature Methods*, 21, 914–921.
   [doi:10.1038/s41592-024-02232-7](https://doi.org/10.1038/s41592-024-02232-7).
