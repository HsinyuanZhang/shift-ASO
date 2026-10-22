# Neuropixels: case-bound ingestion and scoring references

## Cases and safe defaults

`configs/datasets.json` (installed mirror: `shift_aso/datasets_registry.json`)
records the two official Figshare archives, byte sizes and reference types.
See [source identities](data.md). Medium-drift has no supplied golden computed
MD5 in the pinned metadata. Its observed build-time MD5 is recorded with
`md5_match: null`, not reported as a verified golden match.

```bash
python scripts/download_data.py --case both --metadata-only
python -m shift_aso.neuropixels.datasets --case nodrift --data-dir data/neuropixels
python -m shift_aso.neuropixels.datasets --case meddrift --data-dir data/neuropixels
```

Default inspection reads only the ZIP directory and `.ap.ch` / `.ap.meta`
members. It does not decode CBIN, read GT NPZ arrays, hash the entire archive,
or create outputs (even if `--output` is passed). Required suffixes are unique
and tolerate nested directory prefixes. Archive filename/size, compressed
member size vs chunk offsets, 385 recorded channels, 383 active AP channels,
30 kHz and coverage of the first 300 s are checked. Size/metadata inspection
is not cryptographic proof of archive identity.

Explicit downloads require `--download`; existing unknown files and failed
`.part` files are preserved, not overwritten or resumed. A download checks
size and an available computed MD5, and requires file size +20 GiB disk space.

## Native format and labels

The native CBIN chain is zlib → int16 time differences → Fortran-order
reshape → int32 time cumulative sum → int16. The first 384 columns are AP;
active-channel masking uses `~snsShankMap`. Geometry is
`x=column*32+(27 if even row else 11)` µm, `y=row*20` µm. The reader keeps
native chunk boundaries, which matter to the dynamic reference's window/rank
semantics. Upstream recordings may exceed 300 s; only the first 9,000,000
samples form this benchmark interval.

Source `st` is the native zero-origin integer sample index at 30 kHz, and
`cl` is the simulator unit ID. The source is not resampled, relabeled, or
shifted. Prepared labels contain:

- `sample_index`, `unit_id`, `estimated_local_channel`;
- `is_good` (unit 0–599), `is_background` (600–1199), `scoring_group`;
- `geometry_um`, `local_to_original_AP`.

Ingestion retains all source labels in the interval. Which groups count as
TP/FN or background-ignore events is a **separate evaluator policy**. The
legacy `canonical_score` counts all 1200 units and has not been upgraded to
good-600/background-ignore scoring by this preparation package. Unknown
reference channel **−1 stays unknown**: no channel-zero replacement. A
compatible evaluator must exclude unknowns from its spatial index while
retaining applicable GT in the FN denominator; the legacy scorer must not be
assumed to do this safely.

## No-drift reference

Use only the packaged `_unit_to_main_channel.json` map (1200 entries) for
`nodrift`. It was estimated from labeled 30–60 s waveforms, including some of
the benchmark TEST interval. The map is a scoring reference, not a simulator
channel and not detector calibration. Its provenance/hash is included in the
explicit reference-build receipt. The medium-drift builder is refused for
this case.

## Medium-drift reference

Use only its own **GT-label-driven 10 s block template reference**, ported
from accepted `np_drift_template_reference_v1.py`, source SHA-256
`d312d459616ddba090cec6a098f9b47efe169086335b645a9db6a5ed7bffccee`.
The static no-drift map is refused for medium-drift.

Locked production method:

1. Thirty 10 s blocks ×1200 units. A first pass computes rough GT-aligned raw
   means and picks their negative-trough channels. Retain 13 nearest active
   channels by geometry around each rough channel.
2. Windows are **[−50,+70] inclusive** (121 samples). Select 16 uniformly
   spaced occurrence ranks per time parity, preserving source rank order.
3. Compute a pointwise median of combined snippets and choose its most
   negative trough channel. At least four combined snippets are required;
   odd/even diagnostic templates need at least two each.
4. Invalid cells use the nearest valid 10 s block for the same unit (ties go
   to the lower block index). A unit with no valid block keeps channel −1
   and fallback distance −1. Validity/fallback/unknown diagnostics are saved.

The accepted source computes same-unit overlap diagnostics but **does not
exclude overlapping snippets**. A window must fit wholly in a native CBIN
chunk; boundary visits consume ranks, including repeated visits across chunk
boundaries. These semantics are intentionally preserved, not silently
“corrected.” The params NPZ `drift` field indexes a waveform bank on a 2 s
grid; it is not confirmed physical µm displacement and is not projected into
channels by this method.

All GT times/IDs, **including TEST**, participate only in the scorer reference.
No detector CAL, threshold, configuration or feature is fitted from this
reference. It is not a fresh-blind evaluation claim.

## Explicit reference preparation (heavy; not run on real data for this release)

```bash
python -m shift_aso.neuropixels.datasets --case nodrift --data-dir data/neuropixels \
  --output data/np_reference_nodrift --build-reference
python -m shift_aso.neuropixels.datasets --case meddrift --data-dir data/neuropixels \
  --output data/np_reference_meddrift --build-reference
```

No-drift loads GT and the frozen map without recording decode. Medium-drift
uses two sequential decode passes through the first 300 s, with several GiB
of memory for the snippet bank. CLI library threads are fixed to one before
NumPy import. Existing output directories are refused. Failures preserve
partials and a failure report; no retry or alternate reference is selected.

Outputs include `labels.npz`, `labels_metadata.json`, and a commit-last
`reference_receipt.json`; medium-drift also writes
`template_dynamic_reference.npz` and its diagnostic JSON. Receipts bind output,
source-module, registry and applicable static-map hashes, case/member identity,
observed MD5, versions, and label exposure.

These paths are source-backed and synthetically tested. **Real-archive
reference construction has not been executed for the portable release.** No
new detector scores or medium-drift performance results accompany this update.
The legacy detector command remains no-drift only; do not feed medium-drift
archives/references into its external-label override and assume correctness.
