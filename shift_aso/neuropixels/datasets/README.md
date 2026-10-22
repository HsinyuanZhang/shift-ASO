# Neuropixels public datasets package (nodrift / meddrift)

Data preparation for the public Shift-ASO Neuropixels benchmark cases
(`sim_no_drift.zip`, `sim_med_drift.zip`; first 300 s at 30 kHz, 383 active AP
channels, 384 AP entries, 385 recorded channels). This package owns **only**
dataset ingestion and scorer-reference construction. The legacy detector
(`shift_aso.neuropixels.core`, `_kernels`, `_matching`, `config.json`) is
frozen and untouched, and the legacy replay results are not re-labelled.

## CLI

```bash
python -m shift_aso.neuropixels.datasets --case nodrift --data-dir PATH
python -m shift_aso.neuropixels.datasets --case meddrift --data-dir PATH
python -m shift_aso.neuropixels.datasets --case meddrift --data-dir PATH \
    --output PATH --build-reference   # explicit opt-in, user-run only
```

* Default (no `--build-reference`): **metadata inspection only**. Validates
  the case archive against the frozen registry (`shift_aso/datasets_registry.json`,
  read-only): archive filename/size (case identity), suffix-based nested ZIP
  member discovery, the `.ap.ch` header (zlib, time-differenced, 385 recorded
  channels, monotone bounds/offsets, entry size vs offsets), the `.ap.meta`
  shank map (384 AP entries, 383 active, 30 kHz), and first-300 s coverage
  (upstream recordings may be **longer**; coverage of the first 9,000,000
  samples is required, not exact equality). Prints a JSON report, writes
  nothing, never decodes the recording, never reads the GT `params` NPZ, and
  never computes the 43 GB archive MD5 (that is done only by an explicit
  build).
* `--build-reference` (requires `--output`): full case-bound scorer-reference
  generation. This flag authorizes a **future user** run; this package's
  author never executed it against real archives. Outputs are created once
  (`mkdir(exist_ok=False)`), the receipt is committed **last**, and a failed
  run keeps its partials plus `_build_failure.json`.

## Case identities and reference semantics (accepted; do not change)

| case | archive | scorer reference |
|---|---|---|
| `nodrift` | `sim_no_drift.zip` | packaged frozen `_unit_to_main_channel.json` map (scorer-only; original 30–60 s extraction support). No recording decode. |
| `meddrift` | `sim_med_drift.zip` | its OWN dynamic GT template reference (`gt_template_10s_blocks`) — port of the accepted `np_drift_template_reference_v1.py` (SHA-256 `d312d459616ddba090cec6a098f9b47efe169086335b645a9db6a5ed7bffccee`). |

The static nodrift map is **refused** for meddrift and the dynamic template
builder is **refused** for nodrift (`static_map_channels` /
`template_reference_allowed`). The older `np_drift_reference_v1.py`
trajectory projection is neither ported nor called. The registry's meddrift
reference parameters (13 local channels, 16 snippets per parity, minimum 4
combined, window [-50, 70]) are cross-checked against the locked production
constants; a mismatch is a reported blocker, not an adaptation.

### meddrift template builder (`_template_reference.py`)

Verbatim port of the accepted source: 10 s blocks x 1200 units; rough
per-(block, unit) channel from argmin of mean GT-aligned raw samples
(negative trough); 13 nearest local channels; windows [-50, +70] (121
samples); 16 uniformly spaced ranks per time parity (`rank & 1` parity,
`rank // 2` within-parity rank); pointwise median over combined snippets,
most negative sample per local channel, argmin channel; validity requires
>= 4 combined snippets; odd/even parity templates need >= 2 each
(diagnostic only); nearest-valid-10s-block fallback with ties resolved to
the lower block index; unknown cells stay `-1` with fallback distance `-1`;
the same-unit overlap statistic is computed but **never** applied as an
exclusion; the `drift` params field is a 2 s discrete waveform-bank index
and is never read or projected to micrometers.

Chunk-boundary semantics are preserved exactly: a snippet is stored only
when its whole window fits inside one yielded chunk, and a boundary spike
consumes one rank increment **per visiting chunk** even though nothing is
stored (the accepted source depends on this; it is not "fixed").

All GT times/IDs — including TEST — feed **scorer-only** references. They
are never detector CAL/DEV inputs or settings, and the builder accepts no
detector configuration of any kind.

### Native decode (`_cbin.py`)

Identical decode semantics to the accepted runtime and the legacy public
reader — zlib int16 blobs, per-chunk time differencing undone by Fortran
reshape + int32 cumulative sum cast to int16, first 384 columns AP, masked
to the active channels, yielded as float32 — but the ZIP member name is
discovered by suffix instead of the hardcoded `sim.imec0.ap.cbin`. The
stream yields complete native `chunk_bounds` chunks truncated at the stop
sample (the accepted reference runtime's granularity), is reiterable (one
fresh ZIP handle per pass; two passes per build), and closes its handles
per pass. The legacy detector keeps its own 1 s re-blocked reader.

### Labels (`_labels.py`)

Native `st`/`cl` are loaded only during explicit reference preparation:
sorted, non-negative, unit ids within [0, 1200), restricted to the first
300 s. Labels carry `sample_index`, `unit_id`, `estimated_local_channel`,
`is_good` (units 0–599), `is_background` (600–1199), `scoring_group` (0/1),
`geometry_um`, and `local_to_original_AP`; group bounds come from the
registry and are echoed in `labels_metadata.json` with the scorer-only
exposure statement.

### Build outputs and receipt (`_build.py`)

meddrift: `template_dynamic_reference.npz` (exact accepted field set),
`labels.npz`, `template_dynamic_reference_diagnostic.json`,
`labels_metadata.json`. nodrift: `labels.npz`, `labels_metadata.json`.
Both: `reference_receipt.json` committed last with applicable accepted-source
name/hash and production constants (meddrift only), registry SHA-256, applicable
static-map SHA-256 (nodrift only), archive identity (size, observed MD5,
registry MD5 and match), discovered members, source-module/output SHA-256,
environment (python/numpy/platform/thread env/UTC), and diagnostics. With no
golden MD5, match is null and status is observed-only, never verified. The nodrift build
fails if the archive MD5 disagrees with the registry checksum (case
identity); meddrift records its computed MD5 (the registry lists none).

## Limitations and provenance

* The dynamic-template build allocates the accepted source's snippet bank
  (30 x 1200 x 2 x 16 x 121 x 13 int16, ~3.6 GiB virtual; sparse pages stay
  uncommitted) plus ~110 MiB of per-block means; expect a few GiB of
  resident memory and two full sequential decode passes over the first
  300 s per build.
* Archive MD5 verification reads the full archive; it is performed only by
  the explicit build (minutes for the ~43 GiB cases), never by inspection.
* The builder was validated on tiny synthetic streams and a synthetic
  full-coverage (9,000,000-sample) archive with locked production
  constants. No real Neuropixels recording, GT, events, pairs, or results
  were read, downloaded, generated, or relabelled by the author, and the
  production build was never executed here.
* `ZipCbinStream` chunk granularity (native `chunk_bounds` segments,
  truncated at stop) was source-checked against the accepted runtime. Changing
  chunk emission can change boundary/rank selection and therefore requires
  a separately versioned reference, not an unnoticed reader refactor.
* Group labels do not upgrade the legacy all-1200-unit scorer to a
  good-600/background-ignore evaluator. Unknown channels remain -1; compatible
  scoring must keep applicable unknown GT in the FN denominator without
  indexing geometry at -1.
