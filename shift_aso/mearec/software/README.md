# MEArec software: one responsibility per module

Start with [the public workflow](../../../docs/mearec.md) in the source checkout.
The current CLI is `python -m shift_aso.mearec dev|test --help`.

| Module | Responsibility |
|---|---|
| `grid.py` | Five lags, three nonzero EMA coefficients, coarse K and window constants |
| `features.py` | Signed float64 centered NEO/EMA; explicit halo/state; CAL statistic |
| `detect.py` | Spatial NMS, exact ties, post-H polarity, retiming and eligibility |
| `api.py` | Validate explicit configurations; CAL and small-array batch API |
| `stream.py` | Bounded-memory full-origin replay of all frozen methods in one pass |
| `scoring.py` | MEArec matching policy, full GT denominator, original GT row pairs |
| `search.py` | DEV-only coarse/refinement/extension search with stable IDs |
| `records.py` | Caller manifests, source/input hashes and create-once records |
| `cli.py` | All-case DEV freeze barrier, exact DEV replay, then TEST |
| `tests/` | Nonempty synthetic semantic and end-to-end tests |

Plain NEO uses rho 0; EMA uses 0.5, 0.875 or 0.9375. Lags are 1, 2, 4, 6, 8.
Both current families share center-only negative polarity and window-minimum
timing. No neighbor energy is added. Any finite nonnegative K, including
refined factors, can be replayed.

FP and LC6 inputs are supplied by the caller; no encoder/noise generator is
invoked. Mathematical settings are published; selected winners and observed
benchmark evidence are not. Generated choices/events/scores stay in external
local output directories and must never be committed.

The old HDF5 SI/ASO interface is explicitly separate:
`python -m shift_aso.mearec legacy --help`. It is not a current-method alias.
