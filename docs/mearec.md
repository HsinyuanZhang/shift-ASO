# Legacy MEArec HDF5 software interface

This initial clean snapshot retains the legacy fixed SI10, NEO10/EMA and
LC6 Shift-ASO implementation. Method parameters are in configs/mearec.json.
It is separate from dataset generation and stores no observed benchmark
outcomes. The current-method integration is applied in the following commit.

```bash
python -m shift_aso.mearec --recording input.h5 --output ../runs/legacy
```

Use the caller-supplied HDF5 format described by the source reader. CAL and DEV
are separate from TEST; matching is one-to-one within the specified temporal
and spatial tolerances. Generated events and scores belong in external local
output directories, never in the public repository.
