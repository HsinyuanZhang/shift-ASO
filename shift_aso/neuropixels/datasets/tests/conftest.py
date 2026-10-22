"""Test configuration: one thread, own basetemp/cache, no real NP data anywhere.

Thread environment variables are set before NumPy is imported by any test
module (conftest is imported first). The suite pins itself to a single worker
and never reads, downloads, or generates real Neuropixels recordings, GT, or
results; every archive is synthetic and tiny.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_var] = "1"

import sys
from pathlib import Path

# Run against this worktree without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
