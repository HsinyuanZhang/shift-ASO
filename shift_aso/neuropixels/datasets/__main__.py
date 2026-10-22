"""Command-line interface for the public Neuropixels datasets package.

Default invocation is metadata/validation ONLY: it inspects ZIP member names,
the ``.ap.ch`` header and the ``.ap.meta`` text, validates them against the
parent dataset registry, prints a JSON report and writes nothing. It never
decodes the recording and never loads the GT ``params`` NPZ.

``--build-reference`` authorizes full reference generation for a future user;
it is never run by this package's author and is refused without ``--output``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Parent packages and this package are lazy: these controls precede NumPy.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

from ._archive import ArchiveContractError, inspect_case
from ._build import BuildContractError, reference_build
from ._cbin import CbinContractError
from ._labels import LabelContractError
from ._meta import MetaContractError
from ._registry import RegistryError
from ._template_reference import ReferenceContractError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m shift_aso.neuropixels.datasets",
        description=(
            "Public Neuropixels dataset inspection and scorer-reference preparation "
            "(nodrift / meddrift, first 300 s benchmark)."
        ),
    )
    parser.add_argument("--case", required=True, choices=("nodrift", "meddrift"),
                        help="benchmark case; selects the case-bound scorer reference")
    parser.add_argument("--data-dir", required=True, type=str,
                        help="directory holding the case archive ZIP named in the registry")
    parser.add_argument("--output", type=str, default=None,
                        help="output directory for --build-reference (required with it; "
                             "ignored and unwritten by the default metadata inspection)")
    parser.add_argument("--build-reference", action="store_true",
                        help="explicit opt-in: generate the case scorer reference "
                             "(future user action; heavy: reads and decodes the archive)")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.build_reference:
        try:
            report = inspect_case(args.case, args.data_dir)
        except (RegistryError, ArchiveContractError, CbinContractError, MetaContractError) as exc:
            print(json.dumps({"schema": "shift_aso_np_datasets_inspection_v1",
                              "status": "invalid", "case": args.case,
                              "error": f"{type(exc).__name__}: {exc}"}, indent=2))
            print(f"invalid dataset: {exc}", file=sys.stderr)
            return 1
        if args.output:
            report["output_ignored"] = (
                "metadata inspection writes nothing; pass --build-reference to generate outputs"
            )
        print(json.dumps(report, indent=2))
        return 0
    if not args.output:
        parser.error("--build-reference requires --output")
    try:
        receipt = reference_build(args.case, args.data_dir, args.output)
    except (RegistryError, ArchiveContractError, CbinContractError, MetaContractError,
            BuildContractError, LabelContractError, ReferenceContractError) as exc:
        print(f"reference build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": receipt["status"], "case": receipt["case"],
                      "receipt": "reference_receipt.json",
                      "outputs": sorted(receipt["outputs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
