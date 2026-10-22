"""CLI for the public MEArec multistage raw-data generator.

Usage::

    python -m shift_aso.mearec.generation --dataset D0|D1|D2|D3|D4 --output PATH
        [--assets-dir PATH] [--stage all|select|train|clean|noise|calibrate]
        [--execute] [--config PATH] [--workers N] [--reserve-bytes N]
        [--scratch-dir PATH]

Without ``--execute`` the command prints a dry-run plan built from config and
asset metadata only: no output creation, no MEArec import, no generation.
With ``--execute`` every stage runs in a fresh subprocess whose thread
environment is fixed before the first NumPy import (spatial stage: BLAS=2
with at most 2 workers = 4 compute threads; everything else 1 thread).

This module imports nothing heavy at import time; the thread environment for
the requested stage is resolved and exported BEFORE NumPy is imported (via
``shift_aso.mearec.generation.core``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _thread_env_for(stage: str) -> dict:
    # Local import of the pure mapping without touching NumPy: the mapping
    # lives in core, so replicate the tiny rule here to keep cli import-light.
    if stage in ("noise", "noise_spatial"):
        return {"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}
    return {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m shift_aso.mearec.generation",
        description="Public multistage MEArec raw-data generator (floor 5 uVrms, "
                    "total CAL SNR 10/15/20 dB, datasets D0-D4 only).")
    parser.add_argument("--dataset", required=True, choices=["D0", "D1", "D2", "D3", "D4"])
    parser.add_argument("--output", required=True,
                        help="output root; the bundle is written to <output>/<dataset>")
    parser.add_argument("--assets-dir", default=None,
                        help="directory containing assets/mearec (templates.h5, "
                             "spiketrains.npz, recording_params.json, "
                             "spiketrain_annotations.json)")
    parser.add_argument("--stage", default="all",
                        choices=["all", "select", "train", "clean", "noise", "calibrate",
                                 "noise_spatial", "noise_unit"],
                        help="noise_spatial/noise_unit are internal child stages of "
                             "the public noise stage")
    parser.add_argument("--execute", action="store_true",
                        help="actually generate; default is a metadata-only dry run")
    parser.add_argument("--config", default=None,
                        help="frozen config path (default: packaged config.json)")
    parser.add_argument("--workers", type=int, default=4,
                        help="pool size (spatial clamped to 2)")
    parser.add_argument("--reserve-bytes", type=int, default=20 * (1 << 30),
                        help="minimum free-space reserve before large writes (at least 20 GiB)")
    parser.add_argument("--scratch-dir", default=None,
                        help="scratch root (default: <output>/.scratch)")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Fix the thread environment for THIS stage before NumPy comes in.
    for key, value in _thread_env_for(args.stage).items():
        os.environ[key] = value
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    from . import core, native  # NumPy imported here, after the env is fixed

    def emit(event: dict) -> None:
        print(json.dumps(event, indent=1, sort_keys=True), flush=True)

    try:
        config = core.load_config(args.config)
        assets_dir = core.resolve_assets_dir(config, args.assets_dir)
        if not args.execute:
            plan = core.plan_dataset(config, args.dataset, assets_dir, args.stage,
                                     args.output)
            emit(plan)
            return 0
        ctx = native.GenerationContext(
            config, Path(args.config) if args.config else Path(__file__).with_name("config.json"),
            args.dataset, assets_dir, args.output, args.workers, args.reserve_bytes,
            args.scratch_dir)
        if args.stage == "all":
            # Fresh subprocess per stage with the correct env before NumPy.
            for stage in core.STAGE_ORDER:
                native.run_child(ctx, stage)
                emit({"event": "stage_done", "stage": stage})
        else:
            native.run_stage(ctx, args.stage)
        emit({"event": "end", "dataset": args.dataset, "stage": args.stage,
              "execute": True, "status": "EXECUTED"})
        return 0
    except core.StopGeneration as stop:
        emit({"event": "stop", "dataset": args.dataset, "stage": args.stage,
              "execute": bool(args.execute), "reason": str(stop)})
        return 2


if __name__ == "__main__":
    sys.exit(main())
