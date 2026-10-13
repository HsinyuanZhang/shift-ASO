#!/usr/bin/env python3
"""Generate the public MEArec data condition from its scientific assets."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import MEArec as mr
import neo
import numpy as np
import quantities as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "mearec_generation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"checksum mismatch for {path}: {actual}")


def load_public_inputs(config: dict, config_path: Path) -> tuple[Path, dict, dict, list[dict]]:
    asset_dir = (config_path.parent.parent / config["assets"]["directory"]).resolve()
    template_path = asset_dir / config["assets"]["templates_file"]
    trains_path = asset_dir / config["assets"]["spiketrains_file"]
    info_path = asset_dir / config["assets"]["recording_info_file"]
    annotation_path = asset_dir / config["assets"]["annotations_file"]
    for path, name in ((template_path, "templates_sha256"), (trains_path, "spiketrains_sha256"),
                       (info_path, "recording_info_sha256"), (annotation_path, "annotations_sha256")):
        require_file(path, config["assets"][name])
    with np.load(trains_path) as archive:
        trains = {key: archive[key] for key in archive.files}
    info = json.loads(info_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    if len(annotations) != 64 or len([key for key in trains if key.startswith("unit_")]) != 64:
        raise RuntimeError("public spike-train assets do not describe 64 units")
    return template_path, trains, info, annotations


def fixed_trains(trains: dict, annotations: list[dict], duration_s: float) -> list[neo.SpikeTrain]:
    result = []
    for unit in range(64):
        values = np.asarray(trains[f"unit_{unit:02d}_times_s"], dtype=np.float64)
        values = values[values < duration_s]
        train = neo.SpikeTrain(values * pq.s, t_start=0.0 * pq.s, t_stop=duration_s * pq.s)
        train.annotations = copy.deepcopy(annotations[unit])
        result.append(train)
    return result


def build_params(info: dict, duration_s: float) -> dict:
    params = copy.deepcopy(info)
    params["recordings"]["duration"] = duration_s
    params["recordings"]["chunk_duration"] = 5
    params["spiketrains"]["duration"] = duration_s
    params["seeds"] = {key: 20261007 for key in params["seeds"]}
    return params


def source_params() -> dict:
    """The original 60 s construction overrides the MEArec 1.9.3 defaults."""
    params = mr.get_default_recordings_params()
    params["spiketrains"].update(n_exc=51, n_inh=13, f_exc=8, f_inh=15,
                                 st_exc=2, st_inh=3, duration=60.0, t_start=0)
    params["seeds"] = {key: 20261007 for key in params["seeds"]}
    params["templates"].update(min_dist=25, min_amp=50)
    params["recordings"].update(fs=20000, noise_level=10, noise_mode="uncorrelated",
                                 filter=True, filter_cutoff=[300, 6000], filter_order=3,
                                 filter_mode="filtfilt", chunk_duration=5, drifting=False)
    return params


def run(output: Path, template_path: Path, trains: dict, info: dict, annotations: list[dict],
        duration_s: float, workers: int) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="mearec_public_", dir=str(output.parent)))
    try:
        if duration_s == 60.0:
            recgen = mr.gen_recordings(templates=str(template_path), params=source_params(),
                                       verbose=True, n_jobs=workers)
        else:
            tempgen = mr.load_templates(template_path, verbose=False)
            params = build_params(info, duration_s)
            spgen = mr.SpikeTrainGenerator(spiketrains=fixed_trains(trains, annotations, duration_s),
                                           params=params["spiketrains"], verbose=False)
            recgen = mr.gen_recordings(tempgen=tempgen, spgen=spgen, params=params, verbose=True,
                                       tmp_mode="memmap", tmp_folder=str(temporary), n_jobs=workers)
        for train in recgen.spiketrains:
            train.annotations = {key: value.item() if isinstance(value, np.generic) else value
                                 for key, value in train.annotations.items()}
        mr.save_recording_generator(recgen, filename=str(output), include_spike_traces=False, verbose=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--duration-s", type=float, choices=(60.0, 300.0), default=300.0)
    parser.add_argument("--output", type=Path, default=Path("recording_static_64u_300s.h5"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    template_path, trains, info, annotations = load_public_inputs(config, config_path)
    workers = args.workers if args.workers is not None else (4 if args.duration_s == 60.0 else 8)
    report = {"duration_s": args.duration_s, "template_generator": template_path.name, "units": 64,
              "mearec_version": mr.__version__, "fixed_spiketrains": args.duration_s == 300.0,
              "workers": workers}
    if args.dry_run:
        print(json.dumps(report, sort_keys=True))
        return
    run(args.output, template_path, trains, info, annotations, args.duration_s, workers)
    print(json.dumps({**report, "output": str(args.output), "sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
