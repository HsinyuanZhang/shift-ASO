#!/usr/bin/env python3
"""Inspect public Kilosort4 file metadata; download only with --download."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from urllib.request import urlopen

ARTICLE_URL = "https://api.figshare.com/v2/articles/25298815"
CASES = {"nodrift": "sim_no_drift.zip", "meddrift": "sim_med_drift.zip"}
CHUNK_BYTES = 8 << 20


def article_metadata():
    with urlopen(ARTICLE_URL, timeout=60) as response:
        return json.load(response)


def selected_files(article, include_waveforms=False, case="nodrift"):
    registry = json.loads((Path(__file__).resolve().parents[1] / "configs/datasets.json").read_text())
    names = list(CASES) if case == "both" else [case]
    available = {entry["name"]: entry for entry in article.get("files", [])}
    selected = []
    for key in names:
        expected = registry["neuropixels"]["cases"][key]
        name = CASES[key]
        if name not in available:
            raise RuntimeError(f"Figshare article does not list {name}")
        entry = dict(available[name])
        if entry["id"] != expected["file_id"] or entry["size"] != expected["bytes"]:
            raise RuntimeError(f"Figshare identity changed for {name}; review before downloading")
        golden = expected.get("computed_md5")
        observed = entry.get("computed_md5") or None
        if golden and observed != golden:
            raise RuntimeError(f"Figshare computed MD5 changed for {name}")
        entry["expected_md5"] = golden or observed
        selected.append(entry)
    if include_waveforms:
        if "all_waveforms.npz" not in available:
            raise RuntimeError("Figshare article does not list all_waveforms.npz")
        entry = dict(available["all_waveforms.npz"])
        entry["expected_md5"] = entry.get("computed_md5") or None
        selected.append(entry)
    return selected


def md5(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = entry["name"]
    if Path(name).name != name:
        raise RuntimeError("download name must be a filename")
    target = output / name
    partial = output / (name + ".part")
    expected = entry.get("expected_md5") or entry.get("computed_md5") or None
    if target.exists():
        if (target.is_file() and target.stat().st_size == entry["size"]
                and expected and md5(target) == expected):
            print(f"verified existing {target}")
            return target
        raise RuntimeError(f"preserving unverified existing file: {target}")
    if partial.exists():
        raise RuntimeError(f"preserving previous partial: {partial}; no automatic resume")
    if shutil.disk_usage(output).free < int(entry["size"]) + 20 * (1 << 30):
        raise RuntimeError("insufficient disk space: file size plus 20 GiB reserve required")
    digest = hashlib.md5()
    # Exclusive create: failures retain the partial; never silently retry.
    with urlopen(entry["download_url"], timeout=120) as source, partial.open("xb") as sink:
        for block in iter(lambda: source.read(CHUNK_BYTES), b""):
            sink.write(block)
            digest.update(block)
    actual = digest.hexdigest()
    if partial.stat().st_size != entry["size"] or (expected and actual != expected):
        raise RuntimeError(f"size/checksum mismatch for {name}; partial preserved")
    # Link is create-once even if another process creates the target meanwhile.
    os.link(partial, target)
    partial.unlink()
    print(json.dumps({"file": str(target), "md5": actual,
                      "md5_status": "verified" if expected else "observed_only_no_golden_checksum"}))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("downloads"))
    parser.add_argument("--case", choices=("nodrift", "meddrift", "both"), default="nodrift")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--metadata-only", action="store_true", help="default: no download or writes")
    mode.add_argument("--download", action="store_true", help="explicit opt-in to large downloads")
    parser.add_argument("--include-waveforms", action="store_true")
    args = parser.parse_args()
    article = article_metadata()
    selected = selected_files(article, args.include_waveforms, args.case)
    print(json.dumps({"article": {key: article.get(key) for key in
                                 ("id", "title", "doi", "url_public_html")},
                      "files": selected, "download_requested": args.download}, indent=2))
    if args.download:
        for entry in selected:
            download(entry, args.output)


if __name__ == "__main__":
    main()
