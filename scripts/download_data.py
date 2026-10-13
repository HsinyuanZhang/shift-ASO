#!/usr/bin/env python3
"""Fetch public Kilosort4 simulation metadata or selected files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen


ARTICLE_URL = "https://api.figshare.com/v2/articles/25298815"
SELECTED = "sim_no_drift.zip"
WAVEFORMS = "all_waveforms.npz"
CHUNK_BYTES = 8 << 20


def article_metadata() -> dict[str, Any]:
    with urlopen(ARTICLE_URL, timeout=60) as response:
        return json.load(response)


def selected_files(article: dict[str, Any], include_waveforms: bool) -> list[dict[str, Any]]:
    wanted = {SELECTED}
    if include_waveforms:
        wanted.add(WAVEFORMS)
    available = {entry["name"]: entry for entry in article.get("files", [])}
    missing = wanted.difference(available)
    if missing:
        raise RuntimeError(f"Figshare article does not list: {', '.join(sorted(missing))}")
    return [available[name] for name in sorted(wanted)]


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry: dict[str, Any], output: Path) -> Path:
    target = output / entry["name"]
    expected = entry.get("supplied_md5") or entry.get("computed_md5")
    if target.is_file() and expected and md5(target) == expected:
        print(f"verified existing {target}")
        return target
    partial = target.with_suffix(target.suffix + ".part")
    output.mkdir(parents=True, exist_ok=True)
    with urlopen(entry["download_url"], timeout=120) as source, partial.open("wb") as sink:
        while True:
            block = source.read(CHUNK_BYTES)
            if not block:
                break
            sink.write(block)
    if expected and md5(partial) != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 mismatch for {entry['name']}")
    partial.replace(target)
    print(f"downloaded {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("downloads"))
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--include-waveforms", action="store_true")
    args = parser.parse_args()
    article = article_metadata()
    selected = selected_files(article, args.include_waveforms)
    metadata = {
        "article": {key: article.get(key) for key in ("id", "title", "doi", "url_public_html")},
        "files": [{key: entry.get(key) for key in ("id", "name", "size", "download_url", "supplied_md5", "computed_md5")} for entry in selected],
    }
    print(json.dumps(metadata, indent=2))
    if not args.metadata_only:
        for entry in selected:
            download(entry, args.output)


if __name__ == "__main__":
    main()
