from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description="Fixed public MEArec software replay")
    parser.add_argument("--recording", required=True); parser.add_argument("--case", choices=("all", "base", "independent_gaussian_10uVrms", "common_mode_gaussian_10uVrms"), default="all")
    parser.add_argument("--stop-s", type=int, choices=(60, 300), default=300); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--config", type=Path)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    cases = ("base", "independent_gaussian_10uVrms", "common_mode_gaussian_10uVrms") if args.case == "all" else (args.case,)
    report = {}
    for case in cases:
        outcome = evaluate(args.recording, case, stop_s=args.stop_s, config_path=str(args.config) if args.config else None); report[case] = {}
        for name, value in outcome.items():
            np.savez_compressed(args.output / f"{case}_{name}_events.npz", events=value["events"])
            report[case][name] = value["metrics"]
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
if __name__ == "__main__": main()
