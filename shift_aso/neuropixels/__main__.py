from __future__ import annotations
import argparse
from .core import run

def main():
 p=argparse.ArgumentParser(description="Frozen Neuropixels SI10, NEO10, and LC6 replay")
 p.add_argument("--data-dir",required=True)
 p.add_argument("--output",required=True)
 p.add_argument("--config", default=None, help="optional frozen public configuration JSON")
 p.add_argument("--stop-s", type=float, default=None, help="replay duration in seconds")
 a=p.parse_args(); run(a.data_dir,a.output,a.config,a.stop_s)
if __name__=="__main__": main()
