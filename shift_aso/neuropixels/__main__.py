from __future__ import annotations
import argparse
from .core import run

def main():
 p=argparse.ArgumentParser(description="Frozen Neuropixels LC6 Shift-ASO replay")
 p.add_argument("--data-dir",required=True)
 p.add_argument("--output",required=True)
 a=p.parse_args(); run(a.data_dir,a.output)
if __name__=="__main__": main()
