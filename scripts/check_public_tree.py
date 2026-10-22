"""Fail closed when the TRACKED release tree contains private evidence.

This checks the current index/tree, not historical commits. History removal is
an independent operation and is never performed by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

FORBIDDEN_DIRS = {'results','result','experiment_records','experiments','paper_notes',
                  'output','outputs','tracking','.planning'}
FORBIDDEN_SUFFIXES = {'.csv','.jsonl','.log','.vcd','.fsdb','.saif','.pdf','.png','.jpg',
                      '.tar','.gz','.zip','.npy','.npz','.h5'}
PUBLIC_INPUT_ASSETS = {'assets/mearec/templates.h5','assets/mearec/spiketrains.npz'}
PRIVATE_TEXT = [re.compile(r'/(?:data/\w*work|home/\w+)/(?:iscas_ws|Work_host|si10mad_fp)(?:/|\b)'),
                re.compile(r'gdrive:\w+'),re.compile(r'\bolp_[A-Za-z0-9]{20,}\b'),
                re.compile(r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}\b')]
RESULT_KEYS = {'TP','FP','FN','A','accuracy','precision','recall','F1','FP_per_s',
               'metrics','worst_A','pooled','macro','test_accuracy'}


def json_has_results(value):
    if isinstance(value,dict):
        return any(k in RESULT_KEYS or json_has_results(v) for k,v in value.items())
    return isinstance(value,list) and any(json_has_results(v) for v in value)


def audit(root, paths):
    failures=[]
    for name in paths:
        path=Path(name)
        if name in PUBLIC_INPUT_ASSETS:
            continue
        if FORBIDDEN_DIRS.intersection(path.parts) or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append((name,'forbidden evidence/artifact path'));continue
        local=root/path
        if local.is_symlink():
            failures.append((name,'symlink cannot be audited as release source'));continue
        if not local.is_file():
            failures.append((name,'tracked file missing'));continue
        if path.suffix.lower() in ('.py','.md','.toml','.json','.yml','.yaml','.txt') or path.name=='.gitignore':
            text=local.read_text(encoding='utf8')
            if any(pattern.search(text) for pattern in PRIVATE_TEXT):
                failures.append((name,'private path or credential pattern'))
            if path.suffix.lower()=='.json' and not name.startswith('assets/') and json_has_results(json.loads(text)):
                failures.append((name,'observed-result JSON keys'))
    return failures


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();root=args.root.resolve()
    output=subprocess.check_output(['git','-C',str(root),'ls-files','-z'])
    paths=output.decode().split('\0');paths=[p for p in paths if p]
    failures=audit(root,paths)
    for path,reason in failures:print(path+': '+reason,file=sys.stderr)
    print('PUBLIC_TREE_'+('FAIL' if failures else 'PASS')+f': {len(paths)} tracked files')
    return 1 if failures else 0

if __name__=='__main__':
    raise SystemExit(main())
