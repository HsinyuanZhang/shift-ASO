"""Two-stage CLI: globally finish/freeze DEV, then exact replay and TEST."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import numpy as np
import scipy
import numba
from .api import calibrate
from .search import search
from .stream import replay
from .scoring import Scorer
from .records import (array_sha,bindings,load_case,manifest,sha_file,source_hashes,
                      validate_samples,verify_complete,write_complete,write_json)


def environment():
    return dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__)


def select_dev(manifest_path, output):
    cases = manifest(manifest_path)
    sources, runtime = source_hashes(), environment()
    output = Path(output);output.mkdir(parents=True,exist_ok=False)
    frozen, files = [], []
    for case in cases:
        bound = bindings(case)
        raw,geometry,times,channels = load_case(case)
        validate_samples(raw,case['front'],case['dev_stop'])
        evidence = search(raw,geometry,times,channels,
            cal_samples=case['cal_samples'],dev_stop=case['dev_stop'])
        directory = output/case['id'];directory.mkdir()
        arrays, roles = {}, {}
        for name,value in evidence['selected'].items():
            roles[name] = dict(configuration=value['configuration'],metrics=value['metrics'])
            for field in ('H','cal_stat','events','pairs'):
                arrays[name+'__'+field] = value[field]
        array_path = directory/'evidence.npz'
        with array_path.open('xb') as stream:
            np.savez_compressed(stream,**arrays)
        grid_path = directory/'grid.jsonl'
        with grid_path.open('x') as stream:
            for row in evidence['rows']:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        if bindings(case)!=bound:
            raise ValueError('input changed during DEV')
        frozen.append(dict(case=case,bindings=bound,roles=roles,evidence=str(array_path.relative_to(output))))
        files.extend((array_path,grid_path))
    choices = output/'choices.json'
    write_json(choices,dict(schema='mearec_software_frozen_v1',cases=frozen,environment=runtime))
    files.append(choices)
    if source_hashes()!=sources:
        raise ValueError('implementation changed during DEV')
    # Last write is the global barrier; failed/partial cases never produce it.
    write_complete(output,files,dict(status='PASS_DEV_FROZEN',cases=len(cases),sources=sources))
    return dict(status='PASS_DEV_FROZEN',cases=len(cases),barrier_sha256=sha_file(output/'COMPLETE.json'))


def evaluate_test(dev_run, output):
    dev_run, output = Path(dev_run),Path(output)
    barrier = verify_complete(dev_run)
    frozen = json.loads((dev_run/'choices.json').read_text())
    if frozen['environment']!=environment():
        raise ValueError('numerical runtime changed since DEV')
    output.mkdir(parents=True,exist_ok=False)
    files = []
    for entry in frozen['cases']:
        case, roles = entry['case'],entry['roles']
        if bindings(case)!=entry['bindings']:
            raise ValueError('input changed since DEV')
        raw,geometry,times,channels = load_case(case)
        configurations = {name:role['configuration'] for name,role in roles.items()}
        prefix = raw[:case['dev_stop']]
        thresholds = {name:calibrate(prefix,cfg,case['cal_samples'])*cfg['K']
                      for name,cfg in configurations.items()}
        dev_events = replay(prefix,geometry,configurations,thresholds,
            bounds=(case['cal_samples'],case['dev_stop']))
        scorer = Scorer(times,channels,geometry,(case['cal_samples'],case['dev_stop']))
        with np.load(dev_run/entry['evidence'],allow_pickle=False) as expected:
            for name,cfg in configurations.items():
                base = calibrate(prefix,cfg,case['cal_samples'])
                metrics,pairs = scorer.score(dev_events[name])
                actual = dict(H=thresholds[name],cal_stat=base,events=dev_events[name],pairs=pairs)
                for field,value in actual.items():
                    if not np.array_equal(value,expected[name+'__'+field]):
                        raise ValueError('exact DEV replay mismatch: '+name+'/'+field)
                if metrics!=roles[name]['metrics']:
                    raise ValueError('exact DEV metrics mismatch')
        directory = output/case['id'];directory.mkdir()
        checked = directory/'DEV_REPLAY_EXACT.json'
        write_json(checked,dict(status='PASS_EXACT_DEV_REPLAY',roles=sorted(roles)))
        files.append(checked)
        # No TEST sample validation/detection/scoring occurs before DEV replay.
        validate_samples(raw,case['front'])
        events = replay(raw,geometry,configurations,thresholds,bounds=(case['dev_stop'],len(raw)))
        scorer = Scorer(times,channels,geometry,(case['dev_stop'],len(raw)))
        arrays, metrics = {}, {}
        for name,cfg in configurations.items():
            metrics[name],pairs = scorer.score(events[name])
            arrays.update({name+'__events':events[name],name+'__pairs':pairs,name+'__H':thresholds[name]})
        array_path = directory/'evidence.npz'
        with array_path.open('xb') as stream:
            np.savez_compressed(stream,**arrays)
        record = directory/'metrics.json'
        write_json(record,dict(configurations=configurations,metrics=metrics,
                              arrays={key:array_sha(value) for key,value in arrays.items()}))
        files.extend((array_path,record))
        if bindings(case)!=entry['bindings']:
            raise ValueError('input changed during TEST')
    if source_hashes()!=barrier['sources']:
        raise ValueError('implementation changed during TEST')
    write_complete(output,files,dict(status='PASS_TEST_FROZEN',cases=len(frozen['cases']),
        global_dev_barrier_sha256=sha_file(dev_run/'COMPLETE.json'),sources=barrier['sources']))
    return dict(status='PASS_TEST_FROZEN',cases=len(frozen['cases']))


def main(argv=None):
    parser = argparse.ArgumentParser(description='MEArec plain NEO/EMA: DEV selection then frozen TEST',
        epilog='Legacy HDF5 replay is separate: python -m shift_aso.mearec legacy --help')
    commands = parser.add_subparsers(dest='stage',required=True)
    dev = commands.add_parser('dev',help='complete every DEV case, then freeze all choices')
    dev.add_argument('--manifest',type=Path,required=True)
    dev.add_argument('--output',type=Path,required=True)
    test = commands.add_parser('test',help='exactly replay frozen DEV before any TEST scoring')
    test.add_argument('--dev-run',type=Path,required=True)
    test.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    result = select_dev(args.manifest,args.output) if args.stage=='dev' else evaluate_test(args.dev_run,args.output)
    print(json.dumps(result))
