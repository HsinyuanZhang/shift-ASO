"""User-supplied array manifests and create-once local run records.

Generated records are local user outputs, never bundled publication evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import numpy as np


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path, document):
    with Path(path).open('x') as stream:
        json.dump(document,stream,indent=2,allow_nan=False,
                  default=lambda v:v.item() if isinstance(v,np.generic) else list(v))
        stream.write('\n')


def source_hashes():
    root = Path(__file__).parent
    files = sorted(root.glob('*.py'))
    hashes = {path.name:sha_file(path) for path in files}
    hashes['shared_matching'] = sha_file(root.parent.parent/'neuropixels/_matching.py')
    return hashes


def manifest(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    cases = document.get('cases')
    if not isinstance(cases,list) or not cases:
        raise ValueError('manifest must contain a nonempty cases list')
    seen, result = set(), []
    for item in cases:
        name = item.get('id','')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name) or name in seen:
            raise ValueError('case IDs must be unique safe directory names')
        seen.add(name)
        front = item.get('front')
        if front not in ('FP','LC6'):
            raise ValueError('front must explicitly be FP or LC6')
        allowed = {'id','front','signal','geometry','labels','cal_samples','dev_stop'}
        if set(item)-allowed:
            raise ValueError('unknown manifest case fields')
        cal,stop = item.get('cal_samples',120000),item.get('dev_stop',400000)
        if any(isinstance(v,bool) or not isinstance(v,int) for v in (cal,stop)):
            raise ValueError('split sample indices must be integers, not coerced values')
        row = dict(id=name,front=front,cal_samples=cal,dev_stop=stop)
        for key in ('signal','geometry','labels'):
            value = Path(item[key]);value = value if value.is_absolute() else path.parent/value
            row[key] = str(value.resolve())
        result.append(row)
    return result


def bindings(case):
    return {key:sha_file(case[key]) for key in ('signal','geometry','labels')}


def load_case(case):
    raw = np.load(case['signal'],mmap_mode='r',allow_pickle=False)
    geometry = np.load(case['geometry'],allow_pickle=False)
    if raw.ndim!=2 or len(geometry)!=raw.shape[1] or raw.shape[1]<1:
        raise ValueError('invalid signal/geometry shapes')
    if case['front']=='LC6' and raw.dtype!=np.int8:
        raise ValueError('LC6 must be a supplied frozen int8 Q array; no encoding is performed')
    if case['front']=='FP' and raw.dtype not in (np.dtype('float32'),np.dtype('float64')):
        raise ValueError('FP must be float32 or float64')
    if not 16 < case['cal_samples'] < case['dev_stop'] < len(raw):
        raise ValueError('input must cover nonempty CAL, DEV, and TEST')
    with np.load(case['labels'],allow_pickle=False) as data:
        times,channels = data['times'],data['channels']
    if times.ndim!=1 or times.shape!=channels.shape or not np.issubdtype(times.dtype,np.integer) or not np.issubdtype(channels.dtype,np.integer):
        raise ValueError('labels require equal-length integer times/channels')
    if np.any(times<0) or np.any(times>=len(raw)):
        raise ValueError('GT times outside recording')
    return raw,geometry,times,channels


def validate_samples(raw, front, stop=None, block_samples=30000):
    previous = np.zeros(raw.shape[1],np.int16)
    stop = len(raw) if stop is None else stop
    for first in range(0,stop,block_samples):
        block = raw[first:min(first+block_samples,stop)]
        if not np.all(np.isfinite(block)):
            raise ValueError('nonfinite signal')
        if front=='LC6':
            q = block.astype(np.int16)
            if np.any(np.abs(q)>31) or np.any(np.abs(q[0]-previous)>8) or np.any(np.abs(np.diff(q,axis=0))>8):
                raise ValueError('supplied Q violates range31 or emitted slew8, including zero-origin state')
            previous = q[-1]


def write_complete(root, files, document):
    document = dict(document,files={str(p.relative_to(root)):sha_file(p) for p in files})
    write_json(root/'COMPLETE.json',document)


def verify_complete(root):
    document = json.loads((root/'COMPLETE.json').read_text())
    if document.get('status')!='PASS_DEV_FROZEN':
        raise ValueError('DEV run is not globally frozen')
    for name,expected in document['files'].items():
        path = (root/name).resolve()
        if root.resolve() not in path.parents or sha_file(path)!=expected:
            raise ValueError('DEV file binding drift')
    if document['sources']!=source_hashes():
        raise ValueError('implementation changed since DEV freeze')
    return document
