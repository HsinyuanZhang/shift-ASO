"""End-to-end tests use only short, synthetic caller-supplied arrays."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from shift_aso.mearec.software.cli import select_dev, evaluate_test
from shift_aso.mearec.software.records import validate_samples, manifest as read_manifest


def inputs(tmp_path):
    raw=np.random.default_rng(12).normal(0,.03,(640,2)).astype(np.float32)
    q=np.zeros((640,2),np.int8)
    for t in [160,240,360,520,600]:
        raw[t,0]-=8;q[t,0]=-8
    np.save(tmp_path/'raw.npy',raw);np.save(tmp_path/'q.npy',q)
    np.save(tmp_path/'geometry.npy',np.array([[0,0],[36,0]]))
    np.savez(tmp_path/'labels.npz',times=np.array([160,240,360,520,600,639]),channels=np.array([0,0,0,0,0,-1]))
    cases=[dict(id=front,front=front,signal=signal,geometry='geometry.npy',labels='labels.npz',cal_samples=128,dev_stop=400)
           for front,signal in [('FP','raw.npy'),('LC6','q.npy')]]
    path=tmp_path/'inputs.json';path.write_text(json.dumps(dict(cases=cases)))
    return path


def test_all_DEV_freeze_then_exact_replay_before_TEST(tmp_path):
    manifest=inputs(tmp_path);dev=tmp_path/'dev';test=tmp_path/'test'
    assert select_dev(manifest,dev)==dict(status='PASS_DEV_FROZEN',cases=2,
        barrier_sha256=__import__('hashlib').sha256((dev/'COMPLETE.json').read_bytes()).hexdigest())
    assert evaluate_test(dev,test)==dict(status='PASS_TEST_FROZEN',cases=2)
    for front in ('FP','LC6'):
        assert json.loads((test/front/'DEV_REPLAY_EXACT.json').read_text())['status']=='PASS_EXACT_DEV_REPLAY'
        with np.load(test/front/'evidence.npz',allow_pickle=False) as data:
            assert len(data['plain__events'])>0 and len(data['ema__events'])>0
        metrics=json.loads((test/front/'metrics.json').read_text())['metrics']
        assert metrics['plain']['good_GT_count']==3  # includes unknown/end-guard GT
    with pytest.raises(FileExistsError):select_dev(manifest,dev)
    with pytest.raises(FileExistsError):evaluate_test(dev,test)


def test_TEST_rejects_mutated_input_and_mutated_DEV_evidence(tmp_path):
    path=inputs(tmp_path);dev=tmp_path/'dev';select_dev(path,dev)
    original=(tmp_path/'raw.npy').read_bytes()
    raw=np.load(tmp_path/'raw.npy');raw[500,0]+=1;np.save(tmp_path/'raw.npy',raw)
    with pytest.raises(ValueError,match='input changed'):evaluate_test(dev,tmp_path/'changed_input')
    (tmp_path/'raw.npy').write_bytes(original)
    with (dev/'FP/grid.jsonl').open('a') as stream:stream.write('{}\n')
    with pytest.raises(ValueError,match='binding drift'):evaluate_test(dev,tmp_path/'changed_grid')


def test_global_barrier_absent_after_a_failed_case(tmp_path):
    path=inputs(tmp_path);doc=json.loads(path.read_text());doc['cases'][1]['signal']='missing.npy';path.write_text(json.dumps(doc))
    dev=tmp_path/'partial'
    with pytest.raises(FileNotFoundError):select_dev(path,dev)
    assert not (dev/'COMPLETE.json').exists()
    with pytest.raises(FileNotFoundError):evaluate_test(dev,tmp_path/'forbidden_TEST')


def test_Q_validation_never_reencodes_or_resets_state():
    q=np.zeros((100,1),np.int8);q[40]=-8;validate_samples(q,'LC6',block_samples=41)
    q[41]=8
    with pytest.raises(ValueError,match='slew8'):validate_samples(q,'LC6',block_samples=41)


def test_current_and_legacy_CLI_help_are_unambiguous():
    root=Path(__file__).resolve().parents[4]
    current=subprocess.run([sys.executable,'-m','shift_aso.mearec','--help'],cwd=root,text=True,capture_output=True,check=True)
    assert 'DEV selection then frozen TEST' in current.stdout and '{dev,test}' in current.stdout
    legacy=subprocess.run([sys.executable,'-m','shift_aso.mearec','legacy','--help'],cwd=root,text=True,capture_output=True,check=True)
    assert '--recording' in legacy.stdout


def test_manifest_never_truncates_fractional_splits_or_ignores_protocol_fields(tmp_path):
    path=inputs(tmp_path);doc=json.loads(path.read_text())
    doc['cases'][0]['cal_samples']=128.5;path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='integers'):read_manifest(path)
    doc['cases'][0]['cal_samples']=128;doc['cases'][0]['fs']=30000;path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='unknown manifest'):read_manifest(path)
