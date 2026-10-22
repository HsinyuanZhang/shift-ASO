"""Publication checks test file classes, not private benchmark values."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('public_safety',ROOT/'scripts/check_public_tree.py')
safety=importlib.util.module_from_spec(spec);spec.loader.exec_module(safety)


def test_gate_refuses_observed_tables_and_runtime_records(tmp_path):
    paths=['results/table.csv','outputs/evidence.npz','experiment_records/study/notes.md']
    assert len(safety.audit(tmp_path,paths))==3
    path=tmp_path/'settings.json';path.write_text(json.dumps({'metrics':{'TP':1}}))
    assert safety.audit(tmp_path,['settings.json'])
    path.write_text(json.dumps({'A':0.25}))
    assert safety.audit(tmp_path,['settings.json'])


def test_gate_preserves_method_settings_and_public_input_assets(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'lag':4,'rho':.875,'K':8}))
    assert not safety.audit(tmp_path,['config.json','assets/mearec/spiketrains.npz'])


def test_method_docs_have_current_entrypoint_not_old_result_claims():
    text=(ROOT/'docs/mearec.md').read_text()
    assert 'python -m shift_aso.mearec dev' in text
    assert 'python -m shift_aso.mearec test' in text
    assert 'legacy --help' in text and 'not bundled' in text
    assert 'existing result tables remain' not in (ROOT/'docs/mearec_data.md').read_text()
