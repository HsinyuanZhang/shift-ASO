import json
from pathlib import Path
import numpy as np
from shift_aso.quiroga.runner import encode, match

ROOT=Path(__file__).resolve().parents[1]
def test_frozen_matrix_has_twelve_methods():
    cfg=json.loads((ROOT/'configs'/'quiroga.json').read_text())
    assert len(cfg['matrix']) == 12
    for key in ('Shift_ASO_LC4','Shift_ASO_LC5','Shift_ASO_LC6'):
        assert cfg['matrix'][key]['k'] == .5

def test_lc_encoder_and_matcher_contracts():
    q,r,d=encode(np.array([0.,-1.,-2.,0.]),4,7.)
    assert d == 1. and q.tolist()==[0,-1,-2,0] and r.tolist()==[0.,-1.,-2.,0.]
    assert match(np.array([10,30]),np.array([11,31]),1)['TP']==2
