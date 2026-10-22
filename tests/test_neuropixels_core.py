import numpy as np
import json
from pathlib import Path
from shift_aso.neuropixels.core import load_config
from shift_aso.neuropixels import _kernels as k

def test_packaged_configuration_matches_public_copy():
    package=load_config()
    public=json.loads((Path(__file__).parents[1]/"configs"/"neuropixels.json").read_text())
    assert package == public
    assert set(package["methods"]) == {"SI10_MAD", "NEO10_EMA", "LC6_ShiftASO"}
    assert package["methods"]["LC6_ShiftASO"]["full_scale"] == 2827.6876831054688

def test_global_encoder_and_feature_are_stateful():
    x=np.array([[0.,1.],[2.,-3.],[-4.,5.]])
    cfg=k.ranges_from_cal(x.min(0),x.max(0),np.array([0,1]),10.,"global",1.0)
    q,_,_=k.encode_range(x,cfg["delta"],cfg["offset_codes"],cfg["initial"])
    e,_,_=k.feature_range(q,cfg["offset_codes"],.875,cfg["initial"],np.zeros(2))
    assert q.shape==x.shape and e.shape==x.shape and np.all(e>=0)
