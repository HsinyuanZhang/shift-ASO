import numpy as np
from shift_aso.neuropixels import _kernels as k

def test_frozen_bank_encoder_and_feature_are_stateful():
    x=np.array([[0.,1.],[2.,-3.],[-4.,5.]])
    cfg=k.ranges_from_cal(x.min(0),x.max(0),np.array([0,1]),10.,"bank8_power2",1.25)
    q,_,_=k.encode_range(x,cfg["delta"],cfg["offset_codes"],cfg["initial"])
    e,_,_=k.feature_range(q,cfg["offset_codes"],.875,cfg["initial"],np.zeros(2))
    assert q.shape==x.shape and e.shape==x.shape and np.all(e>=0)
