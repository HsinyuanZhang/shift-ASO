"""Synthetic semantic tests, never stored benchmark outcomes."""
import numpy as np
import pytest
from shift_aso.mearec.software import centered_neo, detect, ema, run, replay, calibrate
from shift_aso.mearec.software.api import validate
from shift_aso.mearec.software.features import activity
from shift_aso.mearec.software.detect import EVENT, peak_pool
from shift_aso.mearec.software.scoring import Scorer
from shift_aso.mearec.software.search import search, rank, refinement_points


def fixture(channels=2):
    raw = np.zeros((160,channels))
    raw[64] = -3
    energy = np.zeros_like(raw)
    geometry = np.column_stack((np.arange(channels)*36,np.zeros(channels)))
    return raw,energy,geometry


def test_refined_K_is_valid_without_becoming_a_stored_winner():
    cfg = validate('ema',4,.875,float(np.sqrt(8*16)))
    assert cfg['K']==np.sqrt(8*16)
    for method,lag,rho,k in [('plain',4,.5,1),('ema',12,.5,1),('plain',True,0,1),('plain',4,0,np.nan),('plain',4,0,-1)]:
        with pytest.raises(ValueError):validate(method,lag,rho,k)


def test_nearby_spatial_competition_and_far_electrodes():
    raw,energy,geometry = fixture();energy[64]=[9,16]
    events = detect(raw,energy,np.ones(2),geometry)
    assert len(events)==1 and events['channel_index'].tolist()==[1]
    geometry[1,0]=100
    assert len(detect(raw,energy,np.ones(2),geometry))==2


def test_same_time_ties_are_direct_not_transitive():
    raw,energy,geometry = fixture(3);energy[64]=9
    assert detect(raw,energy,np.ones(3),geometry)['channel_index'].tolist()==[0,2]


def test_polarity_is_after_ties_and_not_an_NMS_input_mask():
    raw,energy,geometry = fixture();energy[64]=9;raw[64,0]=3
    assert len(detect(raw,energy,np.ones(2),geometry))==0


def test_losing_H_or_polarity_does_not_remove_competition():
    raw,energy,geometry = fixture();energy[64]=[16,9]
    assert len(detect(raw,energy,np.array([20,1]),geometry))==0
    energy[:]=0;energy[62,0]=16;energy[64,1]=9;raw[62,0]=3
    assert len(detect(raw,energy,np.ones(2),geometry))==0


def test_strict_H_and_asymmetric_plateau_timing():
    raw,energy,geometry = fixture(1);energy[64:66]=9;raw[65]=0
    assert len(detect(raw,energy,np.array([9.]),geometry))==0
    events = detect(raw,energy,np.array([8.]),geometry)
    assert len(events)==1 and events['feature_peak'][0]==64


def test_earliest_raw_minimum_and_full_event_metadata():
    raw,energy,geometry = fixture(1);energy[64]=9;raw[61]=raw[66]=-4
    events = detect(raw,energy,np.ones(1),geometry,lag=2)
    assert events.dtype==EVENT and len(events)==1
    assert events['sample_index'][0]==61 and events['feature_peak'][0]==64
    assert events['availability_sample'][0]==74


def test_guard_is_strict_and_availability_respects_last_needed_sample():
    raw,energy,geometry = fixture(1);energy[120]=9;raw[120]=-3
    assert len(detect(raw,energy,np.ones(1),geometry))==0
    energy[:]=0;energy[150]=9;raw[150]=-3
    assert len(detect(raw,energy,np.ones(1),geometry,guard=0))==1
    energy[:]=0;energy[151]=9;raw[151]=-3
    assert len(detect(raw,energy,np.ones(1),geometry,guard=0))==0


def test_invalid_input_is_not_silently_adopted():
    raw,energy,geometry = fixture()
    with pytest.raises(ValueError):detect(raw,energy,np.array([np.nan,1]),geometry)
    with pytest.raises(ValueError):detect(raw,energy,np.ones(2),geometry[:1])
    with pytest.raises(ValueError):activity(raw,1,.5,start=5,stop=20)
    with pytest.raises(ValueError):activity(raw[:10],8,0,start=0,stop=10,total_frames=100)


@pytest.mark.parametrize('lag',[1,2,4,6,8])
@pytest.mark.parametrize('rho',[0.,.5,.875,.9375])
def test_feature_block_halos_and_state_match_batch_exactly(lag,rho):
    raw = np.random.default_rng(4).integers(-31,32,(400,3),dtype=np.int8)
    full,last = activity(raw,lag,rho)
    pieces=[];state=np.zeros(3)
    for first in range(0,len(raw),23):
        stop=min(first+23,len(raw));left=max(0,first-lag);right=min(len(raw),stop+lag)
        part,state=activity(raw[left:right],lag,rho,buffer_start=left,start=first,stop=stop,total_frames=len(raw),state=state)
        pieces.append(part)
    np.testing.assert_array_equal(np.concatenate(pieces),full)
    np.testing.assert_array_equal(state,last)
    np.testing.assert_array_equal(ema(centered_neo(raw,lag),rho),full)


def test_signed_arithmetic_casts_before_multiply_and_empty_input():
    raw=np.zeros((100,1),np.int8);raw[40]=2;raw[42]=3
    assert centered_neo(raw,1)[41,0]==-6
    assert ema(centered_neo(raw,1),.5)[41,0]==-4
    raw[60]=-31;assert centered_neo(raw,1)[60,0]==961
    assert ema(np.empty((0,2)),.5).shape==(0,2)


@pytest.mark.parametrize('block',[1,7,17,31,400,1000])
@pytest.mark.parametrize('method,lag,rho',[('plain',8,0.),('ema',1,.5),('ema',6,.9375)])
def test_nonempty_stream_and_batch_exact_with_no_split_reset(block,method,lag,rho):
    raw=np.random.default_rng(5).normal(0,.1,(400,3)).astype(np.float32)
    raw[80,0]-=8;raw[180,2]-=12;raw[260,1]-=10
    geometry=np.array([[0,0],[36,0],[72,0]])
    cfg=validate(method,lag,rho,1);H=np.full(3,.1)
    batch=run(raw,geometry,H,method,lag,rho,1,bounds=(100,400))
    streamed=replay(raw,geometry,{'arm':cfg},{'arm':H},bounds=(100,400),block_samples=block)['arm']
    assert len(batch)>0
    np.testing.assert_array_equal(streamed,batch)


def test_CAL_is_unmasked_unlabeled_and_has_no_future_dependency():
    raw=np.zeros((300,2),np.float32);raw[80]=[-8,8]
    cfg=validate('ema',4,.875,1)
    base=calibrate(raw,cfg,128)
    altered=raw.copy();altered[128:]=100
    np.testing.assert_array_equal(base,calibrate(altered,cfg,128))
    assert np.all(base>0) and base[0]==base[1]


def test_unknown_and_end_guard_GT_stay_in_full_denominator():
    raw,energy,geometry=fixture(1);energy[64]=9
    events=detect(raw,energy,np.ones(1),geometry)
    scorer=Scorer(np.array([64,80,159]),np.array([0,-1,0]),geometry,(0,160))
    metrics,pairs=scorer.score(events)
    assert metrics['TP']==1 and metrics['FN']==2 and metrics['good_GT_count']==3
    assert pairs.tolist()==[[0,0]]


def test_refinement_bracket_is_not_family_global_and_rank_uses_exact_ratio():
    assert refinement_points([0,1,2,4],2)==[float(k) for k in np.geomspace(1,4,5)[1:-1] if float(k)!=2.]
    assert rank(dict(TP=2,FP=1,FN=1),4)>rank(dict(TP=1,FP=1,FN=0),5)


def test_DEV_selection_with_pruning_matches_full_scoring_and_ignores_TEST_labels():
    raw=np.random.default_rng(8).normal(0,.1,(640,2)).astype(np.float32)
    for t in [160,240,360,520]:raw[t,0]-=8
    geometry=np.array([[0,0],[36,0]])
    times=np.array([160,240,360,520,600]);channels=np.array([0,0,0,0,-1])
    a=search(raw,geometry,times,channels,cal_samples=128,dev_stop=400)
    b=search(raw,geometry,times,channels,cal_samples=128,dev_stop=400,pruning=False)
    times[3:]=[570,610];channels[3:]=[1,1]
    c=search(raw,geometry,times,channels,cal_samples=128,dev_stop=400)
    for name in a['selected']:
        for other in (b,c):
            assert a['selected'][name]['configuration']==other['selected'][name]['configuration']
            for field in ('H','cal_stat','events'):
                np.testing.assert_array_equal(a['selected'][name][field],other['selected'][name][field])
    ids=[r['configuration']['config_id'] for r in a['rows']]
    assert len(ids)==len(set(ids)) and len(a['groups'])==20


def test_upper_endpoint_extension_keeps_unique_ids_across_rounds():
    raw=np.zeros((900,1),np.float64)
    raw[:128]=np.random.default_rng(31).normal(0,.5,(128,1))
    geometry=np.zeros((1,2))
    base=calibrate(raw,validate('plain',1,0,1),128)[0]
    for sample,ratio in ((220,28),(320,60),(500,500)):
        raw[sample,0]=-np.sqrt(base*ratio)
    result=search(raw,geometry,np.array([500]),np.array([0]),cal_samples=128,dev_stop=700)
    extensions=[row for row in result['rows'] if row['configuration']['group']==0 and row['configuration']['stage']=='upper_extension']
    assert len(extensions)>4  # at least two real doubling rounds
    ids=[row['configuration']['config_id'] for row in result['rows']]
    assert len(ids)==len(set(ids))
    assert max(row['configuration']['K'] for row in extensions)>=64
    assert result['selected']['plain']['metrics']['A']==1.


def test_current_API_rejects_instead_of_silently_substituting_legacy_policy():
    raw=np.zeros((160,1));geometry=np.zeros((1,2))
    cfg=validate('plain',4,0,1);cfg['policy']='ungated'
    with pytest.raises(ValueError,match='unsupported current-method policy'):
        replay(raw,geometry,{'reference':cfg},{'reference':np.ones(1)})
    with pytest.raises(ValueError):peak_pool(raw,geometry,lag=1.5)
