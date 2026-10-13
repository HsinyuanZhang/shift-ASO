"""Frozen streaming Neuropixels SI and NEO controls."""
from __future__ import annotations
import numpy as np
from numba import njit
from scipy.signal import lfilter
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

FS=30000
MAD=.6744897501960817
RADIUS=40.
CONTEXT=80
EVENT_DTYPE=np.dtype([('sample_index','<i8'),('channel_index','<i4'),('amplitude','<f4')])

@njit(cache=True)
def _encode(x,full,state):
    delta=full/31.; q=np.empty(x.shape,np.int8); limited=0; saturated=0
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            requested=int(np.trunc((float(x[n,c])-state[c]*delta)/delta))
            limited+=abs(requested)>8
            step=max(-8,min(8,requested)); nxt=max(-31,min(31,state[c]+step))
            saturated+=nxt!=state[c]+step; q[n,c]=nxt; state[c]=nxt
    return q,state,limited,saturated

def encode_stateful(x,full,state):
    return _encode(np.ascontiguousarray(x,dtype=np.float32),float(full),np.ascontiguousarray(state,dtype=np.int16))

def cal_statistics(cal):
    cal=np.asarray(cal,dtype=np.float32); median=np.median(cal,axis=0)
    return (np.median(np.abs(cal-median),axis=0)/MAD).astype(np.float32)

def qmad_from_cal(cal,full):
    q,state,limited,saturated=encode_stateful(cal,full,np.zeros(cal.shape[1],np.int16)); median=np.median(q,axis=0)
    return (np.median(np.abs(q-median),axis=0)/MAD).astype(np.float32),{'limited_updates_cal':int(limited),'saturated_samples_cal':int(saturated),'final_state_cal':state.tolist()}

def neo_scales_from_cal(cal):
    cal=np.asarray(cal,dtype=np.float32); x=np.vstack((cal[:1],cal,cal[-1:])); psi=x[1:-1]*x[1:-1]-x[:-2]*x[2:]
    rho=np.float32(np.exp(-1./(FS*72.13475e-6))); ema=lfilter([1-rho],[1,-rho],psi,axis=0).astype(np.float32)
    return np.maximum(np.mean(np.abs(psi),axis=0),1e-9).astype(np.float32),np.maximum(np.mean(np.abs(ema),axis=0),1e-9).astype(np.float32)

def _recording(x,geometry):
    rec=NumpyRecording(np.ascontiguousarray(x,dtype=np.float32),sampling_frequency=FS);rec.set_channel_locations(geometry);return rec

def si_peaks(x,geometry,sign,noise,threshold):
    return detect_peaks(_recording(x,geometry),method='locally_exclusive',method_kwargs={'peak_sign':sign,'detect_threshold':float(threshold),'exclude_sweep_ms':.4,'radius_um':RADIUS,'noise_levels':noise},job_kwargs={'n_jobs':1,'chunk_duration':'1s','progress_bar':False})

def _append(parts,events,offset,lo,hi):
    if not len(events): return
    z=events.copy();z['sample_index']+=int(offset);z=z[(z['sample_index']>=lo)&(z['sample_index']<hi)]
    if len(z):
        out=np.empty(len(z),EVENT_DTYPE)
        for f in EVENT_DTYPE.names: out[f]=z[f]
        parts.append(out)

def _combine(parts): return np.concatenate(parts) if parts else np.empty(0,EVENT_DTYPE)

def _resolve_quantized_ties(peaks,geometry,radius=40.):
    peaks=np.asarray(peaks); channels=np.asarray(peaks['channel_index'],np.int64); geo=np.asarray(geometry,float); removed=np.zeros(len(peaks),bool)
    keys=np.lexsort((channels,peaks['amplitude'],peaks['sample_index'])); start=0
    while start<len(keys):
        end=start+1
        while end<len(keys) and peaks['sample_index'][keys[end]]==peaks['sample_index'][keys[start]] and peaks['amplitude'][keys[end]]==peaks['amplitude'][keys[start]]: end+=1
        ids=keys[start:end]
        if len(ids)>1:
            order=ids[np.argsort(channels[ids],kind='stable')];live=np.ones(len(ids),bool);position={int(v):j for j,v in enumerate(ids)}
            for anchor in order:
                ap=position[int(anchor)]
                if not live[ap]: continue
                dist=np.linalg.norm(geo[channels[ids]]-geo[channels[anchor]],axis=1);direct=ids[live&(dist<=radius)]
                for victim in direct:
                    if victim!=anchor: live[position[int(victim)]]=False;removed[victim]=True
        start=end
    return peaks[~removed]


def stream_candidates(stream_factory,stop,geometry,full,sigma,qmad,psi_scale,ema_scale):
    """Return rawSI, LC6SI, bare NEO, and NEO+EMA events in global samples."""
    raw_parts,q_parts,psi_parts,ema_parts=[],[],[],[]; channels=len(geometry); state=np.zeros(channels,np.int16)
    raw_tail=np.empty((0,channels),np.float32);q_tail=np.empty((0,channels),np.int8);psi_tail=np.empty((0,channels),np.float32);z_tail=np.empty((0,channels),np.float32)
    ema_state=np.zeros(channels,np.float32);rho=np.float32(np.exp(-1./(FS*72.13475e-6)));pending=None
    stats={'limited_updates':0,'saturated_samples':0,'chunks':0,'max_chunk_samples':0,'max_context_bytes':0,'rho':float(rho)}
    for start,x in stream_factory.chunks(stop):
        x=np.asarray(x,dtype=np.float32);q,state,limited,saturated=encode_stateful(x,full,state);stats['limited_updates']+=int(limited);stats['saturated_samples']+=int(saturated);stats['chunks']+=1;stats['max_chunk_samples']=max(stats['max_chunk_samples'],len(x))
        if pending is None:
            pending={'start':start,'raw':x,'q':q,'left':x[:1].copy()};continue
        plus=np.vstack((pending['left'],pending['raw'],x[:1]));psi=(plus[1:-1]*plus[1:-1]-plus[:-2]*plus[2:]).astype(np.float32);z,final=lfilter([1-rho],[1,-rho],psi,axis=0,zi=ema_state[None,:]);ema_state=final[0];pending['psi']=psi;pending['z']=z.astype(np.float32)
        right_n=min(CONTEXT,len(x));core_lo=pending['start'];core_hi=core_lo+len(pending['raw']);nx=x[:right_n];nplus=np.vstack((pending['raw'][-1:],nx,x[right_n:right_n+1] if len(x)>right_n else nx[-1:]));npsi=(nplus[1:-1]*nplus[1:-1]-nplus[:-2]*nplus[2:]).astype(np.float32);nz=lfilter([1-rho],[1,-rho],npsi,axis=0,zi=ema_state[None,:])[0].astype(np.float32)
        xx=np.vstack((raw_tail,pending['raw'],nx));qq=np.vstack((q_tail,pending['q'],q[:right_n]));zz=np.vstack((z_tail,pending['z'],nz));ppsi=np.vstack((psi_tail,pending['psi'],npsi))
        _append(raw_parts,si_peaks(xx,geometry,'neg',sigma,3.),core_lo-len(raw_tail),core_lo,core_hi)
        _append(q_parts,si_peaks(qq.astype(np.float32),geometry,'neg',np.ones(channels,np.float32),1.5),core_lo-len(q_tail),core_lo,core_hi)
        _append(psi_parts,si_peaks(ppsi,geometry,'pos',psi_scale,8.),core_lo-len(psi_tail),core_lo,core_hi)
        _append(ema_parts,si_peaks(zz,geometry,'pos',ema_scale,8.),core_lo-len(z_tail),core_lo,core_hi)
        raw_tail=pending['raw'][-CONTEXT:].copy();q_tail=pending['q'][-CONTEXT:].copy();psi_tail=pending['psi'][-CONTEXT:].copy();z_tail=pending['z'][-CONTEXT:].copy();stats['max_context_bytes']=max(stats['max_context_bytes'],xx.nbytes+qq.nbytes+zz.nbytes)
        pending={'start':start,'raw':x,'q':q,'left':pending['raw'][-1:].copy()}
    if pending is not None:
        plus=np.vstack((pending['left'],pending['raw'],pending['raw'][-1:]));psi=(plus[1:-1]*plus[1:-1]-plus[:-2]*plus[2:]).astype(np.float32);z,_=lfilter([1-rho],[1,-rho],psi,axis=0,zi=ema_state[None,:]);z=z.astype(np.float32);core_lo=pending['start'];core_hi=core_lo+len(pending['raw']);xx=np.vstack((raw_tail,pending['raw'],pending['raw'][-CONTEXT:]));qq=np.vstack((q_tail,pending['q'],pending['q'][-CONTEXT:]));zz=np.vstack((z_tail,z,z[-CONTEXT:]));ppsi=np.vstack((psi_tail,psi,psi[-CONTEXT:]))
        _append(raw_parts,si_peaks(xx,geometry,'neg',sigma,3.),core_lo-len(raw_tail),core_lo,core_hi);_append(q_parts,si_peaks(qq.astype(np.float32),geometry,'neg',np.ones(channels,np.float32),1.5),core_lo-len(q_tail),core_lo,core_hi);_append(psi_parts,si_peaks(ppsi,geometry,'pos',psi_scale,8.),core_lo-len(psi_tail),core_lo,core_hi);_append(ema_parts,si_peaks(zz,geometry,'pos',ema_scale,8.),core_lo-len(z_tail),core_lo,core_hi)
    pools={'rawSI_MAD':_combine(raw_parts),'LC6SI_global':_combine(q_parts),'NEO_bare':_combine(psi_parts),'NEO_EMA':_combine(ema_parts)}
    answer={}
    for key,scale,k in (('rawSI_MAD',sigma,4.),('NEO_bare',psi_scale,12.),('NEO_EMA',ema_scale,8.)):
        e=pools[key]; answer[key]=e[np.abs(e['amplitude'])>=k*scale[e['channel_index']]]
    p=pools['LC6SI_global']; keep=np.abs(p['amplitude'])>=3.*np.maximum(qmad[p['channel_index']],1.)
    answer['LC6SI_global']=_resolve_quantized_ties(p[keep],geometry,40.)
    return answer,stats

def run_baselines(stream_factory,cal,geometry,full,stop):
    """Freeze CAL statistics then stream the four formal controls once."""
    cal=np.asarray(cal,dtype=np.float32);sigma=cal_statistics(cal);qmad,frontend=qmad_from_cal(cal,full);psi,ema=neo_scales_from_cal(cal)
    events,stream=stream_candidates(stream_factory,int(stop),np.asarray(geometry,float),float(full),sigma,qmad,psi,ema)
    return {'events':events,'calibration':{'raw_sigma':sigma,'qmad':qmad,'psi_scale':psi,'ema_scale':ema,'frontend':frontend},'stream':stream,'parameters':{'rawSI_MAD':{'k':4.},'LC6SI_global':{'k':3.,'qmad_floor_codes':1.,'bits':6,'jmax':8},'NEO_bare':{'k':12.},'NEO_EMA':{'k':8.,'rho':float(np.exp(-1./(FS*72.13475e-6)))}}}
