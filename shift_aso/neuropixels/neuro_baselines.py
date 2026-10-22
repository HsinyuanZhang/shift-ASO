"""Public Q10 SI and NEO+EMA controls used by the Neuropixels replay."""
from __future__ import annotations
import numpy as np
import tempfile
from pathlib import Path
from numba import njit

MAD = 0.6744897501960817

@njit(cache=True)
def _nms(x, h, positive, near, sweep):
    n, nc = x.shape
    out_t = np.empty(n * nc, np.int64); out_c = np.empty(n * nc, np.int32); out_v = np.empty(n * nc, np.float32); k = 0
    for t in range(sweep, n-sweep):
        for c in range(nc):
            v=x[t,c]
            if (positive and v <= h[c]) or ((not positive) and v >= -h[c]): continue
            ok=True
            for other in range(nc):
                if not near[c,other]: continue
                if other != c:
                    if (positive and v < x[t,other]) or ((not positive) and v > x[t,other]): ok=False
                for d in range(1,sweep+1):
                    past=x[t-d,other]; future=x[t+d,other]
                    if positive:
                        if v <= past or v < future: ok=False
                    else:
                        if v >= past or v > future: ok=False
                    if not ok: break
                if not ok: break
            if ok: out_t[k]=t;out_c[k]=c;out_v[k]=v;k+=1
    return out_t[:k],out_c[:k],out_v[:k]

def q10(raw, center, full_scale):
    """Centered 10-bit signed quantizer: round-to-nearest-even then clip."""
    delta=np.float32(float(full_scale)/511.)
    return np.clip(np.rint((np.asarray(raw,np.float32)-center)/delta),-511,511).astype(np.int16)

def neighbours(geometry, radius=40.):
    d=np.linalg.norm(geometry[:,None]-geometry[None,:],axis=2)
    return d <= float(radius)

def events_from_array(values, threshold, positive, geometry, offset=0):
    t,c,v=_nms(np.ascontiguousarray(values,np.float32),np.asarray(threshold,np.float32),bool(positive),neighbours(geometry),12)
    dtype=np.dtype([('sample_index','<i8'),('channel_index','<i4'),('amplitude','<f4')])
    out=np.empty(len(t),dtype);out['sample_index']=t+int(offset);out['channel_index']=c;out['amplitude']=v
    return out

def controls(raw, geometry, full_scale, cal_samples, rho=.75, c=6.):
    """Compute the two public controls on a continuous in-memory raw prefix."""
    raw=np.asarray(raw,np.float32); center=np.median(raw[:int(cal_samples)],axis=0).astype(np.float32)
    q=q10(raw,center,full_scale); qcal=q[:int(cal_samples)]
    sigma=np.median(np.abs(qcal-np.median(qcal,axis=0)),axis=0)/MAD
    si=events_from_array(q,np.floor(3.5*sigma),False,geometry)
    qi=q.astype(np.int32); psi=(qi[1:-1]*qi[1:-1]-qi[:-2]*qi[2:]).astype(np.float32)
    ema=np.empty_like(psi); state=np.zeros(q.shape[1],np.float64)
    for i in range(len(psi)):
        state=float(rho)*state+(1.-float(rho))*psi[i];ema[i]=state
    scale=np.floor(float(c)*np.mean(np.abs(ema[:max(1,int(cal_samples)-2)]),axis=0,dtype=np.float64)).astype(np.float32)
    neo=events_from_array(ema,scale,True,geometry,offset=1)
    return {'SI10_MAD':si,'NEO10_EMA':neo}, {'center':center,'sigma':sigma,'neo_scale':scale}

def controls_stream(factory, stop, geometry, full_scale, cal, rho=.75, c=6., temp_dir=None):
    """Replay controls with disk-backed continuous Q10 and EMA trajectories."""
    cal=np.asarray(cal,np.float32); center=np.median(cal,axis=0).astype(np.float32)
    qcal=q10(cal,center,full_scale); sigma=np.median(np.abs(qcal-np.median(qcal,0)),axis=0)/MAD
    # CAL-only NEO scale, evaluated on its continuous trajectory.
    qc=qcal.astype(np.int32); p=qc[1:-1]*qc[1:-1]-qc[:-2]*qc[2:]; state=np.zeros(qc.shape[1],np.float32)
    ecal=np.empty_like(p,dtype=np.float32)
    for i in range(len(p)): state=np.float32(.75)*state+np.float32(.25)*p[i];ecal[i]=state
    neo_h=np.floor(float(c)*np.mean(np.abs(ecal),axis=0,dtype=np.float64)).astype(np.float32)
    nchan=len(center); dtype=np.dtype([('sample_index','<i8'),('channel_index','<i4'),('amplitude','<f4')])
    with tempfile.TemporaryDirectory(dir=temp_dir) as work:
        q=np.memmap(Path(work)/'q10.dat',dtype=np.int16,mode='w+',shape=(stop,nchan))
        for start,raw in factory.chunks(stop): q[start:start+len(raw)]=q10(raw,center,full_scale)
        ema=np.memmap(Path(work)/'ema.dat',dtype=np.float32,mode='w+',shape=(stop,nchan)); state=np.zeros(nchan,np.float64)
        # Samples 0 and stop-1 are invalid NEO endpoints and are never owned.
        for lo in range(1,stop-1,30000):
            hi=min(stop-1,lo+30000); x=np.asarray(q[lo-1:hi+1],np.int32)
            psi=(x[1:-1]*x[1:-1]-x[:-2]*x[2:]).astype(np.float32)
            for i in range(len(psi)): state=float(rho)*state+(1.-float(rho))*psi[i];ema[lo+i]=state
        si_parts=[]; neo_parts=[]
        for lo in range(0,stop,30000):
            hi=min(stop,lo+30000); a=max(0,lo-12); b=min(stop,hi+12)
            ev=events_from_array(np.asarray(q[a:b],np.float32),np.floor(3.5*sigma),False,geometry,offset=a)
            si_parts.append(ev[(ev['sample_index']>=lo)&(ev['sample_index']<hi)])
            a=max(1,lo-12); b=min(stop-1,hi+12)
            if b-a>24:
                ev=events_from_array(np.asarray(ema[a:b]),neo_h,True,geometry,offset=a)
                neo_parts.append(ev[(ev['sample_index']>=max(lo,1))&(ev['sample_index']<min(hi,stop-1))])
        del q,ema
    return {'SI10_MAD':np.concatenate(si_parts) if si_parts else np.empty(0,dtype),'NEO10_EMA':np.concatenate(neo_parts) if neo_parts else np.empty(0,dtype)}, {'center':center,'sigma':sigma,'neo_scale':neo_h}
