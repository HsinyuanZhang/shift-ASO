"""Standalone frozen LC6 bank-range Shift-ASO replay for KS4 sim_no_drift."""
from __future__ import annotations
import hashlib, json, re, zipfile, zlib
from pathlib import Path
import numpy as np

FS=30000; CAL=10*FS; DEV=20*FS; STOP=300*FS; GUARD=60; RADIUS_UM=40.0
FROZEN={"branch":"LC6_bank8_power2_shiftASO","rho":0.875,"K_mad":12.9,"bank_size":8,"range_margin":1.25,"J":8,"codes":31,"window_before":6,"window_after":10,"peak_left":12,"peak_right":10,"coherence_half_window":6,"weak_alpha":0.75,"weak_neighbor_count":2,"weak_contrast":2.0,"boundary_beta":2.0,"pmin":2.0,"tv_kappa":3.0,"prominence_floor":1.0,"nms_samples":6,"ptp_rescue":False}
EVENT_DTYPE=np.dtype([("sample_index","<i8"),("channel_index","<i4"),("feature_peak","<i8"),("amplitude","<f8"),("availability_sample","<i8")])

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def _need(root, relative):
 p=root/relative
 if not p.is_file(): raise FileNotFoundError(f"required dataset file missing: {p}")
 return p

def dataset_paths(data_dir):
 root=Path(data_dir)
 archive=_need(root,"sim_no_drift.zip")
 labels=root/"evaluation_labels.npz"
 return {"zip":archive,"labels":labels if labels.is_file() else None}

def zip_text(archive, member):
 with zipfile.ZipFile(archive) as z:
  return z.read(member).decode()

def geometry(meta):
 text=Path(meta).read_text() if isinstance(meta,Path) else str(meta)
 line=next((x for x in text.splitlines() if x.startswith("~snsShankMap=")),None)
 if line is None: raise ValueError("meta lacks ~snsShankMap")
 rows=re.findall(r"\((\d+):(\d+):(\d+):(\d+)\)",line)
 if len(rows)<384: raise ValueError("meta has fewer than 384 AP map entries")
 active=np.asarray([i for i,(_,_,_,used) in enumerate(rows[:384]) if int(used)],np.int64)
 g=np.asarray([[int(rows[i][1])*32.0+(27.0 if int(rows[i][2])%2==0 else 11.0),int(rows[i][2])*20.0] for i in active],np.float64)
 return active,g

def stream_cbin(zip_path, header_path, active, stop):
 header=json.loads(Path(header_path).read_text() if isinstance(header_path,Path) else header_path)
 if header.get("algorithm")!="zlib" or not header.get("do_time_diff"): raise ValueError("expected zlib time-differenced .ch")
 bounds=np.asarray(header["chunk_bounds"],np.int64); offsets=np.asarray(header["chunk_offsets"],np.int64); nch=int(header["n_channels"])
 if nch!=385: raise ValueError(f"expected 385 recorded channels, got {nch}")
 with zipfile.ZipFile(zip_path) as archive, archive.open("sim.imec0.ap.cbin") as handle:
  cursor=0
  for i in range(len(bounds)-1):
   start,end=int(bounds[i]),int(bounds[i+1])
   if start>=stop: break
   target=int(offsets[i+1]); blob=handle.read(target-cursor); cursor=target
   raw=np.frombuffer(zlib.decompress(blob),dtype=np.int16)
   if raw.size!=(end-start)*nch: raise ValueError(f"CBIN chunk {i} size mismatch")
   full=np.cumsum(raw.reshape((end-start,nch),order="F"),axis=0,dtype=np.int32).astype(np.int16)
   use=full[:min(end,stop)-start,:384][:,active]
   for offset in range(0,len(use),FS):
    yield start+offset,use[offset:offset+FS].astype(np.float64,copy=False)

def load_labels(path, active, g):
 z=np.load(path,allow_pickle=False)
 required=("sample_index","estimated_local_channel","geometry_um","local_to_original_AP")
 if any(k not in z for k in required): raise ValueError("labels NPZ misses public reference-channel fields")
 t=np.asarray(z["sample_index"],np.int64).ravel(); c=np.asarray(z["estimated_local_channel"],np.int64).ravel()
 if not np.array_equal(np.asarray(z["local_to_original_AP"],np.int64),active) or not np.allclose(np.asarray(z["geometry_um"],float),g): raise ValueError("labels and AP geometry disagree")
 return t,c

def public_labels(archive, stop):
 """Derive public KS4 event times and frozen reference channels from ZIP members."""
 with zipfile.ZipFile(archive) as z:
  with z.open("sim.imec0.ap_params.npz") as f: params=np.load(f,allow_pickle=False); times=np.asarray(params["st"],np.int64); units=np.asarray(params["cl"],np.int64)
 mapping=json.loads((Path(__file__).with_name("_unit_to_main_channel.json")).read_text())["unit_to_main_channel"]
 mapping=np.asarray(mapping,np.int64)
 if mapping.shape!=(1200,) or units.max(initial=-1)>=len(mapping): raise ValueError("invalid frozen unit reference map")
 keep=times<stop
 return times[keep],mapping[units[keep]]

# Exact archived range-selection helpers; imported after the public loader so the
# CLI has no dependency on the historical workspace.
def canonical_geometry_sets(g):
    d=np.linalg.norm(g[:,None]-g[None,:],axis=2); ix=np.arange(len(g))
    near=[np.flatnonzero((d[i]<=40.)&(ix!=i)).astype(np.int32) for i in ix]
    ring=[np.flatnonzero((d[i]>40.)&(d[i]<=80.)).astype(np.int32) for i in ix]
    def pad(groups):
        w=max(map(len,groups)); a=np.full((len(groups),w),-1,np.int32); n=np.zeros(len(groups),np.int32)
        for i,v in enumerate(groups): a[i,:len(v)]=v;n[i]=len(v)
        return a,n
    ni,nc=pad(near);ri,rc=pad(ring);return ni,nc,ri,rc,d<=40.

def canonical_detect(q, e, delta, offsets, h, g, lo, hi):
    from . import _kernels as k
    ni,nc,ri,rc,near=canonical_geometry_sets(g); x=k.spatial_input(q,delta,offsets,4524.30029296875/31.)
    low=.75*h; pp,cc=k.peak_indices_range(e,low,6,10,12,10,0,lo,hi)
    pool=k.extrema_range(x,e,pp,cc,ni,nc,ri,rc,6,10,6,0,k.descriptor_dtype(ni.shape[1]))
    scale=delta/(4524.30029296875/31.)
    strong,weak=k.masks(pool,h,scale,ni,nc,.75,2,2.,0,2.,2.,3.,1.,0)
    eligible=np.flatnonzero(strong|weak); competing=pool[eligible]
    sk,wk=k.nms_sorted(competing,strong[eligible],weak[eligible],near,6)
    events=k.events_from(competing,sk,wk,dict(b=6,a=10,right=10,coh=6),dict(nms=6))
    guard=(events['sample_index']>=lo)&(events['sample_index']+60<hi)
    return events[guard],dict(pool_candidates=int(len(pool)),strong=int(sk.sum()),weak=int(wk.sum()))

def canonical_score(events,times,channels,g,lo,hi):
    from ._matching import prepare_gt_index,spatial_pairs_fast
    events=events[(events['sample_index']>=lo)&(events['sample_index']+60<hi)]; keep=(times>=lo)&(times<hi); idx=prepare_gt_index(times[keep],channels[keep],g,40.); pairs=spatial_pairs_fast(events['sample_index'],events['channel_index'],idx,30);tp=len(pairs);fp=len(events)-tp;fn=int(keep.sum())-tp
    return {'TP':int(tp),'FP':int(fp),'FN':int(fn),'detections':int(len(events)),'labels':int(keep.sum()),'A':tp/(tp+fp+fn),'precision':tp/len(events) if len(events) else 0.,'recall':tp/int(keep.sum()),'F1':2*tp/(2*tp+fp+fn),'FP_per_s':fp/((hi-lo)/FS)}

def _sorted_pool(pool):
    return pool[np.lexsort((pool['ch'],pool['p'],pool['t']))]

def _threshold(e_cal):
    med=np.median(e_cal,axis=0); mad=np.median(np.abs(e_cal-med),axis=0)
    return np.maximum(1.,med+12.9*np.maximum(mad,.125)),{'med':med,'mad':mad}

def _extract_part(x,e,low,origin,start,stop,sets):
    from . import _kernels as k
    ni,nc,ri,rc,_=sets
    pp,cc=k.peak_indices_range(e,low,6,10,12,10,origin,start,stop)
    return k.extrema_range(x,e,pp,cc,ni,nc,ri,rc,6,10,6,origin,k.descriptor_dtype(ni.shape[1]))

def _pool_stream(factory, stop, cfg, h, sets):
    """One sequential pass; only fixed-K eligible descriptors are retained."""
    from . import _kernels as k
    state=cfg['initial'].copy();pq=cfg['initial'].copy();pe=np.zeros(len(cfg['initial']))
    pending=tail_x=tail_e=None;parts=[];raw_count=kept_count=0; ni,nc,_,_,_=sets; scale=cfg['delta']/(4524.30029296875/31.)
    for start,raw in factory.chunks(stop):
        q,state,_=k.encode_range(raw,cfg['delta'],cfg['offset_codes'],state)
        e,pq,pe=k.feature_range(q,cfg['offset_codes'],.875,pq,pe)
        x=k.spatial_input(q,cfg['delta'],cfg['offset_codes'],4524.30029296875/31.)
        if pending is not None:
            lo,px,pe0=pending; tx=tail_x if tail_x is not None else px[:0];te=tail_e if tail_e is not None else pe0[:0]
            xx=np.concatenate((tx,px,x[:80])); ee=np.concatenate((te,pe0,e[:80])); origin=lo-len(tx)
            z=_extract_part(xx,ee,.75*h,origin,max(lo,origin+80),lo+len(px),sets);raw_count+=len(z)
            strong,weak=k.masks(z,h,scale,ni,nc,.75,2,2.,0,2.,2.,3.,1.,0);central=(z['ptp']>=2*scale[z['ch']])&(z['tv']<=3*z['ptp']+scale[z['ch']])&(z['e']>=h[z['ch']]);z=z[strong|weak|central];kept_count+=len(z)
            if len(z): parts.append(z)
            tail_x,tail_e=px[-80:].copy(),pe0[-80:].copy()
        pending=(start,x,e)
    if pending is not None:
        lo,px,pe0=pending; xx=np.concatenate((tail_x,px));ee=np.concatenate((tail_e,pe0));z=_extract_part(xx,ee,.75*h,lo-len(tail_x),lo,stop,sets);raw_count+=len(z)
        strong,weak=k.masks(z,h,scale,ni,nc,.75,2,2.,0,2.,2.,3.,1.,0);central=(z['ptp']>=2*scale[z['ch']])&(z['tv']<=3*z['ptp']+scale[z['ch']])&(z['e']>=h[z['ch']]);z=z[strong|weak|central];kept_count+=len(z)
        if len(z):parts.append(z)
    return _sorted_pool(np.concatenate(parts) if parts else np.empty(0,k.descriptor_dtype(sets[0].shape[1]))),{'raw_candidates':raw_count,'eligible_union':kept_count}

def _events(pool,h,sets,neighbor=True):
    from . import _kernels as k
    ni,nc,_,_,near=sets;scale=np.ones(len(h)) # overwritten from caller below
    if neighbor:
        scale=_EVENT_SCALE
        strong,weak=k.masks(pool,h,scale,ni,nc,.75,2,2.,0,2.,2.,3.,1.,0)
    else:
        scale=_EVENT_SCALE; z=pool; strong=(z['ptp']>=2*scale[z['ch']])&(z['tv']<=3*z['ptp']+scale[z['ch']])&(z['e']>=h[z['ch']]);weak=np.zeros(len(pool),bool)
    ix=np.flatnonzero(strong|weak); comp=pool[ix];sk,wk=k.nms_sorted(comp,strong[ix],weak[ix],near,6)
    return k.events_from(comp,sk,wk,dict(b=6,a=10,right=10,coh=6),dict(nms=6)),dict(eligible=int(len(ix)),strong=int(sk.sum()),weak=int(wk.sum()))

def run_full(data_dir, output):
    global _EVENT_SCALE
    paths=dataset_paths(data_dir);out=Path(output);out.mkdir(parents=True,exist_ok=False);header=zip_text(paths['zip'],'sim.imec0.ap.ch');active,g=geometry(zip_text(paths['zip'],'sim.imec0.ap.meta'))
    class Factory:
        def chunks(self,stop): return stream_cbin(paths['zip'],header,active,stop)
    factory=Factory();cal=np.empty((CAL,len(active)),np.int16)
    for start,x in factory.chunks(CAL):cal[start:start+len(x)]=x.astype(np.int16)
    full=4524.30029296875
    from . import _kernels as k
    cfg=k.ranges_from_cal(cal.min(0),cal.max(0),active,full,'bank8_power2',1.25);qcal,_,_=k.encode_range(cal,cfg['delta'],cfg['offset_codes'],cfg['initial']);ecal,_,_=k.feature_range(qcal,cfg['offset_codes'],.875,cfg['initial'].copy(),np.zeros(len(active)));h,stats=_threshold(ecal);del qcal,ecal
    sets=canonical_geometry_sets(g);_EVENT_SCALE=cfg['delta']/(full/31.);pool,diag=_pool_stream(factory,STOP,cfg,h,sets)
    times,channels=public_labels(paths['zip'],STOP) if paths['labels'] is None else load_labels(paths['labels'],active,g)
    on,ond=_events(pool,h,sets,True);off,offd=_events(pool,h,sets,False);onm=canonical_score(on,times,channels,g,DEV,STOP);offm=canonical_score(off,times,channels,g,DEV,STOP)
    on=on[(on['sample_index']>=DEV)&(on['sample_index']+60<STOP)];off=off[(off['sample_index']>=DEV)&(off['sample_index']+60<STOP)]
    np.savez_compressed(out/'lc6_bank8_neighbor_events.npz',events=on);np.savez_compressed(out/'lc6_bank8_no_neighbor_events.npz',events=off);np.savez_compressed(out/'calibration.npz',H=h,med=stats['med'],mad=stats['mad'],delta=cfg['delta'])
    result={'status':'COMPLETE','protocol':{'CAL':[0,10],'DEV':[10,20],'TEST':[20,300],'guard':60,'time_tolerance':30,'spatial_um':40},'frozen_config':FROZEN,'input':{'zip_sha256':'not_computed; public Figshare file id 44729857','full_scale':full,'channels':len(active)},'pool':diag,'neighbor':{**onm,**ond,'event_sha256':hashlib.sha256((out/'lc6_bank8_neighbor_events.npz').read_bytes()).hexdigest()},'no_neighbor':{**offm,**offd,'event_sha256':hashlib.sha256((out/'lc6_bank8_no_neighbor_events.npz').read_bytes()).hexdigest()}}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result


def run(data_dir, output):
    return run_full(data_dir, output)
