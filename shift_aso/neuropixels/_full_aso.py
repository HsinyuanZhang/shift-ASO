"""Frozen full-resolution exact-ASO Neuropixels control."""
from __future__ import annotations
import numpy as np
from . import _kernels as k

FS=30000
DELTA0_DIVISOR=31.
CONFIG={'rho':0.,'family':'q50q75','K':5.076916772249668,'alpha':.8468820381570884,'count':1,'contrast':.21724201545480137,'policy':0,'beta':3.373855347490842,'pmin':5.642666963421676,'kappa':4.514990358231777,'promfloor':1.6879976384228152,'phase':0,'nms':6}
WINDOW={'b':6,'a':10,'left':12,'right':10,'coh':6}

def _sets(geometry):
 g=np.asarray(geometry,float);d=g[:,None]-g[None,:];distance=np.einsum('ijk,ijk->ij',d,d);ix=np.arange(len(g));near=[np.flatnonzero((distance[i]<=1600.)&(ix!=i)).astype(np.int32) for i in ix];ring=[np.flatnonzero((distance[i]>1600.)&(distance[i]<=6400.)).astype(np.int32) for i in ix]
 def pad(groups):
  width=max(len(v) for v in groups);out=np.full((len(groups),width),-1,np.int32);count=np.zeros(len(groups),np.int32)
  for i,v in enumerate(groups):out[i,:len(v)]=v;count[i]=len(v)
  return out,count
 ni,nc=pad(near);ri,rc=pad(ring);return ni,nc,ri,rc,distance<=1600.

def _feature(x,previous):
 out=np.empty(x.shape,np.float64)
 for n in range(len(x)):
  v=x[n];out[n]=np.maximum(v*(v-previous),0.);previous=v.copy()
 return out,previous

def _stats(cal,full):
 delta0=float(full)/DELTA0_DIVISOR; raw=np.asarray(cal,dtype=np.float64)/delta0;feature,_=_feature(raw,np.zeros(raw.shape[1]))
 med=np.median(feature,axis=0);iqr=np.quantile(feature,.75,axis=0)-med;sigma=np.maximum(np.median(np.abs(raw-np.median(raw,axis=0)),axis=0)/.6744897501960817,1e-12)
 scale=np.maximum(iqr,np.maximum(sigma*sigma*1e-6,1e-12));h=np.maximum(np.maximum(sigma*sigma*1e-6,1e-12),med+CONFIG['K']*scale)
 return delta0,sigma,med,iqr,h

def _pool(stream_factory,stop,geometry,delta0,h,sigma):
 ni,nc,ri,rc,near=_sets(geometry);dtype=k.descriptor_dtype(ni.shape[1]);previous=np.zeros(len(geometry));pending=None;tail_x=tail_e=None;parts=[]
 for start,raw in stream_factory.chunks(stop):
  x=np.asarray(raw,dtype=np.float64)/delta0;e,previous=_feature(x,previous)
  if pending is not None:
   lo,px,pe=pending;tx=tail_x if tail_x is not None else px[:0];te=tail_e if tail_e is not None else pe[:0];xx=np.concatenate((tx,px,x[:80]));ee=np.concatenate((te,pe,e[:80]));origin=lo-len(tx)
   pp,cc=k.peak_indices_range(ee,.45*h,WINDOW['b'],WINDOW['a'],WINDOW['left'],WINDOW['right'],origin,max(lo,origin+80),lo+len(px));z=k.extrema_range(xx,ee,pp,cc,ni,nc,ri,rc,WINDOW['b'],WINDOW['a'],WINDOW['coh'],origin,dtype)
   if len(z):
    strong,weak=k.masks(z,h,sigma,ni,nc,CONFIG['alpha'],CONFIG['count'],CONFIG['contrast'],CONFIG['policy'],CONFIG['beta'],CONFIG['pmin'],CONFIG['kappa'],CONFIG['promfloor'],CONFIG['phase'])
    central=(z['ptp']>=CONFIG['pmin']*sigma[z['ch']])&(z['tv']<=CONFIG['kappa']*z['ptp']+sigma[z['ch']])&(z['e']>=h[z['ch']])
    z=z[strong|weak|central]
    if len(z):parts.append(z)
   tail_x,tail_e=px[-80:].copy(),pe[-80:].copy()
  pending=(start,x,e)
 if pending is not None:
  lo,px,pe=pending;tx=tail_x if tail_x is not None else px[:0];te=tail_e if tail_e is not None else pe[:0];xx=np.concatenate((tx,px));ee=np.concatenate((te,pe));origin=lo-len(tx)
  pp,cc=k.peak_indices_range(ee,.45*h,WINDOW['b'],WINDOW['a'],WINDOW['left'],WINDOW['right'],origin,lo,stop);z=k.extrema_range(xx,ee,pp,cc,ni,nc,ri,rc,WINDOW['b'],WINDOW['a'],WINDOW['coh'],origin,dtype)
  if len(z):
   strong,weak=k.masks(z,h,sigma,ni,nc,CONFIG['alpha'],CONFIG['count'],CONFIG['contrast'],CONFIG['policy'],CONFIG['beta'],CONFIG['pmin'],CONFIG['kappa'],CONFIG['promfloor'],CONFIG['phase'])
   central=(z['ptp']>=CONFIG['pmin']*sigma[z['ch']])&(z['tv']<=CONFIG['kappa']*z['ptp']+sigma[z['ch']])&(z['e']>=h[z['ch']])
   z=z[strong|weak|central]
   if len(z):parts.append(z)
 pool=np.concatenate(parts) if parts else np.empty(0,dtype);pool=pool[np.lexsort((pool['ch'],pool['p'],pool['t']))]
 return pool,(ni,nc,near)

def _events(pool,h,sigma,sets,neighbor):
 ni,nc,near=sets;scale=sigma
 if neighbor:
  strong,weak=k.masks(pool,h,scale,ni,nc,CONFIG['alpha'],CONFIG['count'],CONFIG['contrast'],CONFIG['policy'],CONFIG['beta'],CONFIG['pmin'],CONFIG['kappa'],CONFIG['promfloor'],CONFIG['phase'])
 else:
  z=pool;strong=(z['ptp']>=CONFIG['pmin']*scale[z['ch']])&(z['tv']<=CONFIG['kappa']*z['ptp']+scale[z['ch']])&(z['e']>=h[z['ch']]);weak=np.zeros(len(pool),bool)
 ix=np.flatnonzero(strong|weak);comp=pool[ix];sk,wk=k.nms_sorted(comp,strong[ix],weak[ix],near,CONFIG['nms']);return k.events_from(comp,sk,wk,WINDOW,CONFIG),{'eligible':int(len(ix)),'strong':int(sk.sum()),'weak':int(wk.sum())}

def run_full_aso(stream_factory,cal,geometry,full,stop):
 """Run the frozen full-resolution exact-ASO neighbor and no-neighbor arms."""
 delta0,sigma,median,iqr,h=_stats(cal,full);pool,sets=_pool(stream_factory,int(stop),geometry,delta0,h,sigma);on,on_diag=_events(pool,h,sigma,sets,True);off,off_diag=_events(pool,h,sigma,sets,False)
 return {'events':{'fullprecision_exactASO_neighbor':on,'fullprecision_exactASO_no_neighbor':off},'calibration':{'delta0':delta0,'sigma':sigma,'median':median,'iqr':iqr,'H':h},'protocol':{'config':CONFIG,'window':WINDOW,'input':'raw_float64_over_delta0','feature':'max(x[n]*(x[n]-x[n-1]),0), rho=0','candidate_low_alpha':.45},'diagnostics':{'pool_candidates':int(len(pool)),'neighbor':on_diag,'no_neighbor':off_diag}}
