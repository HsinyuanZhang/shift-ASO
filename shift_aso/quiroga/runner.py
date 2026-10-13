"""Frozen, standalone Quiroga 12-method timing reproduction."""
from __future__ import annotations
import argparse, csv, json, os, ctypes, hashlib, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfilt, lfilter
from scipy.optimize import linear_sum_assignment

FAMILIES=('Easy1','Easy2','Difficult1','Difficult2'); NOISES=('005','01','015','02')
TEST=(18.002,59.998); SHIFT=23; RHO=.875; ARMS=('raw_tol2_rawmask2','raw_tol1_rawmask2','peakminus23_tol1_rawmask2','peakminus23_tol1_correctedmask1')
ASO=(('Shift_ASO_LC4',4,.5),('Shift_ASO_LC5',5,.5),('Shift_ASO_LC6',6,.5),('Ordinary_ASO_uniform4',4,.75),('Ordinary_ASO_uniform5',5,.5),('Ordinary_ASO_uniform6',6,.5),('Ordinary_ASO_full_precision',None,.5),('Exact_product_ASO_LC6',6,.5))

def cells(v, field=0):
 a=np.asarray(v)
 if a.dtype != object:return a
 a=np.asarray(a).ravel()[field]
 while a.dtype==object:a=np.asarray(a).ravel()[0]
 return a

def load(path):
 m=loadmat(path,squeeze_me=False,struct_as_record=False); x=np.asarray(m['data'],float).ravel(); gt=np.asarray(cells(m['spike_times']),float).ravel().astype(np.int64)-1
 labels=np.asarray(cells(m['spike_class']),np.int64).ravel(); sp=np.asarray(m['spike_class']).ravel(); overlap=np.asarray(sp[1]).ravel()!=0 if sp.size>1 and np.asarray(sp[1]).size==gt.size else np.zeros(gt.size,bool)
 fs=1000/float(np.asarray(m['samplingInterval']).ravel()[0]); y=sosfilt(butter(1,[300.,6000.],btype='bandpass',fs=fs,output='sos'),x); ncal=round(6*fs); full=max(1.1*float(np.max(np.abs(y[:ncal]))),1e-12)
 return dict(stem=path.stem,fs=fs,y=y,gt=gt,overlap=overlap,ncal=ncal,full=full)

_LC_KERNEL=None
def _lc_kernel():
 global _LC_KERNEL
 if _LC_KERNEL is None:
  here=Path(__file__).resolve().parent; src=here/'lc_kernel.cpp'; digest=hashlib.sha256(src.read_bytes()).hexdigest()[:16]; target=Path(os.environ.get('SHIFT_ASO_BUILD_DIR',str(Path.home()/'.cache'/'shift_aso'))); target.mkdir(parents=True,exist_ok=True); so=target/('lc_'+digest+('.dylib' if sys.platform=='darwin' else '.so'))
  if not so.exists():
   temporary=so.with_name(so.name+'.'+str(os.getpid())+'.tmp')
   try:
    subprocess.run(['g++','-O3','-std=c++11','-shared','-fPIC',str(src),'-o',str(temporary)],check=True)
    os.replace(temporary,so)
   finally:
    if temporary.exists(): temporary.unlink()
  fun=ctypes.CDLL(str(so)).encode; dptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS'); iptr=np.ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='C_CONTIGUOUS'); fun.argtypes=[dptr,ctypes.c_int64,ctypes.c_int,ctypes.c_double,iptr,dptr]; _LC_KERNEL=fun
 return _LC_KERNEL
def encode(x,bits,full):
 values=np.ascontiguousarray(x,dtype=np.float64);q=np.empty(values.size,np.int32);r=np.empty(values.size,np.float64);_lc_kernel()(values,values.size,bits,full,q,r);return q,r,full/((1<<(bits-1))-1)

def peak_events(trig,wave,fs):
 before,after=round(.0002*fs),round(.0006*fs); trig=np.asarray(trig,np.int64); trig=trig[(trig>=0)&(trig+after<wave.size)]; p=np.empty(len(trig),np.int64)
 off=np.arange(before+after+1)
 for j in range(0,len(trig),8192):
  t=trig[j:j+8192]; left=np.maximum(0,t-before); p[j:j+len(t)]=left+np.argmax(np.abs(wave[left[:,None]+off]),axis=1)
 return dict(t_trigger=trig,t_peak=p,t_emit=trig+after)

def fixed_detect(score,wave,fs,ncal,level):
 # Exact fixed-level threshold_stream state: rearm below half-level, 1ms refractory.
 last=-10**12; armed=True; out=[]; ref=max(1,round(.001*fs))
 for i,v in enumerate(np.asarray(score)[ncal:],ncal):
  if v<=.5*level:armed=True
  if armed and v>level and i-last>=ref: out.append(i);last=i;armed=False
 return peak_events(out,wave,fs)

def adaptive_detect(score,wave,fs,ncal,kappa):
 s=np.asarray(score,float); init=s[:ncal]; mean=float(np.median(init)); floor=max(1e-15,float(np.mean(init))*.02); dev=max(float(np.median(np.abs(init-mean))),floor); alpha=min(1.,1/(fs*.2)); last=-10**12; blank=0; armed=True; out=[]; ref=max(1,round(.001*fs)); bspan=max(1,round(.001*fs))
 for t,v in enumerate(s[ncal:],ncal):
  threshold=max(floor,mean+kappa*dev)
  if v<=.5*threshold: armed=True
  if armed and v>threshold and t-last>=ref: out.append(t);last=t;blank=t+bspan;armed=False
  if t>=blank:
   err=np.clip(v-mean,-3*max(dev,floor),3*max(dev,floor)); mean=max(0.,mean+alpha*err); dev=max(floor,dev+alpha*(abs(err)-dev))
 return peak_events(out,wave,fs)

def threshold_events(wave,fs,ncal,threshold):
 hits=np.flatnonzero(np.abs(wave[ncal:])>threshold).astype(np.int64)+ncal; keep=[]; last=-10**12; gap=round(.001*fs)
 for t in hits:
  if t-last>=gap: keep.append(int(t));last=int(t)
 return peak_events(keep,wave,fs)
def tc_detect(wave,fs,ncal,k):
 sigma=float(np.median(np.abs(wave[:ncal]))/.6744897501960817); return threshold_events(wave,fs,ncal,k*max(sigma,1e-15))

def ema(x): return lfilter([1.],[1.,-RHO],np.asarray(x,float))
def classic(x):
 prev=np.r_[0.,x[:-1]];return np.maximum(x*(x-prev),0.)
def aso_code(q,d,shift=False):
 q=np.asarray(q,np.int32); st=np.r_[q[0],np.diff(q)]; a=(np.sign(st)*np.array([0,1,2,2,4,4,8,8,8],np.int32)[np.minimum(np.abs(st),8)]).astype(np.int32) if shift else st
 return np.maximum(d*d*q.astype(float)*a.astype(float),0.)
def neo(x,fs):
 z=np.zeros(len(x));z[2:]=np.maximum(0,x[1:-1]**2-x[:-2]*x[2:]); a=1-np.exp(-1/(fs*.154403068e-3));return lfilter([a],[1.,-(1-a)],z)

def metric(tp,fp,fn):
 p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.;return dict(TP=int(tp),FP=int(fp),FN=int(fn),A=tp/(tp+fp+fn) if tp+fp+fn else 0.,P=p,R=r,F1=2*p*r/(p+r) if p+r else 0.)
def match(d,g,tol):
 # temporal components and assignment follow the frozen canonical matcher.
 d=np.asarray(d,np.int64);g=np.asarray(g,np.int64)
 if not len(d) or not len(g):return metric(0,len(d),len(g))
 di=np.lexsort((np.arange(len(d)),d));gi=np.lexsort((np.arange(len(g)),g)); pairs=[]; dc=gc=0
 while dc<len(di) or gc<len(gi):
  ds,gs=dc,gc
  if gc==len(gi) or (dc<len(di) and d[di[dc]]<=g[gi[gc]]): de,ge=dc+1,gc; df=(dc,de);gf=None
  else: de,ge=dc,gc+1;df=None;gf=(gc,ge)
  while df is not None or gf is not None:
   if df is not None:
    l,r=df; lo=d[di[l]]-tol;hi=d[di[r-1]]+tol; ne=np.searchsorted(g[gi],hi,'right');gf=(ge,ne) if ne>ge else None;ge=max(ge,ne);df=None
   if gf is not None:
    l,r=gf; lo=g[gi[l]]-tol;hi=g[gi[r-1]]+tol; ne=np.searchsorted(d[di],hi,'right');df=(de,ne) if ne>de else None;de=max(de,ne);gf=None
  dd,gg=di[ds:de],gi[gs:ge]
  if len(dd) and len(gg):
   er=np.abs(d[dd,None]-g[None,gg]); pen=(tol+1)*(len(dd)+len(gg)+1); size=len(dd)+len(gg); cost=np.full((size,size),4*pen); valid=er<=tol;cost[:len(dd),:len(gg)]=np.where(valid,er,4*pen); cost[np.arange(len(dd)),len(gg)+np.arange(len(dd))]=pen;cost[len(dd)+np.arange(len(gg)),np.arange(len(gg))]=pen;cost[len(dd):,len(gg):]=0
   rr,cc=linear_sum_assignment(cost);pairs += [(r,c) for r,c in zip(rr,cc) if r<len(dd) and c<len(gg) and valid[r,c]]
  dc,gc=de,ge
 return metric(len(pairs),len(d)-len(pairs),len(g)-len(pairs))

def eligible(item,ev):
 a,b=round(TEST[0]*item['fs']),round(TEST[1]*item['fs']); p=np.asarray(ev['t_peak']);e=np.asarray(ev['t_emit']);return p[(p>=a)&(p<b)&(e<b)],item['gt'][(item['gt']>=a)&(item['gt']<b)]
def overlap(item,p,radius,shift):
 a,b=round(TEST[0]*item['fs']),round(TEST[1]*item['fs']); flag=np.sort(item['gt'][(item['gt']>=a)&(item['gt']<b)&item['overlap']]); gt=item['gt'][(item['gt']>=a)&(item['gt']<b)&~item['overlap']]
 if len(flag):
  q=p+shift;i=np.searchsorted(flag,q);dist=np.full(len(q),np.inf);hi=i<len(flag);lo=i>0;dist[hi]=np.minimum(dist[hi],np.abs(flag[i[hi]]-q[hi]));dist[lo]=np.minimum(dist[lo],np.abs(flag[i[lo]-1]-q[lo]));p=p[dist>radius]
 return p,gt

def score_arms(item,ev):
 p,g=eligible(item,ev);fs=item['fs'];p2,g2=overlap(item,p,round(.002*fs),0);p1,g1=overlap(item,p,round(.001*fs),-SHIFT);t=round(.001*fs)
 return {'raw_tol2_rawmask2':match(p2,g2,round(.002*fs)),'raw_tol1_rawmask2':match(p2,g2,t),'peakminus23_tol1_rawmask2':match(p2-SHIFT,g2,t),'peakminus23_tol1_correctedmask1':match(p1-SHIFT,g1,t)}
def score_tc_arms(item,ev):
 x=score_arms(item,ev);return {'raw_mask2_2ms':x['raw_tol2_rawmask2'],'raw_mask2_1ms':x['raw_tol1_rawmask2'],'peak_minus23_rawmask2_1ms':x['peakminus23_tol1_rawmask2'],'peak_minus23_correctedmask1_1ms':x['peakminus23_tol1_correctedmask1']}

def worker(arg):
 family,noise,fn=arg; it=load(Path(fn)); out=[]; codes={}
 for b in (4,5,6):codes[b]=encode(it['y'],b,it['full'])
 for name,b,k in ASO:
  if name.startswith('Shift'): q,_,d=codes[b]; f=ema(aso_code(q,d,True)); src='frozen k=0.5, LC/J8, rho=0.875'
  elif name.startswith('Ordinary_ASO_uniform'): q,_,d=codes[b]; f=ema(classic(np.clip(np.rint(it['y']/d),-(1<<(b-1))+1,(1<<(b-1))-1)*d));src='frozen uniform ASO'
  elif name=='Ordinary_ASO_full_precision':f=ema(classic(it['y']));src='frozen full-precision ASO'
  else:q,_,d=codes[6];f=ema(aso_code(q,d,False));src='frozen LC6 exact-product ASO'
  lev=max(k*float(np.quantile(f[:it['ncal']],.99)),1e-18); ev=fixed_detect(f,it['y'],it['fs'],it['ncal'],lev)
  for arm,z in score_arms(it,ev).items():out.append(dict(stem=it['stem'],family=family,noise=noise,method=name,bits='' if b is None else b,k=k,arm=arm,level=lev,**z,parameter_source=src))
 # TC/NEO definitions frozen from current evidence.
 q,xhat,d=codes[6];xhat=xhat-np.median(xhat[:it['ncal']]); sigma=max(float(np.median(np.abs(xhat[:it['ncal']]))/.6744897501960817),d/np.sqrt(12))
 controls=[('raw_tc','original',tc_detect(it['y'],it['fs'],it['ncal'],4.5),'TC MAD kappa=4.5'),('lc_tc',6,threshold_events(xhat,it['fs'],it['ncal'],5*sigma),'TC quantum-floor kappa=5'),('raw_neo','original',adaptive_detect(neo(it['y'],it['fs']),it['y'],it['fs'],it['ncal'],20),'NEO EMA kappa=20'),('lc_neo',6,adaptive_detect(neo(xhat,it['fs']),xhat,it['fs'],it['ncal'],20),'LC NEO EMA kappa=20')]
 for name,b,ev,src in controls:
  for arm,z in score_tc_arms(it,ev).items():out.append(dict(stem=it['stem'],family=family,noise=noise,method=name,bits=b,kappa=4.5 if name=='raw_tc' else 5 if name=='lc_tc' else 20,arm=arm,**z,parameter_source=src))
 return out

def aggregate(rows):
 tp=sum(r['TP'] for r in rows);fp=sum(r['FP'] for r in rows);fn=sum(r['FN'] for r in rows);m=metric(tp,fp,fn);return {'records':len(rows),'macro':{k:float(np.mean([r[k] for r in rows])) for k in ('A','P','R','F1')},'pooled':m,'worst_A':float(min(r['A'] for r in rows))}
def run(data_dir,output,workers=8):
 data=Path(data_dir);output=Path(output);output.mkdir(parents=True,exist_ok=True);tasks=[(f,n,str(data/f'C_{f}_noise{n}.mat')) for f in FAMILIES for n in NOISES]
 if any(not Path(t[2]).is_file() for t in tasks): raise FileNotFoundError('data-dir must contain the 16 native C_*_noise*.mat Quiroga records')
 with ProcessPoolExecutor(max_workers=workers) as ex: rows=[r for part in ex.map(worker,tasks) for r in part]
 rows.sort(key=lambda r:(r['method'],r['stem'],r['arm'])); fields=list(dict.fromkeys(k for r in rows for k in r));
 with (output/'per_record.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 groups={}
 for r in rows:groups.setdefault((r['method'],str(r['bits']),r['arm']),[]).append(r)
 summary={'status':'complete','contract':{'calibration_s':[0,6],'development_s':[6.002,18],'test_s':list(TEST),'main_arm':'peakminus23_tol1_correctedmask1','peak_shift_samples':SHIFT,'overlap':'original eligibility; corrected-coordinate 1ms mask for main arm','no_parameter_search':True},'methods':{ '|'.join(k):aggregate(v) for k,v in sorted(groups.items())}}
 (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args();
 os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');run(a.data_dir,a.output,a.workers)
if __name__=='__main__':main()
