"""Frozen Neuropixels range, feature, descriptor, and NMS kernels."""
import numpy as np
from numba import njit

EVENT = np.dtype([('sample_index', '<i8'), ('channel_index', '<i4'), ('feature_peak', '<i8'), ('amplitude', '<f8'), ('availability_sample', '<i8')])

def descriptor_dtype(max_neighbors):
    return np.dtype([('p','<i8'),('t','<i8'),('ch','<i4'),('amp','<f8'),('e','<f8'),('ptp','<f8'),('tv','<f8'),('near','<f8'),('ring','<f8'),('refs','<i2'),('prom','<f8'),('phase','?'),('emax','<f8',(int(max_neighbors),))])


def ranges_from_cal(cal_min, cal_max, ap_ids, full0, mode, margin):
    """Return the frozen global or bank-of-eight power-of-two LC ranges."""
    cal_min=np.asarray(cal_min,np.float64)
    cal_max=np.asarray(cal_max,np.float64)
    ap_ids=np.asarray(ap_ids,np.int64)
    bank=ap_ids//8
    if mode=='global':
        requested_lo=np.full(len(ap_ids),-full0)
        requested_hi=np.full(len(ap_ids),full0)
        delta=np.full(len(ap_ids),full0/31.)
    elif mode=='bank8_power2':
        maximum=np.maximum(np.abs(cal_min),np.abs(cal_max))
        maximum=np.asarray([maximum[bank==k].max() for k in bank])
        requested_hi=margin*maximum
        requested_lo=-requested_hi
        halfspan=np.clip(requested_hi,full0/4.,2.*full0)
        halfspan=full0*np.power(2.,np.ceil(np.log2(halfspan/full0)))
        delta=halfspan/31.
    else:
        raise ValueError("mode must be 'global' or 'bank8_power2'")
    center=np.zeros(len(ap_ids))
    offset_codes=np.zeros(len(ap_ids),np.int16)
    initial=-offset_codes
    required_delta=(requested_hi-requested_lo)/62.
    floor_active=required_delta<full0/124.
    return dict(delta=delta,center=center,initial=initial,offset_codes=offset_codes,
                required_delta=required_delta,bank=bank,
                lower=center-31.*delta,upper=center+31.*delta,
                requested_lower=requested_lo,requested_upper=requested_hi,
                floor_active=floor_active,
                ceiling_active=required_delta>2.*full0/31.,
                requested_lower_covered=center-31.*delta<=requested_lo,
                requested_upper_covered=center+31.*delta>=requested_hi,
                lower_margin_shortfall_counts=np.maximum(center-31.*delta-requested_lo,0.),
                upper_margin_shortfall_counts=np.maximum(requested_hi-center-31.*delta,0.),
                substantive_lower_margin_loss=center-31.*delta-requested_lo>1.e-10,
                substantive_upper_margin_loss=requested_hi-center-31.*delta>1.e-10)


@njit(cache=True)
def encode_range(x, delta, offset_codes, initial):
    state=initial.copy()
    q=np.empty(x.shape,np.int8)
    counters=np.zeros((x.shape[1],5),np.int64)
    for c in range(x.shape[1]):
        for n in range(len(x)):
            requested=int(np.trunc((float(x[n,c])-(int(state[c])+int(offset_codes[c]))*delta[c])/delta[c]))
            counters[c,0]+=abs(requested)>8
            step=max(-8,min(8,requested))
            nxt=max(-31,min(31,int(state[c])+step))
            counters[c,1]+=nxt!=int(state[c])+step
            actual=nxt-int(state[c])
            counters[c,2]+=actual!=0
            counters[c,3]+=abs(actual)
            counters[c,4]+=abs(nxt)==31
            q[n,c]=nxt
            state[c]=nxt
    return q,state,counters


@njit(cache=True)
def feature_range(q, offset_codes, rho, previous_q, previous_e):
    result=np.empty(q.shape,np.float64)
    lut=np.asarray((0,1,2,2,4,4,8,8,8),np.int64)
    for c in range(q.shape[1]):
        for n in range(len(q)):
            code=int(q[n,c])
            d=code-int(previous_q[c])
            v=float(code)+offset_codes[c]
            drive=0.
            if (v>0 and d>0) or (v<0 and d<0):
                drive=abs(v)*lut[abs(d)]
            previous_e[c]=rho*previous_e[c]+drive
            previous_q[c]=code
            result[n,c]=previous_e[c]
    return result,previous_q,previous_e


def spatial_input(q,delta,offset_codes,delta0):
    """The global arm yields old code values exactly, without multiply/divide."""
    return (q.astype(np.float64)+offset_codes)*(delta/delta0)


@njit(cache=True)
def range_errors(x,q,delta,offset_codes,lower,upper):
    total=np.zeros((x.shape[1],9),np.float64)
    for c in range(x.shape[1]):
        for n in range(len(x)):
            actual=float(x[n,c]);code=int(q[n,c])
            error=actual-(code+int(offset_codes[c]))*delta[c]
            total[c,0]+=error*error
            total[c,1]+=abs(error)
            total[c,2]=max(total[c,2],abs(error))
            total[c,3]+=actual>upper[c]
            total[c,4]+=actual<lower[c]
            total[c,5]+=code==31
            total[c,6]+=code==-31
            total[c,7]+=abs(error)>delta[c]+1.e-10
            total[c,8]+=1
    return total


@njit(cache=True)
def extrema_range(x, e, pp, cc, near, nc, ring, rc, b, a, coh, origin, dtype):
    out = np.empty(len(pp), dtype)
    for i in range(len(pp)):
        p, c = pp[i], cc[i]
        low, high, first = x[p-b, c], x[p-b, c], 0
        tv = 0.
        previous = low
        for d in range(1, b+a+1):
            v = x[p-b+d, c]
            tv += abs(v-previous)
            previous = v
            if v < low:
                low, first = v, d
            high = max(high, v)
        t = p-b+first
        near_ptp = 0.
        out[i]['emax'][:] = 0.
        for j in range(nc[c]):
            other = near[c, j]
            lo, hi, em = x[p-b, other], x[p-b, other], 0.
            for d in range(1, b+a+1):
                v = x[p-b+d, other]
                lo, hi = min(lo, v), max(hi, v)
            near_ptp = max(near_ptp, hi-lo)
            for d in range(-coh, coh+1):
                if x[p+d, other] < 0:
                    em = max(em, e[p+d, other])
            out[i]['emax'][j] = em
        best, second = 0., 0.
        ref_values = np.empty(rc[c], np.float64)
        for j in range(rc[c]):
            other = ring[c, j]
            lo, hi = x[p-b, other], x[p-b, other]
            for d in range(1, b+a+1):
                v = x[p-b+d, other]
                lo, hi = min(lo, v), max(hi, v)
            value = hi-lo
            if value >= best:
                second, best = best, value
            elif value > second:
                second = value
            ref_values[j] = x[t, other]
        prom = np.median(ref_values)-low if rc[c] >= 2 else 1.e10
        out[i]['p'], out[i]['t'], out[i]['ch'] = origin+p, origin+t, c
        out[i]['amp'], out[i]['e'] = low, e[p, c]
        out[i]['ptp'], out[i]['tv'] = high-low, tv
        out[i]['near'], out[i]['ring'], out[i]['refs'] = near_ptp, second, rc[c]
        out[i]['prom'], out[i]['phase'] = prom, x[p, c] < 0
    return out



@njit(cache=True)
def peak_indices_range(e, low, b, a, left, right, origin, start, stop):
    capacity=(len(e)+min(left,right))//(min(left,right)+1)
    locations=np.empty((e.shape[1],capacity),np.int64)
    count=np.zeros(e.shape[1],np.int64)
    for c in range(e.shape[1]):
        for p in range(max(b,left),len(e)-max(a,right)):
            if not start<=origin+p<stop or e[p,c]<low[c]:
                continue
            good=True
            for d in range(1,left+1):
                if e[p,c]<=e[p-d,c]:
                    good=False
                    break
            if good:
                for d in range(1,right+1):
                    if e[p,c]<e[p+d,c]:
                        good=False
                        break
            if good:
                assert count[c]<capacity
                locations[c,count[c]]=p
                count[c]+=1
    total=int(count.sum())
    pp=np.empty(total,np.int64);cc=np.empty(total,np.int32);n=0
    for c in range(e.shape[1]):
        for j in range(count[c]):
            pp[n]=locations[c,j];cc[n]=c;n+=1
    return pp,cc


@njit(cache=True)
def masks(pool, h, scale, ns, nc, alpha, count, contrast, policy, beta, pmin, kappa, promfloor, phase):
    strong=np.zeros(len(pool),np.bool_); weak=strong.copy()
    for i in range(len(pool)):
        z=pool[i]; c=z['ch']
        if z['ptp']<pmin*scale[c] or z['tv']>kappa*z['ptp']+scale[c]: continue
        if promfloor>0 and z['refs']>=2 and z['prom']<promfloor*scale[c]: continue
        if phase and not z['phase']: continue
        support=0
        for j in range(nc[c]): support+=int(z['emax'][j]>=.5*h[ns[c,j]])
        sb=z['e']>=h[c]
        if sb:
            passed=True
            if z['e']<beta*h[c]:
                if policy==0: passed=z['near']>=2*scale[c]
                elif policy==1: passed=support>=1
                elif policy==2: passed=support>=1 and z['near']>=2*scale[c]
            strong[i]=passed
        elif z['e']>=alpha*h[c] and support>=count:
            weak[i]=(z['refs']>=2 and z['ptp']-z['ring']>=contrast*scale[c])
    return strong,weak

@njit(cache=True)
def better(pool,j,i):
    if pool[j]['amp']!=pool[i]['amp']: return pool[j]['amp']<pool[i]['amp']
    if pool[j]['t']!=pool[i]['t']: return pool[j]['t']<pool[i]['t']
    if pool[j]['ch']!=pool[i]['ch']: return pool[j]['ch']<pool[i]['ch']
    if pool[j]['p']!=pool[i]['p']: return pool[j]['p']<pool[i]['p']
    return j<i

@njit(cache=True)
def nms_sorted(pool,strong,weak,near,radius):
    sk,wk=strong.copy(),weak.copy();lo=0;hi=0
    for i in range(len(pool)):
        if not strong[i] and not weak[i]: continue
        while lo<len(pool) and pool[lo]['t']<pool[i]['t']-radius: lo+=1
        hi=max(hi,i+1)
        while hi<len(pool) and pool[hi]['t']<=pool[i]['t']+radius: hi+=1
        for j in range(lo,hi):
            if j==i or not near[pool[i]['ch'],pool[j]['ch']]: continue
            if better(pool,j,i):
                if strong[i] and strong[j]: sk[i]=False;break
                if weak[i] and weak[j]: wk[i]=False;break
    lo=0;hi=0
    for i in range(len(pool)):
        if not wk[i]: continue
        while lo<len(pool) and pool[lo]['t']<pool[i]['t']-radius: lo+=1
        hi=max(hi,i+1)
        while hi<len(pool) and pool[hi]['t']<=pool[i]['t']+radius: hi+=1
        for j in range(lo,hi):
            if sk[j] and near[pool[i]['ch'],pool[j]['ch']]: wk[i]=False;break
    return sk,wk

def events_from(pool,sk,wk,cfg,params):
    ix=np.flatnonzero(sk|wk); e=np.empty(len(ix),EVENT)
    for dst,src in [('sample_index','t'),('channel_index','ch'),('feature_peak','p'),('amplitude','amp')]: e[dst]=pool[src][ix]
    e['availability_sample']=e['sample_index']+cfg['b']+max(cfg['a'],cfg['right'],cfg['coh'])+params['nms']*np.where(wk[ix],2,1)
    return e
