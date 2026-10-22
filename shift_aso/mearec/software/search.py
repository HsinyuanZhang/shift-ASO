"""Small DEV-only search, with stable IDs and exact GROUP-local safe pruning.

Only the true coarse winner sets the refinement bracket. Neither TEST labels
nor historical result tables are used to propose or select configurations.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import numpy as np
from .api import validate
from .features import activity, cal_stat
from .detect import peak_pool, activate, eligible
from .grid import LAGS, EMA_RHOS, COARSE_K
from .scoring import Scorer


def rank(metrics, config_id):
    denominator = metrics['TP']+metrics['FP']+metrics['FN']
    return (Fraction(metrics['TP'], max(1,denominator)), -metrics['FP'], -config_id)


def refinement_points(coarse, winner, count=3):
    distinct = sorted(set(coarse)); position = distinct.index(winner)
    lower = distinct[max(0,position-1)]; upper = distinct[min(len(distinct)-1,position+1)]
    if not upper > lower:
        return []
    points = np.geomspace(lower,upper,count+2)[1:-1] if lower>0 else np.linspace(lower,upper,count+2)[1:-1]
    return [float(k) for k in points if lower < k < upper and float(k) not in distinct]


def search(signal, geometry, times, channels, *, cal_samples=120000, dev_stop=400000,
           pruning=True, seed_k=8.):
    """Select plain and EMA ONLY on [cal_samples,dev_stop).

    Large inputs should be NPY memmaps. This function reads at most the DEV
    prefix and holds only one dense energy matrix at a time. Status rows may
    be pruned/aliased; never replace unknown metrics with zeros.
    """
    raw = np.asarray(signal[:dev_stop])
    if len(raw) != dev_stop or not 2*max(LAGS) < cal_samples < dev_stop:
        raise ValueError('signal does not cover valid CAL and DEV intervals')
    scorer = Scorer(times,channels,geometry,(cal_samples,dev_stop))
    rows, groups, selected, aliases = [], [], {}, {}
    for lag_index,lag in enumerate(LAGS):
        for rho_index,rho in enumerate((0.,)+EMA_RHOS):
            group = lag_index*4+rho_index
            method = 'plain' if rho==0 else 'ema'
            energy,_ = activity(raw,lag,rho)
            base = cal_stat(energy,cal_samples,lag)
            pool = peak_pool(energy,geometry,lag)
            del energy
            winner = None

            def evaluate(k, config_id, stage):
                nonlocal winner
                cfg = validate(method,lag,rho,k)
                cfg.update(config_id=config_id,group=group,stage=stage)
                H = k*base
                events = eligible(activate(pool,raw,H,geometry),cal_samples,dev_stop)
                row = dict(configuration=cfg, n_events=len(events), metrics=None)
                key = hashlib.sha256(np.ascontiguousarray(events['sample_index']).tobytes()
                    + np.ascontiguousarray(events['channel_index']).tobytes()).digest()
                if key in aliases:
                    metrics,pairs,origin = aliases[key]
                    row.update(status='SCORE_ALIAS_TIMES_CHANNELS',metrics=metrics,alias_of=origin)
                else:
                    if pruning and winner is not None:
                        cap = min(scorer.count,scorer.feasible_count(events))
                        upper = Fraction(cap,max(1,scorer.count+max(0,len(events)-cap)))
                        if upper < rank(winner['metrics'],winner['configuration']['config_id'])[0]:
                            row.update(status='PRUNED_OPTIMISTIC_BOUND',A_upper=[upper.numerator,upper.denominator],
                                       prune_scope='group')
                            rows.append(row)
                            return
                    metrics,pairs = scorer.score(events)
                    aliases[key] = (metrics,pairs,config_id)
                    row.update(status='SCORED',metrics=metrics)
                rows.append(row)
                evidence = dict(configuration=cfg,cal_stat=base,H=H,events=events,pairs=pairs,metrics=metrics)
                if winner is None or rank(metrics,config_id)>rank(winner['metrics'],winner['configuration']['config_id']):
                    winner = evidence
                if method not in selected or rank(metrics,config_id)>rank(selected[method]['metrics'],selected[method]['configuration']['config_id']):
                    selected[method] = evidence

            coarse = [float(k) for k in COARSE_K]
            seed = min(range(len(coarse)),key=lambda i:(abs(coarse[i]-seed_k),i))
            for index in [seed]+[i for i in range(len(coarse)) if i!=seed]:
                evaluate(coarse[index],group*10000+index,'coarse')
            evaluated = list(coarse)
            for index,k in enumerate(refinement_points(coarse,winner['configuration']['K'])):
                evaluate(k,group*10000+5000+index,'refine')
                evaluated.append(k)
            positive = base[pool['channel_index']] > 0
            ratios = pool['amplitude'][positive]/base[pool['channel_index'][positive]]
            zero_bound = float(np.nextafter(np.max(ratios),np.inf)) if len(ratios) else 0.
            if not np.isfinite(zero_bound):
                raise RuntimeError('nonfinite positive-stat no-output threshold bound')
            extension_index = 0
            while winner['configuration']['K']==max(evaluated) and len(winner['events'])>0 and np.any(base>0) and max(evaluated)<zero_bound:
                maximum = max(evaluated); doubled = 2*maximum
                if not np.isfinite(doubled) or doubled <= maximum:
                    raise RuntimeError('cannot extend a still-winning endpoint')
                points = list(np.geomspace(maximum,doubled,5)[1:-1])+[doubled]
                points = sorted(set(min(float(k),zero_bound) for k in points if min(float(k),zero_bound)>maximum))
                if not points:
                    raise RuntimeError('cannot extend a still-winning endpoint')
                for k in points:
                    if extension_index>=4000:
                        raise RuntimeError('reserved extension candidate IDs exhausted')
                    evaluate(k,group*10000+6000+extension_index,'upper_extension')
                    evaluated.append(k);extension_index+=1
            groups.append(dict(group=group, winner_config_id=winner['configuration']['config_id'],
                               K=winner['configuration']['K']))
    return dict(selected=selected,rows=rows,groups=groups)
