"""Compiled sparse-edge collector with exact NP spatial assignment."""
from __future__ import annotations
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, min_weight_full_bipartite_matching

@njit(cache=True)
def _edge_counts(dt, dc, times, offsets, neighbors, ncount, tol):
    out=np.zeros(len(dt),np.int64)
    for i in range(len(dt)):
        c=dc[i]
        if c<0 or c>=len(offsets)-1: continue
        for k in range(ncount[c]):
            ch=neighbors[c,k]; a=offsets[ch]; b=offsets[ch+1]
            lo=np.searchsorted(times[a:b],dt[i]-tol,side='left')+a
            hi=np.searchsorted(times[a:b],dt[i]+tol,side='right')+a
            out[i]+=hi-lo
    return out

@njit(cache=True)
def _edge_fill(dt, dc, times, original, offsets, neighbors, ncount, tol, starts, rows, cols, costs):
    for i in range(len(dt)):
        p=starts[i]; c=dc[i]
        if c<0 or c>=len(offsets)-1: continue
        for k in range(ncount[c]):
            ch=neighbors[c,k]; a=offsets[ch]; b=offsets[ch+1]
            lo=np.searchsorted(times[a:b],dt[i]-tol,side='left')+a
            hi=np.searchsorted(times[a:b],dt[i]+tol,side='right')+a
            for j in range(lo,hi):
                rows[p]=i; cols[p]=original[j]; costs[p]=1+abs(times[j]-dt[i]); p+=1

def prepare_gt_index(gt_t, gt_ch, geometry, radius_um):
    gt_t=np.asarray(gt_t,dtype=np.int64);gt_ch=np.asarray(gt_ch,dtype=np.int64);geo=np.asarray(geometry,float)
    n=len(geo); valid=(gt_ch>=0)&(gt_ch<n); parts=[]; ids=[]; offsets=[0]
    for c in range(n):
        ix=np.flatnonzero(valid&(gt_ch==c)); ix=ix[np.argsort(gt_t[ix],kind='stable')]
        parts.append(gt_t[ix]);ids.append(ix);offsets.append(offsets[-1]+len(ix))
    times=np.concatenate(parts) if offsets[-1] else np.empty(0,np.int64); original=np.concatenate(ids) if offsets[-1] else np.empty(0,np.int64)
    d=geo[:,None]-geo[None,:]; near=np.einsum('ijk,ijk->ij',d,d)<=radius_um*radius_um
    counts=near.sum(1).astype(np.int64); width=int(counts.max(initial=0)); neighbors=np.full((n,width),-1,np.int64)
    for c in range(n): neighbors[c,:counts[c]]=np.flatnonzero(near[c])
    return {'times':times,'original':original,'offsets':np.asarray(offsets,np.int64),'neighbors':neighbors,'ncount':counts,'geometry':geo,'radius_um':float(radius_um),'gt_count':len(gt_t)}

def spatial_pairs_fast(det_t, det_ch, index, tolerance_samples):
    dt=np.asarray(det_t,dtype=np.int64);dc=np.asarray(det_ch,dtype=np.int64);tol=int(tolerance_samples)
    if len(dt)!=len(dc) or tol<0: raise ValueError('invalid detection vectors or tolerance')
    n_gt=index['gt_count']
    if not len(dt) or not len(index['times']): return np.empty((0,2),np.int64)
    count=_edge_counts(dt,dc,index['times'],index['offsets'],index['neighbors'],index['ncount'],tol)
    starts=np.r_[0,np.cumsum(count[:-1])]; total=int(count.sum())
    if not total:return np.empty((0,2),np.int64)
    rows=np.empty(total,np.int64);cols=np.empty(total,np.int64);cost=np.empty(total,np.int64)
    _edge_fill(dt,dc,index['times'],index['original'],index['offsets'],index['neighbors'],index['ncount'],tol,starts,rows,cols,cost)
    adjacency=coo_matrix((np.ones(2*total,np.uint8),(np.r_[rows,len(dt)+cols],np.r_[len(dt)+cols,rows])),shape=(len(dt)+n_gt,len(dt)+n_gt)).tocsr()
    _,labels=connected_components(adjacency,directed=False,return_labels=True); edge_labels=labels[rows]; order=np.argsort(edge_labels,kind='stable'); lab=edge_labels[order]; bounds=np.r_[0,np.flatnonzero(np.diff(lab))+1,len(order)]
    out=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        e=order[a:b]; rr=rows[e];cc=cols[e]
        if len(e)==1: out.append(np.array([[rr[0],cc[0]]],np.int64));continue
        ur=np.unique(rr);uc=np.unique(cc); lr=np.searchsorted(ur,rr);lc=np.searchsorted(uc,cc)
        dummy=(min(len(ur),len(uc))+1)*(tol+1)
        graph=coo_matrix((np.r_[cost[e],np.full(len(ur),dummy,np.int64)],(np.r_[lr,np.arange(len(ur))],np.r_[lc,len(uc)+np.arange(len(ur))])),shape=(len(ur),len(uc)+len(ur))).tocsr()
        mr,mc=min_weight_full_bipartite_matching(graph); keep=mc<len(uc);out.append(np.c_[ur[mr[keep]],uc[mc[keep]]])
    return np.concatenate(out).astype(np.int64,copy=False)


from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, min_weight_full_bipartite_matching
def _as_integral_vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError(f"{name} must contain integer values")
    return array.astype(np.int64, copy=False)


def resolve_quantized_ties(peaks: np.ndarray, geometry: object, radius_um: float) -> np.ndarray:
    """Keep channel-ascending winners of direct same-code spatial ties.

    ``peaks`` must be a structured array with ``sample_index``, ``amplitude``,
    and ``channel_index`` fields.  Events are returned in their original order.
    Each live channel-ascending anchor removes only live neighbors in its own
    ``(sample_index, amplitude)`` group that lie within ``radius_um``.
    """
    peaks = np.asarray(peaks)
    required_fields = {"sample_index", "amplitude", "channel_index"}
    if peaks.ndim != 1 or peaks.dtype.names is None or not required_fields.issubset(peaks.dtype.names):
        raise ValueError("peaks must be a one-dimensional structured array with sample_index, amplitude, and channel_index")
    if radius_um < 0:
        raise ValueError("radius_um must be non-negative")
    coordinates = np.asarray(geometry, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 1 or not np.all(np.isfinite(coordinates)):
        raise ValueError("geometry must be a finite (channels, dimensions) array")
    channels = _as_integral_vector(peaks["channel_index"], "peaks.channel_index")
    if np.any(channels < 0) or np.any(channels >= coordinates.shape[0]):
        raise ValueError("peaks.channel_index is outside geometry")

    removed = np.zeros(peaks.size, dtype=bool)
    keys = np.lexsort((channels, peaks["amplitude"], peaks["sample_index"]))
    start = 0
    while start < peaks.size:
        end = start + 1
        while (end < peaks.size
               and peaks["sample_index"][keys[end]] == peaks["sample_index"][keys[start]]
               and peaks["amplitude"][keys[end]] == peaks["amplitude"][keys[start]]):
            end += 1
        ids = keys[start:end]
        if ids.size > 1:
            order = ids[np.argsort(channels[ids], kind="stable")]
            live = np.ones(ids.size, dtype=bool)
            position = {int(index): local_index for local_index, index in enumerate(ids)}
            for anchor in order:
                anchor_position = position[int(anchor)]
                if not live[anchor_position]:
                    continue
                distances = np.linalg.norm(coordinates[channels[ids]] - coordinates[channels[anchor]], axis=1)
                direct = ids[live & (distances <= radius_um)]
                for victim in direct:
                    if victim != anchor:
                        live[position[int(victim)]] = False
                        removed[victim] = True
        start = end
    return peaks[~removed]


def spatial_pairs(
    det_t: object,
    det_ch: object,
    gt_t: object,
    gt_ch: object,
    geometry: object,
    tolerance_samples: int,
    radius_um: float,
) -> np.ndarray:
    """Return optimal ``(det_index, gt_index)`` pairs in original array order.

    Feasible pairs obey both the inclusive sample-time tolerance and the
    inclusive Euclidean channel-distance radius.  The matching first maximizes
    pair count, then minimizes total absolute sample-time error.
    """
    det_t = _as_integral_vector(det_t, "det_t")
    det_ch = _as_integral_vector(det_ch, "det_ch")
    gt_t = _as_integral_vector(gt_t, "gt_t")
    gt_ch = _as_integral_vector(gt_ch, "gt_ch")
    if det_t.size != det_ch.size or gt_t.size != gt_ch.size:
        raise ValueError("time and channel arrays must have equal lengths")
    if tolerance_samples < 0 or int(tolerance_samples) != tolerance_samples:
        raise ValueError("tolerance_samples must be a non-negative integer")
    if radius_um < 0:
        raise ValueError("radius_um must be non-negative")
    tolerance_samples = int(tolerance_samples)
    coordinates = np.asarray(geometry, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 1:
        raise ValueError("geometry must have shape (channels, dimensions)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("geometry must be finite")

    det_count, gt_count = det_t.size, gt_t.size
    if det_count == 0 or gt_count == 0:
        return np.empty((0, 2), dtype=np.int64)

    channel_count = coordinates.shape[0]
    valid_det = (det_ch >= 0) & (det_ch < channel_count)
    # A negative ground-truth channel deliberately has no candidate edge.
    valid_gt = (gt_ch >= 0) & (gt_ch < channel_count)
    if not np.any(valid_det) or not np.any(valid_gt):
        return np.empty((0, 2), dtype=np.int64)

    gt_by_channel: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for channel in np.unique(gt_ch[valid_gt]):
        original = np.flatnonzero(gt_ch == channel)
        order = np.argsort(gt_t[original], kind="stable")
        gt_by_channel[int(channel)] = (gt_t[original][order], original[order])

    # The number of physical channels is small; form only the channel-neighbor
    # table used by present detections, never an event-by-event dense matrix.
    neighbor_channels: dict[int, np.ndarray] = {}
    radius_squared = float(radius_um) * float(radius_um)
    for channel in np.unique(det_ch[valid_det]):
        delta = coordinates - coordinates[int(channel)]
        nearby = np.flatnonzero(np.einsum("ij,ij->i", delta, delta) <= radius_squared)
        neighbor_channels[int(channel)] = nearby

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    cost_parts: list[np.ndarray] = []
    for det_index in np.flatnonzero(valid_det):
        det_time = det_t[det_index]
        for channel in neighbor_channels[int(det_ch[det_index])]:
            indexed_gt = gt_by_channel.get(int(channel))
            if indexed_gt is None:
                continue
            times, indices = indexed_gt
            begin = np.searchsorted(times, det_time - tolerance_samples, side="left")
            end = np.searchsorted(times, det_time + tolerance_samples, side="right")
            if begin == end:
                continue
            selected_times = times[begin:end]
            selected_indices = indices[begin:end]
            row_parts.append(np.full(selected_indices.size, det_index, dtype=np.int64))
            col_parts.append(selected_indices.astype(np.int64, copy=False))
            cost_parts.append(1 + np.abs(selected_times - det_time))

    if not row_parts:
        return np.empty((0, 2), dtype=np.int64)

    real_rows = np.concatenate(row_parts)
    real_cols = np.concatenate(col_parts)
    real_costs = np.concatenate(cost_parts).astype(np.int64, copy=False)

    # Solve independent real-edge components separately.  This keeps a sparse
    # event stream with many isolated events from becoming one giant assignment.
    adjacency = coo_matrix(
        (np.ones(2 * real_rows.size, dtype=np.uint8),
         (np.concatenate((real_rows, det_count + real_cols)),
          np.concatenate((det_count + real_cols, real_rows)))),
        shape=(det_count + gt_count, det_count + gt_count),
    ).tocsr()
    _, labels = connected_components(adjacency, directed=False, return_labels=True)
    edge_labels = labels[real_rows]
    order = np.argsort(edge_labels, kind="stable")
    sorted_labels = edge_labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1, order.size]
    pair_parts: list[np.ndarray] = []
    for start, end in zip(starts[:-1], starts[1:]):
        edge_indices = order[start:end]
        component_rows = real_rows[edge_indices]
        component_cols = real_cols[edge_indices]
        if edge_indices.size == 1:
            pair_parts.append(np.array([[component_rows[0], component_cols[0]]], dtype=np.int64))
            continue
        unique_rows = np.unique(component_rows)
        unique_cols = np.unique(component_cols)
        local_rows = np.searchsorted(unique_rows, component_rows)
        local_cols = np.searchsorted(unique_cols, component_cols)
        # Each detection has a private dummy column.  Its cost exceeds every
        # possible component real-edge cost, so it enforces cardinality first.
        dummy_cost = (min(unique_rows.size, unique_cols.size) + 1) * (tolerance_samples + 1)
        if dummy_cost > np.iinfo(np.int64).max:
            raise ValueError("tolerance_samples is too large")
        local_count = unique_rows.size
        rows = np.concatenate((local_rows, np.arange(local_count, dtype=np.int64)))
        cols = np.concatenate((local_cols, unique_cols.size + np.arange(local_count, dtype=np.int64)))
        costs = np.concatenate((real_costs[edge_indices], np.full(local_count, dummy_cost, dtype=np.int64)))
        graph = coo_matrix((costs, (rows, cols)), shape=(local_count, unique_cols.size + local_count)).tocsr()
        matched_rows, matched_cols = min_weight_full_bipartite_matching(graph)
        keep = matched_cols < unique_cols.size
        pair_parts.append(np.column_stack((unique_rows[matched_rows[keep]], unique_cols[matched_cols[keep]])))
    return np.concatenate(pair_parts).astype(np.int64, copy=False)
