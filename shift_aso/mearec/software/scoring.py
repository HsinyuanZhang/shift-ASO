"""MEArec one-to-one scoring. Unknown GT channels remain in the denominator."""
from __future__ import annotations

import numpy as np
# Shared sparse matching primitive; the MEArec policy is defined entirely here.
from ...neuropixels._matching import prepare_gt_index, spatial_pairs_fast, _edge_counts


class Scorer:
    def __init__(self, times, channels, geometry, bounds, *, fs=20000):
        times, channels = np.asarray(times), np.asarray(channels)
        if times.ndim != 1 or times.shape != channels.shape or not np.issubdtype(times.dtype,np.integer) or not np.issubdtype(channels.dtype,np.integer):
            raise ValueError('GT times/channels must be equal-length integer vectors')
        lo, hi = bounds
        if not 0 <= lo < hi or fs <= 0:
            raise ValueError('invalid scoring interval or sampling rate')
        self.original = np.flatnonzero((times >= lo) & (times < hi))
        self.index = prepare_gt_index(times[self.original], channels[self.original], geometry, 54.)
        self.count, self.duration = len(self.original), (hi-lo)/float(fs)

    def feasible_count(self, events):
        """Optimistic only; overlapping label neighborhoods may count repeatedly."""
        index = self.index
        edges = _edge_counts(events['sample_index'], events['channel_index'],
            index['times'], index['offsets'], index['neighbors'], index['ncount'], 20)
        return int(np.count_nonzero(edges))

    def score(self, events):
        pairs = spatial_pairs_fast(events['sample_index'], events['channel_index'], self.index, 20)
        pairs = pairs.copy()
        if len(pairs):
            pairs[:,1] = self.original[pairs[:,1]]
        tp, fp, fn = len(pairs), len(events)-len(pairs), self.count-len(pairs)
        den = tp+fp+fn
        metrics = dict(TP=int(tp), FP=int(fp), FN=int(fn),
            A=tp/den if den else 0., recall=tp/self.count if self.count else 0.,
            precision=tp/len(events) if len(events) else 0., FP_per_s=fp/self.duration,
            good_GT_count=self.count, raw_output_count=len(events), scored_output_count=len(events))
        return metrics, pairs
