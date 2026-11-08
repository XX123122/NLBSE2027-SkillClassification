"""Relative ranking between two observed-positive-overlap prototype pairs.

No zero-overlap pair is used as a negative. Lower overlap means lower observed
similarity, not proven skill absence. This is an explicit weak-supervision
assumption, independent of the rejected fully-disjoint-negative recipe.
"""
import numpy as np

from skillcomp.retrieval_evidence import EvidenceNeighbors


def positive_overlap_pools(targets, repositories, groups, minimum_gap=.1):
    y = np.asarray(targets)
    repos, groups = np.asarray(repositories), np.asarray(groups)
    if y.ndim != 2 or not len(y) or len(repos) != len(y) or len(groups) != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('Aligned observed binary proxy targets/repositories/groups required')
    weights = np.log((len(y) + 1) / (y.sum(0) + 1))
    weighted = y.astype(float) * weights
    intersection = weighted @ y.astype(float).T
    sizes = weighted.sum(1)
    union = sizes[:, None] + sizes[None, :] - intersection
    jaccard = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    eligible = (intersection > 0) & (groups[:, None] != groups[None, :]) & (repos[:, None] == repos[None, :])
    anchors, low, high, gaps = [], [], [], []
    for i in range(len(y)):
        candidates = np.flatnonzero(eligible[i])
        if len(candidates) < 4:
            continue
        bottom, top = np.quantile(jaccard[i, candidates], [.25, .75])
        if top - bottom < minimum_gap:
            continue
        anchors.append(i)
        low.append(candidates[jaccard[i, candidates] <= bottom])
        high.append(candidates[jaccard[i, candidates] >= top])
        gaps.append(float(top - bottom))
    return {'anchors': np.asarray(anchors, dtype=int), 'low': low, 'high': high,
            'label_weights': weights, 'weighted_jaccard': jaccard,
            'minimum_gap': minimum_gap, 'interquartile_gaps': np.asarray(gaps)}


def metric_transform(vectors, left, right):
    x = np.asarray(vectors, dtype=np.float32)
    u, v = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    if x.ndim != 2 or u.shape != v.shape or u.ndim != 2 or x.shape[1] != u.shape[0] or not all(np.isfinite(a).all() for a in [x, u, v]):
        raise ValueError('Finite input vectors and aligned low-rank factors required')
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    value = x + (x @ v) @ u.T
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if (norm == 0).any():
        raise ValueError('Metric must preserve nonzero embeddings')
    return value / norm


class PositiveOverlapMetricNeighbors(EvidenceNeighbors):
    def __init__(self, left, right, blend_original=False, **kwargs):
        super().__init__(**kwargs)
        self.left = np.asarray(left, np.float32)
        self.right = np.asarray(right, np.float32)
        self.blend_original = blend_original

    def _transform(self, vectors):
        x = np.asarray(vectors, np.float32)
        transformed = metric_transform(x, self.left, self.right)
        if self.blend_original:
            normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            return np.concatenate([normalized, transformed], axis=1) / np.float32(np.sqrt(2))
        return transformed

    def fit(self, vectors, targets, repositories, fit_groups):
        return super().fit(self._transform(vectors), targets, repositories, fit_groups)

    def predict_evidence(self, vectors, repositories):
        return self._evidence(self._transform(vectors), repositories)

    def fit_centre_evidence(self, vectors, repositories, fit_groups):
        return self._evidence(self._transform(vectors), repositories, np.asarray(fit_groups))
