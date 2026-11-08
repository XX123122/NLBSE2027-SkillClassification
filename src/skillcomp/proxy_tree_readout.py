"""Fit-only projection and shared multi-output probability trees for proxies."""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor


class ProxyTreeReadout:
    def __init__(self, dimensions=128, estimators=256, depth=12, min_leaf=5, seed=1193, jobs=16):
        self.dimensions = dimensions;self.estimators = estimators;self.depth = depth
        self.min_leaf = min_leaf;self.seed = seed;self.jobs = jobs

    def fit_projection(self, vectors, repositories):
        x = np.asarray(vectors, dtype=np.float32);r = np.asarray(repositories).astype(str)
        if x.ndim != 2 or len(x) != len(r) or not np.isfinite(x).all():raise ValueError('Aligned fit-only vectors and repositories required')
        self.projection_ = TruncatedSVD(n_components=min(self.dimensions, min(x.shape) - 1), n_iter=3, random_state=self.seed).fit(x)
        self.repository_vocabulary_ = sorted(set(r))
        return self

    def input_features(self, vectors, repositories):
        x = np.asarray(vectors, dtype=np.float32);r = np.asarray(repositories).astype(str)
        if x.ndim != 2 or len(x) != len(r) or not np.isfinite(x).all():raise ValueError('Aligned input-only vectors/repositories required')
        projected = self.projection_.transform(x)
        indicators = np.column_stack([r == name for name in self.repository_vocabulary_]).astype(np.float32)
        return np.concatenate([projected, indicators], axis=1)

    def fit_head(self, vectors, targets, repositories):
        y = np.asarray(targets)
        if y.ndim != 2 or len(y) != len(vectors) or not len(y) or not np.isin(y, [0, 1]).all():raise ValueError('Observed-only binary fit proxies required')
        self.head_ = ExtraTreesRegressor(n_estimators=self.estimators, max_depth=self.depth, min_samples_leaf=self.min_leaf,
                                         max_features='sqrt', criterion='squared_error', bootstrap=False,
                                         random_state=self.seed, n_jobs=self.jobs).fit(self.input_features(vectors, repositories), y)
        self.known_rows_ = len(y);self.positive_counts_ = y.sum(0)
        return self

    def predict_proba(self, vectors, repositories):
        p = np.asarray(self.head_.predict(self.input_features(vectors, repositories)))
        if p.ndim == 1:p = p[:, None]
        p[:, self.positive_counts_ == 0] = 0;p[:, self.positive_counts_ == self.known_rows_] = 1
        if not np.isfinite(p).all() or (p < -1e-12).any() or (p > 1 + 1e-12).any():raise ValueError('Tree means must be finite probabilities')
        return np.clip(p, 0, 1)
