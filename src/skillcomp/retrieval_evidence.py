"""Input-only neighbor evidence and component-excluded training statistics.

The original deployed neighbor class is deliberately untouched. Group identifiers
are accepted only by the explicitly named fit-centre method, never query APIs.
"""
import numpy as np

from skillcomp.annotation_channel import apply_channel
from skillcomp.known_proxy_neighbors import KnownProxyNeighbors


class EvidenceNeighbors(KnownProxyNeighbors):
    def __init__(self, same_repository=True, repository_prior=True, **kwargs):
        super().__init__(**kwargs)
        self.same_repository = same_repository
        self.repository_prior = repository_prior

    def fit(self, vectors, targets, repositories, fit_groups):
        super().fit(vectors, targets, repositories, fit_groups)
        self.group_counts_ = {}
        self.repo_counts_ = {}
        self.group_repo_counts_ = {}
        for group in np.unique(self.fit_groups_):
            mask = self.fit_groups_ == group
            self.group_counts_[group] = (int(mask.sum()), self.targets_[mask].sum(0))
            for repo in np.unique(self.repositories_[mask]):
                both = mask & (self.repositories_ == repo)
                self.group_repo_counts_[(group, str(repo))] = (int(both.sum()), self.targets_[both].sum(0))
        for repo in np.unique(self.repositories_):
            mask = self.repositories_ == repo
            self.repo_counts_[str(repo)] = (int(mask.sum()), self.targets_[mask].sum(0))
        return self

    def _evidence(self, vectors, repositories, exclude_groups=None):
        x = np.asarray(vectors, dtype=np.float32)
        repos = np.asarray(repositories).astype(str)
        if x.ndim != 2 or x.shape != (len(repos), self.vectors_.shape[1]) or not np.isfinite(x).all():
            raise ValueError('Aligned finite query vectors/repositories required')
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError('Nonzero query vectors required')
        if exclude_groups is not None and len(exclude_groups) != len(x):
            raise ValueError('Aligned fit-only groups required')
        sim = (x / norms) @ self.vectors_.T
        result = {key: [] for key in ['probability', 'positive_mass', 'zero_mass', 'effective_count',
                                      'top1', 'top5', 'span', 'variance', 'prior', 'weight_sum']}
        zero = np.zeros(self.targets_.shape[1], dtype=np.float32)
        for i, repo in enumerate(repos):
            eligible = np.ones(len(self.targets_), dtype=bool)
            count, positives = len(self.targets_), self.positive_counts_.copy()
            local_n, local_y = self.repo_counts_.get(str(repo), (0, zero))
            if exclude_groups is not None:
                group = exclude_groups[i]
                eligible &= self.fit_groups_ != group
                n_g, y_g = self.group_counts_.get(group, (0, zero))
                count, positives = count - n_g, positives - y_g
                n_gr, y_gr = self.group_repo_counts_.get((group, str(repo)), (0, zero))
                local_n, local_y = local_n - n_gr, local_y - y_gr
            prior = positives / count if count else np.full(len(zero), .5)
            if self.repository_prior and local_n:
                prior = (local_y + self.prior_strength * prior) / (local_n + self.prior_strength)
            same = eligible & (self.repositories_ == repo)
            if self.same_repository and same.any():
                eligible = same
            values = sim[i].copy()
            values[~eligible] = -np.inf
            indices = np.argsort(-values, kind='stable')[:min(self.neighbors, int(eligible.sum()))]
            similarities = values[indices]
            weights = np.maximum(similarities, 0) ** self.power
            mass = weights @ self.targets_[indices]
            total = float(weights.sum())
            score = (mass + self.pseudocount * prior) / (total + self.pseudocount)
            if count:
                score[positives == 0] = 0
                score[positives == count] = 1
            empirical = mass / total if total > 0 else np.asarray(prior)
            for key, value in {
                'probability': score, 'positive_mass': mass,
                'zero_mass': np.maximum(total - mass, 0),
                'effective_count': total * total / max(float(weights @ weights), 1e-12),
                'top1': float(similarities[0]) if len(indices) else 0.,
                'top5': float(similarities[:5].mean()) if len(indices) else 0.,
                'span': float(similarities[0] - similarities[-1]) if len(indices) else 0.,
                'variance': empirical * (1 - empirical), 'prior': prior, 'weight_sum': total,
            }.items():
                result[key].append(value)
        return {key: np.asarray(value) for key, value in result.items()}

    def predict_evidence(self, vectors, repositories):
        return self._evidence(vectors, repositories)

    def predict_proba(self, vectors, repositories):
        return self.predict_evidence(vectors, repositories)['probability']

    def fit_centre_evidence(self, vectors, repositories, fit_groups):
        return self._evidence(vectors, repositories, np.asarray(fit_groups))

    def predict_fit_centre(self, vectors, repositories, fit_groups):
        return self.fit_centre_evidence(vectors, repositories, fit_groups)['probability']


class ExcludedChannelStatistics:
    """Subtract component sufficient counts; numerically match refitting channel.

    This excludes the centre row's component from mapping counts and priors.
    It does not refit the supervised encoder or claim independent centring.
    """
    def __init__(self, published, auxiliary, matched, repositories, groups, strength=20):
        y, a = np.asarray(published), np.asarray(auxiliary)
        m, r, g = np.asarray(matched, bool), np.asarray(repositories).astype(str), np.asarray(groups)
        if y.shape != a.shape or len(y) != len(m) or len(r) != len(y) or len(g) != len(y):
            raise ValueError('Aligned fit targets, repositories and groups required')
        if not np.isin(y, [0, 1]).all() or not np.isin(a[m], [0, 1]).all():
            raise ValueError('Binary published and observed proxy targets required')
        self.strength = strength
        self.repositories = sorted(set(r[m]))
        self.total = self._counts(y, a, m)
        self.by_group = {group: self._counts(y[g == group], a[g == group], m[g == group]) for group in np.unique(g)}
        self.local = {repo: self._counts(y[r == repo], a[r == repo], m[r == repo]) for repo in self.repositories}
        self.group_local = {(group, repo): self._counts(y[(g == group) & (r == repo)], a[(g == group) & (r == repo)], m[(g == group) & (r == repo)])
                            for group in np.unique(g) for repo in set(r[g == group])}

    @staticmethod
    def _counts(y, a, m):
        known = a[m].astype(bool)
        den = np.stack([(~known).sum(0), known.sum(0)])
        num = np.stack([(y[m] * ~known).sum(0), (y[m] * known).sum(0)])
        return {'n': len(y), 'y': y.sum(0), 'known': int(m.sum()), 'den': den, 'num': num}

    @staticmethod
    def _subtract(total, part):
        return {key: value - part[key] for key, value in total.items()} if part is not None else total

    def excluding(self, group):
        total = self._subtract(self.total, self.by_group.get(group))
        if total['n'] <= 0:
            raise ValueError('At least one other fit component required for centring')
        mean = total['y'] / total['n']
        rates = np.where(total['den'] > 0, (total['num'] + .5) / (total['den'] + 1), np.tile(mean, (2, 1)))
        local = {}
        for repo in self.repositories:
            stat = self._subtract(self.local[repo], self.group_local.get((group, repo)))
            if stat['known']:
                local[repo] = (stat['num'] + self.strength * rates) / (stat['den'] + self.strength)
        return {'global_rates': rates, 'repository_rates': local, 'mean': mean, 'strength': self.strength,
                'known_rows': total['known'], 'fit_rows': total['n']}

    def map_fit_centre(self, probabilities, repositories, groups):
        p, r, g = np.asarray(probabilities), np.asarray(repositories), np.asarray(groups)
        output = np.empty(p.shape, dtype=float)
        for group in np.unique(g):
            mask = g == group
            output[mask] = apply_channel(self.excluding(group), p[mask], r[mask])
        return output


def evidence_features(evidence, kind):
    """Small cell-wise feature tensor; raw zeros remain evidence, not negatives."""
    positive = np.asarray(evidence['positive_mass'])
    features = [np.log1p(positive), np.log1p(evidence['zero_mass'])]
    if kind == 'A2':
        for key in ['effective_count', 'top1', 'top5', 'span']:
            features.append(np.broadcast_to(np.asarray(evidence[key])[:, None], positive.shape))
        features.extend([np.asarray(evidence['variance']), np.asarray(evidence['prior'])])
    elif kind != 'A1':
        raise ValueError('Only preregistered A1/A2 features allowed')
    return np.stack(features, axis=-1)
