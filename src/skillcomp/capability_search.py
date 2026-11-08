"""Joint feature construction and train-only RF-gated capability selection."""
import numpy as np
from scipy import sparse

from skillcomp.threshold_frontier import exact_frontier, choose_frontier


def repo_features(training_repos, query_repos):
    names = sorted(set(np.asarray(training_repos).tolist()))
    lookup = {name: i for i, name in enumerate(names)}
    rows, columns = [], []
    for row, name in enumerate(query_repos):
        if name in lookup:
            rows.append(row)
            columns.append(lookup[name])
    return sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(query_repos), len(names)))


def joint_features(lexical, embeddings, repository, spec):
    parts = []
    if spec['lexical']:
        parts.append(lexical)
    if spec['embedding_scale']:
        parts.append(sparse.csr_matrix(embeddings * spec['embedding_scale']))
    if spec['repo_scale']:
        parts.append(repository * spec['repo_scale'])
    return sparse.hstack(parts, format='csr')


def threshold_choice(y, scores, rf):
    return choose_frontier(exact_frontier(y, scores, rf, {}), z=0.)


def select_candidate(y, candidates, prior, rf, weights):
    records = []
    for name in sorted(candidates):
        for weight in weights:
            choice = threshold_choice(y, weight * candidates[name] + (1 - weight) * prior, rf)
            records.append({'candidate': name, 'weight': float(weight), **choice})
    best = max(records, key=lambda r: (r['micro_f1'], r['precision'], not r['fallback_rf'], r['candidate'], r['weight']))
    return best, records
