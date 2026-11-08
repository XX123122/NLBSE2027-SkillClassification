"""Input-only estimate of auxiliary evidence availability; not a skill target."""
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from skillcomp.capability_search import repo_features


def matchability_features(title,body,fit_repositories,query_repositories):
    if title.shape[0]!=body.shape[0] or title.shape[0]!=len(query_repositories):raise ValueError('Aligned input-only query feature rows required')
    return normalize(sparse.hstack([title,body,repo_features(fit_repositories,query_repositories)],format='csr'),norm='l2')


def predicted_coverage_weights(model,features,fit_coverage):
    if not 0<fit_coverage<=1:raise ValueError('Positive fit-only known-evidence fraction required')
    probability=np.asarray(model.predict_proba(features))[:,1]
    if not np.isfinite(probability).all():raise ValueError('Finite matchability predictions required')
    return np.clip(probability/fit_coverage,0,2)
