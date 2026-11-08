"""Fit-only repository coverage of the auxiliary evidence source."""
import numpy as np


def fit_coverage(known,repositories,strength=20):
    known=np.asarray(known);repos=np.asarray(repositories)
    if known.ndim!=1 or len(known)!=len(repos) or not len(known) or not np.isin(known,[0,1]).all() or not known.any() or strength<=0:raise ValueError('Nonempty binary known-evidence fit mask required')
    mean=float(known.mean());local={}
    for name in np.unique(repos):
        rows=repos==name;local[str(name)]={'fit_rows':int(rows.sum()),'known_rows':int(known[rows].sum()),'relative_coverage':float(np.clip((known[rows].sum()+strength*mean)/(rows.sum()+strength)/mean,0,2))}
    return {'global_coverage':mean,'repositories':local,'strength':strength,'fit_rows':len(known),'query_evidence_mask_used':False}


def coverage_weights(bundle,repositories,mix):
    if not 0<=mix<=1:raise ValueError('Coverage gate mixture must be in [0,1]')
    return np.array([(1-mix)+mix*bundle['repositories'].get(str(name),{'relative_coverage':1})['relative_coverage'] for name in repositories])
