"""Joint mixture of observed evidence availability and known-only proxy skills."""
import numpy as np


def fit_missing_publication(published,matched,repositories,strength=20):
    y=np.asarray(published);known=np.asarray(matched,dtype=bool);repos=np.asarray(repositories)
    if y.ndim!=2 or known.shape!=(len(y),) or repos.shape!=known.shape or not np.isin(y,[0,1]).all():raise ValueError('Aligned fully observed fit publication labels required')
    missing=~known;mean=y.mean(0);global_rate=(y[missing].sum(0)+.5)/(missing.sum()+1) if missing.any() else mean;local={}
    for repo in np.unique(repos):
        rows=(repos==repo)&missing;local[str(repo)]=(y[rows].sum(0)+strength*global_rate)/(rows.sum()+strength)
    return {'global_rate':global_rate,'repository_rates':local,'fit_rows':len(y),'missing_fit_rows':int(missing.sum()),'mean':mean,'strength':strength}


def joint_publication(bundle,availability,known_channel,repositories):
    p=np.asarray(availability,dtype=float);known=np.asarray(known_channel,dtype=float);repos=np.asarray(repositories)
    if known.shape!=(len(p),len(bundle['mean'])) or repos.shape!=p.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.isfinite(known).all() or ((known<0)|(known>1)).any():raise ValueError('Aligned input-only model probabilities required')
    missing=np.array([bundle['repository_rates'].get(str(repo),bundle['global_rate']) for repo in repos]);score=p[:,None]*known+(1-p[:,None])*missing;score[:,bundle['mean']==0]=0;score[:,bundle['mean']==1]=1;return score
