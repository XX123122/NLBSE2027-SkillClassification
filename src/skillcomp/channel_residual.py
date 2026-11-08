"""Centered auxiliary annotation evidence, with fit-only repository expectations."""
import numpy as np
from scipy.special import expit,logit


def fit_expectation(scores,repositories):
    scores=np.asarray(scores,dtype=float);repositories=np.asarray(repositories)
    if scores.ndim!=2 or len(scores)!=len(repositories) or not len(scores) or not np.isfinite(scores).all():raise ValueError('Nonempty aligned fit predictions required')
    return {'mean':scores.mean(0),'repositories':{str(name):scores[repositories==name].mean(0) for name in np.unique(repositories)},'fit_rows':len(scores)}


def apply_residual(base,channel,repositories,expectation,weight,label_weights=None,row_weights=None):
    base=np.asarray(base,dtype=float);channel=np.asarray(channel,dtype=float);repositories=np.asarray(repositories)
    if base.shape!=channel.shape or base.ndim!=2 or len(base)!=len(repositories) or weight<0:raise ValueError('Aligned probabilities and nonnegative weight required')
    if not np.isfinite(base).all() or not np.isfinite(channel).all() or ((base<0)|(base>1)).any() or ((channel<0)|(channel>1)).any():raise ValueError('Finite probabilities required')
    if weight==0:return base.copy()
    center=np.tile(expectation['mean'],(len(base),1))
    for name,value in expectation['repositories'].items():center[repositories==name]=value
    residual=logit(np.clip(channel,1e-5,1-1e-5))-logit(np.clip(center,1e-5,1-1e-5))
    if label_weights is not None:
        label_weights=np.asarray(label_weights,dtype=float)
        if label_weights.shape!=(base.shape[1],) or not np.isfinite(label_weights).all() or ((label_weights<0)|(label_weights>1)).any():raise ValueError('One finite support weight in [0,1] per label required')
        residual=residual*label_weights
    if row_weights is not None:
        row_weights=np.asarray(row_weights,dtype=float)
        if row_weights.shape!=(len(base),) or not np.isfinite(row_weights).all() or ((row_weights<0)|(row_weights>2)).any():raise ValueError('One finite evidence weight in [0,2] per query row required')
        residual=residual*row_weights[:,None]
    score=expit(logit(np.clip(base,1e-5,1-1e-5))+weight*residual);score[base==0]=0;score[base==1]=1
    if label_weights is not None:score[:,label_weights==0]=base[:,label_weights==0]
    if row_weights is not None:score[row_weights==0]=base[row_weights==0]
    return score


def support_weights(auxiliary_fit_targets,strength):
    targets=np.asarray(auxiliary_fit_targets)
    if targets.ndim!=2 or not len(targets) or not np.isin(targets,[0,1]).all() or strength<0:raise ValueError('Known binary auxiliary fit targets and nonnegative shrinkage required')
    if strength==0:return np.ones(targets.shape[1])
    positive=targets.sum(0);negative=len(targets)-positive;effective=positive*negative/len(targets)
    return effective/(effective+strength)
