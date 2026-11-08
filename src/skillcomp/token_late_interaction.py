"""Frozen contextual token matching, without target-dependent retrieval inputs."""
import numpy as np


def document_frequencies(rows):
    counts={}
    for ids in rows:
        for token in set(map(int,ids)):counts[token]=counts.get(token,0)+1
    return counts


def idf_values(ids,counts,documents):
    if documents<1:raise ValueError('Nonempty own-fit corpus required')
    return np.asarray([np.log((documents+1)/(counts.get(int(token),0)+1))+1 for token in ids],np.float64)


def token_relative_votes(original,scores,valid,temperature=.1,blend=.5):
    w=np.asarray(original,float);s=np.asarray(scores,float);valid=np.asarray(valid,bool)
    if w.ndim!=1 or w.shape!=s.shape or w.shape!=valid.shape or not len(w) or not np.isfinite(w).all() or (w<0).any() or temperature<=0 or not 0<=blend<=1:raise ValueError('Aligned finite nonnegative mass, score validity and fixed temperature')
    # Missing token evidence abstains for the entire query, preserving all mass.
    if not valid.all() or not np.isfinite(s).all():return w.copy()
    positive=w>0
    if not positive.any() or np.ptp(s[positive])==0:return w.copy()
    logits=np.log(w[positive])+(s[positive]-s[positive].max())/temperature
    p=np.exp(logits-logits.max());p=p/p.sum()*w.sum();new=np.zeros_like(w);new[positive]=p
    return (1-blend)*w+blend*new


def independent_score(query,document,weights):
    q=np.asarray(query,np.float64);d=np.asarray(document,np.float64);w=np.asarray(weights,np.float64)
    if q.ndim!=2 or d.ndim!=2 or q.shape[1]!=d.shape[1] or len(w)!=len(q):raise ValueError('Aligned contextual token matrices and query IDF')
    if not len(q) or not len(d):return np.nan
    return float((q@d.T).max(1)@w/w.sum())
