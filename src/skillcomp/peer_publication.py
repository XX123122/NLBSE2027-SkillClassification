"""Observed peer contributions to released labels; never missing API truth."""
from collections import defaultdict
import numpy as np


def peer_observed_union(proxy,matched,repositories,bag_ids):
    a=np.asarray(proxy);m=np.asarray(matched,bool);r=np.asarray(repositories).astype(str);g=np.asarray(bag_ids)
    if a.ndim!=2 or m.shape!=(len(a),) or r.shape!=m.shape or g.shape!=m.shape or not len(a) or not np.isin(a[m],[0,1]).all():raise ValueError('Aligned role-owned observed API annotations and bag metadata')
    bags=defaultdict(list)
    for i,bag in enumerate(g):bags[str(bag)].append(i)
    result=np.zeros(a.shape,np.uint8);union=np.zeros_like(result)
    for rows in bags.values():
        if len(set(r[rows]))!=len(rows):raise ValueError('Only one released issue per repository/PR bag expected')
        known=np.asarray(rows)[m[rows]];counts=a[known].astype(np.int64).sum(0);total=counts>0
        for i in rows:
            # Missing values are never read or converted into local API negatives.
            remaining=counts-a[i].astype(np.int64) if m[i] else counts
            result[i]=remaining>0;union[i]=total
    return result,union


def fit_peer_prior(targets,repositories,strength=20.):
    y=np.asarray(targets);r=np.asarray(repositories).astype(str)
    if y.ndim!=2 or len(y)!=len(r) or not len(y) or not np.isin(y,[0,1]).all():raise ValueError('Observed binary peer-publication targets required')
    mean=y.mean(0);local={}
    for repo in sorted(set(r)):
        mask=r==repo;local[repo]=(y[mask].sum(0)+strength*mean)/(mask.sum()+strength)
    return {'mean':mean,'repositories':local,'positive_counts':y.sum(0),'fit_rows':len(y),'strength':strength}


def peer_prior_scores(bundle,repositories):
    r=np.asarray(repositories).astype(str);global_scores=np.tile(bundle['mean'],(len(r),1));local=np.array([bundle['repositories'].get(repo,bundle['mean']) for repo in r]);local[:,bundle['positive_counts']==0]=0;local[:,bundle['positive_counts']==bundle['fit_rows']]=1
    return global_scores,local
