"""Peer observed contribution conditional on no own observed contribution."""
import numpy as np


class ConditionalPeerPrior:
    def __init__(self,strength=20):self.strength=strength

    @staticmethod
    def counts(f,s,pub):
        zero=~s.astype(bool);return {'n':len(f),'F':f.sum(0),'Y':pub.sum(0),'den':zero.sum(0),'num':(f*zero).sum(0)}

    @staticmethod
    def subtract(total,part):
        return {key:total[key]-part[key] for key in total} if part is not None else total

    def fit(self,peer,own_observed,repositories,groups,published):
        f=np.asarray(peer);s=np.asarray(own_observed);pub=np.asarray(published);r=np.asarray(repositories).astype(str);g=np.asarray(groups)
        if f.ndim!=2 or f.shape!=s.shape or f.shape!=pub.shape or len(r)!=len(f) or len(g)!=len(f) or not len(f) or not np.isin(f,[0,1]).all() or not np.isin(s,[0,1]).all() or not np.isin(pub,[0,1]).all():raise ValueError('Aligned observed mechanism targets required')
        if not np.array_equal(np.maximum(f,s),pub):raise ValueError('Own/peer observed union does not reconstruct published target')
        self.total_=self.counts(f,s,pub);self.groups_={str(group):self.counts(f[g==group],s[g==group],pub[g==group]) for group in np.unique(g)};self.local_={repo:self.counts(f[r==repo],s[r==repo],pub[r==repo]) for repo in set(r)};self.group_local_={(str(group),repo):self.counts(f[(g==group)&(r==repo)],s[(g==group)&(r==repo)],pub[(g==group)&(r==repo)]) for group in np.unique(g) for repo in set(r[g==group])};return self

    def probabilities(self,repositories,exclude_groups=None):
        r=np.asarray(repositories).astype(str);g=None if exclude_groups is None else np.asarray(exclude_groups)
        if g is not None and len(g)!=len(r):raise ValueError('Fit-only exclusion groups aligned')
        result=[];counts=[];ns=[]
        for i,repo in enumerate(r):
            group=None if g is None else str(g[i]);total=self.subtract(self.total_,self.groups_.get(group))
            if total['n']<=0:raise ValueError('Need another fit component')
            mean=total['F']/total['n'];global_rate=np.where(total['den']>0,(total['num']+.5)/(total['den']+1),mean);local=self.local_.get(repo)
            if local is not None:
                local=self.subtract(local,self.group_local_.get((group,repo)))
                score=(local['num']+self.strength*global_rate)/(local['den']+self.strength) if local['n'] else global_rate
            else:score=global_rate
            result.append(score);counts.append(total['Y']);ns.append(total['n'])
        return np.asarray(result),np.asarray(counts),np.asarray(ns)

    def predict(self,repositories):return self.probabilities(repositories)
    def fit_centre(self,repositories,groups):return self.probabilities(repositories,groups)
