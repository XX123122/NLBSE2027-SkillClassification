"""Role-owned peer priors and a stated conditional-independence union model."""
import numpy as np
from skillcomp.peer_publication import fit_peer_prior,peer_prior_scores


class ExcludedPeerPrior:
    def __init__(self,strength=20):self.strength=strength

    def fit(self,targets,repositories,groups,published):
        y=np.asarray(targets);r=np.asarray(repositories).astype(str);g=np.asarray(groups);pub=np.asarray(published)
        if y.shape!=pub.shape or y.ndim!=2 or len(g)!=len(y) or len(r)!=len(y) or not len(y) or not np.isin(y,[0,1]).all() or not np.isin(pub,[0,1]).all():raise ValueError('Aligned observed peer/released targets and fit roles')
        self.prior_=fit_peer_prior(y,r,self.strength);self.published_counts_=pub.sum(0);self.count_=len(y);self.total_=y.sum(0);self.groups_={};self.repos_={};self.group_repos_={}
        for group in np.unique(g):
            rows=g==group;self.groups_[str(group)]=(int(rows.sum()),y[rows].sum(0),pub[rows].sum(0))
            for repo in set(r[rows]):
                both=rows&(r==repo);self.group_repos_[(str(group),repo)]=(int(both.sum()),y[both].sum(0))
        for repo in set(r):
            rows=r==repo;self.repos_[repo]=(int(rows.sum()),y[rows].sum(0))
        return self

    def predict(self,repositories):
        return peer_prior_scores(self.prior_,repositories)[1],np.tile(self.published_counts_,(len(repositories),1)),np.repeat(self.count_,len(repositories))

    def fit_centre(self,repositories,groups):
        r=np.asarray(repositories).astype(str);g=np.asarray(groups);output=[];counts=[];ns=[];zero=np.zeros_like(self.total_)
        if len(r)!=len(g):raise ValueError('Aligned fit-only groups')
        for repo,group in zip(r,g):
            ng,yg,pg=self.groups_.get(str(group),(0,zero,zero));n=self.count_-ng
            if n<=0:raise ValueError('At least one other fit component required')
            mean=(self.total_-yg)/n;nr,yr=self.repos_.get(repo,(0,zero));ngr,ygr=self.group_repos_.get((str(group),repo),(0,zero));nr-=ngr;local=(yr-ygr+self.strength*mean)/(nr+self.strength) if nr else mean;output.append(local);counts.append(self.published_counts_-pg);ns.append(n)
        return np.asarray(output),np.asarray(counts),np.asarray(ns)


def union_probability(peer,availability,proxy,published_counts,fit_counts):
    f=np.asarray(peer,float);a=np.asarray(availability,float);q=np.asarray(proxy,float);counts=np.asarray(published_counts);n=np.asarray(fit_counts)
    if f.ndim!=2 or f.shape!=q.shape or a.shape!=(len(f),) or counts.shape!=f.shape or n.shape!=(len(f),) or (n<=0).any() or not all(np.isfinite(x).all() for x in [f,a,q]) or any(((x<0)|(x>1)).any() for x in [f,a,q]):raise ValueError('Aligned legal input-only probabilities and fit counts')
    p=f+(1-f)*(a[:,None]*q);p[counts==0]=0;p[counts==n[:,None]]=1;return p
