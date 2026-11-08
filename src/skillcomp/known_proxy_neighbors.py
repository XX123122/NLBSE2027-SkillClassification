"""Known-label proxy neighbors; group exclusion is limited to fit centering."""
import numpy as np


class KnownProxyNeighbors:
    def __init__(self,neighbors=20,power=4,pseudocount=1,prior_strength=20):
        self.neighbors=neighbors;self.power=power;self.pseudocount=pseudocount;self.prior_strength=prior_strength

    def fit(self,vectors,targets,repositories,fit_groups):
        x=np.asarray(vectors,dtype=np.float32);y=np.asarray(targets)
        if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)!=len(repositories) or len(x)!=len(fit_groups) or not len(x) or not np.isin(y,[0,1]).all() or not np.isfinite(x).all():raise ValueError('Aligned observed-only proxy training examples required')
        norms=np.linalg.norm(x,axis=1,keepdims=True)
        if (norms==0).any():raise ValueError('Nonzero input embeddings required')
        self.vectors_=x/norms;self.targets_=y.astype(np.float32);self.repositories_=np.asarray(repositories).astype(str);self.fit_groups_=np.asarray(fit_groups);self.mean_=self.targets_.mean(0);self.positive_counts_=y.sum(0);self.local_prior_={}
        for repo in np.unique(self.repositories_):
            rows=self.repositories_==repo;self.local_prior_[str(repo)]=(self.targets_[rows].sum(0)+self.prior_strength*self.mean_)/(rows.sum()+self.prior_strength)
        return self

    def _predict(self,vectors,repositories,exclude_groups=None):
        x=np.asarray(vectors,dtype=np.float32);repos=np.asarray(repositories).astype(str)
        if x.ndim!=2 or len(x)!=len(repos) or x.shape[1]!=self.vectors_.shape[1] or not np.isfinite(x).all():raise ValueError('Aligned input-only query vectors/repositories required')
        norms=np.linalg.norm(x,axis=1,keepdims=True)
        if (norms==0).any():raise ValueError('Nonzero input embeddings required')
        similarities=(x/norms)@self.vectors_.T;output=[]
        for i,repo in enumerate(repos):
            eligible=np.ones(len(self.targets_),dtype=bool)
            if exclude_groups is not None:eligible&=self.fit_groups_!=exclude_groups[i]
            same=eligible&(self.repositories_==repo)
            if same.any():eligible=same
            values=similarities[i].copy();values[~eligible]=-np.inf;indices=np.argsort(-values,kind='stable')[:min(self.neighbors,int(eligible.sum()))];weights=np.maximum(values[indices],0)**self.power;prior=self.local_prior_.get(str(repo),self.mean_);score=(weights@self.targets_[indices]+self.pseudocount*prior)/(weights.sum()+self.pseudocount);output.append(score)
        scores=np.asarray(output);scores[:,self.positive_counts_==0]=0;scores[:,self.positive_counts_==len(self.targets_)]=1;return scores

    def predict_proba(self,vectors,repositories):
        return self._predict(vectors,repositories)

    def predict_fit_centre(self,vectors,repositories,fit_groups):
        if len(fit_groups)!=len(vectors):raise ValueError('Fit-only group exclusions must align with centering rows')
        return self._predict(vectors,repositories,np.asarray(fit_groups))
