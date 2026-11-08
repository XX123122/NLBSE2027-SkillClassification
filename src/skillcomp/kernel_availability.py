"""Train-only kernel landmarks augment the fixed input-only availability task."""
import numpy as np
from scipy import sparse
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


class KernelAvailabilityClassifier:
    def __init__(self,gamma=2,components=256,c=1,seed=1234):
        self.gamma=gamma;self.components=components;self.C=c;self.seed=seed

    def _transform(self,x):
        return normalize(sparse.hstack([sparse.csr_matrix(x),sparse.csr_matrix(self.kernel_.transform(x))],format='csr'),norm='l2')

    def fit(self,x,y):
        y=np.asarray(y)
        if x.shape[0]!=len(y) or len(np.unique(y))!=2:raise ValueError('Aligned binary availability supervision required')
        self.n_features_in_=x.shape[1];self.kernel_=Nystroem(kernel='rbf',gamma=self.gamma,n_components=min(self.components,len(y)),random_state=self.seed).fit(x);self.model_=LogisticRegression(C=self.C,solver='liblinear',max_iter=2000,random_state=42).fit(self._transform(x),y);self.classes_=self.model_.classes_.copy();return self

    def predict_proba(self,x):
        if x.shape[1]!=self.n_features_in_:raise ValueError('Availability query dimensions differ from fit')
        return self.model_.predict_proba(self._transform(x))
