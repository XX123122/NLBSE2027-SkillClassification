"""Save fit-only text-structure scaling alongside the input-only availability gate."""
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


class StructuredAvailabilityClassifier:
    def __init__(self,dimensions=5,scale=.25,c=1):
        self.dimensions=dimensions;self.scale=scale;self.C=c

    def _transform(self,x):
        x=sparse.csr_matrix(x);structure=x[:,-self.dimensions:].toarray();z=np.clip((structure-self.mean_)/self.std_,-3,3)*self.scale
        return normalize(sparse.hstack([x[:,:-self.dimensions],sparse.csr_matrix(z)],format='csr'),norm='l2')

    def fit(self,x,y):
        x=sparse.csr_matrix(x)
        if x.shape[1]<=self.dimensions or len(y)!=x.shape[0]:raise ValueError('Aligned lexical/semantic inputs with appended structure required')
        self.n_features_in_=x.shape[1];values=x[:,-self.dimensions:].toarray();self.mean_=values.mean(0);self.std_=values.std(0);self.std_[self.std_<1e-10]=1;self.model_=LogisticRegression(C=self.C,solver='liblinear',max_iter=2000,random_state=42).fit(self._transform(x),y);self.classes_=self.model_.classes_.copy();return self

    def predict_proba(self,x):
        if x.shape[1]!=self.n_features_in_:raise ValueError('Availability input dimensions differ from fit')
        return self.model_.predict_proba(self._transform(x))
