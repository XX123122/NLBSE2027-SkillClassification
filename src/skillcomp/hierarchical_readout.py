"""Fit independent roots and child-given-parent proxy skill classifiers."""
import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from skillcomp.label_hierarchy import parents_from_labels


class ConditionalHierarchyClassifier:
    def __init__(self,labels,c=1):
        self.labels=list(labels);self.c=c

    def fit(self,x,y):
        y=np.asarray(y)
        if y.ndim!=2 or len(y)!=x.shape[0] or y.shape[1]!=len(self.labels) or not len(y) or not np.isin(y,[0,1]).all():raise ValueError('Aligned known binary proxy labels required')
        self.parents_=np.array(parents_from_labels(self.labels));self.n_features_in_=x.shape[1]
        for child,parent in enumerate(self.parents_):
            if parent>=0 and ((y[:,child]==1)&(y[:,parent]==0)).any():raise ValueError('Known fit child implies parent constraint violated')
        self.coef_=np.zeros((len(self.labels),x.shape[1]));self.intercept_=np.zeros(len(self.labels));self.constants_=np.full(len(self.labels),np.nan);self.fit_rows_=np.zeros(len(self.labels),dtype=int)
        for column,parent in enumerate(self.parents_):
            rows=np.ones(len(y),dtype=bool) if parent<0 else y[:,parent].astype(bool);target=y[rows,column];self.fit_rows_[column]=len(target)
            if not len(target) or not target.any():self.constants_[column]=0;continue
            if target.all():self.constants_[column]=1;continue
            model=LogisticRegression(C=self.c,solver='liblinear',max_iter=2000,random_state=42).fit(x[rows],target);self.coef_[column]=model.coef_[0];self.intercept_[column]=model.intercept_[0]
        return self

    def predict_proba(self,x):
        if x.ndim!=2 or x.shape[1]!=self.n_features_in_:raise ValueError('Feature dimensions differ from fit')
        values=x.data if sparse.issparse(x) else np.asarray(x)
        if not np.isfinite(values).all():raise ValueError('Finite query features required')
        scores=expit(x@self.coef_.T+self.intercept_);known=np.isfinite(self.constants_);scores[:,known]=self.constants_[known]
        child=np.flatnonzero(self.parents_>=0);scores[:,child]*=scores[:,self.parents_[child]]
        return np.asarray(scores)
