"""Append fit-vocabulary repo indicators while preserving existing text geometry."""
import numpy as np
from scipy import sparse


def append_repository_indicators(values,fit_repositories,query_repositories,scale=1):
    if values.shape[0]!=len(query_repositories) or scale<=0:raise ValueError('Aligned features and positive repository scale required')
    names=sorted(set(str(name) for name in fit_repositories));lookup={name:i for i,name in enumerate(names)};rows=[];columns=[]
    for row,name in enumerate(query_repositories):
        if str(name) in lookup:rows.append(row);columns.append(lookup[str(name)])
    indicators=sparse.csr_matrix((np.full(len(rows),scale,dtype=float),(rows,columns)),shape=(len(query_repositories),len(names)));return sparse.hstack([sparse.csr_matrix(values),indicators],format='csr')
