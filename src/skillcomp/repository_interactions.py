"""Shrunk domain-specific dense-feature deviations with fit-only categories."""
import numpy as np
from scipy import sparse


def append_repository_interactions(values,dense_values,fit_repositories,query_repositories,scale=.25):
    dense=sparse.csr_matrix(dense_values)
    if dense.shape[0]!=values.shape[0] or dense.shape[0]!=len(query_repositories) or scale<=0:raise ValueError('Aligned query-only dense inputs and positive deviation scale required')
    names=sorted(set(str(name) for name in fit_repositories));lookup={name:i for i,name in enumerate(names)};categories=np.array([lookup.get(str(name),-1) for name in query_repositories],dtype=np.int32);counts=np.diff(dense.indptr);expanded=np.repeat(categories,counts);keep=expanded>=0;indices=dense.indices[keep]+expanded[keep]*dense.shape[1];indptr=np.concatenate([[0],np.cumsum(np.where(categories>=0,counts,0))]);interaction=sparse.csr_matrix((dense.data[keep]*scale,indices,indptr),shape=(dense.shape[0],dense.shape[1]*len(names)));return sparse.hstack([sparse.csr_matrix(values),interaction],format='csr')
