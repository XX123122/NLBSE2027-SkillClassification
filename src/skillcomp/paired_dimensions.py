"""Matryoshka truncation must preserve both separately normalized fields."""
import numpy as np


def truncate_paired_embeddings(values,dimensions):
    values=np.asarray(values)
    if values.ndim!=2:raise ValueError('Two-dimensional paired embeddings required')
    width=values.shape[1]//2
    if values.shape[1]%2 or not 0<dimensions<=width:raise ValueError('Valid per-field embedding dimension required')
    first=values[:,:dimensions].copy();second=values[:,width:width+dimensions].copy()
    first/=np.maximum(np.linalg.norm(first,axis=1,keepdims=True),1e-12);second/=np.maximum(np.linalg.norm(second,axis=1,keepdims=True),1e-12)
    return np.concatenate([first,second],axis=1)/np.float32(np.sqrt(2))
