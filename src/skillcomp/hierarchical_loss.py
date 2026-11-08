"""Bernoulli tree factorization for root and child-given-parent skill targets."""
import torch
from torch.nn import functional as F
from skillcomp.label_hierarchy import parents_from_labels


def conditional_prevalence(targets,parents):
    import numpy as np
    targets=np.asarray(targets);result=np.zeros(targets.shape[1])
    for child,parent in enumerate(parents):
        selected=np.ones(len(targets),dtype=bool) if parent<0 else targets[:,parent].astype(bool)
        if parent>=0 and ((targets[:,child]==1)&~selected).any():raise ValueError('Child implies parent constraint violated in known fit targets')
        result[child]=(targets[selected,child].sum()+.5)/(int(selected.sum())+1)
    return result


def conditional_hierarchy_bce(logits,targets,parents):
    parent=torch.as_tensor(parents,dtype=torch.long,device=targets.device);child=parent>=0;mask=torch.ones_like(targets)
    mask[:,child]=targets[:,parent[child]]
    if ((targets[:,child]>0)&(mask[:,child]==0)).any():raise ValueError('Inconsistent child-positive known training target')
    # A constant denominator preserves a per-example joint likelihood objective.
    return (F.binary_cross_entropy_with_logits(logits.float(),targets.float(),reduction='none')*mask).mean()


def hierarchy_marginals(conditional,parents):
    parent=torch.as_tensor(parents,dtype=torch.long,device=conditional.device);child=parent>=0;probability=conditional.clone();probability[:,child]=conditional[:,child]*conditional[:,parent[child]]
    return probability
