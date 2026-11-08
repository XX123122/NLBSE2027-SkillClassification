"""Auxiliary weak targets are supervised only where extraction evidence exists."""
import torch
from torch.nn import functional as F


def masked_auxiliary_bce(logits,targets,matched):
    matched=torch.as_tensor(matched,dtype=torch.bool,device=logits.device)
    if matched.shape!=(len(logits),) or logits.shape!=targets.shape:raise ValueError('Aligned logits, auxiliary targets and evidence mask required')
    count=int(matched.sum())
    if not count:return logits.sum()*0,0
    return F.binary_cross_entropy_with_logits(logits[matched].float(),targets[matched].float()),count


def proxy_distillation_losses(logits,targets,matched,teacher_probabilities):
    matched=torch.as_tensor(matched,dtype=torch.bool,device=logits.device)
    if logits.shape!=teacher_probabilities.shape:raise ValueError('Aligned teacher probability vectors required')
    known_loss,known_count=masked_auxiliary_bce(logits,targets,matched);unknown=~matched;count=int(unknown.sum())
    if not count:return known_loss,known_count,logits.sum()*0,0
    teacher=teacher_probabilities[unknown].float()
    if not torch.isfinite(teacher).all() or not ((teacher>0)&(teacher<1)).all():raise ValueError('Strictly soft teacher probabilities required for unobserved proxy rows')
    return known_loss,known_count,F.binary_cross_entropy_with_logits(logits[unknown].float(),teacher),count
