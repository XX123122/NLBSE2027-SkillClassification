"""Combine two frozen policy scores while reusing the same input encoders."""
import numpy as np

from skillcomp.availability_inference import refine_availability
from skillcomp.residual_inference import refine_scores


def refine_policy_ensemble(bundle,component,base_scores,current_scores,embeddings,repositories,records,semantic_embeddings=None):
    fraction=component['fraction']
    if not 0<=fraction<=1:raise ValueError('Score mixture must be in the unit interval')
    reference=refine_scores(component['reference_residual_component'],base_scores,embeddings,repositories,records=records);reference=refine_availability(bundle,component['reference_availability_component'],reference,records,semantic_embeddings=semantic_embeddings);current=np.asarray(current_scores,dtype=float)
    if current.shape!=reference.shape or not np.isfinite(current).all() or ((current<0)|(current>1)).any():raise ValueError('Aligned finite policy probabilities required')
    if fraction==0:return reference.copy()
    if fraction==1:return current.copy()
    return (1-fraction)*reference+fraction*current
