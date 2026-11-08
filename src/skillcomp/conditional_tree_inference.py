"""Bundle-only tree and conditional observed-publication inference."""
import numpy as np
from skillcomp.annotation_channel import apply_channel
from skillcomp.availability_inference import availability_input_features,semantic_availability_features
from skillcomp.availability_calibration import calibrate_availability
from skillcomp.peer_union_model import union_probability
from skillcomp.channel_residual import apply_residual
from skillcomp.evidence_coverage import coverage_weights


def conditional_tree_mapping(component,embeddings,repositories,availability):
    q=component['neighbors'].predict_proba(embeddings,repositories);f,counts,n=component['conditional_prior'].predict(repositories);conditional=union_probability(f,availability,q,counts,n);tree=component['tree'].predict_proba(embeddings,repositories);mapped=apply_channel(component['channel'],tree,repositories);return .5*mapped+.5*conditional


def refine_conditional_tree_union(bundle,component,base_scores,embeddings,repositories,records,semantic_embeddings):
    av=bundle['availability_component'];x=availability_input_features(bundle,records,av['fit_repository_names']);x=semantic_availability_features(x,semantic_embeddings);a=av['gate'].predict_proba(x)[:,1]
    if av.get('probability_calibration') is not None:a=calibrate_availability(av['probability_calibration'],a,repositories)
    mapped=conditional_tree_mapping(component,embeddings,repositories,a);row=coverage_weights(component['coverage'],repositories,component['selection']['mix']);return apply_residual(base_scores,mapped,repositories,component['expectation'],.5,row_weights=row)
