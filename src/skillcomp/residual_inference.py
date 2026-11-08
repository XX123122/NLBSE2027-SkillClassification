"""Inference of a centered auxiliary component without target or corpus access."""
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from skillcomp.annotation_channel import apply_channel
from skillcomp.channel_residual import apply_residual
from skillcomp.frozen_inference import assemble_features
from skillcomp.paired_dimensions import truncate_paired_embeddings
from skillcomp.evidence_coverage import coverage_weights
from skillcomp.repository_conditioning import append_repository_indicators


class ProbabilityReadout:
    """Identity readout for a supervised proxy head whose sigmoid ran on the GPU."""
    def predict_proba(self,values):
        values=np.asarray(values,dtype=float)
        if values.ndim!=2 or not np.isfinite(values).all() or ((values<0)|(values>1)).any():raise ValueError('Finite proxy probabilities required')
        return values.copy()


def fuse_auxiliary_embeddings(arrays):
    arrays=[np.asarray(values) for values in arrays]
    if not arrays or any(values.ndim!=2 or len(values)!=len(arrays[0]) or not np.isfinite(values).all() for values in arrays):raise ValueError('Aligned finite auxiliary embedding matrices required')
    return normalize(sparse.hstack([sparse.csr_matrix(values) for values in arrays],format='csr'),norm='l2').toarray()


def auxiliary_features(component,records,embeddings,repositories=None):
    spec=component.get('feature_spec')
    if spec is None:values=embeddings
    elif spec['title_scale'] or spec['body']:
        if records is None:raise ValueError('Text records required by auxiliary lexical features')
        values=assemble_features({'spec':spec,'title_vectorizer':component['title_vectorizer'],'body_vectorizer':component['body_vectorizer']},records,embeddings)
    else:values=truncate_paired_embeddings(embeddings,spec['embedding_dimension_per_field']) if spec.get('embedding_dimension_per_field') else embeddings
    if 'repository_vocabulary' in component:
        if repositories is None:
            if records is None:raise ValueError('Repository inputs required by the saved conditioned proxy head')
            repositories=[row['Repo Name'] for row in records]
        values=append_repository_indicators(values,component['repository_vocabulary'],repositories,component['repository_scale'])
    return values


def refine_scores(component, base_scores, auxiliary_embeddings, repositories,records=None):
    scores = np.asarray(component['head'].predict_proba(auxiliary_features(component,records,auxiliary_embeddings,repositories)))
    counts = component['positive_counts']; scores[:, counts == 0] = 0; scores[:, counts == component['matched_training_rows']] = 1
    channel = apply_channel(component['channel'], scores, repositories)
    weights=coverage_weights(component['coverage'],repositories,component['selection']['mix']) if 'coverage' in component else None
    return apply_residual(base_scores, channel, repositories, component['expectation'], component['selection']['residual_weight'],row_weights=weights)
