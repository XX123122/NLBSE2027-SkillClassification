"""Publication-cohort correction from input-only evidence-availability estimates."""
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from skillcomp.annotation_channel import apply_channel
from skillcomp.channel_residual import apply_residual
from skillcomp.matchability import matchability_features
from skillcomp.technical_text import normalize_raw
from skillcomp.availability_calibration import calibrate_availability


def availability_input_features(bundle,records,fit_repository_names):
    title=normalize(bundle['title_vectorizer'].transform([normalize_raw(row['issue text']) for row in records]),norm='l2');body=normalize(bundle['body_vectorizer'].transform([normalize_raw(row['issue description']) for row in records]),norm='l2')
    return matchability_features(title,body,fit_repository_names,[row['Repo Name'] for row in records])


def semantic_availability_features(lexical,semantic):
    values=np.asarray(semantic)
    if values.ndim!=2 or len(values)!=lexical.shape[0] or not np.isfinite(values).all():raise ValueError('Aligned finite semantic input vectors required')
    return normalize(sparse.hstack([lexical,sparse.csr_matrix(normalize(values,norm='l2'))],format='csr'),norm='l2')


def refine_availability(bundle,component,scores,records,semantic_embeddings=None):
    repos=[row['Repo Name'] for row in records];x=availability_input_features(bundle,records,component['fit_repository_names'])
    if component.get('semantic_feature_dimensions'):
        if semantic_embeddings is None or np.asarray(semantic_embeddings).shape!=(len(records),component['semantic_feature_dimensions']):raise ValueError('Saved semantic availability head requires its input encoder vectors')
        x=semantic_availability_features(x,semantic_embeddings)
    p=np.asarray(component['gate'].predict_proba(x))[:,1]
    if component.get('probability_calibration') is not None:p=calibrate_availability(component['probability_calibration'],p,repos)
    mapped=apply_channel(component['channel'],np.repeat(p[:,None],len(component['channel']['mean']),axis=1),repos)
    return apply_residual(scores,mapped,repos,component['expectation'],component['selection']['residual_weight'])
