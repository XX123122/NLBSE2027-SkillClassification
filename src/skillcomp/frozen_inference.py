"""Inference-only feature/prior assembly, without database or training labels."""
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from skillcomp.paired_dimensions import truncate_paired_embeddings
from skillcomp.technical_text import normalize_raw


def validate_records(records):
    expected={'issue text','issue description','Repo Name'}
    if not isinstance(records,list) or any(not isinstance(row,dict) or set(row)!=expected or any(not isinstance(value,str) for value in row.values()) for row in records):
        raise ValueError('JSON array of issue text, issue description and Repo Name strings required')


def assemble_features(bundle,records,embeddings):
    spec=bundle['spec'];parts=[]
    if spec['title_scale']:
        title=normalize(bundle['title_vectorizer'].transform([normalize_raw(row['issue text']) for row in records]),norm='l2');parts.append(title*spec['title_scale'])
    if spec['body']:
        body=normalize(bundle['body_vectorizer'].transform([normalize_raw(row['issue description']) for row in records]),norm='l2');parts.append(body)
    if spec.get('embedding_scale'):
        values=np.asarray(embeddings)
        if spec.get('embedding_dimension_per_field'):values=truncate_paired_embeddings(values,spec['embedding_dimension_per_field'])
        parts.append(sparse.csr_matrix(values*spec['embedding_scale']))
    return normalize(sparse.hstack(parts,format='csr'),norm='l2')


def repository_probabilities(bundle,repositories):
    mean=bundle['prior_mean'];strength=bundle['prior_strength'];output=np.tile(mean,(len(repositories),1))
    for index,repo in enumerate(repositories):
        if repo in bundle['repository_counts']:
            count,positive=bundle['repository_counts'][repo];output[index]=(positive+strength*mean)/(count+strength)
    return output


def predict_with_embeddings(bundle,records,embeddings,clean_text):
    validate_records(records)
    if bundle['selection']['fallback_rf']:
        text=[clean_text(row['issue text'])+' '+clean_text(row['issue description']) for row in records]
        prediction=bundle['rf_model'].predict(bundle['rf_vectorizer'].transform(text).toarray()).astype(bool)
        return prediction,prediction.astype(float)
    x=assemble_features(bundle,records,embeddings);score=np.asarray(bundle['model'].predict_proba(x));positive=bundle['positive_counts'];score[:,positive==0]=0;score[:,positive==bundle['training_rows']]=1
    prior=repository_probabilities(bundle,[row['Repo Name'] for row in records]);weight=bundle['selection']['weight'];fused=weight*score+(1-weight)*prior
    return fused>=bundle['selection']['threshold'],fused
