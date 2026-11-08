"""Separate title and body channels, fitted only in the assigned training role."""
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from skillcomp.models import estimator, continuous_scores
from skillcomp.runtime import json_write
from skillcomp.capability_search import select_candidate
from skillcomp.learned_features import load_context_embeddings
from skillcomp.paired_dimensions import truncate_paired_embeddings
from capability_search_workers import digest

DATA=None


def initialize(root, out):
    global DATA
    threadpool_limits(1); root=Path(root); out=Path(out)
    data=np.load(root/'cache/workshop_v1/dataset.npz',allow_pickle=True)
    fields=json.loads((out/'input_fields.json').read_text())
    config=json.loads((out/'config.json').read_text())
    DATA={'root':root,'out':out,'y':data['y'],'repos':data['repos'],
          'title':np.array(fields['title']),'body':np.array(fields['body']),
          'embeddings':np.load(root/config['embedding_path'],mmap_mode='r') if config.get('embedding_path') else None,'config':config}


def prepare(job):
    folder=DATA['out']/'contexts'/job['context'];folder.mkdir(parents=True)
    role=np.load(DATA['root']/job['role_path']);train=role['train'];query=role[job['query_key']]
    if job['config'].get('reuse_feature_run'):
        source=DATA['root']/'outputs/experiments'/job['config']['reuse_feature_run']/'contexts'/job['context']
        saved=np.load(source/'roles.npz');assert np.array_equal(saved['train'],train) and np.array_equal(saved['query'],query)
        import os
        for filename in ['title.npz','body.npz','title_vectorizer.joblib','body_vectorizer.joblib','roles.npz']:
            os.link(source/filename,folder/filename)
        record={'context':job['context'],'role_path':job['role_path'],'role_sha256':digest(DATA['root']/job['role_path']),
                'reused_feature_source':str(source.relative_to(DATA['root'])),'fit_rows':len(train),'query_rows':len(query),'vocabulary_fit_only':True}
        json_write(folder/'prepared.json',record);return record
    for key,features in [('title',job['config']['title_features']),('body',job['config']['body_features'])]:
        vec=FeatureUnion([
          ('word',TfidfVectorizer(max_features=features['word'],ngram_range=(1,2),min_df=2,sublinear_tf=True)),
          ('char',TfidfVectorizer(max_features=features['char'],analyzer='char_wb',ngram_range=(3,5),min_df=3,sublinear_tf=True))])
        vec.fit(DATA[key][train]);matrix=normalize(vec.transform(DATA[key]),norm='l2')
        sparse.save_npz(folder/(key+'.npz'),matrix.tocsr());joblib.dump(vec,folder/(key+'_vectorizer.joblib'))
    np.savez_compressed(folder/'roles.npz',train=train,query=query)
    record={'context':job['context'],'role_path':job['role_path'],'role_sha256':digest(DATA['root']/job['role_path']),
            'fit_rows':len(train),'query_rows':len(query),'vocabulary_fit_only':True}
    json_write(folder/'prepared.json',record);return record


def features(folder,spec,embedding_override=None):
    parts=[]
    if spec['title_scale']:parts.append(sparse.load_npz(folder/'title.npz')*spec['title_scale'])
    if spec['body']:parts.append(sparse.load_npz(folder/'body.npz'))
    if spec.get('embedding_scale'):
        values=DATA['embeddings'] if embedding_override is None else embedding_override
        if embedding_override is None and spec.get('embedding_path'):values=np.load(DATA['root']/spec['embedding_path'],mmap_mode='r')
        if embedding_override is None and DATA['config'].get('context_embedding_pattern'):
            values=load_context_embeddings(folder,DATA['config'],spec['epoch'],len(DATA['y']))
        if spec.get('embedding_dimension_per_field'):values=truncate_paired_embeddings(values,spec['embedding_dimension_per_field'])
        parts.append(sparse.csr_matrix(values*spec['embedding_scale']))
    return normalize(sparse.hstack(parts,format='csr'),norm='l2')


def fit(job):
    started=time.monotonic();folder=DATA['out']/'contexts'/job['context'];role=np.load(folder/'roles.npz');train,query=role['train'],role['query']
    target=folder/job['spec']['name'];target.mkdir();mapping=None;transformed=None
    if job['spec'].get('kernel_gamma'):
        mapping=Nystroem(kernel='rbf',gamma=job['spec']['kernel_gamma'],n_components=256,random_state=1234,n_jobs=1)
        mapping.fit(DATA['embeddings'][train]);transformed=mapping.transform(DATA['embeddings'])
    x=features(folder,job['spec'],transformed);jobs=job.get('parallel_labels',1)
    with warnings.catch_warnings(),joblib.parallel_backend('threading',n_jobs=jobs):
        warnings.filterwarnings('ignore',message=r'Label .* present in all training examples\.',category=UserWarning)
        model=estimator('lr',c=job['spec']['C'],jobs=jobs);model.fit(x[train],DATA['y'][train]);model.set_params(n_jobs=1)
    score=continuous_scores(model,x[query],DATA['y'][train],'lr')
    np.savez_compressed(target/'scores.npz',positions=query,scores=score);joblib.dump({'model':model,'spec':job['spec'],'train':train,'kernel_map':mapping},target/'model.joblib')
    record={'context':job['context'],'candidate':job['spec']['name'],'runtime_seconds':time.monotonic()-started,'features':x.shape[1],
            'model_sha256':digest(target/'model.joblib'),'scores_sha256':digest(target/'scores.npz'),'independent_label_jobs':jobs,
            'kernel_fit_rows':len(train) if mapping is not None else 0,'kernel_gamma':job['spec'].get('kernel_gamma')}
    json_write(target/'fit.json',record);return record


def select(job):
    name=job['name'];cfg=job['config'];root=DATA['root'];out=DATA['out'];original=root/'outputs/experiments/20261021_input_preservation_inner_kd_v1'/name
    source=np.load(original/'inner_oof_inputs.npz');train=source['positions'];lookup={int(p):i for i,p in enumerate(train)}
    scores={spec['name']:np.zeros(source['y_true'].shape) for spec in cfg['candidates']};seen=np.zeros(len(train),int)
    for index in range(3):
        folder=out/'contexts'/f'{name}_inner{index}';query=np.load(folder/'roles.npz')['query'];local=np.array([lookup[int(p)] for p in query]);seen[local]+=1
        for candidate in scores:
            a=np.load(folder/candidate/'scores.npz');assert np.array_equal(query,a['positions']);scores[candidate][local]=a['scores']
    assert (seen==1).all();choice,records=select_candidate(source['y_true'],scores,source['prior_scores'],source['rf_prediction'],cfg['text_weights'])
    reference_run=cfg.get('reference_run')
    reference=np.load(root/'outputs/experiments'/(reference_run or '20261023_capability_search_v1')/(name+'_oof_scores.npz'))
    assert np.array_equal(reference['positions'],train)
    if reference_run:
        locked=json.loads((root/'outputs/experiments'/reference_run/'all_training_selections_locked.json').read_text())['selections'][name]['new_model']
        reference_score=locked['weight']*reference[locked['candidate']]+(1-locked['weight'])*source['prior_scores']
    else:reference_score=reference['reference_adaptive']
    control,_=select_candidate(source['y_true'],{'reference_adaptive':reference_score},source['prior_scores'],source['rf_prediction'],[1])
    portfolio=max([choice,control],key=lambda r:(r['micro_f1'],r['precision'],r['candidate']))
    json_write(out/(name+'_oof_search.json'),records);np.savez_compressed(out/(name+'_oof_scores.npz'),positions=train,y_true=source['y_true'],rf_prediction=source['rf_prediction'],prior_scores=source['prior_scores'],**scores)
    return name,{'new_model':choice,'portfolio':portfolio,'reference':control}


def predict(job):
    root=DATA['root'];out=DATA['out'];folder=out/'contexts'/job['context'];role=np.load(root/job['role_path']);test=role['test']
    assert (out/'all_calibrations_locked.json').is_file()
    saved=joblib.load(folder/job['spec']['name']/'model.joblib');np.testing.assert_array_equal(role['train'],saved['train'])
    mapping=saved.get('kernel_map');transformed=None if mapping is None else mapping.transform(DATA['embeddings'])
    x=features(folder,job['spec'],transformed);score=continuous_scores(saved['model'],x[test],DATA['y'][saved['train']],'lr')
    np.savez_compressed(folder/job['spec']['name']/'test_scores.npz',positions=test,scores=score)
    return {'context':job['context'],'rows':len(test)}
