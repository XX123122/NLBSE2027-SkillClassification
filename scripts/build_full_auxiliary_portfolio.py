"""Deploy a training-OOF-selected auxiliary family without refitting encoders."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import warnings

import joblib
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from skillcomp.annotation_channel import fit_channel,apply_channel
from skillcomp.benchmark import priors
from skillcomp.capability_search import threshold_choice
from skillcomp.channel_residual import fit_expectation,apply_residual
from skillcomp.frozen_inference import predict_with_embeddings
from skillcomp.models import estimator,continuous_scores
from skillcomp.paired_dimensions import truncate_paired_embeddings
from skillcomp.residual_inference import refine_scores,fuse_auxiliary_embeddings
from skillcomp.hierarchical_readout import ConditionalHierarchyClassifier
from skillcomp.repository_conditioning import append_repository_indicators
from skillcomp.runtime import ROOT,require_server,provenance,json_write,status,official_module


def feature_spec(candidate):
    return {'title_scale':int(candidate.endswith('_hybrid')),'body':candidate.endswith('_hybrid'),'embedding_scale':1,**({'embedding_dimension_per_field':1024} if candidate=='qwen4_mrl' else {})}


def cached_features(cfg,context,candidate,train):
    base=ROOT/cfg['base_deployment']/'contexts'/context;learned=ROOT/cfg.get('learned_features_root',cfg['learned_deployment'])/'contexts'/context;role=np.load(learned/'roles.npz');np.testing.assert_array_equal(np.load(base/'roles.npz')['train'],train)
    if cfg.get('learned_encoder_known_subset'):
        np.testing.assert_array_equal(role['context_train'],train);aux=np.load(ROOT/cfg['learned_deployment']/'auxiliary_targets.npz');np.testing.assert_array_equal(role['train'],train[aux['matched'][train].astype(bool)])
    else:np.testing.assert_array_equal(role['train'],train)
    if candidate.startswith('learned'):
        path=learned/f'learned_e{cfg.get("epoch",2)}.npy';metadata=json.loads(path.with_suffix('.json').read_text());assert metadata['feature_sha256']==digest(path) and metadata['training_positions_sha256']==__import__('hashlib').sha256(role['train'].tobytes()).hexdigest();values=np.load(path,mmap_mode='r')
        if 'fused' in candidate:values=fuse_auxiliary_embeddings([values,np.load(ROOT/cfg['frozen_features'],mmap_mode='r')])
    else:values=np.load(ROOT/cfg['frozen_features'],mmap_mode='r')
    spec=feature_spec(candidate)
    if spec.get('embedding_dimension_per_field'):values=truncate_paired_embeddings(values,spec['embedding_dimension_per_field'])
    if spec['body']:values=normalize(sparse.hstack([sparse.load_npz(base/'title.npz'),sparse.load_npz(base/'body.npz'),sparse.csr_matrix(values)],format='csr'),norm='l2')
    if cfg.get('auxiliary_repository_scale'):
        repos=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];values=append_repository_indicators(values,repos[train],repos,cfg['auxiliary_repository_scale'])
    return values


def fit_case(job):
    threadpool_limits(1);out=Path(job['out']);cfg=job['config'];context=job['context'];candidate=job['candidate'];role=np.load(ROOT/cfg['base_deployment']/'contexts'/context/'roles.npz');train,query=role['train'],role['query'];data=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True);y,repos=data['y'],data['repos'];aux=np.load(ROOT/cfg['learned_deployment']/'auxiliary_targets.npz');known=aux['matched'][train].astype(bool);target=aux['targets'][train[known]];x=cached_features(cfg,context,candidate,train)
    labels_parallel=job.get('parallel_labels',1)
    with warnings.catch_warnings(),joblib.parallel_backend('threading',n_jobs=labels_parallel):
        warnings.filterwarnings('ignore',message=r'Label .* present in all training examples\.',category=UserWarning);head=ConditionalHierarchyClassifier(json.loads((ROOT/'cache/workshop_v1/labels.json').read_text()),c=cfg['head_C']) if '_tree' in candidate else estimator('lr',c=cfg['head_C'],jobs=labels_parallel);head.fit(x[train[known]],target)
        if hasattr(head,'set_params'):head.set_params(n_jobs=1)
    channel=fit_channel(y[train],aux['targets'][train],known,repos[train],cfg['channel_strength']);expectation=fit_expectation(apply_channel(channel,continuous_scores(head,x[train],target,'lr'),repos[train]),repos[train]);score=apply_channel(channel,continuous_scores(head,x[query],target,'lr'),repos[query]);folder=out/'contexts'/context/candidate;folder.mkdir(parents=True)
    component={'head':head,'channel':channel,'expectation':expectation,'positive_counts':target.sum(0),'matched_training_rows':len(target),'training_rows':len(train),'train':train,'feature_spec':feature_spec(candidate)}
    if cfg.get('auxiliary_repository_scale'):component.update(repository_vocabulary=sorted(set(str(name) for name in repos[train])),repository_scale=cfg['auxiliary_repository_scale'])
    joblib.dump(component,folder/'component.joblib');np.savez_compressed(folder/'scores.npz',positions=query,scores=score);return {'context':context,'candidate':candidate,'training_rows':len(train),'known_rows':len(target),'model_sha256':digest(folder/'component.joblib')}


def run(args,out):
    started=time.monotonic();cfg=json.loads((ROOT/args.config).read_text());provenance(out,cfg);status(out,'running',stage='validate_existing_fit_role_assets');base_root=ROOT/cfg['base_deployment'];learned_root=ROOT/cfg['learned_deployment'];base_manifest=json.loads((base_root/'manifest.json').read_text());learned_manifest=json.loads((learned_root/'manifest.json').read_text());assert digest(base_root/'model.joblib')==base_manifest['model_sha256'];assert digest(learned_root/'model.joblib')==learned_manifest['model_sha256'];assert json.loads((learned_root/'database_independent_inference_verification.json').read_text())['state']=='passed';assert json.loads((ROOT/cfg['development_reference']/'verification.json').read_text())['state']=='passed'
    if cfg.get('learned_encoder_known_subset'):assert learned_manifest['training_OOF_selection']['epoch']==cfg['epoch']
    base=joblib.load(base_root/'model.joblib');learned=joblib.load(learned_root/'model.joblib');manifest=json.loads((ROOT/cfg['frozen_manifest']).read_text());assert digest(ROOT/cfg['frozen_features'])==manifest['combined_sha256'];assert digest(ROOT/'outputs/experiments/20261023_markdown_semantic_features_v1/original_fields.json')==manifest['encoded_original_fields_sha256'];data=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True);y,repos=data['y'],data['repos'];assert digest(ROOT/'cache/workshop_v1/dataset.npz')==base['dataset_sha256'];jobs=[{'out':str(out),'config':cfg,'context':f'full_inner{i}','candidate':candidate} for i in range(3) for candidate in cfg['candidates']]
    with ProcessPoolExecutor(max_workers=cfg['CPU_workers']) as executor:
        records=[]
        for future in as_completed([executor.submit(fit_case,job) for job in jobs]):records.append(future.result());status(out,'running',stage='training_OOF_auxiliary_heads',completed=len(records),total=len(jobs))
        json_write(out/'inner_fit_records.json',records);scores={candidate+'_w'+str(w):np.zeros(y.shape) for candidate in cfg['candidates'] for w in cfg['weights']};rf=np.zeros_like(y,dtype=bool);seen=np.zeros(len(y),int)
        for i in range(3):
            context=f'full_inner{i}';role=np.load(base_root/'contexts'/context/'roles.npz');train,query=role['train'],role['query'];seen[query]+=1;raw=np.load(base_root/'contexts'/context/base['spec']['name']/'scores.npz');np.testing.assert_array_equal(raw['positions'],query);control=np.load(base_root/'contexts'/context/'rf_scores.npz');np.testing.assert_array_equal(control['positions'],query);rf[query]=control['prediction'];baseline=base['selection']['weight']*raw['scores']+(1-base['selection']['weight'])*priors(y[train],repos[train],repos[query],strength=20)
            for candidate in cfg['candidates']:
                folder=out/'contexts'/context/candidate;component=joblib.load(folder/'component.joblib');a=np.load(folder/'scores.npz');np.testing.assert_array_equal(a['positions'],query)
                for w in cfg['weights']:scores[candidate+'_w'+str(w)][query]=apply_residual(baseline,a['scores'],repos[query],component['expectation'],w)
        assert (seen==1).all();search=[]
        for candidate in cfg['candidates']:
            for w in cfg['weights']:search.append(threshold_choice(y,scores[candidate+'_w'+str(w)],rf)|{'candidate':candidate,'residual_weight':w})
        choice=max(search,key=lambda row:(row['micro_f1'],row['precision'],-row['residual_weight'],row['candidate']));json_write(out/'training_selection_locked.json',{'selection':choice,'search':search,'calibration':'training OOF selection, not independent test','new_test_score':None});candidate=choice['candidate'];final=executor.submit(fit_case,{'out':str(out),'config':cfg,'context':'full_final','candidate':candidate,'parallel_labels':12}).result()
    component=joblib.load(out/'contexts/full_final'/candidate/'component.joblib');np.testing.assert_array_equal(component.pop('train'),np.arange(len(y)));component['selection']=choice
    if candidate.endswith('_hybrid'):
        component['title_vectorizer']=base['title_vectorizer'];component['body_vectorizer']=base['body_vectorizer']
    frozen_encoder=dict(base['encoder'],kind='frozen_fields',model='${PRETRAINED_ROOT}/Qwen3-Embedding-4B_'+manifest['model_revision'],revision=manifest['model_revision'],dtype='bfloat16',batch_size=1,whitespace='original',minimum_GPU_MiB=10000)
    assert frozen_encoder['instruction']==json.loads((ROOT/'outputs/experiments/20261023_qwen4_embedding_features_v1/encoder_settings.json').read_text())['instruction']
    if candidate.startswith('learned'):
        encoder=dict(learned['residual_component']['encoder'])
        if cfg.get('learned_encoder_known_subset'):encoder['kind']='learned_encoder'
        if 'fused' in candidate:encoder={'kind':'feature_fusion','encoders':[encoder,frozen_encoder],'minimum_GPU_MiB':10000}
    else:
        encoder=frozen_encoder
        assert encoder['instruction']==json.loads((ROOT/'outputs/experiments/20261023_qwen4_embedding_features_v1/encoder_settings.json').read_text())['instruction']
    component['encoder']=encoder;bundle=dict(base,residual_component=component,kind='full_development_auxiliary_portfolio',independent_score=None,historical_development_score_reference=cfg['development_reference']+'/results.json is historical CV, not a score for this full-data model');joblib.dump(bundle,out/'model.joblib')
    inputs=json.loads((learned_root/'verification_inputs.json').read_text());json_write(out/'verification_inputs.json',inputs);positions=np.arange(len(inputs));base_config=json.loads((base_root/'config.json').read_text());base_features=np.load(ROOT/base_config['representations'][base['spec']['representation']]['embedding_path'],mmap_mode='r')[positions];_,baseline=predict_with_embeddings(base,inputs,base_features,official_module().clean_text);feature_root=ROOT/cfg.get('learned_features_root',cfg['learned_deployment']);aux_features=np.load(feature_root/'contexts/full_final'/f'learned_e{cfg.get("epoch",2)}.npy',mmap_mode='r')[positions] if candidate.startswith('learned') else np.load(ROOT/cfg['frozen_features'],mmap_mode='r')[positions]
    if 'fused' in candidate:aux_features=fuse_auxiliary_embeddings([aux_features,np.load(ROOT/cfg['frozen_features'],mmap_mode='r')[positions]])
    expected_scores=refine_scores(component,baseline,aux_features,repos[positions],records=inputs)
    expected=expected_scores>=choice['threshold']
    if choice['fallback_rf']:expected,_=predict_with_embeddings(dict(base,selection={'fallback_rf':True}),inputs,base_features,official_module().clean_text)
    with (out/'inference_verification.log').open('x') as stream:subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/predict_frozen_candidate.py'),'--bundle',str(out/'model.joblib'),'--input',str(out/'verification_inputs.json'),'--output',str(out/'verification_predictions.json'),'--scores',str(out/'verification_scores.npy')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    prediction=np.array([r['vector'] for r in json.loads((out/'verification_predictions.json').read_text())['predictions']],dtype=bool);actual=np.load(out/'verification_scores.npy');np.testing.assert_array_equal(prediction,expected);delta=float(np.abs(actual-expected_scores).max());assert delta<=1e-5
    result={'state':'complete','model_path':str(out/'model.joblib'),'model_sha256':digest(out/'model.joblib'),'training_rows':len(y),'labels':y.shape[1],'auxiliary_candidate':candidate,'encoder':encoder,'training_OOF_selection':choice,'training_OOF_not_test_score':True,'independent_score':None,'new_benchmark_score_claimed':False,'raw_input_predictions_equal_cached_feature_predictions':True,'verification_rows':len(inputs),'score_max_absolute_difference':delta,'new_encoder_fits':0,'new_base_or_RF_fits':0,'final_fit':final,'runtime_seconds':time.monotonic()-started};json_write(out/'manifest.json',result);report=ROOT/cfg['report_directory'];report.mkdir(exist_ok=True);json_write(report/'manifest.json',result);status(out,'complete',model_path=result['model_path'],verification_passed=True,independent_score=None)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--config',default='configs/full_auxiliary_portfolio_v1.json');parser.add_argument('--launch',action='store_true');args=parser.parse_args();require_server()
    if args.launch:
        assert not (ROOT/'outputs/deployment'/args.run_id).exists();env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
        for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
        with (ROOT/'logs'/(args.run_id+'.log')).open('x') as stream:child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--run-id',args.run_id,'--config',args.config],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'state':'launched','PID':child.pid,'run_id':args.run_id}))
    else:
        out=ROOT/'outputs/deployment'/args.run_id;out.mkdir(parents=True)
        try:run(args,out)
        except BaseException:
            (out/'error.txt').write_text(traceback.format_exc());status(out,'failed',error=traceback.format_exc());raise
