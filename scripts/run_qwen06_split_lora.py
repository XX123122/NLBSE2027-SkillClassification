"""Bounded three-GPU warm-start LoRA with train-only policy selection."""
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
from scipy.special import expit
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from skillcomp.models import estimator,continuous_scores
from skillcomp.benchmark import fast_micro
from skillcomp.capability_search import select_candidate,threshold_choice
from skillcomp.runtime import ROOT,require_server,json_write,status,provenance,encoder_python
from skillcomp.component_targets import align_targets
from skillcomp.schema import validate_metadata_inputs
from skillcomp.gpu_policy import snapshot,permitted,reserved_free


def export_head(job):
    threadpool_limits(1);out=Path(job['out']);cfg=job['config'];folder=out/'contexts'/job['context'];folder.mkdir(parents=True)
    roles=np.load(ROOT/job['roles']);train=roles['train'];query=roles[job['query_key']]
    saved_roles={'train':train,'query':query}
    if 'test' in roles.files:saved_roles['test']=roles['test']
    np.savez_compressed(folder/'roles.npz',**saved_roles)
    data=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True);y=data['y'];x=normalize(sparse.csr_matrix(np.load(ROOT/cfg['embedding_path'])),norm='l2')
    if job['inner']:
        source=ROOT/'outputs/experiments'/cfg['reference_run']/'contexts'/job['context']/cfg['warm_head_candidate']/'model.joblib'
        bundle=joblib.load(source);np.testing.assert_array_equal(bundle['train'],train);model=bundle['model'];source_reference={'path':str(source.relative_to(ROOT)),'sha256':digest(source)}
    else:
        with warnings.catch_warnings(),joblib.parallel_backend('threading',n_jobs=8):
            warnings.filterwarnings('ignore',message=r'Label .* present in all training examples\.',category=UserWarning)
            model=estimator('lr',c=1,jobs=8);model.fit(x[train],y[train]);model.set_params(n_jobs=1)
        joblib.dump({'model':model,'train':train},folder/'warm_model.joblib');source_reference={'new_fit_C':1,'path':str((folder/'warm_model.joblib').relative_to(ROOT)),'sha256':digest(folder/'warm_model.joblib')}
    coefficients=np.zeros((y.shape[1],x.shape[1]));intercepts=np.zeros(y.shape[1]);counts=y[train].sum(0);absent=counts==0;universal=counts==len(train)
    for index,head in enumerate(model.estimators_):
        if hasattr(head,'coef_'):coefficients[index]=head.coef_[0];intercepts[index]=head.intercept_[0]
        else:intercepts[index]=np.log(np.clip(counts[index]/len(train),1e-6,1-1e-6)/(1-np.clip(counts[index]/len(train),1e-6,1-1e-6)))
    score=expit(x[query]@coefficients.T+intercepts);score[:,absent]=0;score[:,universal]=1
    expected=continuous_scores(model,x[query],y[train],'lr');np.testing.assert_allclose(score,expected,atol=1e-12,rtol=0)
    np.savez_compressed(folder/'initial_head.npz',coef=coefficients,intercept=intercepts,absent=absent,universal=universal,train=train)
    np.savez_compressed(folder/'warm_scores.npz',positions=query,scores=score)
    json_write(folder/'head_export.json',{'source':source_reference,'roles_sha256':digest(ROOT/job['roles']),'training_rows':len(train),
               'query_rows':len(query),'manual_logits_max_absolute_difference':float(np.abs(score-expected).max()),'initial_head_sha256':digest(folder/'initial_head.npz')})
    json_write(out/(job['context']+'_job.json'),{key:value for key,value in job.items() if key not in ['out','config']})
    return {'context':job['context'],'inner':job['inner'],'training_rows':len(train)}


def gpu_environment(gpu):
    gpus,_,_=snapshot()
    if gpu not in permitted(gpus):raise RuntimeError('GPU is reserved for the other user benchmark: '+str(gpu))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT/'src'),CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false')
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:env[key]='1'
    return env


def gpu_resources():
    gpus,_,_=snapshot()
    return {index:[0,100] if row['excluded'] else [row['free_MiB'],row['utilization']] for index,row in gpus.items()}


def reusable_job(out,job,mode,cfg):
    if not cfg.get('reuse_complete_jobs'):return False
    folder=out/'contexts'/job['context'];role=np.load(folder/'roles.npz')
    if mode=='predict':
        path=folder/job['candidate']/'test_scores.npz'
        if not path.is_file():return False
        a=np.load(path);np.testing.assert_array_equal(a['positions'],role['test']);assert np.isfinite(a['scores']).all();return True
    if mode!='fit':return False
    names=[f'{cfg.get("candidate_prefix","modern")}_e{epoch}' for epoch in cfg['epochs']] if job.get('inner') else [job['candidate']]
    for name in names:
        target=folder/name;required=['model.pt','fit.json','scores.npz']+(['fit_reference_scores.npz'] if cfg.get('save_fit_reference_scores') else [])
        if not all((target/key).is_file() for key in required):return False
        record=json.loads((target/'fit.json').read_text());assert record.get('fit_rows',record.get('training_rows'))==len(role['train']) and record['candidate']==name
        a=np.load(target/'scores.npz');np.testing.assert_array_equal(a['positions'],role['query']);assert np.isfinite(a['scores']).all()
        if cfg.get('save_fit_reference_scores'):
            a=np.load(target/'fit_reference_scores.npz');np.testing.assert_array_equal(a['positions'],role['context_train']);assert np.isfinite(a['scores']).all()
    return True


def preserve_incomplete_job(out,job,mode,cfg):
    folder=out/'contexts'/job['context'];stamp=str(time.time_ns());archive=folder/('interrupted_'+stamp)
    names=[f'{cfg.get("candidate_prefix","modern")}_e{epoch}' for epoch in cfg['epochs']] if job.get('inner') else [job.get('candidate','')]
    paths=[folder/(mode+'.log')]
    if mode=='fit':paths += [folder/name for name in names if name]
    for path in paths:
        if path.exists():archive.mkdir(exist_ok=True);path.rename(archive/path.name)


def reservation_entries(path):
    rows=json.loads(path.read_text()) if path.exists() else [];kept=[]
    for row in rows:
        try:command=Path(f'/proc/{row["PID"]}/cmdline').read_bytes()
        except (FileNotFoundError,ProcessLookupError):continue
        if str(ROOT).encode() in command:kept.append(row)
    return kept


def gpu_queue(out,jobs,mode,cfg):
    import fcntl
    pending=[];running={};records=[];failed=[];last_update=0;status_directory=Path(cfg.get('queue_status_directory',out));registry=ROOT/'cache/gpu_dispatch_reservations.json'
    for job in jobs:
        if reusable_job(out,job,mode,cfg):records.append({'context':job['context'],'mode':mode,'exit_code':0,'reused_completed_fit_or_prediction':True})
        else:
            if cfg.get('reuse_complete_jobs'):preserve_incomplete_job(out,job,mode,cfg)
            pending.append(job)
    json_write(out/(mode+'_queue_records.json'),records)
    while pending or running:
        for pid,(gpu,process,job,stream) in list(running.items()):
            code=process.poll()
            if code is None:continue
            stream.close();record={'context':job['context'],'GPU':gpu,'PID':process.pid,'exit_code':code,'mode':mode};records.append(record)
            if code:failed.append(record)
            del running[pid];json_write(out/(mode+'_queue_records.json'),records)
        if failed and not running:raise RuntimeError('GPU worker failure: '+json.dumps(failed))
        if pending and not failed:
            deferred=set()
            if cfg.get('defer_until_run_terminal'):
                foreign=ROOT/'outputs/experiments'/cfg['defer_until_run_terminal']/'status.json'
                if not foreign.is_file() or json.loads(foreign.read_text())['state'] not in ['complete','failed']:deferred=set(cfg.get('defer_GPU_slots',[]))
            while pending:
                with (ROOT/'cache/gpu_dispatch.lock').open('a+') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);gpus,processes,policy=snapshot();slots=list(permitted(gpus)) if policy.get('expand_to_all_nonexcluded_GPUs') else cfg['GPU_slots'];limit=policy.get('max_parallel_workers_per_queue',cfg.get('max_parallel_GPU',len(slots)))
                    if len(running)>=limit:break
                    needed=cfg['minimum_free_GPU_MiB']
                    if 'codebert' in str(cfg.get('model','')).lower():needed=policy.get({'predict':'CodeBERT_predict_reservation_MiB','extract':'CodeBERT_extract_reservation_MiB'}.get(mode,'CodeBERT_fit_reservation_MiB'),needed)
                    qwen06='qwen3-embedding-0.6b' in str(cfg.get('model','')).lower()
                    if qwen06:needed=policy.get('Qwen06_reservation_MiB',needed)
                    rows=reservation_entries(registry);options=[]
                    for gpu in slots:
                        if gpu not in permitted(gpus) or gpu in deferred:continue
                        if qwen06 and gpus[gpu]['UUID'] in policy.get('Qwen06_excluded_GPU_UUIDs',[]):continue
                        if 'qwen3-embedding-4b' in str(cfg.get('model','')).lower() and gpus[gpu]['UUID'] in policy.get('Qwen4_excluded_GPU_UUIDs',[]):continue
                        row=gpus[gpu];claims=[entry for entry in rows if entry['UUID']==row['UUID']];free=reserved_free(row,processes,claims)
                        if len(claims)>=policy.get('max_workers_per_GPU_per_queue',1) or free<needed:continue
                        if not policy.get('allow_busy_GPU_sharing') and row['utilization']>cfg['maximum_GPU_utilization_at_start']:continue
                        options.append((-len(claims),free,gpu))
                    if not options:json_write(registry,rows);break
                    _,_,gpu=max(options);job=pending.pop(0);json_write(out/(job['context']+'_job.json'),{key:value for key,value in job.items() if key not in ['out','config']});logfile=out/'contexts'/job['context']/(mode+'.log');stream=logfile.open('x')
                    command=[encoder_python(),'-u',str(ROOT/cfg.get('worker_script','scripts/qwen06_split_lora_worker.py')),'--out',str(out.relative_to(ROOT)),'--mode',mode,'--job',job['context']+'_job.json'];process=subprocess.Popen(command,cwd=ROOT,env=gpu_environment(gpu),stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);running[process.pid]=(gpu,process,job,stream);rows.append({'PID':process.pid,'UUID':gpus[gpu]['UUID'],'reservation_MiB':needed,'run_id':out.name,'mode':mode});json_write(registry,rows)
        if time.monotonic()-last_update>10:
            status(status_directory,'running',stage=mode+'_GPU_queue',completed=len(records),total=len(jobs),pending=len(pending),
                   active=[{'GPU':gpu,'PID':process.pid,'context':job['context']} for gpu,process,job,_ in running.values()]);last_update=time.monotonic()
        if pending or running:time.sleep(2)
    return records


def select_outer(job):
    out=Path(job['out']);cfg=job['config'];name=job['name'];source=ROOT/'outputs/experiments'/cfg['reference_run'];original=np.load(ROOT/'outputs/experiments/20261021_input_preservation_inner_kd_v1'/name/'inner_oof_inputs.npz')
    candidates=[f'modern_e{epoch}' for epoch in cfg['epochs']] if cfg.get('family')=='modernbert' else ['warm','head_e1','head_e2','adapter_e1','adapter_e2']
    train=original['positions'];lookup={int(position):local for local,position in enumerate(train)};scores={key:np.zeros(original['y_true'].shape) for key in candidates};seen=np.zeros(len(train),int)
    for index in range(3):
        folder=out/'contexts'/f'{name}_inner{index}';query=np.load(folder/'roles.npz')['query'];local=np.array([lookup[int(p)] for p in query]);seen[local]+=1
        for candidate in scores:
            path=folder/'warm_scores.npz' if candidate=='warm' else folder/candidate/'scores.npz';saved=np.load(path);np.testing.assert_array_equal(query,saved['positions']);scores[candidate][local]=saved['scores']
    assert (seen==1).all()
    new,rows=select_candidate(original['y_true'],scores,original['prior_scores'],original['rf_prediction'],cfg['score_weights'])
    source_oof=np.load(source/(name+'_oof_scores.npz'));np.testing.assert_array_equal(source_oof['positions'],train)
    choice=json.loads((source/'all_training_selections_locked.json').read_text())['selections'][name]['new_model'];reference=choice['weight']*source_oof[choice['candidate']]+(1-choice['weight'])*original['prior_scores']
    control,_=select_candidate(original['y_true'],{'reference_fused':reference},original['prior_scores'],original['rf_prediction'],[1])
    best=max([new,control],key=lambda value:(value['micro_f1'],value['precision'],value['candidate']))
    json_write(out/(name+'_OOF_search.json'),rows);np.savez_compressed(out/(name+'_oof_scores.npz'),positions=train,y_true=original['y_true'],prior_scores=original['prior_scores'],rf_prediction=original['rf_prediction'],reference_fused=reference,**scores)
    return name,{'new_policy':new,'best_policy':best,'reference':control}


def warm_scores(out,folder,positions,cfg):
    initial=np.load(folder/'initial_head.npz');x=normalize(sparse.csr_matrix(np.load(ROOT/cfg['embedding_path'])[positions]),norm='l2')
    scores=expit(x@initial['coef'].T+initial['intercept']);scores[:,initial['absent']]=0;scores[:,initial['universal']]=1
    return scores


def selected_score(out,name,selection,query,mode,cfg):
    source=ROOT/'outputs/experiments'/cfg['reference_run'];candidate=selection['candidate'];folder=out/'contexts'/(name+'_final')
    if candidate=='reference_fused':
        saved=np.load(source/(name+('_calibration.npz' if mode=='calibration' else '_assessment.npz')))
        np.testing.assert_array_equal(saved['positions'],query);return saved['new_model'] if mode=='calibration' else saved['new_model_scores']
    if candidate=='warm':return warm_scores(out,folder,query,cfg)
    saved=np.load(folder/candidate/('scores.npz' if mode=='calibration' else 'test_scores.npz'));np.testing.assert_array_equal(saved['positions'],query);return saved['scores']


def run(args,out):
    started=time.monotonic();cfg=json.loads((ROOT/args.config).read_text());provenance(out,cfg|{'run_id':args.run_id})
    source=ROOT/'outputs/experiments'/cfg['reference_run'];assert json.loads((source/'verification.json').read_text())['state']=='passed'
    features=json.loads((ROOT/cfg['feature_manifest']).read_text());assert digest(ROOT/cfg['embedding_path'])==features['combined_sha256']
    data=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True);y=data['y'];groups=np.load(ROOT/'cache/workshop_v1/key_component_groups.npy');jobs=[];outer={};seen=np.zeros(len(y),int)
    aux_identity=None
    if cfg.get('auxiliary'):
        auxiliary=cfg['auxiliary'];path=ROOT/auxiliary['source'];assert digest(path)==auxiliary['sha256']
        aligned=align_targets(data,np.load(path));matched=aligned['matched'];assert int(matched.sum())==auxiliary['expected_matched_rows']
        targets=np.zeros_like(y);targets[matched]=aligned['companion'][matched]
        np.savez_compressed(out/'auxiliary_targets.npz',targets=targets,matched=matched)
        aux_identity={'source_sha256':digest(path),'matched_rows':int(matched.sum()),'aligned_sha256':digest(out/'auxiliary_targets.npz'),'missing_rows_used_as_negatives':False,'used_at_prediction':False}
    for fold in range(5):
        name=f'fold{fold}';path=f'cache/workshop_v1/grouped5_pr_components/{name}.npz';role=np.load(ROOT/path);outer[name]={key:role[key] for key in ['train','validation','test']};seen[role['test']]+=1
        for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not set(groups[role[a]])&set(groups[role[b]])
        for index in range(3):
            roles=f'outputs/experiments/20261021_input_preservation_inner_kd_v1/{name}/inner_{index}/split.npz';r=np.load(ROOT/roles)
            assert set(r['train'])|set(r['held'])==set(role['train']) and not set(groups[r['train']])&set(groups[r['held']])
            jobs.append({'context':f'{name}_inner{index}','roles':roles,'query_key':'held','inner':True,'out':str(out),'config':cfg})
        jobs.append({'context':name+'_final','roles':path,'query_key':'validation','inner':False,'out':str(out),'config':cfg})
    assert (seen==1).all();json_write(out/'input_identity.json',{'dataset_sha256':digest(ROOT/'cache/workshop_v1/dataset.npz'),'frozen_features_sha256':features['combined_sha256'],
                   'fields_sha256':digest(ROOT/cfg['fields_path']),'role_contexts':20,'outer_test_coverage_once':True,'auxiliary':aux_identity})
    status(out,'running',stage='prepare_train_roles' if cfg.get('family')=='modernbert' else 'export_train_only_warm_heads')
    records=[]
    if cfg.get('family')=='modernbert':
        assert json.loads((ROOT/'reports/modernbert_capability_preparation_v1/training_smoke.json').read_text())['state']=='passed'
        verified=json.loads((ROOT/cfg['model']/'source_verified.json').read_text());assert verified['all_files_verified'] and verified['revision']==cfg['model_revision']
        for job in jobs:
            folder=out/'contexts'/job['context'];folder.mkdir(parents=True);role=np.load(ROOT/job['roles']);values={'train':role['train'],'query':role[job['query_key']]}
            if 'test' in role.files:values['test']=role['test']
            np.savez_compressed(folder/'roles.npz',**values);records.append({'context':job['context'],'initialization':'pretrained backbone + fit-only logit prior head bias'})
    else:
        with ProcessPoolExecutor(max_workers=5) as executor:
            for future in as_completed([executor.submit(export_head,job) for job in jobs]):
                records.append(future.result());status(out,'running',stage='export_train_only_warm_heads',completed=len(records),total=len(jobs))
    json_write(out/('role_initialization_records.json' if cfg.get('family')=='modernbert' else 'head_export_records.json'),records)
    if cfg.get('repository_prefix'):validate_metadata_inputs(['Repo Name','issue text','issue description'],json.loads((ROOT/'cache/workshop_v1/labels.json').read_text()),allow_repo_metadata=True)
    if cfg.get('teacher'):
        with (out/'teacher_preparation.log').open('x') as stream:subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/prepare_training_teacher.py'),'--out',str(out.relative_to(ROOT))],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    command=[encoder_python(),'-u',str(ROOT/cfg.get('worker_script','scripts/qwen06_split_lora_worker.py')),'--out',str(out.relative_to(ROOT)),'--mode','tokenize']
    with (out/'tokenization.log').open('x') as stream:subprocess.run(command,cwd=ROOT,env=gpu_environment(cfg['GPU_slots'][0]),stdout=stream,stderr=subprocess.STDOUT,check=True)
    inner=[job for job in jobs if job['inner']]
    if cfg.get('family')!='modernbert':gpu_queue(out,[inner[0]],'smoke',cfg)
    gpu_queue(out,inner,'fit',cfg)
    with ProcessPoolExecutor(max_workers=5) as executor:selected=dict(executor.map(select_outer,[{'out':str(out),'config':cfg,'name':name} for name in outer]))
    json_write(out/'all_training_selections_locked.json',{'selections':selected,'validation_used':False,'test_used':False})
    final_jobs=[]
    for name in outer:
        candidate=selected[name]['new_policy']['candidate']
        if candidate=='warm':continue
        job=next(value for value in jobs if value['context']==name+'_final');final_jobs.append(dict(job,candidate=candidate))
    gpu_queue(out,final_jobs,'fit',cfg);choices={}
    for name,role in outer.items():
        saved=np.load(ROOT/'outputs/experiments/20261021_input_preservation_assessment_kd_v1'/name/'calibration_inputs.npz');query=role['validation'];np.testing.assert_array_equal(query,saved['positions']);choices[name]={};scores={}
        for family,selection in selected[name].items():
            if family=='reference':continue
            score=selected_score(out,name,selection,query,'calibration',cfg);weight=selection['weight'];scores[family]=weight*score+(1-weight)*saved['prior_scores'];choices[name][family]=threshold_choice(y[query],scores[family],saved['rf_prediction'])
        np.savez_compressed(out/(name+'_calibration.npz'),positions=query,y_true=y[query],rf_prediction=saved['rf_prediction'],**scores)
    json_write(out/'all_calibrations_locked.json',{'selections':selected,'choices':choices,'test_used':False,'all_five_contexts_locked':True})
    gpu_queue(out,final_jobs,'predict',cfg)
    methods=['RF','previous_best','new_policy','best_policy'];pooled={key:np.zeros_like(y,dtype=bool) for key in methods};folds=[]
    for name,role in outer.items():
        test=role['test'];reference=np.load(source/(name+'_assessment.npz'));np.testing.assert_array_equal(reference['positions'],test)
        base=np.load(ROOT/'outputs/experiments/20261021_input_preservation_assessment_kd_v1'/name/'test_inputs.npz');np.testing.assert_array_equal(base['positions'],test)
        pred={'RF':base['rf_prediction'],'previous_best':reference['new_model_prediction']};scores={}
        for family in ['new_policy','best_policy']:
            selection=selected[name][family];score=selected_score(out,name,selection,test,'test',cfg);weight=selection['weight'];scores[family]=weight*score+(1-weight)*base['prior_scores'];choice=choices[name][family]
            pred[family]=base['rf_prediction'] if choice['fallback_rf'] else scores[family]>=choice['threshold']
        for method in methods:pooled[method][test]=pred[method]
        np.savez_compressed(out/(name+'_assessment.npz'),positions=test,y_true=y[test],**{key+'_prediction':value for key,value in pred.items()},**{key+'_scores':value for key,value in scores.items()})
        folds.append({'name':name,'selections':selected[name],'choices':choices[name],'metrics':{key:fast_micro(y[test],value) for key,value in pred.items()}})
    metrics={key:fast_micro(y,value) for key,value in pooled.items()};gates={key:all(value[m]>=metrics['RF'][m] for m in metrics['RF']) for key,value in metrics.items()}
    np.savez_compressed(out/'pooled_predictions.npz',y_true=y,**pooled);result={'run_id':args.run_id,'protocol':cfg['protocol'],'rows':len(y),'metrics':metrics,'RF_point_gates':gates,'folds':folds,
                 'all_choices_locked_before_assessment':True,'inner_full_encoder_fits':15 if cfg.get('family')=='modernbert' else 0,
                 'inner_adapter_fits':0 if cfg.get('family')=='modernbert' else 15,'inner_head_only_fits':0 if cfg.get('family')=='modernbert' else 15,'runtime_seconds':time.monotonic()-started,'limits':cfg['limits']}
    json_write(out/'results.json',result);report=ROOT/cfg['report_directory'];report.mkdir(exist_ok=True)
    for filename in ['results.json','input_identity.json','all_training_selections_locked.json','all_calibrations_locked.json']:json_write(report/filename,json.loads((out/filename).read_text()))
    status(out,'complete',metrics=metrics,RF_point_gates=gates)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--launch',action='store_true');p.add_argument('--config',default='configs/qwen06_split_lora_v1.json');args=p.parse_args();require_server()
    if args.launch:
        assert not (ROOT/'outputs/experiments'/args.run_id).exists();env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
        for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:env[key]='1'
        with (ROOT/'logs'/(args.run_id+'.log')).open('x') as stream:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--run-id',args.run_id,'--config',args.config],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'state':'launched','PID':child.pid,'run_id':args.run_id}))
    else:
        out=ROOT/'outputs/experiments'/args.run_id;out.mkdir()
        try:run(args,out)
        except BaseException:
            (out/'error.txt').write_text(traceback.format_exc());status(out,'failed',error=traceback.format_exc());raise
