"""Own predicted auxiliary skills mapped to published annotations, fit-only."""
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
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from run_qwen06_split_lora import gpu_queue
from skillcomp.annotation_channel import fit_channel,apply_channel
from skillcomp.benchmark import fast_micro,priors
from skillcomp.capability_search import select_candidate,threshold_choice
from skillcomp.component_targets import align_targets
from skillcomp.learned_features import load_context_embeddings
from skillcomp.models import estimator,continuous_scores
from skillcomp.runtime import ROOT,require_server,json_write,status,provenance


def feature_array(out,cfg,context,candidate,epoch):
    if candidate=='frozen_qwen':return np.load(ROOT/cfg['frozen_features'],mmap_mode='r')
    source=ROOT/'outputs/experiments'/cfg['learned_feature_run'];folder=source/'contexts'/context
    if context.endswith('_final') and (out/'contexts'/context/f'learned_e{epoch}.npy').is_file():folder=out/'contexts'/context
    return load_context_embeddings(folder,{'context_embedding_pattern':'learned_e{epoch}.npy','context_embedding_dimensions':768},epoch,7154)


def fit_case(job):
    threadpool_limits(1);out=Path(job['out']);cfg=job['config'];context=job['context'];role=np.load(out/'contexts'/context/'roles.npz');train,query=role['train'],role['query'];data=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True);y,repos=data['y'],data['repos'];aux=np.load(out/'auxiliary_targets.npz');known=aux['matched'][train].astype(bool);assert known.any()
    x=feature_array(out,cfg,context,job['candidate'],job['epoch']);aux_y=aux['targets'][train[known]]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message=r'Label .* present in all training examples\.',category=UserWarning);head=estimator('lr',c=cfg['auxiliary_head_C'],jobs=1);head.fit(x[train[known]],aux_y)
    channel=fit_channel(y[train],aux['targets'][train],known,repos[train],cfg['channel_strength']);aux_score=continuous_scores(head,x[query],aux_y,'lr');score=apply_channel(channel,aux_score,repos[query])
    folder=out/'contexts'/context/job['candidate'];folder.mkdir();joblib.dump({'head':head,'channel':channel,'train':train,'auxiliary_positive_counts':aux_y.sum(0),'auxiliary_fit_rows':len(aux_y)},folder/'model.joblib');np.savez_compressed(folder/'scores.npz',positions=query,scores=score)
    result={'context':context,'candidate':job['candidate'],'auxiliary_fit_rows':int(known.sum()),'primary_fit_rows':len(train),'contradictory_known_positive_cells':channel['auxiliary_positive_published_negative_known_cells'],'model_sha256':digest(folder/'model.joblib')};json_write(folder/'fit.json',result);return result


def select_outer(job):
    out=Path(job['out']);cfg=job['config'];name=job['name'];source=ROOT/'outputs/experiments'/cfg['reference_run'];original=np.load(ROOT/'outputs/experiments/20261021_input_preservation_inner_kd_v1'/name/'inner_oof_inputs.npz');train=original['positions'];lookup={int(value):index for index,value in enumerate(train)}
    scores={key:np.zeros(original['y_true'].shape) for key in cfg['feature_candidates']};seen=np.zeros(len(train),int)
    for index in range(3):
        folder=out/'contexts'/f'{name}_inner{index}';query=np.load(folder/'roles.npz')['query'];local=np.array([lookup[int(value)] for value in query]);seen[local]+=1
        for candidate in scores:
            a=np.load(folder/candidate/'scores.npz');np.testing.assert_array_equal(a['positions'],query);scores[candidate][local]=a['scores']
    assert (seen==1).all();choice,rows=select_candidate(original['y_true'],scores,original['prior_scores'],original['rf_prediction'],cfg['channel_weights']);reference_oof=np.load(source/(name+'_oof_scores.npz'));np.testing.assert_array_equal(reference_oof['positions'],train)
    reference_selection=json.loads((source/'all_training_selections_locked.json').read_text())['selections'][name]['new_model'];ref=reference_selection['weight']*reference_oof[reference_selection['candidate']]+(1-reference_selection['weight'])*original['prior_scores'];control,_=select_candidate(original['y_true'],{'reference':ref},original['prior_scores'],original['rf_prediction'],[1])
    best=max([choice,control],key=lambda row:(row['micro_f1'],row['precision'],row['candidate']));json_write(out/(name+'_OOF_search.json'),rows);np.savez_compressed(out/(name+'_oof_scores.npz'),positions=train,y_true=original['y_true'],**scores)
    return name,{'new_policy':choice,'best_policy':best,'reference':control}


def test_case(job):
    threadpool_limits(1);out=Path(job['out']);cfg=job['config'];name=job['name'];context=name+'_final';assert (out/'all_calibrations_locked.json').is_file();role=np.load(ROOT/'cache/workshop_v1/grouped5_pr_components'/(name+'.npz'));train,test=role['train'],role['test'];saved=joblib.load(out/'contexts'/context/job['candidate']/'model.joblib');np.testing.assert_array_equal(saved['train'],train)
    x=feature_array(out,cfg,context,job['candidate'],job['epoch']);aux_score=np.asarray(saved['head'].predict_proba(x[test]));count=saved['auxiliary_positive_counts'];aux_score[:,count==0]=0;aux_score[:,count==saved['auxiliary_fit_rows']]=1
    repositories=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];scores=apply_channel(saved['channel'],aux_score,repositories[test]);np.savez_compressed(out/'contexts'/context/job['candidate']/'test_scores.npz',positions=test,scores=scores)
    return {'name':name,'test_rows':len(test)}


def pool(executor,function,jobs,out,stage):
    records=[]
    for future in as_completed([executor.submit(function,job) for job in jobs]):records.append(future.result());status(out,'running',stage=stage,completed=len(records),total=len(jobs))
    json_write(out/(stage+'_records.json'),records);return records


def run(args,out):
    started=time.monotonic();cfg=json.loads((ROOT/'configs/annotation_channel_capability_v1.json').read_text())
    learned=ROOT/'outputs/experiments'/cfg['learned_feature_run'];assert json.loads((learned/'verification.json').read_text())['state']=='passed';learned_cfg=json.loads((learned/'config.json').read_text())
    for key in ['model','model_revision','context_embedding_pattern','context_embedding_dimensions']:cfg[key]=learned_cfg[key]
    provenance(out,cfg|{'run_id':args.run_id});data=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True);y,repos=data['y'],data['repos'];path=ROOT/cfg['auxiliary_source'];assert digest(path)==cfg['auxiliary_sha256']
    aligned=align_targets(data,np.load(path));assert int(aligned['matched'].sum())==3070;targets=np.zeros_like(y);targets[aligned['matched']]=aligned['companion'][aligned['matched']];np.savez_compressed(out/'auxiliary_targets.npz',targets=targets,matched=aligned['matched'])
    encoder_source=ROOT/'outputs/experiments'/learned_cfg['encoder_source_run'];encoder_selection=json.loads((encoder_source/'all_training_selections_locked.json').read_text())['selections'];epochs={name:int(value['new_policy']['candidate'].split('_e')[1]) for name,value in encoder_selection.items()}
    source=ROOT/'outputs/experiments'/cfg['reference_run'];assert json.loads((source/'verification.json').read_text())['state']=='passed';groups=np.load(ROOT/'cache/workshop_v1/key_component_groups.npy');jobs=[];outer={};extractions=[];seen=np.zeros(len(y),int)
    for fold in range(5):
        name=f'fold{fold}';role=np.load(ROOT/'cache/workshop_v1/grouped5_pr_components'/(name+'.npz'));outer[name]={key:role[key] for key in ['train','validation','test']};seen[role['test']]+=1
        for index in range(3):
            context=f'{name}_inner{index}';original=np.load(learned/'contexts'/context/'roles.npz');assert np.isin(original['train'],role['train']).all() and np.isin(original['query'],role['train']).all() and not set(groups[original['train']])&set(groups[original['query']])
            folder=out/'contexts'/context;folder.mkdir(parents=True);np.savez_compressed(folder/'roles.npz',train=original['train'],query=original['query'])
            for candidate in cfg['feature_candidates']:jobs.append({'out':str(out),'config':cfg,'context':context,'candidate':candidate,'epoch':epochs[name]})
        context=name+'_final';folder=out/'contexts'/context;folder.mkdir();np.savez_compressed(folder/'roles.npz',train=role['train'],query=role['validation']);epoch=epochs[name];cached=learned/'contexts'/context/f'learned_e{epoch}.npy';checkpoint=encoder_source/'contexts'/context/f'modern_e{epoch}'/'model.pt'
        original_final=np.load(learned/'contexts'/context/'roles.npz');np.testing.assert_array_equal(original_final['train'],role['train']);np.testing.assert_array_equal(original_final['query'],role['validation'])
        if not cached.exists() or json.loads(cached.with_suffix('.json').read_text())['checkpoint']!=str(checkpoint.relative_to(ROOT)):
            label=context+'_encode';(out/'contexts'/label).mkdir();extractions.append({'context':label,'feature_context':context,'epoch':epoch,'checkpoint':str(checkpoint.relative_to(ROOT)),'encoder_roles':str((encoder_source/'contexts'/context/'roles.npz').relative_to(ROOT)),'token_source':str(encoder_source.relative_to(ROOT))})
    assert (seen==1).all();json_write(out/'input_identity.json',{'dataset_sha256':digest(ROOT/'cache/workshop_v1/dataset.npz'),'auxiliary_source_sha256':digest(path),'matched_rows':3070,'source_epochs':epochs})
    if extractions:
        queue_cfg=learned_cfg|{'worker_script':'scripts/extract_learned_context_features.py','minimum_free_GPU_MiB':4000,'GPU_slots':[4,6,1],'max_parallel_GPU':3}
        records=gpu_queue(out,extractions,'extract',queue_cfg);json_write(out/'final_feature_extractions.json',records)
    with ProcessPoolExecutor(max_workers=cfg['CPU_workers']) as executor:
        pool(executor,fit_case,jobs,out,'inner_auxiliary_fits');selected=dict(pool(executor,select_outer,[{'out':str(out),'config':cfg,'name':name} for name in outer],out,'training_selection'))
        json_write(out/'all_training_selections_locked.json',{'selections':selected,'validation_used':False,'test_used':False,'source_epoch_policy':'fixed from previous own-outer-train source policy'})
        finals=[{'out':str(out),'config':cfg,'context':name+'_final','candidate':selection['new_policy']['candidate'],'epoch':epochs[name]} for name,selection in selected.items()];pool(executor,fit_case,finals,out,'final_auxiliary_fits')
        choices={}
        for name,role in outer.items():
            base=np.load(source/(name+'_calibration.npz'));np.testing.assert_array_equal(base['positions'],role['validation']);candidate=selected[name]['new_policy']['candidate'];channel=np.load(out/'contexts'/(name+'_final')/candidate/'scores.npz');np.testing.assert_array_equal(channel['positions'],role['validation']);prior=priors(y[role['train']],repos[role['train']],repos[role['validation']],strength=cfg['prior_strength']);choices[name]={};scores={}
            for family in ['new_policy','best_policy']:
                selection=selected[name][family];scores[family]=base['new_model'] if selection['candidate']=='reference' else selection['weight']*channel['scores']+(1-selection['weight'])*prior;choices[name][family]=threshold_choice(y[role['validation']],scores[family],base['rf_prediction'])
            np.savez_compressed(out/(name+'_calibration.npz'),positions=role['validation'],y_true=y[role['validation']],rf_prediction=base['rf_prediction'],**scores)
        json_write(out/'all_calibrations_locked.json',{'choices':choices,'selections':selected,'test_used':False,'all_five_contexts_locked':True})
        pool(executor,test_case,[job|{'name':job['context'].removesuffix('_final')} for job in finals],out,'test_auxiliary_predictions')
    methods=['RF','previous_best','new_policy','best_policy'];pooled={key:np.zeros_like(y,dtype=bool) for key in methods};folds=[]
    for name,role in outer.items():
        test=role['test'];old=np.load(source/(name+'_assessment.npz'));np.testing.assert_array_equal(old['positions'],test);candidate=selected[name]['new_policy']['candidate'];channel=np.load(out/'contexts'/(name+'_final')/candidate/'test_scores.npz');np.testing.assert_array_equal(channel['positions'],test);prior=priors(y[role['train']],repos[role['train']],repos[test],strength=cfg['prior_strength']);pred={'RF':old['RF_prediction'],'previous_best':old['new_model_prediction']};scores={}
        for family in ['new_policy','best_policy']:
            selection=selected[name][family];scores[family]=old['new_model_scores'] if selection['candidate']=='reference' else selection['weight']*channel['scores']+(1-selection['weight'])*prior;choice=choices[name][family];pred[family]=old['RF_prediction'] if choice['fallback_rf'] else scores[family]>=choice['threshold']
        for method in methods:pooled[method][test]=pred[method]
        np.savez_compressed(out/(name+'_assessment.npz'),positions=test,y_true=y[test],**{key+'_prediction':value for key,value in pred.items()},**{key+'_scores':value for key,value in scores.items()});folds.append({'name':name,'selections':selected[name],'choices':choices[name],'metrics':{key:fast_micro(y[test],value) for key,value in pred.items()}})
    metrics={key:fast_micro(y,value) for key,value in pooled.items()};gates={key:all(value[m]>=metrics['RF'][m] for m in metrics['RF']) for key,value in metrics.items()};np.savez_compressed(out/'pooled_predictions.npz',y_true=y,**pooled)
    result={'run_id':args.run_id,'protocol':cfg['protocol'],'rows':len(y),'metrics':metrics,'RF_point_gates':gates,'folds':folds,'all_choices_locked_before_assessment':True,'new_encoder_fits':0,'runtime_seconds':time.monotonic()-started,'limits':cfg['limits']};json_write(out/'results.json',result);report=ROOT/cfg['report_directory'];report.mkdir(exist_ok=True)
    for filename in ['results.json','input_identity.json','all_training_selections_locked.json','all_calibrations_locked.json']:json_write(report/filename,json.loads((out/filename).read_text()))
    status(out,'complete',metrics=metrics,RF_point_gates=gates)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--launch',action='store_true');args=p.parse_args();require_server()
    if args.launch:
        assert not (ROOT/'outputs/experiments'/args.run_id).exists();env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
        for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
        with (ROOT/'logs'/(args.run_id+'.log')).open('x') as stream:child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--run-id',args.run_id],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'state':'launched','PID':child.pid,'run_id':args.run_id}))
    else:
        out=ROOT/'outputs/experiments'/args.run_id;out.mkdir()
        try:run(args,out)
        except BaseException:
            (out/'error.txt').write_text(traceback.format_exc());status(out,'failed',error=traceback.format_exc());raise
