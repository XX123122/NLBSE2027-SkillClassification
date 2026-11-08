"""Use centered auxiliary-model evidence to refine the frozen portfolio."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from run_annotation_channel_capability import feature_array
from skillcomp.annotation_channel import apply_channel
from skillcomp.benchmark import fast_micro
from skillcomp.capability_search import threshold_choice
from skillcomp.channel_residual import fit_expectation,apply_residual
from skillcomp.runtime import ROOT,require_server,json_write,status,provenance


def prepare_center(job):
    threadpool_limits(1);out=Path(job['out']);source=ROOT/'outputs/experiments'/job['config']['channel_run'];cfg=json.loads((source/'config.json').read_text());context=job['context'];role=np.load(source/'contexts'/context/'roles.npz');train=role['train'];folder=source/'contexts'/context/job['candidate']
    expected=json.loads((folder/'fit.json').read_text());assert digest(folder/'model.joblib')==expected['model_sha256'];bundle=joblib.load(folder/'model.joblib');np.testing.assert_array_equal(bundle['train'],train)
    x=feature_array(source,cfg,context,job['candidate'],job['epoch']);aux=np.asarray(bundle['head'].predict_proba(x[train]));counts=bundle['auxiliary_positive_counts'];aux[:,counts==0]=0;aux[:,counts==bundle['auxiliary_fit_rows']]=1
    repositories=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];prediction=apply_channel(bundle['channel'],aux,repositories[train]);center=fit_expectation(prediction,repositories[train]);destination=out/'centers'/context;destination.mkdir(parents=True);joblib.dump(center,destination/'expectation.joblib')
    return {'context':context,'fit_rows':len(train),'fit_positions_sha256':__import__('hashlib').sha256(train.tobytes()).hexdigest(),'source_model_sha256':expected['model_sha256'],'query_predictions_used_for_center':False}


def base_oof(cfg,name):
    root=ROOT/'outputs/experiments'/cfg['base_run'];a=np.load(root/(name+'_oof_scores.npz'));selection=json.loads((root/'all_training_selections_locked.json').read_text())['selections'][name]['portfolio'];original=np.load(ROOT/'outputs/experiments/20261021_input_preservation_inner_kd_v1'/name/'inner_oof_inputs.npz');np.testing.assert_array_equal(a['positions'],original['positions'])
    if selection['candidate']=='reference_adaptive':
        source_cfg=json.loads((root/'config.json').read_text());reference=ROOT/'outputs/experiments'/source_cfg['reference_run'];selection=json.loads((reference/'all_training_selections_locked.json').read_text())['selections'][name]['new_model'];a=np.load(reference/(name+'_oof_scores.npz'));np.testing.assert_array_equal(a['positions'],original['positions'])
    scores=selection['weight']*a[selection['candidate']]+(1-selection['weight'])*original['prior_scores'];return original,scores


def train_choices(job):
    threadpool_limits(1);out=Path(job['out']);cfg=job['config'];name=job['name'];original,base=base_oof(cfg,name);train=original['positions'];lookup={int(p):i for i,p in enumerate(train)};source=ROOT/'outputs/experiments'/cfg['channel_run'];candidate=job['candidate'];repo=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];scores={str(weight):np.zeros(base.shape) for weight in cfg['weights']};seen=np.zeros(len(train),int)
    for index in range(3):
        context=f'{name}_inner{index}';query=np.load(source/'contexts'/context/'roles.npz')['query'];local=np.array([lookup[int(p)] for p in query]);seen[local]+=1;channel=np.load(source/'contexts'/context/candidate/'scores.npz');np.testing.assert_array_equal(channel['positions'],query);center=joblib.load(out/'centers'/context/'expectation.joblib')
        for weight in cfg['weights']:scores[str(weight)][local]=apply_residual(base[local],channel['scores'],repo[query],center,weight)
    assert (seen==1).all();records=[]
    for weight in cfg['weights']:records.append(threshold_choice(original['y_true'],scores[str(weight)],original['rf_prediction'])|{'residual_weight':weight})
    key=lambda row:(row['micro_f1'],row['precision'],-row['residual_weight']);best=max(records,key=key);new=max([row for row in records if row['residual_weight']>0],key=key)
    json_write(out/(name+'_OOF_search.json'),records);np.savez_compressed(out/(name+'_oof_scores.npz'),positions=train,y_true=original['y_true'],base=base,**{'w'+key:value for key,value in scores.items()});return name,{'new_policy':new,'best_policy':best,'channel_candidate':candidate}


def run(args,out):
    started=time.monotonic();cfg=json.loads((ROOT/'configs/channel_residual_capability_v1.json').read_text());provenance(out,cfg|{'run_id':args.run_id});source=ROOT/'outputs/experiments'/cfg['channel_run'];base_root=ROOT/'outputs/experiments'/cfg['base_run'];assert json.loads((source/'verification.json').read_text())['state']=='passed' and json.loads((base_root/'verification.json').read_text())['state']=='passed'
    source_selection=json.loads((source/'all_training_selections_locked.json').read_text())['selections'];epochs=json.loads((source/'input_identity.json').read_text())['source_epochs'];jobs=[]
    for name,selection in source_selection.items():
        for suffix in ['inner0','inner1','inner2','final']:jobs.append({'out':str(out),'config':cfg,'context':name+'_'+suffix,'candidate':selection['new_policy']['candidate'],'epoch':epochs[name]})
    with ProcessPoolExecutor(max_workers=cfg['CPU_workers']) as executor:
        records=[]
        for future in as_completed([executor.submit(prepare_center,job) for job in jobs]):records.append(future.result());status(out,'running',stage='fit_prediction_expectations',completed=len(records),total=len(jobs))
        json_write(out/'expectation_records.json',records);selected=dict(executor.map(train_choices,[{'out':str(out),'config':cfg,'name':name,'candidate':selection['new_policy']['candidate']} for name,selection in source_selection.items()]))
    json_write(out/'all_training_selections_locked.json',{'selections':selected,'validation_used':False,'test_used':False});data=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True);y,repos=data['y'],data['repos'];roles={};choices={}
    for name,selection in selected.items():
        role=np.load(ROOT/'cache/workshop_v1/grouped5_pr_components'/(name+'.npz'));roles[name]={key:role[key] for key in ['train','validation','test']};query=role['validation'];base=np.load(base_root/(name+'_calibration.npz'));np.testing.assert_array_equal(base['positions'],query)
        channel=np.load(source/'contexts'/(name+'_final')/selection['channel_candidate']/'scores.npz');np.testing.assert_array_equal(channel['positions'],query);center=joblib.load(out/'centers'/(name+'_final')/'expectation.joblib');choices[name]={};scores={}
        for family in ['new_policy','best_policy']:
            weight=selection[family]['residual_weight'];scores[family]=apply_residual(base['portfolio'],channel['scores'],repos[query],center,weight);choices[name][family]=threshold_choice(y[query],scores[family],base['rf_prediction'])
        np.savez_compressed(out/(name+'_calibration.npz'),positions=query,y_true=y[query],rf_prediction=base['rf_prediction'],**scores)
    json_write(out/'all_calibrations_locked.json',{'selections':selected,'choices':choices,'test_used':False,'all_five_contexts_locked':True});methods=['RF','previous_best','new_policy','best_policy'];pooled={key:np.zeros_like(y,dtype=bool) for key in methods};seen=np.zeros(len(y),int);folds=[]
    for name,role in roles.items():
        test=role['test'];seen[test]+=1;selection=selected[name];base=np.load(base_root/(name+'_assessment.npz'));np.testing.assert_array_equal(base['positions'],test);channel=np.load(source/'contexts'/(name+'_final')/selection['channel_candidate']/'test_scores.npz');np.testing.assert_array_equal(channel['positions'],test);center=joblib.load(out/'centers'/(name+'_final')/'expectation.joblib');pred={'RF':base['RF_prediction'],'previous_best':base['portfolio_prediction']};scores={}
        for family in ['new_policy','best_policy']:
            scores[family]=apply_residual(base['portfolio_scores'],channel['scores'],repos[test],center,selection[family]['residual_weight']);choice=choices[name][family];pred[family]=base['RF_prediction'] if choice['fallback_rf'] else scores[family]>=choice['threshold']
        for method in methods:pooled[method][test]=pred[method]
        np.savez_compressed(out/(name+'_assessment.npz'),positions=test,y_true=y[test],**{key+'_prediction':value for key,value in pred.items()},**{key+'_scores':value for key,value in scores.items()});folds.append({'name':name,'selections':selection,'choices':choices[name],'metrics':{key:fast_micro(y[test],value) for key,value in pred.items()}})
    assert (seen==1).all();metrics={key:fast_micro(y,value) for key,value in pooled.items()};gates={key:all(value[m]>=metrics['RF'][m] for m in metrics['RF']) for key,value in metrics.items()};np.savez_compressed(out/'pooled_predictions.npz',y_true=y,**pooled)
    result={'run_id':args.run_id,'protocol':cfg['protocol'],'rows':len(y),'metrics':metrics,'RF_point_gates':gates,'folds':folds,'new_encoder_fits':0,'new_auxiliary_head_fits':0,'expectations_from_fit_rows_only':True,'all_choices_locked_before_assessment':True,'runtime_seconds':time.monotonic()-started,'limits':cfg['limits']};json_write(out/'results.json',result);report=ROOT/cfg['report_directory'];report.mkdir(exist_ok=True)
    for filename in ['results.json','expectation_records.json','all_training_selections_locked.json','all_calibrations_locked.json']:json_write(report/filename,json.loads((out/filename).read_text()))
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
