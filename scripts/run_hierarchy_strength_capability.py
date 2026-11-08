"""Separate root/child auxiliary strengths with existing experts held fixed."""
import argparse
from concurrent.futures import ProcessPoolExecutor
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
from run_channel_residual_capability import base_oof
from run_evidence_coverage_gate_capability import case
from skillcomp.benchmark import fast_micro
from skillcomp.capability_search import threshold_choice
from skillcomp.channel_residual import apply_residual
from skillcomp.evidence_coverage import coverage_weights
from skillcomp.label_hierarchy import parents_from_labels
from skillcomp.runtime import ROOT,require_server,provenance,json_write,status


def publication_source(run,name):
    root=ROOT/'outputs/experiments'/run;cfg=json.loads((root/'config.json').read_text());pick=json.loads((root/'all_training_selections_locked.json').read_text())['selections'][name]['best_policy']
    if cfg.get('channel_kind')!='availability':return None,0
    if pick['candidate']=='reference':return publication_source(cfg['reference_run'],name)
    assert pick['candidate']=='availability';return root,pick['residual_weight']


def prepare_context(out,cfg,name,context,base,mode):
    source=ROOT/'outputs/experiments'/cfg['coverage_run'];source_cfg=json.loads((source/'config.json').read_text());pick=json.loads((source/'all_training_selections_locked.json').read_text())['selections'][name]['best_policy'];query,aux,center,coverage,_=case(source,source_cfg,name,context,mode);repo=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'][query];row=coverage_weights(coverage,repo,pick['mix']);publication,weight=publication_source(cfg['reference_run'],name);folder=out/'contexts'/context;folder.mkdir(parents=True,exist_ok=True);values={'positions':query,'base':base,'auxiliary':aux,'row_weights':row};mapping={'auxiliary_expectation':center,'original_auxiliary_weight':pick['residual_weight'],'publication_weight':weight,'fit_only_row_coverage_mix':pick['mix'],'publication_source':None if publication is None else publication.name}
    if publication is not None:
        old=np.load(publication/'contexts'/context/('scores.npz' if mode=='calibration' else 'test_scores.npz'));np.testing.assert_array_equal(old['positions'],query);pub_map=joblib.load(publication/'contexts'/context/'mapping.joblib');mapping['publication_expectation']=pub_map['expectation'];values['publication']=old['scores']
    np.savez_compressed(folder/('inputs.npz' if mode=='calibration' else 'test_inputs.npz'),**values);joblib.dump(mapping,folder/'mapping.joblib');return query


def corrected(out,context,root_weight,child_weight,roots,repositories,test=False):
    folder=out/'contexts'/context;a=np.load(folder/('test_inputs.npz' if test else 'inputs.npz'));mapping=joblib.load(folder/'mapping.joblib');repos=repositories[a['positions']]
    if root_weight==child_weight:score=apply_residual(a['base'],a['auxiliary'],repos,mapping['auxiliary_expectation'],root_weight,row_weights=a['row_weights'])
    else:score=apply_residual(a['base'],a['auxiliary'],repos,mapping['auxiliary_expectation'],.5,label_weights=np.where(roots,root_weight,child_weight)/.5,row_weights=a['row_weights'])
    if mapping['publication_source'] is not None:score=apply_residual(score,a['publication'],repos,mapping['publication_expectation'],mapping['publication_weight'])
    return a['positions'],score


def prepare_fold(job):
    threadpool_limits(1);out=Path(job['out']);cfg=job['config'];name=job['name'];original,base=base_oof(cfg,name);lookup={int(p):i for i,p in enumerate(original['positions'])};roots=np.asarray(parents_from_labels(json.loads((ROOT/'cache/workshop_v1/labels.json').read_text())))<0;repos=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];seen=np.zeros(len(base),int);control=np.zeros_like(base)
    source=ROOT/'outputs/experiments'/cfg['coverage_run'];source_cfg=json.loads((source/'config.json').read_text());pick=json.loads((source/'all_training_selections_locked.json').read_text())['selections'][name]['best_policy']
    for index in range(3):
        context=f'{name}_inner{index}';query=case(source,source_cfg,name,context,'calibration')[0];local=np.array([lookup[int(p)] for p in query]);seen[local]+=1;prepare_context(out,cfg,name,context,base[local],'calibration');_,control[local]=corrected(out,context,pick['residual_weight'],pick['residual_weight'],roots,repos)
    assert (seen==1).all();choice=threshold_choice(original['y_true'],control,original['rf_prediction']);expected=json.loads((ROOT/'outputs/experiments'/cfg['reference_run']/'all_training_selections_locked.json').read_text())['selections'][name]['best_policy']
    for key in ['precision','recall','micro_f1','threshold']:assert abs(choice[key]-expected[key])<1e-13
    np.savez_compressed(out/(name+'_training.npz'),positions=original['positions'],y_true=original['y_true'],rf=original['rf_prediction'],control=control);return {'name':name,'source_training_choice_exactly_replayed':True,'roots':int(roots.sum()),'children':int((~roots).sum())}


def score_job(job):
    threadpool_limits(1);out=Path(job['out']);name=job['name'];a=np.load(out/(name+'_training.npz'));roots=np.asarray(parents_from_labels(json.loads((ROOT/'cache/workshop_v1/labels.json').read_text())))<0;repos=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];lookup={int(p):i for i,p in enumerate(a['positions'])};scores=np.zeros(a['y_true'].shape);seen=np.zeros(len(scores),int)
    for index in range(3):
        query,value=corrected(out,f'{name}_inner{index}',job['root_weight'],job['child_weight'],roots,repos);local=np.array([lookup[int(p)] for p in query]);scores[local]=value;seen[local]+=1
    assert (seen==1).all();choice=threshold_choice(a['y_true'],scores,a['rf']);return name,choice|{'candidate':'hierarchy_strength','root_weight':job['root_weight'],'child_weight':job['child_weight']}


def run(args,out):
    started=time.monotonic();cfg=json.loads((ROOT/args.config).read_text());provenance(out,cfg);status(out,'running',stage='prepare_fixed_components_and_replay_control');source=ROOT/'outputs/experiments'/cfg['reference_run'];assert json.loads((source/'verification.json').read_text())['state']=='passed';assert digest(ROOT/'cache/workshop_v1/dataset.npz')==cfg['dataset_sha256'];names=[f'fold{i}' for i in range(5)];common={'out':str(out),'config':cfg};selected={}
    with ProcessPoolExecutor(max_workers=cfg['CPU_workers']) as executor:
        replay=list(executor.map(prepare_fold,[common|{'name':name} for name in names]));json_write(out/'source_training_control_replay.json',{'state':'passed','folds':replay});status(out,'running',stage='select_root_child_strengths_on_training_OOF');jobs=[common|{'name':name,'root_weight':rw,'child_weight':cw} for name in names for rw in cfg['weights'] for cw in cfg['weights']];records={name:[] for name in names}
        for name,row in executor.map(score_job,jobs):records[name].append(row)
    for name,rows in records.items():
        a=np.load(out/(name+'_training.npz'));control=threshold_choice(a['y_true'],a['control'],a['rf'])|{'candidate':'reference','root_weight':0,'child_weight':0};key=lambda r:(r['micro_f1'],r['precision'],int(r['candidate']=='reference'),-abs(r['root_weight']-r['child_weight']),-r['root_weight']-r['child_weight']);selected[name]={'new_policy':max([r for r in rows if r['root_weight']!=r['child_weight']],key=key),'best_policy':max(rows+[control],key=key)};json_write(out/(name+'_OOF_search.json'),rows+[control])
    json_write(out/'all_training_selections_locked.json',{'selections':selected,'validation_used':False,'test_used':False});roots=np.asarray(parents_from_labels(json.loads((ROOT/'cache/workshop_v1/labels.json').read_text())))<0;data=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True);y,repos=data['y'],data['repos'];choices={}
    for name in names:
        base=np.load(ROOT/'outputs/experiments'/cfg['base_run']/(name+'_calibration.npz'));reference=np.load(source/(name+'_calibration.npz'));positions=prepare_context(out,cfg,name,name+'_final',base['portfolio'],'calibration');np.testing.assert_array_equal(base['positions'],positions);np.testing.assert_array_equal(reference['positions'],positions);scores={};choices[name]={};original_weight=joblib.load(out/'contexts'/(name+'_final')/'mapping.joblib')['original_auxiliary_weight'];_,original_score=corrected(out,name+'_final',original_weight,original_weight,roots,repos);np.testing.assert_allclose(original_score,reference['best_policy'],rtol=0,atol=1e-13)
        for family,pick in selected[name].items():scores[family]=reference['best_policy'] if pick['candidate']=='reference' else corrected(out,name+'_final',pick['root_weight'],pick['child_weight'],roots,repos)[1];choices[name][family]=threshold_choice(y[positions],scores[family],base['rf_prediction'])
        np.savez_compressed(out/(name+'_calibration.npz'),positions=positions,y_true=y[positions],rf_prediction=base['rf_prediction'],**scores)
    json_write(out/'all_calibrations_locked.json',{'choices':choices,'selections':selected,'test_used':False,'all_five_contexts_locked':True});methods=['RF','previous_best','new_policy','best_policy'];pooled={key:np.zeros_like(y,dtype=bool) for key in methods};seen=np.zeros(len(y),int);folds=[]
    for name in names:
        base=np.load(ROOT/'outputs/experiments'/cfg['base_run']/(name+'_assessment.npz'));reference=np.load(source/(name+'_assessment.npz'));positions=prepare_context(out,cfg,name,name+'_final',base['portfolio_scores'],'test');np.testing.assert_array_equal(base['positions'],positions);np.testing.assert_array_equal(reference['positions'],positions);seen[positions]+=1;pred={'RF':base['RF_prediction'],'previous_best':reference['best_policy_prediction']};scores={};original_weight=joblib.load(out/'contexts'/(name+'_final')/'mapping.joblib')['original_auxiliary_weight'];_,original_score=corrected(out,name+'_final',original_weight,original_weight,roots,repos,test=True);np.testing.assert_allclose(original_score,reference['best_policy_scores'],rtol=0,atol=1e-13)
        for family,pick in selected[name].items():scores[family]=reference['best_policy_scores'] if pick['candidate']=='reference' else corrected(out,name+'_final',pick['root_weight'],pick['child_weight'],roots,repos,test=True)[1];choice=choices[name][family];pred[family]=pred['RF'] if choice['fallback_rf'] else scores[family]>=choice['threshold']
        for key in methods:pooled[key][positions]=pred[key]
        np.savez_compressed(out/(name+'_assessment.npz'),positions=positions,y_true=y[positions],**{key+'_prediction':value for key,value in pred.items()},**{key+'_scores':value for key,value in scores.items()});folds.append({'name':name,'selections':selected[name],'choices':choices[name],'metrics':{key:fast_micro(y[positions],value) for key,value in pred.items()}})
    assert (seen==1).all();metrics={key:fast_micro(y,value) for key,value in pooled.items()};gates={key:all(value[m]>=metrics['RF'][m] for m in metrics['RF']) for key,value in metrics.items()};np.savez_compressed(out/'pooled_predictions.npz',y_true=y,**pooled);result={'run_id':args.run_id,'protocol':cfg['protocol'],'rows':len(y),'metrics':metrics,'RF_point_gates':gates,'folds':folds,'all_choices_locked_before_assessment':True,'new_encoder_fits':0,'new_auxiliary_head_fits':0,'source_training_control_exactly_replayed':True,'runtime_seconds':time.monotonic()-started,'limits':cfg['limits']};json_write(out/'results.json',result);report=ROOT/cfg['report_directory'];report.mkdir(exist_ok=True)
    for file in ['results.json','source_training_control_replay.json','all_training_selections_locked.json','all_calibrations_locked.json']:json_write(report/file,json.loads((out/file).read_text()))
    status(out,'complete',metrics=metrics,RF_point_gates=gates)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--config',default='configs/hierarchy_strength_capability_v1.json');p.add_argument('--launch',action='store_true');args=p.parse_args();require_server()
    if args.launch:
        assert not (ROOT/'outputs/experiments'/args.run_id).exists();env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
        for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
        with (ROOT/'logs'/(args.run_id+'.log')).open('x') as f:child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--run-id',args.run_id,'--config',args.config],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'state':'launched','PID':child.pid,'run_id':args.run_id}))
    else:
        out=ROOT/'outputs/experiments'/args.run_id;out.mkdir()
        try:run(args,out)
        except BaseException:(out/'error.txt').write_text(traceback.format_exc());status(out,'failed',error=traceback.format_exc());raise
