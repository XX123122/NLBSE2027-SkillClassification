"""Modulate auxiliary correction using the source's fit-only domain coverage."""
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
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from run_channel_residual_capability import base_oof
from skillcomp.benchmark import fast_micro
from skillcomp.capability_search import threshold_choice
from skillcomp.channel_residual import apply_residual
from skillcomp.evidence_coverage import fit_coverage,coverage_weights
from skillcomp.matchability import matchability_features,predicted_coverage_weights
from skillcomp.runtime import ROOT,require_server,provenance,json_write,status


def resolve(run,name,method='best_policy'):
    root=ROOT/'outputs/experiments'/run;pick=json.loads((root/'all_training_selections_locked.json').read_text())['selections'][name][method];cfg=json.loads((root/'config.json').read_text())
    if pick['candidate']=='reference':return resolve(cfg['reference_run'],name,cfg.get('reference_method','best_policy'))
    assert pick['candidate'] in ['codebert_dense','codebert_hybrid'];return root,pick


def case(out,cfg,name,context,mode):
    source,pick=resolve(cfg.get('head_reference_run',cfg['reference_run']),name);folder=source/'contexts'/context/pick['candidate'];role=np.load(source/'contexts'/context/'roles.npz');fit=role['train'];model=joblib.load(folder/'model.joblib');assert digest(folder/'model.joblib')==json.loads((folder/'fit.json').read_text())['model_sha256'];np.testing.assert_array_equal(model['train'],fit);repos=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];aux=np.load(ROOT/'outputs/experiments'/cfg['auxiliary_source_run']/'auxiliary_targets.npz');coverage=fit_coverage(aux['matched'][fit],repos[fit],cfg['coverage_strength']);a=np.load(folder/('scores.npz' if mode=='calibration' else 'test_scores.npz'))
    expected=role['query'] if mode=='calibration' else np.load(ROOT/'cache/workshop_v1/grouped5_pr_components'/(name+'.npz'))['test'];np.testing.assert_array_equal(a['positions'],expected)
    predicted=None
    if cfg.get('learned_matchability'):
        lexical=ROOT/'outputs/experiments'/cfg['lexical_run']/'contexts'/context;np.testing.assert_array_equal(np.load(lexical/'roles.npz')['train'],fit);x=matchability_features(sparse.load_npz(lexical/'title.npz'),sparse.load_npz(lexical/'body.npz'),repos[fit],repos);target=out/'gates'/context
        if mode=='calibration':
            target.mkdir(parents=True);known=aux['matched'][fit].astype(bool);gate=LogisticRegression(C=1,solver='liblinear',max_iter=2000,random_state=42).fit(x[fit],known);joblib.dump({'model':gate,'fit':fit,'fit_coverage':float(known.mean()),'target':'known_evidence_availability_not_skill_presence'},target/'gate.joblib');record={'fit_rows':len(fit),'known_rows':int(known.sum()),'fit_role_sha256':__import__('hashlib').sha256(fit.tobytes()).hexdigest(),'query_evidence_mask_used_for_prediction':False}
            if '_inner' in context:
                truth=aux['matched'][expected].astype(bool);record['training_pool_held_matchability_AUC']=float(roc_auc_score(truth,gate.predict_proba(x[expected])[:,1])) if np.unique(truth).size==2 else None
            json_write(target/'fit.json',record)
        gate=joblib.load(target/'gate.joblib');np.testing.assert_array_equal(gate['fit'],fit);predicted=predicted_coverage_weights(gate['model'],x[expected],gate['fit_coverage'])
    return expected,a['scores'],model['expectation'],coverage,predicted


def weights(cfg,name,coverage,repositories,mix,predicted):
    if not cfg.get('learned_matchability'):return coverage_weights(coverage,repositories,mix)
    reference=json.loads((ROOT/'outputs/experiments'/cfg['reference_run']/'all_training_selections_locked.json').read_text())['selections'][name]['best_policy'];baseline=coverage_weights(coverage,repositories,reference['mix']);assert predicted is not None and len(predicted)==len(repositories);return (1-mix)*baseline+mix*predicted


def choose(job):
    threadpool_limits(1);out=Path(job['out']);cfg=job['config'];name=job['name'];original,base=base_oof(cfg,name);repos=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];lookup={int(p):i for i,p in enumerate(original['positions'])};cases=[];seen=np.zeros(len(base),int)
    for i in range(3):
        query,scores,center,coverage,predicted=case(out,cfg,name,f'{name}_inner{i}','calibration');local=np.array([lookup[int(p)] for p in query]);seen[local]+=1;cases.append((local,query,scores,center,coverage,predicted))
    assert (seen==1).all();records=[]
    for mix in cfg['mixes']:
        for weight in cfg['weights']:
            scores=np.zeros_like(base)
            for local,query,channel,center,coverage,predicted in cases:scores[local]=apply_residual(base[local],channel,repos[query],center,weight,row_weights=weights(cfg,name,coverage,repos[query],mix,predicted))
            records.append(threshold_choice(original['y_true'],scores,original['rf_prediction'])|{'mix':mix,'residual_weight':weight})
    key=lambda r:(r['micro_f1'],r['precision'],-r['mix'],-r['residual_weight']);pick={'new_policy':max([r for r in records if r['mix']>0 and r['residual_weight']>0],key=key),'best_policy':max(records,key=key)};json_write(out/(name+'_OOF_search.json'),records);return name,pick


def run(args,out):
    started=time.monotonic();cfg=json.loads((ROOT/args.config).read_text());provenance(out,cfg);status(out,'running',stage='own_training_coverage_selection');source=ROOT/'outputs/experiments'/cfg['reference_run'];assert json.loads((source/'verification.json').read_text())['state']=='passed';data=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True);y,repos=data['y'],data['repos'];assert digest(ROOT/'cache/workshop_v1/dataset.npz')==json.loads((ROOT/'outputs/experiments'/cfg['auxiliary_source_run']/'input_identity.json').read_text())['dataset_sha256']
    with ProcessPoolExecutor(max_workers=cfg['CPU_workers']) as executor:selected=dict(executor.map(choose,[{'out':str(out),'config':cfg,'name':f'fold{i}'} for i in range(5)]))
    json_write(out/'all_training_selections_locked.json',{'selections':selected,'validation_used':False,'test_used':False});choices={}
    for name,pick in selected.items():
        query,channel,center,coverage,predicted=case(out,cfg,name,name+'_final','calibration');base=np.load(ROOT/'outputs/experiments'/cfg['base_run']/(name+'_calibration.npz'));np.testing.assert_array_equal(base['positions'],query);json_write(out/(name+'_coverage.json'),coverage);scores={};choices[name]={}
        for family,p in pick.items():scores[family]=apply_residual(base['portfolio'],channel,repos[query],center,p['residual_weight'],row_weights=weights(cfg,name,coverage,repos[query],p['mix'],predicted));choices[name][family]=threshold_choice(y[query],scores[family],base['rf_prediction'])
        np.savez_compressed(out/(name+'_calibration.npz'),positions=query,y_true=y[query],rf_prediction=base['rf_prediction'],**scores)
    json_write(out/'all_calibrations_locked.json',{'selections':selected,'choices':choices,'test_used':False,'all_five_contexts_locked':True});methods=['RF','previous_best','new_policy','best_policy'];pooled={m:np.zeros_like(y,dtype=bool) for m in methods};seen=np.zeros(len(y),int);folds=[]
    for name,pick in selected.items():
        positions,channel,center,coverage,predicted=case(out,cfg,name,name+'_final','test');seen[positions]+=1;base=np.load(ROOT/'outputs/experiments'/cfg['base_run']/(name+'_assessment.npz'));reference=np.load(source/(name+'_assessment.npz'));np.testing.assert_array_equal(base['positions'],positions);np.testing.assert_array_equal(reference['positions'],positions);pred={'RF':base['RF_prediction'],'previous_best':reference['best_policy_prediction']};scores={}
        for family,p in pick.items():scores[family]=apply_residual(base['portfolio_scores'],channel,repos[positions],center,p['residual_weight'],row_weights=weights(cfg,name,coverage,repos[positions],p['mix'],predicted));choice=choices[name][family];pred[family]=pred['RF'] if choice['fallback_rf'] else scores[family]>=choice['threshold']
        for method in methods:pooled[method][positions]=pred[method]
        np.savez_compressed(out/(name+'_assessment.npz'),positions=positions,y_true=y[positions],**{m+'_prediction':v for m,v in pred.items()},**{m+'_scores':v for m,v in scores.items()});folds.append({'name':name,'selections':pick,'choices':choices[name],'metrics':{m:fast_micro(y[positions],v) for m,v in pred.items()}})
    assert (seen==1).all();metrics={m:fast_micro(y,v) for m,v in pooled.items()};gates={m:all(value[k]>=metrics['RF'][k] for k in metrics['RF']) for m,value in metrics.items()};np.savez_compressed(out/'pooled_predictions.npz',y_true=y,**pooled);result={'run_id':args.run_id,'protocol':cfg['protocol'],'rows':len(y),'metrics':metrics,'RF_point_gates':gates,'folds':folds,'all_choices_locked_before_assessment':True,'new_encoder_fits':0,'new_auxiliary_head_fits':0,'runtime_seconds':time.monotonic()-started,'limits':cfg['limits']};json_write(out/'results.json',result);report=ROOT/cfg['report_directory'];report.mkdir(exist_ok=True)
    for file in ['results.json','all_training_selections_locked.json','all_calibrations_locked.json']:json_write(report/file,json.loads((out/file).read_text()))
    status(out,'complete',metrics=metrics,RF_point_gates=gates)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--config',default='configs/evidence_coverage_gate_capability_v1.json');parser.add_argument('--launch',action='store_true');args=parser.parse_args();require_server()
    if args.launch:
        assert not (ROOT/'outputs/experiments'/args.run_id).exists();env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
        for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
        with (ROOT/'logs'/(args.run_id+'.log')).open('x') as stream:child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--run-id',args.run_id,'--config',args.config],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'state':'launched','PID':child.pid,'run_id':args.run_id}))
    else:
        out=ROOT/'outputs/experiments'/args.run_id;out.mkdir()
        try:run(args,out)
        except BaseException:
            (out/'error.txt').write_text(traceback.format_exc());status(out,'failed',error=traceback.format_exc());raise
