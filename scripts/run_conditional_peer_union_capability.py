"""Fivefold conditional publication union, preserving source expert policy hierarchy."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import time
import traceback

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from availability_channel_workers import inputs as availability_inputs
from run_hierarchy_strength_capability import publication_source
from run_proxy_tree_readout_capability import input_values
from run_peer_publication_information import arrays,lineage_inputs,role_targets
from run_channel_residual_capability import base_oof
from run_auxiliary_lexical_residual_capability import reference_oof
from skillcomp.availability_calibration import fit_availability_intercepts,calibrate_availability
from skillcomp.annotation_channel import apply_channel
from skillcomp.benchmark import fast_micro
from skillcomp.capability_search import threshold_choice
from skillcomp.channel_residual import fit_expectation,apply_residual
from skillcomp.conditional_peer_prior import ConditionalPeerPrior
from skillcomp.peer_union_model import union_probability
from skillcomp.evidence_coverage import fit_coverage,coverage_weights
from skillcomp.runtime import ROOT,require_server,provenance,json_write,status


def source_policy(cfg,name):
    source=ROOT/'outputs/experiments'/cfg['reference_run'];scfg=json.loads((source/'config.json').read_text());pick=json.loads((source/'all_training_selections_locked.json').read_text())['selections'][name]['best_policy']
    if pick['candidate']=='tree':return source,scfg,pick,None,None
    assert pick['candidate']=='reference';nn=ROOT/'outputs/experiments'/scfg['reference_run'];ncfg=json.loads((nn/'config.json').read_text());npick=json.loads((nn/'all_training_selections_locked.json').read_text())['selections'][name]['best_policy'];assert npick['candidate']=='neighbors';return source,scfg,pick,nn,npick


def composition(cfg,context,positions,base,mapped,new_expectation,repositories,mode):
    name=context.split('_')[0];source,scfg,pick,nn,npick=source_policy(cfg,name);role=arrays(source/'contexts'/context/'roles.npz');fit=role['train'];data=arrays(ROOT/'cache/workshop_v1/dataset.npz');aux=arrays(ROOT/'outputs/experiments'/scfg['channel_run']/'auxiliary_targets.npz');repos=data['repos'];coverage_pick=json.loads((ROOT/'outputs/experiments'/scfg['coverage_run']/'all_training_selections_locked.json').read_text())['selections'][name]['best_policy'];row_weights=coverage_weights(fit_coverage(aux['matched'][fit],repos[fit],20),repos[positions],coverage_pick['mix']);pub,weight=publication_source(scfg['publication_source_run'],name);pubscores=arrays(pub/'contexts'/context/('test_scores.npz' if mode=='test' else 'scores.npz'));np.testing.assert_array_equal(pubscores['positions'],positions);pubmap=joblib.load(pub/'contexts'/context/'mapping.joblib')
    if pick['candidate']=='tree':
        oldtree=arrays(source/'contexts'/context/('raw_tree_test_scores.npz' if mode=='test' else 'raw_tree_scores.npz'));oldnn=arrays(source/'contexts'/context/('raw_neighbor_test_scores.npz' if mode=='test' else 'raw_neighbor_scores.npz'));tc=joblib.load(source/'contexts'/context/'model.joblib')['expectation'];nc=joblib.load(source/'contexts'/context/'neighbor_component.joblib')['expectation'];fraction=pick['tree_fraction'];frozen=oldtree['mapped'];old=oldnn['mapped'];fixed_center=tc;old_center=nc;new_weight=1-fraction;fixed_weight=fraction;strength=scfg['residual_weight']
    else:
        nmodel=joblib.load(nn/'contexts'/context/'model.joblib');oldp=arrays(nn/'contexts'/context/('test_scores.npz' if mode=='test' else 'scores.npz'));frozen=oldp['original'];old=oldp['knn'];fixed_center=nmodel['old_expectation'];old_center=nmodel['knn_expectation'];new_weight=npick['neighbor_fraction'];fixed_weight=1-new_weight;strength=npick['residual_weight']
    def combine(current,centre):
        mixed=fixed_weight*frozen+new_weight*current;expected={'mean':fixed_weight*fixed_center['mean']+new_weight*centre['mean'],'repositories':{repo:fixed_weight*fixed_center['repositories'][repo]+new_weight*value for repo,value in centre['repositories'].items()}};score=apply_residual(base,mixed,repos[positions],expected,strength,row_weights=row_weights);return apply_residual(score,pubscores['scores'],repos[positions],pubmap['expectation'],weight)
    reference_formula=combine(old,old_center);new=combine(mapped,new_expectation);return new,reference_formula


def context_values(cfg,context):
    source=ROOT/'outputs/experiments'/cfg['reference_run'];scfg=json.loads((source/'config.json').read_text());fit,query,known,data,aux,groups,x,old=input_values(scfg,context);data={key:data[key] for key in ['y','repos','source_positions']};aux={key:aux[key] for key in ['targets','matched']};nn=joblib.load(source/'contexts'/context/'neighbor_component.joblib');return fit,query,known,data,aux,groups,x,nn


def prepare(job):
    threadpool_limits(1);started=time.monotonic();cfg=job['config'];out=Path(job['out']);context=job['context'];name=context.split('_')[0];folder=out/'contexts'/context;folder.mkdir(parents=True);fit,query,known,data,aux,groups,x,nn=context_values(cfg,context);canonical,annotations,pr,raw_counts=lineage_inputs();np.testing.assert_array_equal(data['y'][fit],canonical['y'][fit]);assert not np.intersect1d(groups[fit],groups[query]).size and not np.intersect1d(pr[fit],pr[query]).size;peer=role_targets(canonical,annotations,pr,raw_counts,fit);own=np.zeros(peer.shape,np.uint8);observed=annotations['matched'][fit].astype(bool);own[observed]=annotations['targets'][fit[observed]];prior=ConditionalPeerPrior(cfg['prior_strength']).fit(peer,own,data['repos'][fit],groups[fit],data['y'][fit]);fq,cq,nq=prior.predict(data['repos'][query]);fc,cc,nc=prior.fit_centre(data['repos'][fit],groups[fit]);qquery=nn['neighbors'].predict_proba(x[query],data['repos'][query]);qcentre=nn['neighbors'].predict_fit_centre(x[fit],data['repos'][fit],groups[fit]);source,scfg,_,_,_=source_policy(cfg,name);pub,_=publication_source(scfg['publication_source_run'],name);avcfg=json.loads((pub/'config.json').read_text());mapping=joblib.load(pub/'contexts'/context/'mapping.joblib');gate=joblib.load(ROOT/mapping['gate_source_path']);assert digest(ROOT/mapping['gate_source_path'])==mapping['gate_sha256'];np.testing.assert_array_equal(gate['fit'],fit);xx,repos=availability_inputs(avcfg,context,fit);aquery=gate['model'].predict_proba(xx[query])[:,1]
    if mapping.get('probability_calibration') is not None:aquery=calibrate_availability(mapping['probability_calibration'],aquery,repos[query])
    np.testing.assert_array_equal(apply_channel(mapping['channel'],np.repeat(aquery[:,None],217,axis=1),repos[query]),arrays(pub/'contexts'/context/'scores.npz')['scores']);acentre=np.zeros(len(fit));seen=np.zeros(len(fit),int)
    for k,(a,b) in enumerate(GroupKFold(3).split(fit,groups=groups[fit])):
        assert not np.intersect1d(groups[fit[a]],groups[fit[b]]).size;model=clone(gate['model']);assert model.C==1 and model.solver=='liblinear';model.fit(xx[fit[a]],observed[a]);pt=model.predict_proba(xx[fit[a]])[:,1];raw=model.predict_proba(xx[fit[b]])[:,1];cal=fit_availability_intercepts(pt,observed[a],repos[fit[a]],cfg['availability_intercept_strength']);p=calibrate_availability(cal,raw,repos[fit[b]]);acentre[b]=p;seen[b]+=1;sub=folder/f'sub{k}';sub.mkdir();joblib.dump({'model':model,'calibration':cal},sub/'model.joblib');np.savez_compressed(sub/'roles.npz',train=fit[a],query=fit[b],train_local=a,query_local=b);np.savez_compressed(sub/'probabilities.npz',positions=fit[b],probability=p,raw_probability=raw)
    assert (seen==1).all();mapped=union_probability(fq,aquery,qquery,cq,nq);centre=union_probability(fc,acentre,qcentre,cc,nc);expectation=fit_expectation(centre,repos[fit]);np.savez_compressed(folder/'roles.npz',train=fit,query=query,known=known);np.savez_compressed(folder/'targets.npz',positions=fit,peer=peer,own_observed=own);np.savez_compressed(folder/'inputs.npz',peer_query=fq,peer_centre=fc,proxy_query=qquery,proxy_centre=qcentre,availability_query=aquery,availability_centre=acentre,published_counts_query=cq,published_counts_centre=cc,fit_counts_query=nq,fit_counts_centre=nc);np.savez_compressed(folder/'scores.npz',positions=query,mapped=mapped,centre_published=centre);joblib.dump({'conditional_prior':prior,'expectation':expectation,'availability_source_run':pub.name,'availability_gate_path':mapping['gate_source_path'],'availability_gate_sha256':mapping['gate_sha256'],'availability_mapping':mapping,'availability_config':avcfg},folder/'model.joblib')
    if '_inner' in context:
        original,base=base_oof(cfg,name);lookup={int(p):i for i,p in enumerate(original['positions'])};local=np.array([lookup[int(p)] for p in query]);reference,_=reference_oof(cfg['reference_run'],name,original['positions']);old=reference[local];new,zero=composition(cfg,context,query,base[local],mapped,expectation,repos[query],'inner')
    else:
        base=arrays(ROOT/'outputs/experiments'/cfg['base_run']/(name+'_calibration.npz'));reference=arrays(source/(name+'_calibration.npz'));np.testing.assert_array_equal(query,base['positions']);old=reference['best_policy'];new,zero=composition(cfg,context,query,base['portfolio'],mapped,expectation,repos[query],'validation')
    np.testing.assert_allclose(zero,old,atol=1e-13,rtol=0);np.savez_compressed(folder/'policy_scores.npz',positions=query,reference=old,candidate=new);record={'context':context,'fit_rows':len(fit),'known_API_rows':len(known),'query_rows':len(query),'new_availability_heads':3,'new_skill_or_encoder_models':0,'reference_formula_max_abs':float(np.max(np.abs(zero-old))),'runtime_seconds':time.monotonic()-started};assert record['runtime_seconds']<=cfg['maximum_context_seconds'];json_write(folder/'fit.json',record);return record


def score_test(cfg,out,name):
    assert (out/'all_calibrations_locked.json').exists();locks=json.loads((out/'all_calibrations_locked.json').read_text());assert locks['all_five_contexts_locked'] and len(locks['choices'])==5;context=name+'_final';fit,_,known,data,aux,groups,x,nn=context_values(cfg,context);rows=arrays(ROOT/'cache/workshop_v1/grouped5_pr_components'/(name+'.npz'))['test'];assert not np.intersect1d(groups[fit],groups[rows]).size;folder=out/'contexts'/context;model=joblib.load(folder/'model.joblib');f,c,n=model['conditional_prior'].predict(data['repos'][rows]);q=nn['neighbors'].predict_proba(x[rows],data['repos'][rows]);gate=joblib.load(ROOT/model['availability_gate_path']);assert digest(ROOT/model['availability_gate_path'])==model['availability_gate_sha256'];xx,repos=availability_inputs(model['availability_config'],context,fit);a=gate['model'].predict_proba(xx[rows])[:,1];mapping=model['availability_mapping']
    if mapping.get('probability_calibration') is not None:a=calibrate_availability(mapping['probability_calibration'],a,repos[rows])
    mapped=union_probability(f,a,q,c,n);base=arrays(ROOT/'outputs/experiments'/cfg['base_run']/(name+'_assessment.npz'));reference=arrays(ROOT/'outputs/experiments'/cfg['reference_run']/(name+'_assessment.npz'));np.testing.assert_array_equal(rows,base['positions']);new,zero=composition(cfg,context,rows,base['portfolio_scores'],mapped,model['expectation'],repos[rows],'test');np.testing.assert_allclose(zero,reference['best_policy_scores'],atol=1e-13,rtol=0);np.savez_compressed(folder/'test_inputs.npz',positions=rows,peer=f,proxy=q,availability=a,published_counts=c,fit_counts=n,mapped=mapped);return rows,new,reference['best_policy_scores'],base['RF_prediction']


def run(args,out):
    threadpool_limits(1);started=time.monotonic();cfg=json.loads((ROOT/args.config).read_text());provenance(out,cfg);pilot=ROOT/'outputs/experiments'/cfg['pilot_run'];assert json.loads((pilot/'audit.json').read_text())['state']=='passed' and json.loads((pilot/'training_only_pilot.json').read_text())['full_five_fold_admitted'];json_write(out/'expansion_lock.json',{'config_sha256':digest(ROOT/args.config),'pilot_audit_sha256':digest(pilot/'audit.json'),'pilot_result_sha256':digest(pilot/'training_only_pilot.json'),'all_hyperparameters_frozen':True,'source_policy_hierarchy_preserved':True,'outer_test_used':False});names=[f'fold{i}' for i in range(5)];jobs=[{'out':str(out),'config':cfg,'context':name+'_'+suffix} for name in names for suffix in ['inner0','inner1','inner2','final']];status(out,'running',stage='twenty_conditional_union_contexts',total=20);records=[]
    with ProcessPoolExecutor(max_workers=cfg['CPU_workers'],mp_context=get_context('spawn')) as pool:
        for future in as_completed([pool.submit(prepare,job) for job in jobs]):records.append(future.result());status(out,'running',stage='twenty_conditional_union_contexts',completed=len(records),total=20)
    selections={}
    for name in names:
        original,_=base_oof(cfg,name);positions=original['positions'];lookup={int(p):i for i,p in enumerate(positions)};scores=np.zeros(original['y_true'].shape);control=np.zeros_like(scores);seen=np.zeros(len(positions),int)
        for i in range(3):
            a=arrays(out/'contexts'/f'{name}_inner{i}'/'policy_scores.npz');local=np.array([lookup[int(p)] for p in a['positions']]);scores[local]=a['candidate'];control[local]=a['reference'];seen[local]+=1
        assert (seen==1).all();reference,_=reference_oof(cfg['reference_run'],name,positions);np.testing.assert_array_equal(control,reference)
        if name=='fold0':
            p=arrays(pilot/'training_scores.npz');np.testing.assert_array_equal(p['positions'],positions);np.testing.assert_array_equal(p['scores'],scores);np.testing.assert_array_equal(p['source'],control)
        candidate=threshold_choice(original['y_true'],scores,original['rf_prediction'])|{'candidate':'conditional_union'};baseline=threshold_choice(original['y_true'],control,original['rf_prediction'])|{'candidate':'reference'};best=max([candidate,baseline],key=lambda r:(r['micro_f1'],r['precision'],r['candidate']=='reference'));selections[name]={'new_policy':candidate,'best_policy':best,'reference':baseline};np.savez_compressed(out/(name+'_oof_scores.npz'),positions=positions,candidate=scores,reference=control)
    json_write(out/'all_training_selections_locked.json',{'selections':selections,'validation_used':False,'test_used':False});choices={}
    for name in names:
        a=arrays(out/'contexts'/(name+'_final')/'policy_scores.npz');base=arrays(ROOT/'outputs/experiments'/cfg['base_run']/(name+'_calibration.npz'));np.testing.assert_array_equal(a['positions'],base['positions']);values={};choices[name]={}
        for family in ['new_policy','best_policy']:
            score=a['reference'] if selections[name][family]['candidate']=='reference' else a['candidate'];values[family]=score;choices[name][family]=threshold_choice(base['y_true'],score,base['rf_prediction'])
        np.savez_compressed(out/(name+'_calibration.npz'),positions=a['positions'],y_true=base['y_true'],rf_prediction=base['rf_prediction'],**values)
    json_write(out/'all_calibrations_locked.json',{'selections':selections,'choices':choices,'all_five_contexts_locked':True,'test_used':False});status(out,'running',stage='all_five_locks_complete_then_test_forward');data=arrays(ROOT/'cache/workshop_v1/dataset.npz');y=data['y'];pooled={key:np.zeros(y.shape,bool) for key in ['RF','previous_best','new_policy','best_policy']};seen=np.zeros(len(y),int);folds=[]
    for name in names:
        rows,candidate,reference,rf=score_test(cfg,out,name);old=arrays(ROOT/'outputs/experiments'/cfg['reference_run']/(name+'_assessment.npz'));pred={'RF':rf,'previous_best':old['best_policy_prediction']};scores={}
        for family in ['new_policy','best_policy']:
            score=reference if selections[name][family]['candidate']=='reference' else candidate;choice=choices[name][family];scores[family]=score;pred[family]=rf if choice['fallback_rf'] else score>=choice['threshold']
        for key,value in pred.items():pooled[key][rows]=value
        seen[rows]+=1;np.savez_compressed(out/(name+'_assessment.npz'),positions=rows,y_true=y[rows],**{key+'_prediction':value for key,value in pred.items()},**{key+'_scores':value for key,value in scores.items()});folds.append({'name':name,'selections':selections[name],'choices':choices[name],'metrics':{key:fast_micro(y[rows],value) for key,value in pred.items()}})
    assert (seen==1).all();metrics={key:fast_micro(y,pred) for key,pred in pooled.items()};gates={key:all(value[k]>=metrics['RF'][k] for k in metrics['RF']) for key,value in metrics.items()};np.savez_compressed(out/'pooled_predictions.npz',y_true=y,**pooled);result={'run_id':args.run_id,'protocol':cfg['protocol'],'rows':len(y),'labels':y.shape[1],'metrics':metrics,'RF_point_gates':gates,'folds':folds,'all_choices_locked_before_assessment':True,'new_availability_heads':60,'new_skill_heads_or_encoders':0,'GPU_work':0,'runtime_seconds':time.monotonic()-started,'limits':cfg['limits']};json_write(out/'results.json',result);json_write(out/'fit_records.json',records);report=ROOT/cfg['report_directory'];report.mkdir(exist_ok=True)
    for file in ['results.json','fit_records.json','expansion_lock.json','all_training_selections_locked.json','all_calibrations_locked.json']:json_write(report/file,json.loads((out/file).read_text()))
    status(out,'complete',metrics=metrics,RF_point_gates=gates)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--config',default='configs/conditional_peer_union_capability_v1.json');p.add_argument('--launch',action='store_true');args=p.parse_args();require_server()
    if args.launch:
        assert not (ROOT/'outputs/experiments'/args.run_id).exists();env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+':'+str(ROOT/'scripts')
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
        with (ROOT/'logs'/(args.run_id+'.log')).open('x') as stream:child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--run-id',args.run_id,'--config',args.config],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'state':'launched','PID':child.pid,'run_id':args.run_id}))
    else:
        out=ROOT/'outputs/experiments'/args.run_id;out.mkdir()
        try:run(args,out)
        except BaseException:(out/'error.txt').write_text(traceback.format_exc());status(out,'failed',error=traceback.format_exc());raise
