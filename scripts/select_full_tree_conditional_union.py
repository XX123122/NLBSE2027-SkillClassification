"""Matching fixed tree-half conditional-NN full-data candidate, no export yet."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import time
import traceback

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from build_full_auxiliary_portfolio import cached_features
from run_peer_publication_information import arrays,lineage_inputs
from skillcomp.proxy_tree_readout import ProxyTreeReadout
from skillcomp.retrieval_evidence import ExcludedChannelStatistics
from skillcomp.annotation_channel import apply_channel
from skillcomp.channel_residual import fit_expectation,apply_residual
from skillcomp.benchmark import fast_micro
from skillcomp.capability_search import threshold_choice
from skillcomp.runtime import ROOT,require_server,provenance,json_write,status


def prepare(job):
    threadpool_limits(1);started=time.monotonic();cfg=job['config'];out=Path(job['out']);context=job['context'];folder=out/'contexts'/context;folder.mkdir(parents=True);union=ROOT/'outputs/deployment'/cfg['conditional_selection_run'];role=arrays(union/'contexts'/context/'roles.npz');fit,query,known=role['train'],role['query'],role['known'];data,aux,pr,raw_counts=lineage_inputs();groups=np.load(ROOT/'cache/workshop_v1/key_component_groups.npy');assert not np.intersect1d(groups[fit],groups[query]).size and not np.intersect1d(pr[fit],pr[query]).size;x=cached_features(dict(cfg,auxiliary_repository_scale=0),context,'learned_fused',fit);template=ProxyTreeReadout(**cfg['tree']).fit_projection(x[fit],data['repos'][fit]);centre=np.zeros((len(fit),217));seen=np.zeros(len(fit),int)
    for k,(a,b) in enumerate(GroupKFold(3).split(fit,groups=groups[fit])):
        train=fit[a];observed=train[aux['matched'][train].astype(bool)];assert not np.intersect1d(groups[observed],groups[fit[b]]).size;head=copy.deepcopy(template).fit_head(x[observed],aux['targets'][observed],data['repos'][observed]);head.head_.set_params(n_jobs=1);centre[b]=head.predict_proba(x[fit[b]],data['repos'][fit[b]]);seen[b]+=1;sub=folder/f'sub{k}';sub.mkdir();joblib.dump(head,sub/'model.joblib');np.savez_compressed(sub/'roles.npz',train=train,known=observed,query=fit[b])
    assert (seen==1).all();head=copy.deepcopy(template).fit_head(x[known],aux['targets'][known],data['repos'][known]);head.head_.set_params(n_jobs=1);q=head.predict_proba(x[query],data['repos'][query]);source=ROOT/cfg['source_deployment'];channel=joblib.load(source/'contexts'/context/'neighbor_component.joblib')['channel'];mapped=apply_channel(channel,q,data['repos'][query]);stats=ExcludedChannelStatistics(data['y'][fit],aux['targets'][fit],aux['matched'][fit],data['repos'][fit],groups[fit]);treecentre=stats.map_fit_centre(centre,data['repos'][fit],groups[fit]);tc=fit_expectation(treecentre,data['repos'][fit]);u=arrays(union/'contexts'/context/'scores.npz');umodel=joblib.load(union/'contexts'/context/'model.joblib');uc=umodel['expectation'];mixed=.5*mapped+.5*u['mapped'];expectation={'mean':.5*tc['mean']+.5*uc['mean'],'repositories':{repo:.5*tc['repositories'][repo]+.5*value for repo,value in uc['repositories'].items()}};cov=apply_residual(u['base'],mixed,data['repos'][query],expectation,.5,row_weights=u['row_weights']);candidate=apply_residual(cov,u['publication'],data['repos'][query],umodel['publication_expectation'],umodel['publication_weight']);np.savez_compressed(folder/'roles.npz',train=fit,query=query,known=known);np.savez_compressed(folder/'tree_probabilities.npz',positions=query,query_proxy=q,centre_proxy=centre,query_mapped=mapped,centre_mapped=treecentre);np.savez_compressed(folder/'scores.npz',positions=query,base=u['base'],row_weights=u['row_weights'],publication=u['publication'],mapped=mixed,reference=u['reference'],candidate=candidate,rf_prediction=u['rf_prediction']);joblib.dump({'head':head,'tree_expectation':tc,'expectation':expectation,'channel':channel,'projection_fit':fit,'conditional_source_scores_sha256':digest(union/'contexts'/context/'scores.npz'),'conditional_source_model_sha256':digest(union/'contexts'/context/'model.joblib')},folder/'model.joblib');record={'context':context,'fit_rows':len(fit),'known_API_rows':len(known),'query_rows':len(query),'new_projection':1,'new_forests':4,'new_availability_heads':0,'new_encoders_or_GPU_work':0,'runtime_seconds':time.monotonic()-started};assert record['runtime_seconds']<=cfg['maximum_context_seconds'];json_write(folder/'fit.json',record);return record


def run(args,out):
    threadpool_limits(1);started=time.monotonic();settings=json.loads((ROOT/args.config).read_text());source=ROOT/settings['source_deployment'];cfg=json.loads((source/'config.json').read_text())|settings;union=ROOT/'outputs/deployment'/cfg['conditional_selection_run'];assert json.loads((union/'selection_audit.json').read_text())['state']=='passed';assert not json.loads((union/'training_selection_locked.json').read_text())['new_fullfit_export_admitted'];assert digest(source/'model.joblib')==cfg['source_model_sha256'];provenance(out,cfg);json_write(out/'selection_contract_locked.json',{'config_sha256':digest(ROOT/args.config),'conditional_selection_audit_sha256':digest(union/'selection_audit.json'),'conditional_selection_result_sha256':digest(union/'training_selection_locked.json'),'single_fixed_tree_fraction':.5,'single_fixed_residual_weight':.5,'source_full_NN_reference_weight':.25,'no_outer_or_new_fivefold_score':True});status(out,'running',stage='twelve_fixed_fulltraining_proxy_forests',CPU_contexts=cfg['CPU_workers'],tree_jobs_per_head=cfg['tree']['jobs']);jobs=[{'out':str(out),'config':cfg,'context':f'full_inner{i}'} for i in range(3)];json_write(out/'context_jobs.json',jobs);records=[]
    with ProcessPoolExecutor(max_workers=cfg['CPU_workers'],mp_context=get_context('spawn')) as pool:
        for future in as_completed([pool.submit(prepare,job) for job in jobs]):records.append(future.result());status(out,'running',stage='twelve_fixed_fulltraining_proxy_forests',completed=len(records),total=3)
    data=arrays(ROOT/'cache/workshop_v1/dataset.npz');y=data['y'];control=np.zeros(y.shape);candidate=np.zeros(y.shape);rf=np.zeros(y.shape,bool);seen=np.zeros(len(y),int);parts=[]
    for i in range(3):
        a=arrays(out/'contexts'/f'full_inner{i}'/'scores.npz');rows=a['positions'];control[rows]=a['reference'];candidate[rows]=a['candidate'];rf[rows]=a['rf_prediction'];seen[rows]+=1;parts.append(rows)
    assert (seen==1).all();old=threshold_choice(y,control,rf);previous=json.loads((union/'training_selection_locked.json').read_text());assert old==previous['control'];new=threshold_choice(y,candidate,rf);rfmetrics=fast_micro(y,rf);admitted=new['micro_f1']>old['micro_f1'] and not new['fallback_rf'] and all(new[key]>=rfmetrics[key] for key in rfmetrics);selection=(new|{'candidate':'tree_conditional_union'}) if admitted else (old|{'candidate':'reference'});deltas=[fast_micro(y[rows],candidate[rows]>=new['threshold'])['micro_f1']-fast_micro(y[rows],control[rows]>=old['threshold'])['micro_f1'] for rows in parts];receipt={'state':'complete','training_OOF_not_test_score':True,'control':old,'candidate':new,'candidate_F1_delta':new['micro_f1']-old['micro_f1'],'partition_F1_deltas':deltas,'RF':rfmetrics,'selection':selection,'new_fullfit_export_admitted':bool(admitted),'new_projections':3,'new_forests':12,'new_availability_or_encoder_models':0,'GPU_work':0,'full_model_exported':False,'fit_records':records,'runtime_seconds':time.monotonic()-started,'limits':cfg['limits']};np.savez_compressed(out/'full_training_OOF.npz',positions=np.arange(len(y)),reference=control,candidate=candidate,rf_prediction=rf);json_write(out/'training_selection_locked.json',receipt);report=ROOT/cfg['report_directory'];report.mkdir(exist_ok=True);json_write(report/'training_selection_locked.json',receipt);status(out,'complete',stage='selection_complete_pending_independent_audit',new_fullfit_export_admitted=bool(admitted),full_model_exported=False,training_OOF_F1=selection['micro_f1'],independent_score=None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--config',default='configs/full_tree_conditional_union_candidate_v1.json');p.add_argument('--launch',action='store_true');args=p.parse_args();require_server()
    if args.launch:
        assert not (ROOT/'outputs/deployment'/args.run_id).exists();env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+':'+str(ROOT/'scripts')
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
        with (ROOT/'logs'/(args.run_id+'.log')).open('x') as stream:child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--run-id',args.run_id,'--config',args.config],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'state':'launched','PID':child.pid,'run_id':args.run_id}))
    else:
        out=ROOT/'outputs/deployment'/args.run_id;out.mkdir()
        try:run(args,out)
        except BaseException:(out/'error.txt').write_text(traceback.format_exc());status(out,'failed',error=traceback.format_exc());raise
