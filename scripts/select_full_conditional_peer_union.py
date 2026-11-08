"""Fixed full-data training OOF choice; no model export before its own audit."""
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
from scipy import sparse
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from build_full_auxiliary_portfolio import cached_features
from capability_search_workers import digest
from run_peer_publication_information import arrays,lineage_inputs,role_targets
from skillcomp.availability_calibration import fit_availability_intercepts,calibrate_availability
from skillcomp.availability_inference import semantic_availability_features
from skillcomp.annotation_channel import apply_channel
from skillcomp.benchmark import priors,fast_micro
from skillcomp.capability_search import threshold_choice
from skillcomp.channel_residual import fit_expectation,apply_residual
from skillcomp.conditional_peer_prior import ConditionalPeerPrior
from skillcomp.peer_union_model import union_probability
from skillcomp.matchability import matchability_features
from skillcomp.retrieval_evidence import EvidenceNeighbors
from skillcomp.evidence_coverage import fit_coverage,coverage_weights
from skillcomp.runtime import ROOT,require_server,provenance,json_write,status


def availability_x(cfg,context,fit,repos):
    folder=ROOT/cfg['base_deployment']/'contexts'/context;x=matchability_features(sparse.load_npz(folder/'title.npz'),sparse.load_npz(folder/'body.npz'),repos[fit],repos);frozen=np.load(ROOT/cfg['frozen_features'],mmap_mode='r');return semantic_availability_features(x,frozen)


def prepare(job):
    threadpool_limits(1);started=time.monotonic();cfg=job['config'];out=Path(job['out']);context=job['context'];folder=out/'contexts'/context;folder.mkdir(parents=True);source=ROOT/cfg['source_deployment'];base_root=ROOT/cfg['base_deployment'];role=arrays(base_root/'contexts'/context/'roles.npz');fit,query=role['train'],role['query'];data,aux,pr,raw_counts=lineage_inputs();groups=np.load(ROOT/'cache/workshop_v1/key_component_groups.npy');assert not np.intersect1d(groups[fit],groups[query]).size and not np.intersect1d(pr[fit],pr[query]).size;known=fit[aux['matched'][fit].astype(bool)];peer=role_targets(data,aux,pr,raw_counts,fit);own=np.zeros(peer.shape,np.uint8);observed=aux['matched'][fit].astype(bool);own[observed]=aux['targets'][fit[observed]];prior=ConditionalPeerPrior(20).fit(peer,own,data['repos'][fit],groups[fit],data['y'][fit]);x=cached_features(dict(cfg,auxiliary_repository_scale=0),context,'learned_fused',fit);nn=EvidenceNeighbors(**cfg['neighbors']).fit(x[known],aux['targets'][known],data['repos'][known],groups[known]);pq=nn.predict_proba(x[query],data['repos'][query]);pc=nn.predict_fit_centre(x[fit],data['repos'][fit],groups[fit]);pub=ROOT/cfg['availability_deployment'];av=joblib.load(pub/'contexts'/context/'component.joblib');np.testing.assert_array_equal(av['fit'],fit);xx=availability_x(cfg,context,fit,data['repos']);aq=av['gate'].predict_proba(xx[query])[:,1]
    if av.get('probability_calibration') is not None:aq=calibrate_availability(av['probability_calibration'],aq,data['repos'][query])
    ac=np.zeros(len(fit));seen=np.zeros(len(fit),int)
    for k,(a,b) in enumerate(GroupKFold(3).split(fit,groups=groups[fit])):
        assert not np.intersect1d(groups[fit[a]],groups[fit[b]]).size;model=clone(av['gate']);assert model.C==1 and model.solver=='liblinear';model.fit(xx[fit[a]],observed[a]);pt=model.predict_proba(xx[fit[a]])[:,1];raw=model.predict_proba(xx[fit[b]])[:,1];cal=fit_availability_intercepts(pt,observed[a],data['repos'][fit[a]],20);p=calibrate_availability(cal,raw,data['repos'][fit[b]]);ac[b]=p;seen[b]+=1;sub=folder/f'sub{k}';sub.mkdir();joblib.dump({'model':model,'calibration':cal},sub/'model.joblib');np.savez_compressed(sub/'roles.npz',train=fit[a],query=fit[b],train_local=a,query_local=b);np.savez_compressed(sub/'probabilities.npz',positions=fit[b],raw=raw,probability=p)
    assert (seen==1).all();fq,cq,nq=prior.predict(data['repos'][query]);fc,cc,nc=prior.fit_centre(data['repos'][fit],groups[fit]);mapped=union_probability(fq,aq,pq,cq,nq);centre=union_probability(fc,ac,pc,cc,nc);expectation=fit_expectation(centre,data['repos'][fit]);bundle=joblib.load(source/'model.joblib');choice=bundle['neighbor_component']['selection'];assert choice['neighbor_fraction']==1 and choice['residual_weight']==.25 and choice['mix']==1;old=joblib.load(source/'contexts'/context/'neighbor_component.joblib');oldscore=arrays(source/'contexts'/context/'scores.npz');np.testing.assert_array_equal(oldscore['positions'],query);raw=arrays(base_root/'contexts'/context/bundle['spec']['name']/'scores.npz');np.testing.assert_array_equal(raw['positions'],query);rf=arrays(base_root/'contexts'/context/'rf_scores.npz');np.testing.assert_array_equal(rf['positions'],query);base=bundle['selection']['weight']*raw['scores']+(1-bundle['selection']['weight'])*priors(data['y'][fit],data['repos'][fit],data['repos'][query],strength=20);coverage=fit_coverage(aux['matched'][fit],data['repos'][fit],20);rows=coverage_weights(coverage,data['repos'][query],choice['mix']);publication=apply_channel(av['channel'],np.repeat(aq[:,None],217,axis=1),data['repos'][query]);pubweight=bundle['availability_component']['selection']['residual_weight'];oldcov=apply_residual(base,oldscore['knn'],data['repos'][query],old['knn_expectation'],choice['residual_weight'],row_weights=rows);control=apply_residual(oldcov,publication,data['repos'][query],av['expectation'],pubweight);newcov=apply_residual(base,mapped,data['repos'][query],expectation,choice['residual_weight'],row_weights=rows);candidate=apply_residual(newcov,publication,data['repos'][query],av['expectation'],pubweight);np.savez_compressed(folder/'roles.npz',train=fit,query=query,known=known);np.savez_compressed(folder/'targets.npz',positions=fit,peer=peer,own_observed=own);np.savez_compressed(folder/'union_inputs.npz',peer_query=fq,peer_centre=fc,proxy_query=pq,proxy_centre=pc,availability_query=aq,availability_centre=ac,published_counts_query=cq,published_counts_centre=cc,fit_counts_query=nq,fit_counts_centre=nc);np.savez_compressed(folder/'scores.npz',positions=query,base=base,row_weights=rows,publication=publication,mapped=mapped,centre_published=centre,source_mapped=oldscore['knn'],reference=control,candidate=candidate,rf_prediction=rf['prediction']);joblib.dump({'prior':prior,'neighbors':nn,'expectation':expectation,'publication_expectation':av['expectation'],'publication_weight':pubweight,'source_knn_expectation':old['knn_expectation'],'source_selection':choice,'availability_source_component':str(pub/'contexts'/context/'component.joblib'),'coverage':coverage},folder/'model.joblib');record={'context':context,'fit_rows':len(fit),'known_API_rows':len(known),'query_rows':len(query),'new_availability_centre_heads':3,'source_metadata_gate_reused':True,'new_encoders_or_GPU_work':0,'runtime_seconds':time.monotonic()-started};assert record['runtime_seconds']<=cfg['maximum_context_seconds'];json_write(folder/'fit.json',record);return record


def run(args,out):
    threadpool_limits(1);started=time.monotonic();settings=json.loads((ROOT/args.config).read_text());source=ROOT/settings['source_deployment'];cfg=json.loads((source/'config.json').read_text())|settings;development=ROOT/'outputs/experiments'/settings['development_run'];assert json.loads((development/'verification.json').read_text())['state']=='passed';result=json.loads((development/'results.json').read_text());gain=result['metrics']['best_policy']['micro_f1']-result['metrics']['previous_best']['micro_f1'];assert gain>=cfg['minimum_verified_development_gain'];assert digest(source/'model.joblib')==cfg['source_model_sha256'];assert json.loads((source/'database_independent_inference_verification.json').read_text())['state']=='passed';provenance(out,cfg);json_write(out/'selection_contract_locked.json',{'source_model_sha256':cfg['source_model_sha256'],'development_verification_sha256':digest(development/'verification.json'),'development_result_sha256':digest(development/'results.json'),'verified_development_gain':gain,'single_candidate':True,'fixed_source_policy':{'neighbor_fraction':1,'residual_weight':.25,'mix':1},'reference_wins_ties':True,'no_test_or_outer_scores_created':True});status(out,'running',stage='three_full_training_conditional_union_OOF_contexts',CPU_workers=cfg['CPU_workers']);jobs=[{'out':str(out),'config':cfg,'context':f'full_inner{i}'} for i in range(3)];json_write(out/'context_jobs.json',jobs);records=[]
    with ProcessPoolExecutor(max_workers=cfg['CPU_workers'],mp_context=get_context('spawn')) as pool:
        for future in as_completed([pool.submit(prepare,job) for job in jobs]):records.append(future.result());status(out,'running',stage='full_training_conditional_union_OOF',completed=len(records),total=3)
    data=arrays(ROOT/'cache/workshop_v1/dataset.npz');y=data['y'];reference=np.zeros(y.shape);candidate=np.zeros(y.shape);rf=np.zeros(y.shape,bool);seen=np.zeros(len(y),int);parts=[]
    for i in range(3):
        a=arrays(out/'contexts'/f'full_inner{i}'/'scores.npz');rows=a['positions'];reference[rows]=a['reference'];candidate[rows]=a['candidate'];rf[rows]=a['rf_prediction'];seen[rows]+=1;parts.append(rows)
    assert (seen==1).all();control=threshold_choice(y,reference,rf);manifest=json.loads((source/'manifest.json').read_text());replay_errors={key:abs(control[key]-manifest['training_OOF_selection'][key]) for key in ['precision','recall','micro_f1','threshold']};assert max(replay_errors.values())<1e-13;new=threshold_choice(y,candidate,rf);rfmetrics=fast_micro(y,rf);admitted=new['micro_f1']>control['micro_f1'] and not new['fallback_rf'] and all(new[key]>=rfmetrics[key] for key in rfmetrics);selection=(new|{'candidate':'conditional_union'}) if admitted else (control|{'candidate':'reference'});deltas=[fast_micro(y[rows],candidate[rows]>=new['threshold'])['micro_f1']-fast_micro(y[rows],reference[rows]>=control['threshold'])['micro_f1'] for rows in parts];receipt={'state':'complete','training_OOF_not_test_score':True,'control':control,'candidate':new,'RF':rfmetrics,'candidate_F1_delta':new['micro_f1']-control['micro_f1'],'partition_F1_deltas':deltas,'selection':selection,'new_fullfit_export_admitted':bool(admitted),'source_control_replay_abs_differences':replay_errors,'fit_records':records,'new_availability_centre_heads':9,'new_encoder_or_skill_heads':0,'GPU_work':0,'full_model_exported':False,'runtime_seconds':time.monotonic()-started,'limits':cfg['limits']};np.savez_compressed(out/'full_training_OOF.npz',positions=np.arange(len(y)),reference=reference,candidate=candidate,rf_prediction=rf);json_write(out/'training_selection_locked.json',receipt);report=ROOT/cfg['report_directory'];report.mkdir(exist_ok=True);json_write(report/'training_selection_locked.json',receipt);status(out,'complete',stage='selection_complete_pending_independent_audit',new_fullfit_export_admitted=bool(admitted),full_model_exported=False,training_OOF_F1=selection['micro_f1'],independent_score=None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--config',default='configs/full_conditional_peer_union_candidate_v1.json');p.add_argument('--launch',action='store_true');args=p.parse_args();require_server()
    if args.launch:
        assert not (ROOT/'outputs/deployment'/args.run_id).exists();env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+':'+str(ROOT/'scripts')
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
        with (ROOT/'logs'/(args.run_id+'.log')).open('x') as stream:child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--run-id',args.run_id,'--config',args.config],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'state':'launched','PID':child.pid,'run_id':args.run_id}))
    else:
        out=ROOT/'outputs/deployment'/args.run_id;out.mkdir()
        try:run(args,out)
        except BaseException:(out/'error.txt').write_text(traceback.format_exc());status(out,'failed',error=traceback.format_exc());raise
