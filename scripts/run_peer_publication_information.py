"""Three fixed forests for the identified observed-peer publication target."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
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
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from run_proxy_tree_readout_capability import input_values
from skillcomp.data import open_readonly
from skillcomp.peer_publication import peer_observed_union,fit_peer_prior,peer_prior_scores
from skillcomp.runtime import ROOT,require_server,provenance,json_write,status


def arrays(path):
    with np.load(path,allow_pickle=True) as f:return {key:f[key] for key in f.files}


def lineage_inputs():
    data=arrays(ROOT/'cache/workshop_v1/dataset.npz');lineage=arrays(ROOT/'data/processed/20261021_join_lineage_v2/aligned_targets.npz');positions=data['source_positions'];np.testing.assert_array_equal(lineage['original_y'][positions],data['y']);np.testing.assert_array_equal(lineage['repos'][positions].astype(str),data['repos'].astype(str));pr=lineage['pr_number'][positions];aux=arrays(ROOT/'outputs/experiments/20261023_annotation_channel_capability_v1/auxiliary_targets.npz')
    with open_readonly(ROOT/'data/raw/skillscope_data.db') as con:raw_counts=dict((int(pr),int(n)) for pr,n in con.execute('SELECT pullNumber,COUNT(*) FROM pull_requests WHERE pullNumber IS NOT NULL GROUP BY pullNumber'))
    return data,aux,pr,raw_counts


def role_targets(data,aux,pr,raw_counts,rows):
    counts=Counter(map(int,pr[rows]));assert all(n==raw_counts[key] for key,n in counts.items()),'Incomplete raw PR bag in role';foreign,published_union=peer_observed_union(aux['targets'][rows],aux['matched'][rows],data['repos'][rows],pr[rows]);np.testing.assert_array_equal(published_union,data['y'][rows]);return foreign


def pack_context(cfg,out,name):
    source=ROOT/'outputs/experiments'/cfg['tree_source_run'];scfg=json.loads((source/'config.json').read_text());fit,query,known,data,aux,groups,x,old=input_values(scfg,name);canonical,annotations,pr,raw_counts=lineage_inputs();np.testing.assert_array_equal(data['y'][fit],canonical['y'][fit]);np.testing.assert_array_equal(aux['targets'][known],annotations['targets'][known]);assert not np.intersect1d(groups[fit],groups[query]).size and not np.intersect1d(pr[fit],pr[query]).size;target=role_targets(canonical,annotations,pr,raw_counts,fit);folder=out/'contexts'/name;folder.mkdir(parents=True);np.savez_compressed(folder/'roles.npz',train=fit,query=query,known_API=known);np.savez_compressed(folder/'fit_targets.npz',positions=fit,peer_target=target);np.savez_compressed(folder/'features.npz',fit_positions=fit,query_positions=query,x_train=x[fit],x_query=x[query],repo_train=canonical['repos'][fit],repo_query=canonical['repos'][query]);bundle=joblib.load(source/'contexts'/name/'model.joblib');np.testing.assert_array_equal(bundle['projection_fit'],fit);job={'context':name,'out':str(out),'config':cfg,'source_model':str(source/'contexts'/name/'model.joblib')};json_write(out/(name+'_job.json'),job);data.close();aux.close();return job


def fit_head(job):
    threadpool_limits(1);started=time.monotonic();cfg=job['config'];folder=Path(job['out'])/'contexts'/job['context'];assert not (folder/'evaluation_targets.npz').exists(),'Held evaluation targets must be created after predictions';features=arrays(folder/'features.npz');target=arrays(folder/'fit_targets.npz');np.testing.assert_array_equal(target['positions'],features['fit_positions']);source=joblib.load(job['source_model']);np.testing.assert_array_equal(source['projection_fit'],features['fit_positions']);model=copy.deepcopy(source['head']);assert (model.dimensions,model.estimators,model.depth,model.min_leaf,model.seed)==(128,256,12,5,1193);model.jobs=cfg['tree_jobs_per_head'];model.fit_head(features['x_train'],target['peer_target'],features['repo_train']);model.head_.set_params(n_jobs=1);probability=model.predict_proba(features['x_query'],features['repo_query']);prior=fit_peer_prior(target['peer_target'],features['repo_train'],cfg['repository_prior_strength']);global_scores,local=peer_prior_scores(prior,features['repo_query']);joblib.dump({'head':model,'prior':prior,'projection_fit':features['fit_positions'],'target':'observed_peer_union_not_local_API_truth'},folder/'model.joblib');np.savez_compressed(folder/'predictions.npz',positions=features['query_positions'],global_prior=global_scores,repository_prior=local,tree=probability);record={'context':job['context'],'fit_target_rows':len(target['peer_target']),'query_rows':len(probability),'trees':len(model.head_.estimators_),'tree_training_jobs':cfg['tree_jobs_per_head'],'tree_prediction_jobs':1,'source_projection_sha256':digest(Path(job['source_model'])),'query_targets_or_PR_or_group_loaded':False,'new_encoder_fits':0,'runtime_seconds':time.monotonic()-started};assert record['runtime_seconds']<=cfg['maximum_head_seconds'];json_write(folder/'fit.json',record);return record


def run(args,out):
    threadpool_limits(1);started=time.monotonic();cfg=json.loads((ROOT/args.config).read_text());provenance(out,cfg);assert json.loads((ROOT/'reports/published_bag_identifiability_v1.json').read_text())['peer_union_target_exactly_identified_on_fit_observations'];assert json.loads((ROOT/'reports/published_bag_raw_coverage_v2.json').read_text())['state']=='passed';json_write(out/'selection_policy_locked.json',{'config_sha256':digest(ROOT/args.config),'fixed_three_forests':True,'target':'other-repository observed API union within original training PR bag','fit_and_evaluation_targets_separate':True,'evaluation_targets_created_only_after_all_model_predictions':True,'no_main_F1_or_outer_scores':True});jobs=[pack_context(cfg,out,f'fold0_inner{i}') for i in range(3)];json_write(out/'head_jobs.json',jobs);status(out,'running',stage='three_fixed_peer_publication_forests',CPU_workers=cfg['CPU_workers'],tree_jobs_per_head=cfg['tree_jobs_per_head'])
    with ProcessPoolExecutor(max_workers=cfg['CPU_workers'],mp_context=get_context('spawn')) as pool:records=list(pool.map(fit_head,jobs))
    data,aux,pr,raw_counts=lineage_inputs();parts=[];truth=[];predictions={key:[] for key in ['global_prior','repository_prior','tree']}
    for job in jobs:
        folder=out/'contexts'/job['context'];a=arrays(folder/'predictions.npz');target=role_targets(data,aux,pr,raw_counts,a['positions']);np.savez_compressed(folder/'evaluation_targets.npz',positions=a['positions'],peer_target=target);metrics={key:{'Brier':float(np.mean((a[key]-target)**2)),'micro_AP':float(average_precision_score(target.ravel(),a[key].ravel()))} for key in predictions};part={'context':job['context'],'query_rows':len(target),'metrics':metrics,'Brier_delta_vs_repository_prior':metrics['tree']['Brier']-metrics['repository_prior']['Brier'],'AP_delta_vs_repository_prior':metrics['tree']['micro_AP']-metrics['repository_prior']['micro_AP']};parts.append(part);truth.append(target)
        for key in predictions:predictions[key].append(a[key])
    y=np.concatenate(truth);metrics={key:{'Brier':float(np.mean((np.concatenate(values)-y)**2)),'micro_AP':float(average_precision_score(y.ravel(),np.concatenate(values).ravel()))} for key,values in predictions.items()};b,n=metrics['repository_prior'],metrics['tree'];gates={'pooled_Brier_vs_repository_prior':b['Brier']-n['Brier']>=cfg['minimum_pooled_Brier_improvement'],'pooled_AP_vs_repository_prior':n['micro_AP']-b['micro_AP']>=cfg['minimum_pooled_AP_improvement'],'partition_Brier_vs_repository_prior':max(r['Brier_delta_vs_repository_prior'] for r in parts)<=cfg['maximum_partition_Brier_loss']};result={'state':'complete','training_only':True,'query_rows':len(y),'target':'other_repository_observed_API_union_not_latent_missing_API','contexts':parts,'pooled_metrics':metrics,'all_fixed_gates':gates,'main_integration_pilot_admitted':all(gates.values()),'new_forests':3,'fit_records':records,'new_encoder_fits':0,'GPU_work':0,'main_F1_candidate_or_outer_predictions_created':False,'runtime_seconds':time.monotonic()-started,'limits':cfg['hypothesis']};json_write(out/'information_gate.json',result);report=ROOT/cfg['report_directory'];report.mkdir(exist_ok=True);json_write(report/'information_gate.json',result);status(out,'complete',main_integration_pilot_admitted=all(gates.values()),benchmark_score_claimed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--config',default='configs/peer_publication_information_v1.json');p.add_argument('--launch',action='store_true');args=p.parse_args();require_server()
    if args.launch:
        assert not (ROOT/'outputs/experiments'/args.run_id).exists();env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+':'+str(ROOT/'scripts')
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
        with (ROOT/'logs'/(args.run_id+'.log')).open('x') as stream:child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--run-id',args.run_id,'--config',args.config],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'state':'launched','PID':child.pid,'run_id':args.run_id}))
    else:
        out=ROOT/'outputs/experiments'/args.run_id;out.mkdir()
        try:run(args,out)
        except BaseException:(out/'error.txt').write_text(traceback.format_exc());status(out,'failed',error=traceback.format_exc());raise
