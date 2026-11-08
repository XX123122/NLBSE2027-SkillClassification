"""Fixed contextual-token information pilot on historical training-only top20."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from download_pinned_model import valid
from run_positive_overlap_metric_pilot import gpu_queue
from skillcomp.token_late_interaction import token_relative_votes
from skillcomp.runtime import ROOT,require_server,provenance,json_write,status


def evaluate(out,cfg,name):
    folder=out/'contexts'/name;a=np.load(folder/'voting_inputs.npz');b=np.load(folder/'pair_scores.npz');r=np.load(folder/'roles.npz');aux=np.load(ROOT/'outputs/experiments/20261023_annotation_channel_capability_v1/auxiliary_targets.npz');source=ROOT/'outputs/experiments'/cfg['tree_source_run'];model=joblib.load(source/'contexts'/name/'neighbor_component.joblib')['neighbors'];query=a['positions'];np.testing.assert_array_equal(b['queries'],np.repeat(query,np.diff(a['offsets'])));np.testing.assert_array_equal(b['documents'],a['document_positions']);scores=[];fallbacks=0
    for i,pos in enumerate(query):
        begin,end=a['offsets'][i:i+2];weights=token_relative_votes(a['cosine_weights'][begin:end],b['token_score'][begin:end],b['valid'][begin:end],cfg['temperature'],cfg['rerank_blend']);fallbacks+=not b['valid'][begin:end].all();indices=a['prototype_indices'][begin:end];p=(weights@model.targets_[indices]+model.pseudocount*a['priors'][i])/(weights.sum()+model.pseudocount);p[model.positive_counts_==0]=0;p[model.positive_counts_==len(r['known'])]=1;scores.append(p)
    scores=np.asarray(scores);y=aux['targets'][query].astype(float);control=a['control'];metrics={key:{'Brier':float(np.mean((value-y)**2)),'micro_AP':float(average_precision_score(y.ravel(),value.ravel()))} for key,value in [('source_cosine',control),('fixed_half_token',scores)]};np.savez_compressed(folder/'proxy_predictions.npz',positions=query,control=control,scores=scores);return {'context':name,'query_rows':len(query),'pairs':len(b['queries']),'fallback_queries':int(fallbacks),'metrics':metrics,'Brier_delta':metrics['fixed_half_token']['Brier']-metrics['source_cosine']['Brier'],'AP_delta':metrics['fixed_half_token']['micro_AP']-metrics['source_cosine']['micro_AP']},y,control,scores


def run(args,out):
    threadpool_limits(1);started=time.monotonic();cfg=json.loads((ROOT/args.config).read_text());provenance(out,cfg);assets=json.loads((ROOT/cfg['model_source_config']).read_text());assert all(valid(Path(cfg['model'])/r['rfilename'],r) for r in assets['files']);json_write(out/'model_source.json',assets);source=ROOT/'outputs/experiments'/cfg['pair_source_run'];assert json.loads((source/'audit.json').read_text())['state']=='passed';fields=ROOT/'outputs/experiments/20261023_markdown_semantic_features_v1/original_fields.json';assert digest(fields)=='bfdfe272215fa99dbf6c5b019ba460cb456e4be2d8dc1331c0caea78ef435530';raw=json.loads(fields.read_text());texts=[f'Title: {a}\nDescription: {b}' for a,b in zip(raw['title'],raw['body'])];assert texts==json.loads((source/'encoder_inputs.json').read_text());json_write(out/'encoder_inputs.json',texts);jobs=[];records=[]
    for i in range(3):
        name=f'fold0_inner{i}';folder=out/'contexts'/name;folder.mkdir(parents=True);src=source/'contexts'/name;hashes={}
        for f in ['roles.npz','voting_inputs.npz','pair_positions.npz']:shutil.copyfile(src/f,folder/f);hashes[f]=digest(folder/f)
        r=np.load(folder/'roles.npz');np.savez_compressed(folder/'idf_fit_positions.npz',positions=r['train']);job={'context':name,'pair_positions':str(folder/'pair_positions.npz'),'idf_fit_positions':str(folder/'idf_fit_positions.npz')};json_write(out/(name+'_job.json'),job);jobs.append(job);records.append({'context':name,'query_rows':len(r['query']),'idf_fit_rows':len(r['train']),'source_file_sha256':hashes})
    json_write(out/'selection_policy_locked.json',{'config_sha256':digest(ROOT/args.config),'source_audit_sha256':digest(source/'audit.json'),'source_pair_runs_unchanged':records,'original_fields_sha256':digest(fields),'encoder_inputs_sha256':digest(out/'encoder_inputs.json'),'temperature':cfg['temperature'],'blend':cfg['rerank_blend'],'empty_any_pair_policy':'whole-query exact original votes','main_F1_candidate_created':False,'outer_validation_or_test_used':False});status(out,'running',stage='frozen_Qwen_contextual_token_features_and_maxsim');gpu=gpu_queue(out,jobs,cfg);parts=[];truth=[];baseline=[];candidate=[]
    for job in jobs:
        part,y,b,c=evaluate(out,cfg,job['context']);parts.append(part);truth.append(y);baseline.append(b);candidate.append(c)
    y=np.concatenate(truth);b=np.concatenate(baseline);c=np.concatenate(candidate);metrics={key:{'Brier':float(np.mean((value-y)**2)),'micro_AP':float(average_precision_score(y.ravel(),value.ravel()))} for key,value in [('source_cosine',b),('fixed_half_token',c)]};gb=metrics['source_cosine']['Brier']-metrics['fixed_half_token']['Brier'];ga=metrics['fixed_half_token']['micro_AP']-metrics['source_cosine']['micro_AP'];gate=gb>=cfg['minimum_pooled_Brier_improvement'] and ga>=cfg['minimum_pooled_AP_improvement'] and max(p['Brier_delta'] for p in parts)<=cfg['maximum_partition_Brier_loss'];result={'state':'complete','training_only':True,'evaluation_sample_observed_proxy_rows':len(y),'pairs_scored':sum(p['pairs'] for p in parts),'contexts':parts,'pooled_metrics':metrics,'pooled_Brier_improvement':gb,'pooled_AP_improvement':ga,'full_training_OOF_pilot_admitted':bool(gate),'GPU_records':gpu,'new_encoder_fits':0,'main_F1_candidate_created':False,'outer_validation_or_test_predictions_created':False,'runtime_seconds':time.monotonic()-started,'limits':cfg['hypothesis']};json_write(out/'information_gate.json',result);report=ROOT/cfg['report_directory'];report.mkdir(exist_ok=True);json_write(report/'information_gate.json',result);status(out,'complete',full_training_OOF_pilot_admitted=bool(gate),benchmark_score_claimed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--config',default='configs/token_late_interaction_v1.json');p.add_argument('--launch',action='store_true');args=p.parse_args();require_server()
    if args.launch:
        assert not (ROOT/'outputs/experiments'/args.run_id).exists();env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+':'+str(ROOT/'scripts')
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
        with (ROOT/'logs'/(args.run_id+'.log')).open('x') as stream:child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--run-id',args.run_id,'--config',args.config],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'state':'launched','PID':child.pid,'run_id':args.run_id}))
    else:
        out=ROOT/'outputs/experiments'/args.run_id;out.mkdir()
        try:run(args,out)
        except BaseException:(out/'error.txt').write_text(traceback.format_exc());status(out,'failed',error=traceback.format_exc());raise
