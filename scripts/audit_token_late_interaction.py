"""Audit legal inputs/own-fit IDF, FP64 MaxSim, votes, metrics and fresh forwards."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from run_positive_overlap_metric_pilot import gpu_queue
from run_proxy_tree_readout_capability import input_values
import run_token_late_interaction as runner
from skillcomp.runtime import ROOT,require_server,encoder_python,json_write,status


def independent_AP(y,s):
    yy=np.asarray(y).ravel().astype(float);ss=np.asarray(s).ravel();order=np.argsort(-ss,kind='stable');yy,ss=yy[order],ss[order];end=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1];tp=np.cumsum(yy)[end];return float((np.diff(np.r_[0,tp/max(yy.sum(),1)])*(tp/(end+1))).sum())


def metrics(y,p):return {'Brier':float(np.mean((np.asarray(p,np.longdouble)-y)**2)),'micro_AP':independent_AP(y,p)}


def main(args):
    require_server();threadpool_limits(1);out=ROOT/'outputs/experiments'/args.run_id;cfg=json.loads((out/'config.json').read_text());result=json.loads((out/'information_gate.json').read_text());assert result['state']=='complete';source=ROOT/'outputs/experiments'/cfg['pair_source_run'];locked=json.loads((out/'selection_policy_locked.json').read_text());assert digest(source/'audit.json')==locked['source_audit_sha256'];raw=json.loads((ROOT/'outputs/experiments/20261023_markdown_semantic_features_v1/original_fields.json').read_text());assert json.loads((out/'encoder_inputs.json').read_text())==[f'Title: {a}\nDescription: {b}' for a,b in zip(raw['title'],raw['body'])];assert digest(out/'encoder_inputs.json')==locked['encoder_inputs_sha256'];scfg=json.loads((ROOT/'outputs/experiments'/cfg['tree_source_run']/'config.json').read_text());records=[];replay=out/'fresh_token_replay_v2_precision_contract';replay.mkdir();json_write(replay/'config.json',cfg|{'worker_script':'scripts/score_token_late_safe_replay.py'});shutil.copyfile(out/'encoder_inputs.json',replay/'encoder_inputs.json');jobs=[];truth=[];baseline=[];candidate=[];auxfile=ROOT/'outputs/experiments/20261023_annotation_channel_capability_v1/auxiliary_targets.npz';aux=np.load(auxfile);maximum_score_error=0.
    for i in range(3):
        name=f'fold0_inner{i}';folder=out/'contexts'/name;src=source/'contexts'/name;r=np.load(folder/'roles.npz');fit,query,known,data,targets,groups,x,old=input_values(scfg,name)
        for f in ['roles.npz','voting_inputs.npz','pair_positions.npz']:assert digest(folder/f)==digest(src/f)
        np.testing.assert_array_equal(r['train'],fit);np.testing.assert_array_equal(r['known'],known);assert np.isin(r['query'],query).all();assert not np.intersect1d(groups[fit],groups[r['query']]).size;np.testing.assert_array_equal(np.load(folder/'idf_fit_positions.npz')['positions'],fit);child=subprocess.run([encoder_python(),str(ROOT/'scripts/audit_token_late_tokenization.py'),'--out',str(out),'--context',name],cwd=ROOT,capture_output=True,text=True);assert child.returncode==0,child.stderr;print(child.stdout,flush=True);metadata=json.loads((folder/'fit.json').read_text());assert metadata['first4_repeat_max_abs']==0 and metadata['targets_loaded'] is False and metadata['strict_safe_loading'];assert metadata['peak_GPU_GiB']<=cfg['hard_GPU_allocated_GiB'] and metadata['runtime_seconds']<=cfg['maximum_inference_seconds'];assert digest(folder/'token_index.npz')==metadata['token_index_sha256'] and digest(folder/'idf.json')==metadata['idf_sha256'];idx=np.load(folder/'token_index.npz');vectors=np.load(folder/'token_vectors.npy',mmap_mode='r');assert vectors.shape==(idx['offsets'][-1],2560);assert np.isfinite(vectors).all();norm_error=0.
        for start in range(0,len(vectors),8192):norm_error=max(norm_error,float(np.abs(np.linalg.norm(vectors[start:start+8192],axis=1)-1).max()))
        assert norm_error<=2e-6;lookup={int(pos):j for j,pos in enumerate(idx['positions'])};pairs=np.load(folder/'pair_positions.npz');scores=np.load(folder/'pair_scores.npz');np.testing.assert_array_equal(pairs['queries'],scores['queries']);np.testing.assert_array_equal(pairs['documents'],scores['documents']);sample=np.unique(np.linspace(0,len(pairs['queries'])-1,8).astype(int));local_error=0.
        for row in sample:
            j=lookup[int(pairs['queries'][row])];a,b=idx['offsets'][j:j+2];q=np.asarray(vectors[a:b],np.float64);weights=idx['weights'][a:b].astype(np.float64);j=lookup[int(pairs['documents'][row])];c,d=idx['offsets'][j:j+2];doc=np.asarray(vectors[c:d],np.float64);ok=bool(len(q) and len(doc));assert ok==scores['valid'][row]
            if ok:
                maxima=np.array([max(float(token@other) for other in doc) for token in q]);value=float(sum(float(w)*float(v) for w,v in zip(weights,maxima))/sum(map(float,weights)));error=abs(value-scores['token_score'][row]);assert error<=cfg['CPU_pair_score_tolerance'];local_error=max(error,local_error)
        maximum_score_error=max(maximum_score_error,local_error);a=np.load(folder/'voting_inputs.npz');model=joblib.load(ROOT/'outputs/experiments'/cfg['tree_source_run']/'contexts'/name/'neighbor_component.joblib')['neighbors'];np.testing.assert_allclose(model.predict_proba(x[r['query']],data['repos'][r['query']]),a['control'],atol=2e-7,rtol=0);new=[];control=[];fallbacks=0
        for j,pos in enumerate(a['positions']):
            begin,end=a['offsets'][j:j+2];w=a['cosine_weights'][begin:end].astype(np.longdouble);s=scores['token_score'][begin:end].astype(np.longdouble);positive=w>0
            if not scores['valid'][begin:end].all() or not positive.any() or np.ptp(s[positive])==0:weights=w.copy();fallbacks+=not scores['valid'][begin:end].all()
            else:
                relative=np.zeros(len(w),np.longdouble);relative[positive]=w[positive]*np.exp((s[positive]-s[positive].max())/np.longdouble(cfg['temperature']));relative*=w.sum()/relative.sum();weights=(1-np.longdouble(cfg['rerank_blend']))*w+np.longdouble(cfg['rerank_blend'])*relative
            assert abs(float(weights.sum()-w.sum()))<1e-12;indices=a['prototype_indices'][begin:end];target=model.targets_[indices].astype(np.longdouble);prior=a['priors'][j].astype(np.longdouble);p=(weights@target+model.pseudocount*prior)/(weights.sum()+model.pseudocount);p[model.positive_counts_==0]=0;p[model.positive_counts_==len(known)]=1;new.append(p);mass=w.astype(np.float32);p=(mass@model.targets_[indices]+model.pseudocount*a['priors'][j])/(mass.sum()+model.pseudocount);p[model.positive_counts_==0]=0;p[model.positive_counts_==len(known)]=1;control.append(p)
        new=np.asarray(new,float);control=np.asarray(control,float);stored=np.load(folder/'proxy_predictions.npz');np.testing.assert_allclose(new,stored['scores'],atol=2e-12,rtol=0);np.testing.assert_allclose(control,stored['control'],atol=2e-12,rtol=0);new=stored['scores'];control=stored['control'];y=aux['targets'][a['positions']].astype(float);truth.append(y);baseline.append(control);candidate.append(new);record=result['contexts'][i]
        for key,p in [('source_cosine',control),('fixed_half_token',new)]:
            for axis,value in metrics(y,p).items():assert abs(value-record['metrics'][key][axis])<1e-12
        assert int(fallbacks)==record['fallback_queries'];selected=np.r_[a['positions'][[0,-1]],np.unique(a['document_positions'])[[0,-1]]];fresh=replay/'contexts'/name;fresh.mkdir(parents=True);job={'context':name,'pair_positions':str(folder/'pair_positions.npz'),'idf_fit_positions':str(folder/'idf_fit_positions.npz'),'replay_only':True,'replay_positions':selected.tolist()};json_write(replay/(name+'_job.json'),job);jobs.append(job);records.append({'context':name,'exact_source_roles_candidate_and_vote_files':True,'independent_all_ownfit_IDF_and_evaluation_tokenization_verified':True,'cached_vectors_sha256':digest(folder/'token_vectors.npy'),'maximum_token_norm_error':norm_error,'independent_CPU_FP64_pairs':len(sample),'maximum_GPU_CPU_score_error':local_error,'all_proxy_votes_and_metrics_recomputed':True})
    gpu=gpu_queue(replay,jobs,cfg|{'worker_script':'scripts/score_token_late_safe_replay.py'});fresh_error=0.;fresh_docs=0
    for job in jobs:
        folder=out/'contexts'/job['context'];fresh=replay/'contexts'/job['context'];a=np.load(folder/'token_index.npz');b=np.load(fresh/'token_index.npz');x=np.load(folder/'token_vectors.npy',mmap_mode='r');y=np.load(fresh/'token_vectors.npy',mmap_mode='r');lookup={int(p):i for i,p in enumerate(a['positions'])}
        for j,pos in enumerate(b['positions']):
            i=lookup[int(pos)];aa,ab=a['offsets'][i:i+2];ba,bb=b['offsets'][j:j+2];np.testing.assert_array_equal(a['ids'][aa:ab],b['ids'][ba:bb]);np.testing.assert_array_equal(a['weights'][aa:ab],b['weights'][ba:bb]);error=float(np.abs(x[aa:ab]-y[ba:bb]).max()) if bb>ba else 0.;assert error<=cfg['fresh_token_forward_max_abs_tolerance'];fresh_error=max(error,fresh_error);fresh_docs+=1
    status(replay,'complete',fresh_documents=fresh_docs,maximum_token_abs_difference=fresh_error)
    # Flip actual held query supervision; prototype assets and text are immutable.
    name='fold0_inner0';folder=out/'contexts'/name;intervention=out/'query_target_intervention';newfolder=intervention/'contexts'/name;newfolder.mkdir(parents=True);query=np.load(folder/'roles.npz')['query'];modified={key:aux[key].copy() for key in aux.files};modified['targets'][query]=1-modified['targets'][query];modified['matched'][query]=~modified['matched'][query].astype(bool)
    for f in ['roles.npz','voting_inputs.npz','pair_scores.npz']:shutil.copyfile(folder/f,newfolder/f)
    original_load=np.load
    def perturb(path,*aa,**kw):return modified if Path(path)==auxfile else original_load(path,*aa,**kw)
    np.load=perturb
    try:runner.evaluate(intervention,cfg,name)
    finally:np.load=original_load
    np.testing.assert_array_equal(np.load(folder/'proxy_predictions.npz')['scores'],np.load(newfolder/'proxy_predictions.npz')['scores']);yy=np.concatenate(truth);bb=np.concatenate(baseline);cc=np.concatenate(candidate);m={key:metrics(yy,p) for key,p in [('source_cosine',bb),('fixed_half_token',cc)]}
    for key in m:
        for axis,value in m[key].items():assert abs(value-result['pooled_metrics'][key][axis])<1e-12
    gain=m['source_cosine']['Brier']-m['fixed_half_token']['Brier'];ap=m['fixed_half_token']['micro_AP']-m['source_cosine']['micro_AP'];gates={'pooled_Brier':gain>=cfg['minimum_pooled_Brier_improvement'],'pooled_AP':ap>=cfg['minimum_pooled_AP_improvement'],'partition_Brier':max(p['Brier_delta'] for p in result['contexts'])<=cfg['maximum_partition_Brier_loss']};assert all(gates.values())==result['full_training_OOF_pilot_admitted'];manifest=json.loads((out/'code_manifest.json').read_text())['files']
    for f in ['scripts/run_token_late_interaction.py','scripts/score_token_late_interaction.py','src/skillcomp/token_late_interaction.py']:assert digest(ROOT/f)==manifest[f]
    assert digest(ROOT/'outputs/deployment/20261024_full_neighbor_candidate_v1/model.joblib')=='40ed37a1c4bacc9a79d72e0266c245f93d1d28e54df3951c9c1a721b9582adf2';assert all(r['GPU'] not in [6,7] and r['exit_code']==0 for r in result['GPU_records']+gpu);report={'state':'passed','contexts':records,'independent_FP64_pairs':sum(r['independent_CPU_FP64_pairs'] for r in records),'maximum_GPU_CPU_score_error':maximum_score_error,'fresh_safe_forward_documents':fresh_docs,'fresh_token_maximum_abs_difference':fresh_error,'fresh_forward_supervised_file_access_forbidden':True,'actual_query_target_missingness_intervention_predictions_exact':True,'all_relative_votes_priors_constants_and_independent_AP_Brier_verified':True,'all_fixed_gates':gates,'full_training_OOF_pilot_admitted':all(gates.values()),'source_export_unchanged':True,'GPU_records':gpu,'outer_or_main_F1_predictions_created':False,'limits':'Repeatedly exposed weak-proxy evaluation subset, not gold or independent. All tokenization/IDF/votes audited; numeric CPU pair and fresh model-forward comparisons are sampled. Frozen last-layer causal tokens have no special late-interaction training.'};json_write(out/'audit.json',report);json_write(ROOT/cfg['report_directory']/'audit.json',report);print(json.dumps({'state':'passed','full_training_OOF_pilot_admitted':all(gates.values()),'CPU_FP64_pairs':report['independent_FP64_pairs'],'maximum_score_error':maximum_score_error,'fresh_documents':fresh_docs,'fresh_max_abs':fresh_error,'gates':gates}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);main(p.parse_args())
