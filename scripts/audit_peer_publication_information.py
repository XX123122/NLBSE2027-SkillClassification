"""Independent peer OR, all forest nodes, priors, metrics and target isolation."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from audit_token_late_interaction import metrics
import run_peer_publication_information as runner
from skillcomp.runtime import ROOT,require_server,json_write


def literal_targets(proxy,matched,repos,pr,rows):
    output=np.zeros((len(rows),proxy.shape[1]),np.uint8)
    for i,pos in enumerate(rows):
        peers=[int(other) for other in rows if other!=pos and pr[other]==pr[pos] and bool(matched[other])];assert all(repos[other]!=repos[pos] for other in peers)
        if peers:output[i]=np.any(proxy[peers]>0,axis=0)
    return output


def forest_equal(first,second):
    assert first.repository_vocabulary_==second.repository_vocabulary_;np.testing.assert_array_equal(first.positive_counts_,second.positive_counts_);assert first.known_rows_==second.known_rows_
    for key in ['components_','singular_values_','explained_variance_','explained_variance_ratio_']:np.testing.assert_array_equal(getattr(first.projection_,key),getattr(second.projection_,key))
    assert first.head_.get_params()==second.head_.get_params();assert len(first.head_.estimators_)==len(second.head_.estimators_)==256
    for a,b in zip(first.head_.estimators_,second.head_.estimators_):
        aa,bb=a.tree_.__getstate__(),b.tree_.__getstate__();assert aa.keys()==bb.keys()
        for key in aa:
            if isinstance(aa[key],np.ndarray):np.testing.assert_array_equal(aa[key],bb[key])
            else:assert aa[key]==bb[key]


def main(args):
    require_server();threadpool_limits(1);out=ROOT/'outputs/experiments'/args.run_id;cfg=json.loads((out/'config.json').read_text());result=json.loads((out/'information_gate.json').read_text());data,aux,pr,raw_counts=runner.lineage_inputs();groups=np.load(ROOT/'cache/workshop_v1/key_component_groups.npy');source=ROOT/'outputs/experiments'/cfg['tree_source_run'];scfg=json.loads((source/'config.json').read_text());records=[];truth=[];predictions={key:[] for key in ['global_prior','repository_prior','tree']};jobs=json.loads((out/'head_jobs.json').read_text());all_fit_times=[];all_eval_times=[]
    for i in range(3):
        name=f'fold0_inner{i}';folder=out/'contexts'/name;role=runner.arrays(folder/'roles.npz');features=runner.arrays(folder/'features.npz');fit_targets=runner.arrays(folder/'fit_targets.npz');eval_targets=runner.arrays(folder/'evaluation_targets.npz');scores=runner.arrays(folder/'predictions.npz');fit,query,known,source_data,source_aux,source_groups,x,_=runner.input_values(scfg,name)
        for key,value in [('train',fit),('query',query),('known_API',known)]:np.testing.assert_array_equal(role[key],value)
        assert not np.intersect1d(pr[fit],pr[query]).size and not np.intersect1d(groups[fit],groups[query]).size;counts=Counter(map(int,pr[fit]));assert all(n==raw_counts[key] for key,n in counts.items());np.testing.assert_array_equal(features['fit_positions'],fit);np.testing.assert_array_equal(features['query_positions'],query);np.testing.assert_array_equal(features['x_train'],x[fit]);np.testing.assert_array_equal(features['x_query'],x[query]);np.testing.assert_array_equal(features['repo_train'],data['repos'][fit]);np.testing.assert_array_equal(features['repo_query'],data['repos'][query]);yf=literal_targets(aux['targets'],aux['matched'],data['repos'],pr,fit);yq=literal_targets(aux['targets'],aux['matched'],data['repos'],pr,query);np.testing.assert_array_equal(yf,fit_targets['peer_target']);np.testing.assert_array_equal(yq,eval_targets['peer_target']);np.testing.assert_array_equal(scores['positions'],query);np.testing.assert_array_equal(eval_targets['positions'],query)
        # Unknown API values, including invalid placeholders, remain inaccessible.
        poison=aux['targets'].astype(float);poison[~aux['matched'].astype(bool)]=np.nan;np.testing.assert_array_equal(runner.role_targets(data,aux|{'targets':poison},pr,raw_counts,fit),yf);saved=joblib.load(folder/'model.joblib');template=joblib.load(source/'contexts'/name/'model.joblib');np.testing.assert_array_equal(template['projection_fit'],fit);np.testing.assert_array_equal(saved['head'].projection_.components_,template['head'].projection_.components_);model=copy.deepcopy(template['head']);model.jobs=cfg['tree_jobs_per_head'];model.fit_head(features['x_train'],yf,features['repo_train']);model.head_.set_params(n_jobs=1);forest_equal(saved['head'],model);np.testing.assert_array_equal(model.predict_proba(features['x_query'],features['repo_query']),scores['tree']);mean=yf.mean(0);np.testing.assert_array_equal(mean,saved['prior']['mean']);np.testing.assert_array_equal(yf.sum(0),saved['prior']['positive_counts']);assert saved['prior']['fit_rows']==len(fit);global_scores=np.tile(mean,(len(query),1));local=[]
        for repo in features['repo_query']:
            own=features['repo_train']==repo;value=(yf[own].sum(0)+cfg['repository_prior_strength']*mean)/(own.sum()+cfg['repository_prior_strength']) if own.any() else mean;local.append(value)
        local=np.asarray(local);local[:,yf.sum(0)==0]=0;local[:,yf.sum(0)==len(fit)]=1;np.testing.assert_array_equal(global_scores,scores['global_prior']);np.testing.assert_array_equal(local,scores['repository_prior']);record=result['contexts'][i]
        for key in predictions:
            for axis,value in metrics(yq,scores[key]).items():assert abs(value-record['metrics'][key][axis])<1e-12
            predictions[key].append(scores[key])
        truth.append(yq);fitmeta=json.loads((folder/'fit.json').read_text());assert fitmeta['query_targets_or_PR_or_group_loaded'] is False and fitmeta['runtime_seconds']<=cfg['maximum_head_seconds'];assert fitmeta['source_projection_sha256']==digest(source/'contexts'/name/'model.joblib');all_fit_times.append((folder/'fit.json').stat().st_mtime_ns);all_eval_times.append((folder/'evaluation_targets.npz').stat().st_mtime_ns);source_data.close();source_aux.close();records.append({'context':name,'fit_and_query_peer_targets_independently_OR_rebuilt':True,'unknown_API_placeholder_poisoning_fit_target_exact':True,'all256_forests_nodes_values_refit_bitwise':True,'source_projection_features_roles_and_priors_exact':True,'serial_prediction_replay_bitwise':True});print(json.dumps(records[-1]),flush=True)
    assert min(all_eval_times)>=max(all_fit_times),'Evaluation labels existed before all predictors finished'
    # Replay preparation and fitting after actual held-label/API/missingness changes.
    name='fold0_inner0';query=runner.arrays(out/'contexts'/name/'roles.npz')['query'];data_changed={key:value.copy() for key,value in data.items()};data_changed['y'][query]=1-data_changed['y'][query];aux_changed={key:value.copy() for key,value in aux.items()};aux_changed['targets'][query]=1-aux_changed['targets'][query];aux_changed['matched'][query]=~aux_changed['matched'][query].astype(bool);lineage_path=ROOT/'data/processed/20261021_join_lineage_v2/aligned_targets.npz';lineage=runner.arrays(lineage_path);lineage['original_y'][data['source_positions'][query]]=1-lineage['original_y'][data['source_positions'][query]];original_arrays=runner.arrays;dataset_path=ROOT/'cache/workshop_v1/dataset.npz';aux_path=ROOT/'outputs/experiments/20261023_annotation_channel_capability_v1/auxiliary_targets.npz'
    def perturb(path):
        path=Path(path)
        if path==dataset_path:return data_changed
        if path==aux_path:return aux_changed
        if path==lineage_path:return lineage
        if path.name=='evaluation_targets.npz':raise RuntimeError('Held-target file read prohibited during replay preparation or fitting')
        return original_arrays(path)
    runner.arrays=perturb;replay=out/'held_supervision_intervention'
    try:
        job=runner.pack_context(cfg,replay,name);runner.fit_head(job)
    finally:runner.arrays=original_arrays
    for f in ['roles.npz','fit_targets.npz','features.npz','predictions.npz']:
        a=runner.arrays(out/'contexts'/name/f);b=runner.arrays(replay/'contexts'/name/f)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
    forest_equal(joblib.load(out/'contexts'/name/'model.joblib')['head'],joblib.load(replay/'contexts'/name/'model.joblib')['head']);y=np.concatenate(truth);pooled={key:metrics(y,np.concatenate(value)) for key,value in predictions.items()}
    for key in pooled:
        for axis,value in pooled[key].items():assert abs(value-result['pooled_metrics'][key][axis])<1e-12
    b,n=pooled['repository_prior'],pooled['tree'];gates={'pooled_Brier_vs_repository_prior':b['Brier']-n['Brier']>=cfg['minimum_pooled_Brier_improvement'],'pooled_AP_vs_repository_prior':n['micro_AP']-b['micro_AP']>=cfg['minimum_pooled_AP_improvement'],'partition_Brier_vs_repository_prior':max(row['Brier_delta_vs_repository_prior'] for row in result['contexts'])<=cfg['maximum_partition_Brier_loss']};assert gates==result['all_fixed_gates'] and all(gates.values())==result['main_integration_pilot_admitted'];manifest=json.loads((out/'code_manifest.json').read_text())['files']
    for f in ['scripts/run_peer_publication_information.py','src/skillcomp/peer_publication.py','src/skillcomp/proxy_tree_readout.py']:assert digest(ROOT/f)==manifest[f]
    assert digest(ROOT/'outputs/deployment/20261024_full_neighbor_candidate_v1/model.joblib')=='40ed37a1c4bacc9a79d72e0266c245f93d1d28e54df3951c9c1a721b9582adf2';audit={'state':'passed','contexts':records,'independent_forest_refits':3,'independent_trees_refit':768,'additional_intervention_forest':1,'all_eval_targets_created_after_all_model_outputs':True,'actual_held_published_API_missingness_intervention_inputs_parameters_and_probabilities_bitwise_exact':True,'independent_AP_Brier_and_all_fixed_gates_verified':True,'all_fixed_gates':gates,'main_integration_pilot_admitted':all(gates.values()),'source_and_export_unchanged':True,'GPU_or_main_F1_or_outer_predictions':False,'limits':'Identified observed annotation contribution, not latent clean skill truth. Inherited original-fit supervised source encoder; query groups excluded from all original fits. Adaptive weak-label training information study, not independent/gold/official.'};json_write(out/'audit.json',audit);json_write(ROOT/cfg['report_directory']/'audit.json',audit);print(json.dumps({'state':'passed','independent_forest_refits':3,'additional_actual_query_intervention_forest':1,'main_integration_pilot_admitted':all(gates.values()),'gates':gates}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);main(p.parse_args())
