"""Independent SVD/12 forests, fixed full-data policy, selection and isolation."""
import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support,average_precision_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from audit_proxy_tree_readout_capability import compare_forests
from audit_proxy_replacement_scores import residual
import select_full_tree_conditional_union as runner
from skillcomp.annotation_channel import fit_channel
from skillcomp.runtime import ROOT,require_server,json_write


arrays=runner.arrays


def metric(y,p):
    a=precision_recall_fscore_support(y,p,average='micro',zero_division=0);return dict(zip(['precision','recall','micro_f1'],map(float,a[:3])))


def main(args):
    require_server();threadpool_limits(1);out=ROOT/'outputs/deployment'/args.run_id;cfg=json.loads((out/'config.json').read_text());result=json.loads((out/'training_selection_locked.json').read_text());union=ROOT/'outputs/deployment'/cfg['conditional_selection_run'];lock=json.loads((out/'selection_contract_locked.json').read_text());assert digest(union/'selection_audit.json')==lock['conditional_selection_audit_sha256'];assert digest(union/'training_selection_locked.json')==lock['conditional_selection_result_sha256'];data,aux,pr,raw_counts=runner.lineage_inputs();y,repos=data['y'],data['repos'];groups=np.load(ROOT/'cache/workshop_v1/key_component_groups.npy');oof=arrays(out/'full_training_OOF.npz');previous=arrays(union/'full_training_OOF.npz');np.testing.assert_array_equal(oof['positions'],np.arange(len(y)));np.testing.assert_array_equal(oof['reference'],previous['reference']);np.testing.assert_array_equal(oof['rf_prediction'],previous['rf_prediction']);records=[];parts=[];seen=np.zeros(len(y),int);literal_checks=0;maxerror=0.
    for i in range(3):
        context=f'full_inner{i}';folder=out/'contexts'/context;r=arrays(folder/'roles.npz');oldrole=arrays(union/'contexts'/context/'roles.npz')
        for key in r:np.testing.assert_array_equal(r[key],oldrole[key])
        fit,query,known=r['train'],r['query'],r['known'];assert not np.intersect1d(groups[fit],groups[query]).size and not np.intersect1d(pr[fit],pr[query]).size;np.testing.assert_array_equal(known,fit[aux['matched'][fit].astype(bool)]);x=runner.cached_features(dict(cfg,auxiliary_repository_scale=0),context,'learned_fused',fit);model=joblib.load(folder/'model.joblib');np.testing.assert_array_equal(model['projection_fit'],fit);template=runner.ProxyTreeReadout(**cfg['tree']).fit_projection(x[fit],repos[fit]);np.testing.assert_array_equal(template.projection_.components_,model['head'].projection_.components_);head=copy.deepcopy(template).fit_head(x[known],aux['targets'][known],repos[known]);head.head_.set_params(n_jobs=1);compare_forests(head,model['head']);pred=head.predict_proba(x[query],repos[query]);savedtree=arrays(folder/'tree_probabilities.npz');np.testing.assert_array_equal(pred,savedtree['query_proxy']);centre=np.zeros((len(fit),217));headseen=np.zeros(len(fit),int)
        for k,(a,b) in enumerate(GroupKFold(3).split(fit,groups=groups[fit])):
            sub=folder/f'sub{k}';roles=arrays(sub/'roles.npz');observed=fit[a][aux['matched'][fit[a]].astype(bool)];np.testing.assert_array_equal(roles['train'],fit[a]);np.testing.assert_array_equal(roles['known'],observed);np.testing.assert_array_equal(roles['query'],fit[b]);assert not np.intersect1d(groups[observed],groups[fit[b]]).size;subhead=copy.deepcopy(template).fit_head(x[observed],aux['targets'][observed],repos[observed]);subhead.head_.set_params(n_jobs=1);compare_forests(subhead,joblib.load(sub/'model.joblib'));centre[b]=subhead.predict_proba(x[fit[b]],repos[fit[b]]);headseen[b]+=1
        assert (headseen==1).all();np.testing.assert_array_equal(centre,savedtree['centre_proxy']);channel=fit_channel(y[fit],aux['targets'][fit],aux['matched'][fit],repos[fit],20);np.testing.assert_array_equal(channel['global_rates'],model['channel']['global_rates'])
        for repo,value in channel['repository_rates'].items():np.testing.assert_array_equal(value,model['channel']['repository_rates'][repo])
        np.testing.assert_array_equal(runner.apply_channel(channel,pred,repos[query]),savedtree['query_mapped']);stats=runner.ExcludedChannelStatistics(y[fit],aux['targets'][fit],aux['matched'][fit],repos[fit],groups[fit]);mappedcentre=stats.map_fit_centre(centre,repos[fit],groups[fit]);np.testing.assert_array_equal(mappedcentre,savedtree['centre_mapped']);unique=np.unique(groups[fit])
        for group in unique[np.unique(np.linspace(0,len(unique)-1,5).astype(int))]:
            keep=groups[fit]!=group;rows=np.flatnonzero(~keep);literal=fit_channel(y[fit[keep]],aux['targets'][fit[keep]],aux['matched'][fit[keep]],repos[fit[keep]],20);np.testing.assert_array_equal(runner.apply_channel(literal,centre[rows],repos[fit[rows]]),mappedcentre[rows]);literal_checks+=1
        expected=runner.fit_expectation(mappedcentre,repos[fit]);np.testing.assert_array_equal(expected['mean'],model['tree_expectation']['mean'])
        for repo,value in expected['repositories'].items():np.testing.assert_array_equal(value,model['tree_expectation']['repositories'][repo])
        assert digest(union/'contexts'/context/'scores.npz')==model['conditional_source_scores_sha256'];assert digest(union/'contexts'/context/'model.joblib')==model['conditional_source_model_sha256'];u=arrays(union/'contexts'/context/'scores.npz');umodel=joblib.load(union/'contexts'/context/'model.joblib');uc=umodel['expectation'];mix=.5*savedtree['query_mapped']+.5*u['mapped'];expectedmix={'mean':.5*expected['mean']+.5*uc['mean'],'repositories':{repo:.5*expected['repositories'][repo]+.5*value for repo,value in uc['repositories'].items()}};np.testing.assert_array_equal(expectedmix['mean'],model['expectation']['mean'])
        for repo,value in expectedmix['repositories'].items():np.testing.assert_array_equal(value,model['expectation']['repositories'][repo])
        saved=arrays(folder/'scores.npz');np.testing.assert_array_equal(mix,saved['mapped']);candidate=residual(residual(u['base'],mix,repos[query],expectedmix,.5,u['row_weights']),u['publication'],repos[query],umodel['publication_expectation'],umodel['publication_weight']);error=float(np.abs(candidate-saved['candidate']).max());assert error<=1e-13;maxerror=max(maxerror,error);np.testing.assert_array_equal(saved['reference'],oof['reference'][query]);np.testing.assert_array_equal(saved['candidate'],oof['candidate'][query]);np.testing.assert_array_equal(saved['rf_prediction'],oof['rf_prediction'][query]);parts.append(query);seen[query]+=1;records.append({'context':context,'SVD_independently_refit_bitwise':True,'all_four_forests_nodes_and_values_refit_bitwise':True,'all_roles_known_targets_and_held_centres_verified':True,'literal_excluded_channel_checks':5,'reused_conditional_NN_source_hashes_unchanged':True,'all_policy_scores_reconstructed':True,'score_max_abs':error});print(json.dumps(records[-1]),flush=True)
    assert (seen==1).all();control=runner.threshold_choice(y,oof['reference'],oof['rf_prediction']);candidate=runner.threshold_choice(y,oof['candidate'],oof['rf_prediction']);assert control==result['control'] and candidate==result['candidate'];rf=metric(y,oof['rf_prediction']);admitted=candidate['micro_f1']>control['micro_f1'] and not candidate['fallback_rf'] and all(candidate[key]>=rf[key] for key in rf);assert admitted==result['new_fullfit_export_admitted'];choice=(candidate|{'candidate':'tree_conditional_union'}) if admitted else(control|{'candidate':'reference'});assert choice==result['selection'];deltas=[metric(y[rows],oof['candidate'][rows]>=candidate['threshold'])['micro_f1']-metric(y[rows],oof['reference'][rows]>=control['threshold'])['micro_f1'] for rows in parts];np.testing.assert_allclose(deltas,result['partition_F1_deltas'],atol=1e-12,rtol=0)
    for key,values in [('control',oof['reference']),('candidate',oof['candidate'])]:
        m=metric(y,values>=result[key]['threshold'])
        for axis,value in m.items():assert abs(value-result[key][axis])<1e-14
    # Original held labels cannot affect this originalfit tree or fixed reused NN assets.
    query=arrays(out/'contexts/full_inner0/roles.npz')['query'];data2={key:value.copy() for key,value in data.items()};aux2={key:value.copy() for key,value in aux.items()};data2['y'][query]=1-data2['y'][query];aux2['targets'][query]=1-aux2['targets'][query];aux2['matched'][query]=~aux2['matched'][query].astype(bool);old=runner.lineage_inputs;runner.lineage_inputs=lambda:(data2,aux2,pr,raw_counts);replay=out/'actual_held_supervision_intervention'
    try:runner.prepare({'out':str(replay),'config':cfg,'context':'full_inner0'})
    finally:runner.lineage_inputs=old
    for f in ['roles.npz','tree_probabilities.npz','scores.npz']:
        a=arrays(out/'contexts/full_inner0'/f);b=arrays(replay/'contexts/full_inner0'/f)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
    compare_forests(joblib.load(out/'contexts/full_inner0/model.joblib')['head'],joblib.load(replay/'contexts/full_inner0/model.joblib')['head'])
    for k in range(3):compare_forests(joblib.load(out/'contexts/full_inner0'/f'sub{k}'/'model.joblib'),joblib.load(replay/'contexts/full_inner0'/f'sub{k}'/'model.joblib'))
    manifest=json.loads((out/'code_manifest.json').read_text())['files']
    for f in ['scripts/select_full_tree_conditional_union.py','src/skillcomp/proxy_tree_readout.py','src/skillcomp/retrieval_evidence.py']:assert digest(ROOT/f)==manifest[f]
    assert digest(ROOT/cfg['source_deployment']/'model.joblib')==cfg['source_model_sha256'];new=oof['candidate']>=candidate['threshold'];old=oof['reference']>=control['threshold'];diagnostic={'state':'complete','training_OOF_not_test_score':True,'reference_Brier':float(np.mean((oof['reference']-y)**2)),'candidate_Brier':float(np.mean((oof['candidate']-y)**2)),'reference_micro_AP':float(average_precision_score(y.ravel(),oof['reference'].ravel())),'candidate_micro_AP':float(average_precision_score(y.ravel(),oof['candidate'].ravel())),'changed_binary_cells':int((new!=old).sum()),'TP_delta':int((new*(y>0)).sum()-(old*(y>0)).sum()),'FP_delta':int((new*(y==0)).sum()-(old*(y==0)).sum()),'no_reselection':True};audit={'state':'passed','contexts':records,'independent_projections_refit':3,'independent_forests_refit':12,'independent_trees_refit':3072,'additional_intervention_projection':1,'additional_intervention_forests':4,'literal_channel_component_deletion_checks':literal_checks,'all7154_scores_control_metrics_thresholds_and_selection_verified':True,'actual_held_supervision_intervention_model_outputs_bitwise_unchanged':True,'source_conditional_assets_and_full_model_sha256_unchanged':True,'new_fullfit_export_admitted':admitted,'selection':choice,'full_model_exported':False,'new_GPU_or_encoders_or_outer_scores':False,'limits':'Ownfit supervised encoder andSVD centre sharing retained; fit counts/heads/components isolated, weak-label fulltraining OOF not independentfivefold/gold/official.'}
    for file,value in [('selection_audit.json',audit),('training_diagnostic.json',diagnostic)]:json_write(out/file,value);json_write(ROOT/cfg['report_directory']/file,value)
    print(json.dumps({'state':'passed','new_fullfit_export_admitted':admitted,'selection':choice,'diagnostic':diagnostic}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);main(p.parse_args())
