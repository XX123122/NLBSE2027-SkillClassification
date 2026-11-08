"""Refit60 metadata heads, verify20 conditional count roles and fivefold locks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from audit_peer_publication_information import literal_targets
from audit_proxy_replacement_scores import residual
import run_conditional_peer_union_capability as runner
from skillcomp.availability_calibration import fit_availability_intercepts,calibrate_availability
from skillcomp.runtime import ROOT,require_server,json_write


arrays=runner.arrays


def metric(y,p):
    a=precision_recall_fscore_support(y,p,average='micro',zero_division=0);return dict(zip(['precision','recall','micro_f1'],map(float,a[:3])))


def conditional_prior(f,s,pub,repos,queries,strength):
    zero=s==0;den=zero.sum(0);num=(f*zero).sum(0);global_rate=np.where(den>0,(num+.5)/(den+1),f.mean(0));out=[]
    for repo in queries:
        mask=repos==repo;out.append(((f[mask]*zero[mask]).sum(0)+strength*global_rate)/(zero[mask].sum(0)+strength) if mask.any() else global_rate)
    return np.asarray(out),np.tile(pub.sum(0),(len(queries),1)),np.repeat(len(f),len(queries))


def classifier_equal(a,b):
    assert a.get_params()==b.get_params()
    for key in ['coef_','intercept_','classes_','n_iter_']:np.testing.assert_array_equal(getattr(a,key),getattr(b,key))


def verify_context(job):
    threadpool_limits(1);cfg=job['config'];out=Path(job['out']);context=job['context'];name=context.split('_')[0];folder=out/'contexts'/context;fit,query,known,data,aux,groups,x,nn=runner.context_values(cfg,context);canonical,annotations,pr,raw_counts=runner.lineage_inputs();role=arrays(folder/'roles.npz')
    for key,value in [('train',fit),('query',query),('known',known)]:np.testing.assert_array_equal(role[key],value)
    assert not np.intersect1d(groups[fit],groups[query]).size and not np.intersect1d(pr[fit],pr[query]).size;f=literal_targets(annotations['targets'],annotations['matched'],data['repos'],pr,fit);own=np.zeros(f.shape,np.uint8);observed=annotations['matched'][fit].astype(bool);own[observed]=annotations['targets'][fit[observed]];np.testing.assert_array_equal(np.maximum(f,own),data['y'][fit]);targets=arrays(folder/'targets.npz');np.testing.assert_array_equal(f,targets['peer']);np.testing.assert_array_equal(own,targets['own_observed']);inputs=arrays(folder/'inputs.npz');model=joblib.load(folder/'model.joblib');fq,cq,nq=conditional_prior(f,own,data['y'][fit],data['repos'][fit],data['repos'][query],cfg['prior_strength']);np.testing.assert_array_equal(fq,inputs['peer_query']);np.testing.assert_array_equal(cq,inputs['published_counts_query']);np.testing.assert_array_equal(nq,inputs['fit_counts_query']);fc,cc,nc=model['conditional_prior'].fit_centre(data['repos'][fit],groups[fit]);np.testing.assert_array_equal(fc,inputs['peer_centre']);np.testing.assert_array_equal(cc,inputs['published_counts_centre']);np.testing.assert_array_equal(nc,inputs['fit_counts_centre']);unique=np.unique(groups[fit]);literal=0
    for group in unique[np.unique(np.linspace(0,len(unique)-1,3).astype(int))]:
        keep=groups[fit]!=group;rows=np.flatnonzero(~keep);p,c,n=conditional_prior(f[keep],own[keep],data['y'][fit[keep]],data['repos'][fit[keep]],data['repos'][fit[rows]],cfg['prior_strength']);np.testing.assert_array_equal(p,fc[rows]);np.testing.assert_array_equal(c,cc[rows]);np.testing.assert_array_equal(n,nc[rows]);literal+=1
    np.testing.assert_array_equal(nn['neighbors'].targets_,annotations['targets'][known]);np.testing.assert_array_equal(nn['neighbors'].fit_groups_,groups[known]);np.testing.assert_array_equal(nn['neighbors'].predict_proba(x[query],data['repos'][query]),inputs['proxy_query']);np.testing.assert_array_equal(nn['neighbors'].predict_fit_centre(x[fit],data['repos'][fit],groups[fit]),inputs['proxy_centre']);xx,repos=runner.availability_inputs(model['availability_config'],context,fit);gate=joblib.load(ROOT/model['availability_gate_path']);assert digest(ROOT/model['availability_gate_path'])==model['availability_gate_sha256'];np.testing.assert_array_equal(gate['fit'],fit);aq=gate['model'].predict_proba(xx[query])[:,1];mapping=model['availability_mapping']
    if mapping.get('probability_calibration') is not None:aq=calibrate_availability(mapping['probability_calibration'],aq,repos[query])
    np.testing.assert_array_equal(aq,inputs['availability_query']);centre=np.zeros(len(fit));seen=np.zeros(len(fit),int)
    for k,(a,b) in enumerate(GroupKFold(3).split(fit,groups=groups[fit])):
        sub=folder/f'sub{k}';r=arrays(sub/'roles.npz');np.testing.assert_array_equal(r['train'],fit[a]);np.testing.assert_array_equal(r['query'],fit[b]);assert not np.intersect1d(groups[fit[a]],groups[fit[b]]).size;refit=clone(gate['model']).fit(xx[fit[a]],observed[a]);saved=joblib.load(sub/'model.joblib');classifier_equal(refit,saved['model']);pt=refit.predict_proba(xx[fit[a]])[:,1];raw=refit.predict_proba(xx[fit[b]])[:,1];cal=fit_availability_intercepts(pt,observed[a],repos[fit[a]],cfg['availability_intercept_strength']);assert cal==saved['calibration'];probability=calibrate_availability(cal,raw,repos[fit[b]]);aout=arrays(sub/'probabilities.npz');np.testing.assert_array_equal(raw,aout['raw_probability']);np.testing.assert_array_equal(probability,aout['probability']);centre[b]=probability;seen[b]+=1
    assert (seen==1).all();np.testing.assert_array_equal(centre,inputs['availability_centre']);scores=arrays(folder/'scores.npz');np.testing.assert_array_equal(runner.union_probability(fq,aq,inputs['proxy_query'],cq,nq),scores['mapped']);np.testing.assert_array_equal(runner.union_probability(fc,centre,inputs['proxy_centre'],cc,nc),scores['centre_published']);expectation=runner.fit_expectation(scores['centre_published'],repos[fit]);np.testing.assert_array_equal(expectation['mean'],model['expectation']['mean'])
    for repo,value in expectation['repositories'].items():np.testing.assert_array_equal(value,model['expectation']['repositories'][repo])
    if context.endswith('_final'):
        assert (folder/'test_inputs.npz').stat().st_mtime_ns>=(out/'all_calibrations_locked.json').stat().st_mtime_ns;test=arrays(folder/'test_inputs.npz');rows=arrays(ROOT/'cache/workshop_v1/grouped5_pr_components'/(name+'.npz'))['test'];np.testing.assert_array_equal(test['positions'],rows);tf,tc,tn=conditional_prior(f,own,data['y'][fit],repos[fit],repos[rows],cfg['prior_strength']);np.testing.assert_array_equal(tf,test['peer']);np.testing.assert_array_equal(tc,test['published_counts']);np.testing.assert_array_equal(tn,test['fit_counts']);tq=nn['neighbors'].predict_proba(x[rows],repos[rows]);ta=gate['model'].predict_proba(xx[rows])[:,1]
        if mapping.get('probability_calibration') is not None:ta=calibrate_availability(mapping['probability_calibration'],ta,repos[rows])
        np.testing.assert_array_equal(tq,test['proxy']);np.testing.assert_array_equal(ta,test['availability']);np.testing.assert_array_equal(runner.union_probability(tf,ta,tq,tc,tn),test['mapped'])
    return {'context':context,'independent_peer_OR_and_conditional_counts_verified':True,'literal_component_deleted_count_checks':literal,'independent_availability_head_refits':3,'all_input_roles_source_NN_fullgate_and_subfit_calibration_exact':True,'all_published_query_centre_expectation_and_test_forward_if_present_verified':True}


def composition(cfg,out,context,positions,base,test=False):
    name=context.split('_')[0];source=ROOT/'outputs/experiments'/cfg['reference_run'];scfg=json.loads((source/'config.json').read_text());pick=json.loads((source/'all_training_selections_locked.json').read_text())['selections'][name]['best_policy'];folder=out/'contexts'/context;role=arrays(folder/'roles.npz');fit=role['train'];data=arrays(ROOT/'cache/workshop_v1/dataset.npz');aux=arrays(ROOT/'outputs/experiments'/scfg['channel_run']/'auxiliary_targets.npz');repos=data['repos'];new=arrays(folder/('test_inputs.npz' if test else 'scores.npz'))['mapped'];newcentre=joblib.load(folder/'model.joblib')['expectation'];m=aux['matched'][fit].astype(bool);global_rate=float(m.mean());rates={str(repo):float((m[repos[fit]==repo].sum()+20*global_rate)/((repos[fit]==repo).sum()+20)) for repo in np.unique(repos[fit])};coverage=np.clip(np.array([rates.get(str(repo),global_rate)/global_rate for repo in repos[positions]]),0,2);cpick=json.loads((ROOT/'outputs/experiments'/scfg['coverage_run']/'all_training_selections_locked.json').read_text())['selections'][name]['best_policy'];row=(1-cpick['mix'])+cpick['mix']*coverage
    if pick['candidate']=='tree':
        fraction=pick['tree_fraction'];fixed=arrays(source/'contexts'/context/('raw_tree_test_scores.npz' if test else 'raw_tree_scores.npz'))['mapped'];fc=joblib.load(source/'contexts'/context/'model.joblib')['expectation'];weight=scfg['residual_weight'];nweight=1-fraction;fweight=fraction
    else:
        nn=ROOT/'outputs/experiments'/scfg['reference_run'];npick=json.loads((nn/'all_training_selections_locked.json').read_text())['selections'][name]['best_policy'];assert npick['candidate']=='neighbors';fixed=arrays(nn/'contexts'/context/('test_scores.npz' if test else 'scores.npz'))['original'];fc=joblib.load(nn/'contexts'/context/'model.joblib')['old_expectation'];nweight=npick['neighbor_fraction'];fweight=1-nweight;weight=npick['residual_weight']
    mixed=fweight*fixed+nweight*new;expected={'mean':fweight*fc['mean']+nweight*newcentre['mean'],'repositories':{repo:fweight*fc['repositories'][repo]+nweight*value for repo,value in newcentre['repositories'].items()}};score=residual(base,mixed,repos[positions],expected,weight,row);pub,pubweight=runner.publication_source(scfg['publication_source_run'],name);pubscore=arrays(pub/'contexts'/context/('test_scores.npz' if test else 'scores.npz'));np.testing.assert_array_equal(pubscore['positions'],positions);mapping=joblib.load(pub/'contexts'/context/'mapping.joblib');return residual(score,pubscore['scores'],repos[positions],mapping['expectation'],pubweight)


def main(args):
    require_server();threadpool_limits(1);out=ROOT/'outputs/experiments'/args.run_id;cfg=json.loads((out/'config.json').read_text());result=json.loads((out/'results.json').read_text());pilot=ROOT/'outputs/experiments'/cfg['pilot_run'];expansion=json.loads((out/'expansion_lock.json').read_text());assert digest(pilot/'audit.json')==expansion['pilot_audit_sha256'];assert digest(pilot/'training_only_pilot.json')==expansion['pilot_result_sha256'];jobs=[{'out':str(out),'config':cfg,'context':f'fold{i}_{suffix}'} for i in range(5) for suffix in ['inner0','inner1','inner2','final']]
    with ProcessPoolExecutor(max_workers=cfg['CPU_workers'],mp_context=get_context('spawn')) as pool:
        checked=[]
        for record in pool.map(verify_context,jobs):checked.append(record);print(json.dumps({'context_verified':record['context'],'refitted_heads':3}),flush=True)
    selections=json.loads((out/'all_training_selections_locked.json').read_text())['selections'];locks=json.loads((out/'all_calibrations_locked.json').read_text());assert locks['all_five_contexts_locked'] and len(locks['choices'])==5;assert (out/'all_training_selections_locked.json').stat().st_mtime_ns<=(out/'all_calibrations_locked.json').stat().st_mtime_ns;pooled=arrays(out/'pooled_predictions.npz');data=arrays(ROOT/'cache/workshop_v1/dataset.npz');y=data['y'];np.testing.assert_array_equal(y,pooled['y_true']);seen=np.zeros(len(y),int);decisions=[]
    for i in range(5):
        name=f'fold{i}';original,base=runner.base_oof(cfg,name);a=arrays(out/(name+'_oof_scores.npz'));np.testing.assert_array_equal(a['positions'],original['positions']);lookup={int(p):j for j,p in enumerate(original['positions'])};reference,_=runner.reference_oof(cfg['reference_run'],name,original['positions']);np.testing.assert_array_equal(reference,a['reference']);source=ROOT/'outputs/experiments'/cfg['reference_run']
        for k in range(3):
            context=f'{name}_inner{k}';local=arrays(out/'contexts'/context/'policy_scores.npz');rows=np.array([lookup[int(p)] for p in local['positions']]);candidate=composition(cfg,out,context,local['positions'],base[rows]);np.testing.assert_allclose(candidate,local['candidate'],atol=1e-13,rtol=0);np.testing.assert_array_equal(local['candidate'],a['candidate'][rows]);np.testing.assert_array_equal(local['reference'],a['reference'][rows])
        candidate=runner.threshold_choice(original['y_true'],a['candidate'],original['rf_prediction'])|{'candidate':'conditional_union'};control=runner.threshold_choice(original['y_true'],a['reference'],original['rf_prediction'])|{'candidate':'reference'};best=max([candidate,control],key=lambda r:(r['micro_f1'],r['precision'],r['candidate']=='reference'));assert selections[name]=={'new_policy':candidate,'best_policy':best,'reference':control}
        for test,suffix,key in [(False,'_calibration.npz','portfolio'),(True,'_assessment.npz','portfolio_scores')]:
            basea=arrays(ROOT/'outputs/experiments'/cfg['base_run']/(name+suffix));current=arrays(out/(name+suffix));prior=arrays(source/(name+suffix));np.testing.assert_array_equal(basea['positions'],current['positions']);score=composition(cfg,out,name+'_final',current['positions'],basea[key],test)
            for family in ['new_policy','best_policy']:
                expected=prior['best_policy_scores' if test else 'best_policy'] if selections[name][family]['candidate']=='reference' else score;np.testing.assert_allclose(current[family+'_scores' if test else family],expected,atol=1e-13,rtol=0);choice=locks['choices'][name][family]
                if not test:assert runner.threshold_choice(current['y_true'],current[family],current['rf_prediction'])==choice
                else:
                    prediction=current['RF_prediction'] if choice['fallback_rf'] else current[family+'_scores']>=choice['threshold'];np.testing.assert_array_equal(prediction,current[family+'_prediction']);np.testing.assert_array_equal(prediction,pooled[family][current['positions']])
            if test:
                assert (out/(name+suffix)).stat().st_mtime_ns>=(out/'all_calibrations_locked.json').stat().st_mtime_ns;np.testing.assert_array_equal(current['RF_prediction'],prior['RF_prediction']);np.testing.assert_array_equal(current['previous_best_prediction'],prior['best_policy_prediction']);seen[current['positions']]+=1
        decisions.append({'fold':name,'all_training_selection_and_validation_thresholds_verified':True,'source_hierarchy_and_all_role_policies_independently_reconstructed':True})
    assert (seen==1).all();p=arrays(pilot/'training_scores.npz');a=arrays(out/'fold0_oof_scores.npz');np.testing.assert_array_equal(p['scores'],a['candidate']);np.testing.assert_array_equal(p['source'],a['reference']);checkedmetrics={key:metric(y,pooled[key]) for key in result['metrics']}
    for method,value in checkedmetrics.items():
        for key,score in value.items():assert abs(score-result['metrics'][method][key])<1e-14
        assert result['RF_point_gates'][method]==all(value[k]>=checkedmetrics['RF'][k] for k in value)
    # Actual held-query supervision intervention on one complete source context.
    oldcontext=runner.context_values;oldlineage=runner.lineage_inputs;name='fold0_inner0';original_values=oldcontext(cfg,name);held=original_values[1];canonical,annotations,pr,raw_counts=oldlineage();canonical2={key:value.copy() for key,value in canonical.items()};annotations2={key:value.copy() for key,value in annotations.items()};canonical2['y'][held]=1-canonical2['y'][held];annotations2['targets'][held]=1-annotations2['targets'][held];annotations2['matched'][held]=~annotations2['matched'][held].astype(bool)
    def context_perturb(config,context):
        values=list(oldcontext(config,context));values[3]=canonical2;values[4]=annotations2;return tuple(values)
    runner.context_values=context_perturb;runner.lineage_inputs=lambda:(canonical2,annotations2,pr,raw_counts);replay=out/'actual_held_supervision_intervention'
    try:runner.prepare({'config':cfg,'out':str(replay),'context':name})
    finally:runner.context_values=oldcontext;runner.lineage_inputs=oldlineage
    for f in ['roles.npz','targets.npz','inputs.npz','scores.npz','policy_scores.npz']:
        a=arrays(out/'contexts'/name/f);b=arrays(replay/'contexts'/name/f)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
    groups=np.load(ROOT/'cache/workshop_v1/key_component_groups.npy');differences={}
    for method in ['new_policy','best_policy']:
        pred=pooled[method];old=pooled['previous_best'];details={}
        for label,clusters in [('component',groups),('repository',data['repos'])]:
            tallies=[]
            for group in np.unique(clusters):
                rows=clusters==group;tallies.append([[(p[rows]&(y[rows]>0)).sum(),p[rows].sum(),y[rows].sum()] for p in [pred,old]])
            tally=np.asarray(tallies,np.int64);draw=np.random.RandomState(9231).randint(len(tally),size=(2000,len(tally)));counts=tally[draw].sum(1);f1=2*counts[:,:,0]/np.maximum(counts[:,:,1]+counts[:,:,2],1);details[label+'_descriptive_paired_F1_delta_95pct']=np.quantile(f1[:,0]-f1[:,1],[.025,.975]).tolist()
        details.update(TP_delta=int((pred*(y>0)).sum()-(old*(y>0)).sum()),FP_delta=int((pred*(y==0)).sum()-(old*(y==0)).sum()),changed_cells=int((pred!=old).sum()),point_F1_delta=checkedmetrics[method]['micro_f1']-checkedmetrics['previous_best']['micro_f1']);differences[method]=details
    manifest=json.loads((out/'code_manifest.json').read_text())['files']
    for f in ['scripts/run_conditional_peer_union_capability.py','src/skillcomp/conditional_peer_prior.py','src/skillcomp/peer_union_model.py']:assert digest(ROOT/f)==manifest[f]
    assert digest(ROOT/'outputs/deployment/20261024_full_neighbor_candidate_v1/model.joblib')=='40ed37a1c4bacc9a79d72e0266c245f93d1d28e54df3951c9c1a721b9582adf2';audit={'state':'passed','contexts':checked,'fold_decisions':decisions,'independent_availability_head_refits':60,'additional_intervention_heads':3,'literal_component_deleted_conditional_prior_checks':sum(r['literal_component_deleted_count_checks'] for r in checked),'all_five_training_then_validation_locks_precede_new_test_artifacts':True,'pilot_fold0_training_scores_bitwise_equal':True,'actual_held_supervision_intervention_inputs_parameters_probabilities_policy_exact':True,'metrics':checkedmetrics,'differences':differences,'source_and_export_unchanged':True,'limits':'All prior and head roles verified, conditional background Repo approximation/source encoder shared-centres persist; adaptive weak-label fivefold development, descriptive CIs omit historical selection/retraining uncertainty, not official/gold/independent significance.'};json_write(out/'verification.json',audit);json_write(ROOT/cfg['report_directory']/'verification.json',audit);print(json.dumps({'state':'passed','metrics':checkedmetrics,'differences':differences,'availability_refits':60}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);main(p.parse_args())
