"""Reuse two existing own-fit predictors; estimate one joint publication mixture."""
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from availability_channel_workers import inputs
from run_auxiliary_lexical_residual_capability import features
from run_evidence_coverage_gate_capability import resolve
from skillcomp.annotation_channel import apply_channel
from skillcomp.availability_calibration import calibrate_availability
from skillcomp.channel_residual import fit_expectation
from skillcomp.joint_evidence_channel import fit_missing_publication,joint_publication
from skillcomp.runtime import ROOT


def predictions(cfg,name,context,fit,positions):
    source,pick=resolve(cfg['proxy_policy_run'],name);source_cfg=json.loads((source/'config.json').read_text());folder=source/'contexts'/context/pick['candidate'];model=joblib.load(folder/'model.joblib');np.testing.assert_array_equal(model['train'],fit);assert digest(folder/'model.joblib')==json.loads((folder/'fit.json').read_text())['model_sha256'];aux=np.load(ROOT/'outputs/experiments'/cfg['auxiliary_source_run']/'auxiliary_targets.npz');known=aux['matched'][fit].astype(bool);assert model['known_rows']==int(known.sum());np.testing.assert_array_equal(model['positive_counts'],aux['targets'][fit[known]].sum(0));x=features(source_cfg,context,pick['candidate'],fit,positions=positions);proxy=np.asarray(model['head'].predict_proba(x[positions]));proxy[:,model['positive_counts']==0]=0;proxy[:,model['positive_counts']==model['known_rows']]=1;availability_root=ROOT/'outputs/experiments'/cfg['availability_source_run'];availability_cfg=json.loads((availability_root/'config.json').read_text());mapping=joblib.load(availability_root/'contexts'/context/'mapping.joblib');np.testing.assert_array_equal(mapping['train'],fit);gate_path=ROOT/mapping['gate_source_path'];assert digest(gate_path)==mapping['gate_sha256'];gate=joblib.load(gate_path);np.testing.assert_array_equal(gate['fit'],fit);x,repos=inputs(availability_cfg,context,fit);probability=gate['model'].predict_proba(x[positions])[:,1]
    if mapping.get('probability_calibration') is not None:probability=calibrate_availability(mapping['probability_calibration'],probability,repos[positions])
    channel=apply_channel(model['channel'],proxy,repos[positions]);return probability,channel,{'proxy_source_run':source.name,'proxy_candidate':pick['candidate'],'proxy_model_sha256':digest(folder/'model.joblib'),'availability_gate_sha256':digest(gate_path)}


def prepare(job):
    threadpool_limits(1);out=Path(job['out']);cfg=job['config'];context=job['context'];name=context.split('_')[0];role=np.load(ROOT/'outputs/experiments'/cfg['lexical_run']/'contexts'/context/'roles.npz');fit,query=role['train'],role['query'];groups=np.load(ROOT/'cache/workshop_v1/key_component_groups.npy');assert not np.intersect1d(groups[fit],groups[query]).size;data=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True);y,repos=data['y'],data['repos'];aux=np.load(ROOT/'outputs/experiments'/cfg['auxiliary_source_run']/'auxiliary_targets.npz');missing=fit_missing_publication(y[fit],aux['matched'][fit],repos[fit],cfg['strength']);positions=np.concatenate([fit,query]);probability,channel,identity=predictions(cfg,name,context,fit,positions);scores=joint_publication(missing,probability,channel,repos[positions]);center=fit_expectation(scores[:len(fit)],repos[fit]);folder=out/'contexts'/context;folder.mkdir(parents=True);joblib.dump({'train':fit,'missing_channel':missing,'expectation':center,**identity},folder/'mapping.joblib');np.savez_compressed(folder/'scores.npz',positions=query,scores=scores[len(fit):]);return {'context':context,'fit_rows':len(fit),'known_proxy_fit_rows':int(aux['matched'][fit].sum()),'missing_publication_fit_rows':missing['missing_fit_rows'],'query_proxy_or_availability_targets_supplied':False,'new_encoder_or_classifier_fits':0,**identity}


def test(job):
    threadpool_limits(1);out=Path(job['out']);cfg=job['config'];name=job['name'];context=name+'_final';assert (out/'all_calibrations_locked.json').exists();role=np.load(ROOT/'cache/workshop_v1/grouped5_pr_components'/(name+'.npz'));model=joblib.load(out/'contexts'/context/'mapping.joblib');np.testing.assert_array_equal(model['train'],role['train']);probability,channel,identity=predictions(cfg,name,context,role['train'],role['test'])
    for key,value in identity.items():assert model[key]==value
    repos=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];scores=joint_publication(model['missing_channel'],probability,channel,repos[role['test']]);np.savez_compressed(out/'contexts'/context/'test_scores.npz',positions=role['test'],scores=scores);return {'name':name,'test_rows':len(role['test']),'query_targets_supplied':False}
