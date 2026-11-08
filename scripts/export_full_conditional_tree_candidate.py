"""Full fit after audited positive OOF; isolated bundle and real-input checks."""
import argparse
import copy
import json
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
from conditional_inference_reservation import run_reserved
from select_full_conditional_peer_union import availability_x
from build_full_auxiliary_portfolio import cached_features
from run_peer_publication_information import arrays,lineage_inputs,role_targets
from skillcomp.availability_calibration import fit_availability_intercepts,calibrate_availability
from skillcomp.annotation_channel import apply_channel
from skillcomp.channel_residual import fit_expectation,apply_residual
from skillcomp.conditional_peer_prior import ConditionalPeerPrior
from skillcomp.peer_union_model import union_probability
from skillcomp.proxy_tree_readout import ProxyTreeReadout
from skillcomp.retrieval_evidence import EvidenceNeighbors,ExcludedChannelStatistics
from skillcomp.evidence_coverage import fit_coverage,coverage_weights
from skillcomp.frozen_inference import predict_with_embeddings
from skillcomp.residual_inference import fuse_auxiliary_embeddings
from skillcomp.availability_inference import refine_availability
from skillcomp.conditional_tree_inference import refine_conditional_tree_union
from skillcomp.runtime import ROOT,require_server,json_write,status,official_module


def model_names(bundle):
    aux=bundle['residual_component']['encoder'];members=aux.get('encoders',[aux]);return [bundle['encoder']['model']]+[m['model'] for m in members if 'model' in m]


def run(args):
    require_server();threadpool_limits(1);started=time.monotonic();out=ROOT/'outputs/deployment'/args.run_id;cfg=json.loads((out/'config.json').read_text());audit=json.loads((out/'selection_audit.json').read_text());selection=json.loads((out/'training_selection_locked.json').read_text())['selection'];assert audit['state']=='passed' and audit['new_fullfit_export_admitted'] and selection['candidate']=='tree_conditional_union';assert not (out/'model.joblib').exists();source=ROOT/cfg['source_deployment'];assert digest(source/'model.joblib')==cfg['source_model_sha256'];bundle=joblib.load(source/'model.joblib');data,aux,pr,raw_counts=lineage_inputs();y,repos=data['y'],data['repos'];fit=np.arange(len(y));known=fit[aux['matched'].astype(bool)];groups=np.load(ROOT/'cache/workshop_v1/key_component_groups.npy');folder=out/'contexts/full_final';folder.mkdir();status(out,'running',stage='full7154_fixed_tree_conditional_union_fit');x=cached_features(dict(cfg,auxiliary_repository_scale=0),'full_final','learned_fused',fit);f=role_targets(data,aux,pr,raw_counts,fit);own=np.zeros(f.shape,np.uint8);own[known]=aux['targets'][known];prior=ConditionalPeerPrior(20).fit(f,own,repos,groups,y);nn=EvidenceNeighbors(**cfg['neighbors']).fit(x[known],aux['targets'][known],repos[known],groups[known]);qcentre=nn.predict_fit_centre(x,repos,groups);fc,cc,nc=prior.fit_centre(repos,groups);xx=availability_x(cfg,'full_final',fit,repos);av=bundle['availability_component'];observed=aux['matched'].astype(bool);ac=np.zeros(len(fit));treecentre=np.zeros(y.shape);seen=np.zeros(len(fit),int);template=ProxyTreeReadout(**cfg['tree']).fit_projection(x,repos)
    for k,(a,b) in enumerate(GroupKFold(3).split(fit,groups=groups)):
        assert not np.intersect1d(groups[a],groups[b]).size;classifier=clone(av['gate']).fit(xx[a],observed[a]);pt=classifier.predict_proba(xx[a])[:,1];raw=classifier.predict_proba(xx[b])[:,1];cal=fit_availability_intercepts(pt,observed[a],repos[a],20);ac[b]=calibrate_availability(cal,raw,repos[b]);meta=folder/f'availability_sub{k}';meta.mkdir();joblib.dump({'model':classifier,'calibration':cal},meta/'model.joblib');np.savez_compressed(meta/'roles.npz',train=a,query=b);np.savez_compressed(meta/'probabilities.npz',positions=b,probability=ac[b],raw=raw);train_known=a[observed[a]];tree=copy.deepcopy(template).fit_head(x[train_known],aux['targets'][train_known],repos[train_known]);tree.head_.set_params(n_jobs=1);treecentre[b]=tree.predict_proba(x[b],repos[b]);sub=folder/f'tree_sub{k}';sub.mkdir();joblib.dump(tree,sub/'model.joblib');np.savez_compressed(sub/'roles.npz',train=a,known=train_known,query=b);seen[b]+=1
    assert (seen==1).all();tree=copy.deepcopy(template).fit_head(x[known],aux['targets'][known],repos[known]);tree.head_.set_params(n_jobs=1);channel=bundle['neighbor_component']['channel'];stats=ExcludedChannelStatistics(y,aux['targets'],aux['matched'],repos,groups);mappedtreecentre=stats.map_fit_centre(treecentre,repos,groups);mappednncentre=union_probability(fc,ac,qcentre,cc,nc);tree_expectation=fit_expectation(mappedtreecentre,repos);nn_expectation=fit_expectation(mappednncentre,repos);expectation={'mean':.5*tree_expectation['mean']+.5*nn_expectation['mean'],'repositories':{repo:.5*tree_expectation['repositories'][repo]+.5*value for repo,value in nn_expectation['repositories'].items()}};coverage=fit_coverage(aux['matched'],repos,20);joblib.dump({'tree':tree,'neighbors':nn,'prior':prior,'expectation':expectation,'tree_expectation':tree_expectation,'NN_expectation':nn_expectation,'channel':channel},folder/'fitted_assets.joblib');np.savez_compressed(folder/'roles.npz',train=fit,known=known);np.savez_compressed(folder/'targets.npz',positions=fit,peer=f,own_observed=own);np.savez_compressed(folder/'centre_probabilities.npz',proxy_NN=qcentre,availability=ac,peer_F0=fc,published_counts=cc,fit_counts=nc,proxy_tree=treecentre,mapped_tree=mappedtreecentre,mapped_NN=mappednncentre);inference_prior=copy.deepcopy(prior);inference_prior.groups_={};inference_prior.group_local_={};inference_nn=copy.deepcopy(nn)
    for attr in ['fit_groups_','group_counts_','group_repo_counts_']:
        if hasattr(inference_nn,attr):delattr(inference_nn,attr)
    probe=np.array([0,1,2]);np.testing.assert_array_equal(nn.predict_proba(x[probe],repos[probe]),inference_nn.predict_proba(x[probe],repos[probe]));p,c,n=prior.predict(repos[probe]);pp,cc2,nn2=inference_prior.predict(repos[probe]);np.testing.assert_array_equal(p,pp);np.testing.assert_array_equal(c,cc2);np.testing.assert_array_equal(n,nn2);component={'tree':tree,'neighbors':inference_nn,'conditional_prior':inference_prior,'channel':channel,'expectation':expectation,'coverage':coverage,'selection':selection|{'mix':1,'tree_fraction':.5,'residual_weight':.5},'training_group_metadata_removed':True};bundle.pop('neighbor_component',None);bundle.pop('policy_ensemble_component',None);bundle.update(conditional_tree_component=component,kind='full_tree_conditional_observed_union_classifier',independent_score=None);joblib.dump(bundle,out/'model.joblib');records=json.loads((source/'verification_inputs.json').read_text());assert len(records)>=8;records=records[:8];json_write(out/'verification_inputs.json',records);rows=np.arange(len(records));base_cfg=json.loads((ROOT/cfg['base_deployment']/'config.json').read_text());main=np.load(ROOT/base_cfg['representations'][bundle['spec']['representation']]['embedding_path'],mmap_mode='r')[rows];_,base=predict_with_embeddings(bundle,records,main,official_module().clean_text);frozen=np.load(ROOT/cfg['frozen_features'],mmap_mode='r')[rows];expected=refine_conditional_tree_union(bundle,component,base,x[rows],repos[rows],records,frozen);expected=refine_availability(bundle,av,expected,records,semantic_embeddings=frozen);np.save(out/'cached_verification_scores.npy',expected);np.save(out/'cached_verification_vectors.npy',expected>=selection['threshold']);json_write(out/'manifest.json',{'state':'exported_pending_verification','model_path':str(out/'model.joblib'),'model_sha256':digest(out/'model.joblib'),'source_model_sha256':cfg['source_model_sha256'],'training_rows':len(y),'labels':217,'training_OOF_selection':selection,'training_OOF_not_test_score':True,'independent_score':None,'new_benchmark_score_claimed':False,'inference_entrypoint':'scripts/predict_conditional_tree_candidate.py','training_group_metadata_removed_from_inference_components':True,'raw_input_predictions_equal_cached_feature_predictions':False,'full_final_fit_seconds':time.monotonic()-started});status(out,'running',stage='raw_text_encoder_vs_cached_verification',model_path=str(out/'model.joblib'))
    with (out/'raw_inference_verification.log').open('x') as stream:gpu=run_reserved([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/predict_conditional_tree_candidate.py'),'--bundle',str(out/'model.joblib'),'--input',str(out/'verification_inputs.json'),'--output',str(out/'verification_predictions.json'),'--scores',str(out/'verification_scores.npy')],stream,model_names(bundle))
    actual=np.load(out/'verification_scores.npy');predictions=json.loads((out/'verification_predictions.json').read_text());binary=np.array([r['vector'] for r in predictions['predictions']],bool);np.testing.assert_array_equal(binary,expected>=selection['threshold']);delta=float(np.abs(actual-expected).max());assert delta<=1e-5;assert predictions['label_order']==bundle['labels'];manifest=json.loads((out/'manifest.json').read_text());manifest.update(state='raw_verified_pending_corpus_and_parameter_audit',verification_rows=len(records),raw_input_predictions_equal_cached_feature_predictions=True,score_max_absolute_difference=delta,raw_verification_GPU=gpu,runtime_seconds=time.monotonic()-started);json_write(out/'manifest.json',manifest);report=ROOT/cfg['report_directory'];json_write(report/'manifest.json',manifest);status(out,'complete',stage='export_raw_checked_pending_corpus_and_final_fit_audit',model_path=str(out/'model.joblib'),model_sha256=manifest['model_sha256'],independent_score=None);print(json.dumps({'state':manifest['state'],'model_sha256':manifest['model_sha256'],'raw_cached_max_abs':delta,'raw_verification_GPU':gpu,'training_OOF_F1':selection['micro_f1']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--launch',action='store_true');args=p.parse_args()
    if args.launch:
        out=ROOT/'outputs/deployment'/args.run_id;assert json.loads((out/'selection_audit.json').read_text())['state']=='passed' and not (out/'model.joblib').exists();env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+':'+str(ROOT/'scripts')
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
        with (out/'export.log').open('x') as stream:child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--run-id',args.run_id],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'state':'launched','PID':child.pid,'run_id':args.run_id}))
    else:
        try:run(args)
        except BaseException:
            out=ROOT/'outputs/deployment'/args.run_id;(out/'export_error.txt').write_text(traceback.format_exc());status(out,'failed',error=traceback.format_exc());raise
