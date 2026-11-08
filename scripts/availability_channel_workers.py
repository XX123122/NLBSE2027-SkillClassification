"""Use inferred evidence availability to model published-label sampling strata."""
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from skillcomp.annotation_channel import fit_channel,apply_channel
from skillcomp.channel_residual import fit_expectation
from skillcomp.matchability import matchability_features
from skillcomp.availability_calibration import fit_availability_intercepts,calibrate_availability
from skillcomp.runtime import ROOT


def inputs(cfg,context,fit):
    lexical=ROOT/'outputs/experiments'/cfg['lexical_run']/'contexts'/context;role=np.load(lexical/'roles.npz');np.testing.assert_array_equal(role['train'],fit);repos=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];x=matchability_features(sparse.load_npz(lexical/'title.npz'),sparse.load_npz(lexical/'body.npz'),repos[fit],repos)
    if cfg.get('availability_embedding'):
        settings=cfg['availability_embedding'];path=ROOT/settings['embedding_path'];manifest=json.loads((ROOT/settings['manifest']).read_text());assert digest(path)==manifest['combined_sha256'] and not manifest['encoder_finetuned'] and not manifest['labels_used_for_encoding'];assert manifest['encoded_original_fields_sha256']==digest(ROOT/'outputs/experiments/20261023_markdown_semantic_features_v1/original_fields.json');values=np.load(path,mmap_mode='r');assert values.shape==(len(repos),manifest['dimensions']);x=normalize(sparse.hstack([x,sparse.csr_matrix(normalize(values,norm='l2'))],format='csr'),norm='l2')
    if cfg.get('availability_structure'):
        from skillcomp.structure_offset import structural_features
        settings=cfg['availability_structure'];fields_path=ROOT/settings['fields_path'];assert digest(fields_path)==settings['fields_sha256'];fields=json.loads(fields_path.read_text());structure=structural_features(fields['title'],fields['body']);assert structure.shape==(len(repos),5);x=sparse.hstack([x,sparse.csr_matrix(structure)],format='csr')
    return x,repos


def prepare(job):
    if job['config'].get('availability_model')=='joint_evidence':
        from joint_evidence_channel_workers import prepare as joint_prepare
        return joint_prepare(job)
    if job['config'].get('availability_model')=='predicted_density':
        from density_channel_workers import prepare as density_prepare
        return density_prepare(job)
    threadpool_limits(1);out=Path(job['out']);cfg=job['config'];context=job['context'];role=np.load(ROOT/'outputs/experiments'/cfg['lexical_run']/'contexts'/context/'roles.npz');fit,query=role['train'],role['query'];gate_path=ROOT/'outputs/experiments'/cfg['gate_source_run']/'gates'/context/'gate.joblib';gate=joblib.load(gate_path);np.testing.assert_array_equal(gate['fit'],fit);assert np.array_equal(gate['model'].classes_,[False,True]);aux=np.load(ROOT/'outputs/experiments'/cfg['auxiliary_source_run']/'auxiliary_targets.npz');known=aux['matched'][fit].astype(bool);assert abs(gate['fit_coverage']-known.mean())<1e-15;y=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['y'];x,repos=inputs(cfg,context,fit)
    # 0 denotes missing API evidence, never an absent skill; all published fit labels are observed.
    if cfg.get('availability_embedding'):
        if cfg.get('availability_structure'):
            from skillcomp.structured_availability import StructuredAvailabilityClassifier
            model=StructuredAvailabilityClassifier(dimensions=5,scale=.25,c=1).fit(x[fit],known)
        elif cfg.get('kernel_availability'):
            from skillcomp.kernel_availability import KernelAvailabilityClassifier
            settings=cfg['kernel_availability'];model=KernelAvailabilityClassifier(gamma=settings['gamma'],components=settings['components'],c=1,seed=settings['seed']).fit(x[fit],known)
        else:model=LogisticRegression(C=1,solver='liblinear',max_iter=2000,random_state=42).fit(x[fit],known)
        gate={'model':model,'fit':fit,'fit_coverage':float(known.mean()),'target':'known_evidence_availability_not_skill_presence'};gate_path=out/'gates'/context/'gate.joblib';gate_path.parent.mkdir(parents=True);joblib.dump(gate,gate_path)
    availability=np.repeat(known[:,None],y.shape[1],axis=1);channel=fit_channel(y[fit],availability,np.ones(len(fit),dtype=bool),repos[fit],cfg['strength']);pfit=gate['model'].predict_proba(x[fit])[:,1];pquery=gate['model'].predict_proba(x[query])[:,1];calibration=None
    if cfg.get('mean_calibrate_availability'):
        calibration=fit_availability_intercepts(pfit,known,repos[fit],cfg['availability_intercept_strength']);pfit=calibrate_availability(calibration,pfit,repos[fit]);pquery=calibrate_availability(calibration,pquery,repos[query])
    center=fit_expectation(apply_channel(channel,np.repeat(pfit[:,None],y.shape[1],axis=1),repos[fit]),repos[fit]);score=apply_channel(channel,np.repeat(pquery[:,None],y.shape[1],axis=1),repos[query]);folder=out/'contexts'/context;folder.mkdir(parents=True);joblib.dump({'channel':channel,'expectation':center,'train':fit,'gate_source_path':str(gate_path.relative_to(ROOT)),'gate_sha256':digest(gate_path),'input_dimensions':x.shape[1],'probability_calibration':calibration},folder/'mapping.joblib');np.savez_compressed(folder/'scores.npz',positions=query,scores=score)
    return {'context':context,'fit_rows':len(fit),'evidence_available_rows':int(known.sum()),'evidence_unavailable_rows':int((~known).sum()),'gate_fit_role_verified':True,'gate_sha256':digest(gate_path),'query_availability_flag_supplied':False,'missing_proxy_rows_used_as_skill_negative':False}


def test(job):
    if job['config'].get('availability_model')=='joint_evidence':
        from joint_evidence_channel_workers import test as joint_test
        return joint_test(job)
    if job['config'].get('availability_model')=='predicted_density':
        from density_channel_workers import test as density_test
        return density_test(job)
    threadpool_limits(1);out=Path(job['out']);cfg=job['config'];name=job['name'];context=name+'_final';assert (out/'all_calibrations_locked.json').exists();role=np.load(ROOT/'cache/workshop_v1/grouped5_pr_components'/(name+'.npz'));mapping=joblib.load(out/'contexts'/context/'mapping.joblib');np.testing.assert_array_equal(mapping['train'],role['train']);gate_path=ROOT/mapping['gate_source_path'];assert digest(gate_path)==mapping['gate_sha256'];gate=joblib.load(gate_path);np.testing.assert_array_equal(gate['fit'],role['train']);x,repos=inputs(cfg,context,role['train']);p=gate['model'].predict_proba(x[role['test']])[:,1]
    if mapping.get('probability_calibration') is not None:p=calibrate_availability(mapping['probability_calibration'],p,repos[role['test']])
    score=apply_channel(mapping['channel'],np.repeat(p[:,None],mapping['channel']['mean'].shape[0],axis=1),repos[role['test']]);np.savez_compressed(out/'contexts'/context/'test_scores.npz',positions=role['test'],scores=score);return {'name':name,'query_availability_flag_supplied':False,'test_rows':len(role['test'])}
