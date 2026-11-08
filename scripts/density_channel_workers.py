"""Own-fit text classifier of coarse label density, with a shrunk mixture decoder."""
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from skillcomp.channel_residual import fit_expectation
from skillcomp.density_channel import density_strata,fit_density_channel,apply_density_channel
from skillcomp.matchability import matchability_features
from skillcomp.runtime import ROOT


def features(cfg,context,fit):
    folder=ROOT/'outputs/experiments'/cfg['lexical_run']/'contexts'/context;role=np.load(folder/'roles.npz');np.testing.assert_array_equal(role['train'],fit);repos=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];x=matchability_features(sparse.load_npz(folder/'title.npz'),sparse.load_npz(folder/'body.npz'),repos[fit],repos);frozen=ROOT/cfg['density_embedding_path'];assert digest(frozen)==cfg['density_embedding_sha256'];dense=normalize(np.load(frozen,mmap_mode='r'),norm='l2');return sparse.hstack([x,sparse.csr_matrix(dense)],format='csr',dtype=np.float32),repos


def prepare(job):
    threadpool_limits(1);out=Path(job['out']);cfg=job['config'];context=job['context'];role=np.load(ROOT/'outputs/experiments'/cfg['lexical_run']/'contexts'/context/'roles.npz');fit,query=role['train'],role['query'];assert not np.intersect1d(fit,query).size;groups=np.load(ROOT/'cache/workshop_v1/key_component_groups.npy');assert not np.intersect1d(groups[fit],groups[query]).size;y=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['y'];x,repos=features(cfg,context,fit);strata,bounds=density_strata(y[fit]);head=LogisticRegression(C=1,solver='lbfgs',max_iter=2000,random_state=42) if len(np.unique(strata))>1 else DummyClassifier(strategy='prior');head.fit(x[fit],strata);channel=fit_density_channel(y[fit],strata,repos[fit],head.classes_,cfg['strength']);center=fit_expectation(apply_density_channel(channel,head.predict_proba(x[fit]),repos[fit]),repos[fit]);mapped=apply_density_channel(channel,head.predict_proba(x[query]),repos[query]);folder=out/'contexts'/context;folder.mkdir(parents=True);joblib.dump({'head':head,'channel':channel,'expectation':center,'train':fit,'density_bounds':bounds,'input_dimensions':x.shape[1],'target':'fit_published_label_density_tertiles','query_target_count_supplied':False},folder/'mapping.joblib');np.savez_compressed(folder/'scores.npz',positions=query,scores=mapped)
    return {'context':context,'fit_rows':len(fit),'query_rows':len(query),'class_counts':channel['class_counts'].tolist(),'density_bounds':bounds.tolist(),'query_target_count_supplied':False,'new_encoder_fits':0,'classifier_C':1,'converged':bool(np.max(getattr(head,'n_iter_',0))<2000)}


def test(job):
    threadpool_limits(1);out=Path(job['out']);cfg=job['config'];name=job['name'];context=name+'_final';assert (out/'all_calibrations_locked.json').exists();role=np.load(ROOT/'cache/workshop_v1/grouped5_pr_components'/(name+'.npz'));model=joblib.load(out/'contexts'/context/'mapping.joblib');np.testing.assert_array_equal(model['train'],role['train']);x,repos=features(cfg,context,role['train']);prediction=model['head'].predict_proba(x[role['test']]);scores=apply_density_channel(model['channel'],prediction,repos[role['test']]);np.savez_compressed(out/'contexts'/context/'test_scores.npz',positions=role['test'],scores=scores)
    return {'name':name,'test_rows':len(role['test']),'query_target_count_supplied':False}
