"""Generate soft teacher targets using only each student's own fit data."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from skillcomp.benchmark import priors
from skillcomp.models import estimator,continuous_scores
from skillcomp.runtime import ROOT,require_server,json_write


def prepare(job):
    threadpool_limits(1);out=Path(job['out']);cfg=job['config'];context=job['context'];folder=out/'contexts'/context;train=np.load(folder/'roles.npz')['train']
    source=ROOT/'outputs/experiments'/cfg['teacher']['source_run']/'contexts'/context
    channels=ROOT/'outputs/experiments/20261023_title_body_capability_v1/contexts'/context
    source_roles=np.load(channels/'roles.npz');np.testing.assert_array_equal(source_roles['train'],train)
    data=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True);embedding=np.load(ROOT/cfg['embedding_path'],mmap_mode='r')
    x=normalize(sparse.hstack([sparse.load_npz(channels/'title.npz')[train],sparse.load_npz(channels/'body.npz')[train],sparse.csr_matrix(embedding[train])],format='csr'),norm='l2')
    path=source/'split_hybrid1_c1/model.joblib'
    if path.is_file():
        bundle=joblib.load(path);np.testing.assert_array_equal(bundle['train'],train);model=bundle['model'];model.set_params(n_jobs=1);reference={'reused':True,'path':str(path.relative_to(ROOT)),'sha256':digest(path)}
    else:
        assert context.endswith('_final')
        with warnings.catch_warnings(),joblib.parallel_backend('threading',n_jobs=4):
            warnings.filterwarnings('ignore',message=r'Label .* present in all training examples\.',category=UserWarning)
            model=estimator('lr',c=1,jobs=4);model.fit(x,data['y'][train]);model.set_params(n_jobs=1)
        path=folder/'teacher_model.joblib';joblib.dump({'model':model,'train':train},path);reference={'reused':False,'path':str(path.relative_to(ROOT)),'sha256':digest(path)}
    probabilities=continuous_scores(model,x,data['y'][train],'lr');prior=priors(data['y'][train],data['repos'][train],data['repos'][train],strength=20)
    weight=cfg['teacher']['text_weight'];scores=weight*probabilities+(1-weight)*prior
    assert scores.shape==(len(train),data['y'].shape[1]) and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    np.savez_compressed(folder/'teacher_targets.npz',positions=train,scores=scores.astype(np.float32))
    record={'context':context,'fit_rows':len(train),'teacher':reference,'teacher_targets_sha256':digest(folder/'teacher_targets.npz'),
            'only_own_fit_rows_used':True,'teacher_scores_are_in_sample_on_teacher_fit_rows':True,'student_held_validation_test_rows_excluded':True}
    json_write(folder/'teacher_source.json',record);return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();require_server();out=ROOT/args.out;cfg=json.loads((out/'config.json').read_text())
    jobs=[{'out':str(out),'config':cfg,'context':folder.name} for folder in sorted((out/'contexts').iterdir()) if folder.is_dir()]
    records=[]
    with ProcessPoolExecutor(max_workers=12) as executor:
        for future in as_completed([executor.submit(prepare,job) for job in jobs]):records.append(future.result());json_write(out/'teacher_progress.json',records)
    json_write(out/'teacher_complete.json',{'state':'complete','records':records,'contexts':len(records),'no_held_or_outer_targets_used':True})
