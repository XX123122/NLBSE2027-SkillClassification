"""Mean-calibrate availability probabilities using only fit metadata."""
import numpy as np
from scipy.special import expit,logit


def fit_availability_intercepts(probability,known,repositories,strength=20):
    p=np.asarray(probability,dtype=float);known=np.asarray(known);repos=np.asarray(repositories)
    if p.shape!=known.shape or p.ndim!=1 or len(p)!=len(repos) or not len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.isin(known,[0,1]).all() or strength<0:raise ValueError('Aligned fit probabilities, binary availability and repository names required')
    mean=float(known.mean());z=logit(np.clip(p,1e-6,1-1e-6))
    def fit(rows,target):
        if target in [0,1]:return {'constant':target,'fit_rows':int(rows.sum())}
        low,high=-50.,50.
        for _ in range(70):
            middle=(low+high)/2
            if float(expit(z[rows]+middle).mean())<target:low=middle
            else:high=middle
        return {'offset':(low+high)/2,'target_rate':target,'fit_rows':int(rows.sum())}
    global_fit=fit(np.ones(len(p),dtype=bool),mean);local={}
    for name in np.unique(repos):
        rows=repos==name;count=int(known[rows].sum());n=int(rows.sum());target=0. if count==0 else 1. if count==n else (count+strength*mean)/(n+strength);local[str(name)]=fit(rows,float(target))
    return {'global':global_fit,'repositories':local,'strength':strength,'fit_rows':len(p),'query_availability_flag_used':False}


def calibrate_availability(bundle,probability,repositories):
    p=np.asarray(probability,dtype=float);repos=np.asarray(repositories)
    if p.shape!=(len(repos),) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Aligned finite query availability estimates required')
    z=logit(np.clip(p,1e-6,1-1e-6))
    def apply(parameters,rows):return np.full(int(np.sum(rows)),parameters['constant']) if 'constant' in parameters else expit(z[rows]+parameters['offset'])
    all_rows=np.ones(len(p),dtype=bool);result=apply(bundle['global'],all_rows)
    for name,parameters in bundle['repositories'].items():
        rows=repos==name;result[rows]=apply(parameters,rows)
    return result
