"""Published-label density is fit supervision; query density is model predicted."""
import numpy as np


def density_strata(published):
    y=np.asarray(published)
    if y.ndim!=2 or not len(y) or not np.isin(y,[0,1]).all():raise ValueError('Binary fit labels required')
    counts=y.sum(1);bounds=np.unique(np.quantile(counts,[1/3,2/3],method='nearest'))
    return np.searchsorted(bounds,counts,side='left'),bounds


def fit_density_channel(published,strata,repositories,classes,strength=20):
    y=np.asarray(published,dtype=float);z=np.asarray(strata);repos=np.asarray(repositories);classes=np.asarray(classes)
    if y.ndim!=2 or z.shape!=(len(y),) or repos.shape!=z.shape or not np.isin(y,[0,1]).all() or not np.isin(z,classes).all():raise ValueError('Aligned fit-only strata, labels and repositories required')
    mean=y.mean(0);sizes=np.array([(z==k).sum() for k in classes]);sums=np.array([y[z==k].sum(0) for k in classes]);rates=(sums+.5)/(sizes[:,None]+1);local={}
    for repo in np.unique(repos):
        mask=repos==repo;n=np.array([(mask&(z==k)).sum() for k in classes]);total=np.array([y[mask&(z==k)].sum(0) for k in classes]);local[str(repo)]=(total+strength*rates)/(n[:,None]+strength)
    return {'classes':classes,'global_rates':rates,'repository_rates':local,'mean':mean,'strength':strength,'fit_rows':len(y),'class_counts':sizes}


def apply_density_channel(model,probability,repositories):
    p=np.asarray(probability,dtype=float);repos=np.asarray(repositories)
    if p.shape!=(len(repos),len(model['classes'])) or not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.allclose(p.sum(1),1):raise ValueError('Normalized model probabilities required')
    scores=p@model['global_rates']
    for name,rates in model['repository_rates'].items():
        mask=repos==name;scores[mask]=p[mask]@rates
    scores[:,model['mean']==0]=0;scores[:,model['mean']==1]=1
    return scores
