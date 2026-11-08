"""Learn an observed proxy-to-published annotation channel on known fit pairs."""
import numpy as np


def fit_channel(published,auxiliary,matched,repositories,strength=20):
    published=np.asarray(published);auxiliary=np.asarray(auxiliary);matched=np.asarray(matched,dtype=bool);repositories=np.asarray(repositories)
    if published.shape!=auxiliary.shape or published.ndim!=2 or matched.shape!=(len(published),) or len(repositories)!=len(published):raise ValueError('Aligned label pairs, evidence and repositories required')
    if not np.isin(published,[0,1]).all() or not np.isin(auxiliary[matched],[0,1]).all() or not matched.any():raise ValueError('Binary known pairs required')
    mean=published.mean(0)
    def conditional(rows,anchor=None):
        a=auxiliary[rows].astype(bool);y=published[rows];den=np.stack([(~a).sum(0),a.sum(0)]);num=np.stack([(y*(~a)).sum(0),(y*a).sum(0)])
        if anchor is None:return np.where(den>0,(num+.5)/(den+1),np.tile(mean,(2,1)))
        return (num+strength*anchor)/(den+strength)
    global_rates=conditional(matched);local={}
    for name in np.unique(repositories[matched]):local[str(name)]=conditional(matched&(repositories==name),global_rates)
    return {'global_rates':global_rates,'repository_rates':local,'mean':mean,'strength':strength,'known_rows':int(matched.sum()),'fit_rows':len(published),
            'auxiliary_positive_published_negative_known_cells':int(((auxiliary[matched]>0)&(published[matched]==0)).sum())}


def apply_channel(bundle,auxiliary_scores,repositories):
    p=np.asarray(auxiliary_scores,dtype=float);repositories=np.asarray(repositories);assert p.ndim==2 and len(p)==len(repositories)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Finite own auxiliary-model probabilities required')
    low=np.tile(bundle['global_rates'][0],(len(p),1));high=np.tile(bundle['global_rates'][1],(len(p),1))
    for name,rates in bundle['repository_rates'].items():low[repositories==name]=rates[0];high[repositories==name]=rates[1]
    score=(1-p)*low+p*high;score[:,bundle['mean']==0]=0;score[:,bundle['mean']==1]=1
    assert np.isfinite(score).all() and ((score>=0)&(score<=1)).all();return score
