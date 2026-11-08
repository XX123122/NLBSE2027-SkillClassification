"""Small issue-dependent logit offsets learned from observable document structure."""
import re
import numpy as np
from scipy.special import expit,logit

NAMES=['log_title_characters','log_body_characters','log_body_words','log_qualified_identifiers','code_fence_fraction']
QUALIFIED=re.compile(r'\b[A-Za-z_]\w*(?:[.$][A-Za-z_]\w*)+\b')
FENCE=re.compile(r'```.*?```',re.S)


def structural_features(titles,bodies):
    if len(titles)!=len(bodies):raise ValueError('Paired title/body arrays required')
    rows=[]
    for title,body in zip(titles,bodies):
        if not isinstance(title,str) or not isinstance(body,str):raise TypeError('Text inputs required')
        title=' '.join(title.split());body=' '.join(body.split())
        rows.append([np.log1p(len(title)),np.log1p(len(body)),np.log1p(len(body.split())),np.log1p(len(QUALIFIED.findall(body))),sum(len(m.group()) for m in FENCE.finditer(body))/max(len(body),1)])
    return np.asarray(rows,dtype=float).reshape(-1,len(NAMES))


def fit_offset(base,features,truth,strength,columns):
    base=np.asarray(base,dtype=float);truth=np.asarray(truth);features=np.asarray(features,dtype=float)[:,columns]
    if base.shape!=truth.shape or base.ndim!=2 or not np.isin(truth,[0,1]).all() or len(features)!=len(base):raise ValueError('Aligned scores, features and targets required')
    if not np.isfinite(base).all() or ((base<0)|(base>1)).any() or not np.isfinite(features).all():raise ValueError('Finite probabilities/features required')
    mean=features.mean(0);scale=features.std(0);scale[scale<1e-10]=1
    x=np.column_stack([np.ones(len(base)),np.clip((features-mean)/scale,-5,5)])
    count=truth.sum(0);eligible=(count>=20)&(len(truth)-count>=20);p=base[:,eligible];y=truth[:,eligible];mask=(p>0)&(p<1)
    offset=logit(np.clip(p,1e-8,1-1e-8));theta=np.zeros(x.shape[1]);n=len(base)
    def evaluate(value):
        z=offset+(x@value)[:,None];prob=expit(z);residual=(prob-y)*mask;weight=prob*expit(-z)*mask
        loss=(np.logaddexp(0,np.where(y.astype(bool),-z,z))*mask).sum()/n+.5*strength*np.square(value).sum()
        gradient=x.T@residual.sum(1)/n+strength*value;hessian=x.T@(weight.sum(1)[:,None]*x)/n+strength*np.eye(x.shape[1])
        return float(loss),gradient,hessian
    for iteration in range(200):
        loss,gradient,hessian=evaluate(theta)
        if np.max(np.abs(gradient))<=1e-7:break
        direction=-np.linalg.solve(hessian,gradient);step=1.;accepted=False
        for _ in range(60):
            proposed=theta+step*direction;new_loss,_,_=evaluate(proposed)
            if new_loss<=loss+1e-4*step*np.dot(gradient,direction)+64*np.finfo(float).eps*max(abs(loss),1):theta=proposed;accepted=True;break
            step*=.5
        if not accepted:raise RuntimeError('Structural offset line search failed')
    else:raise RuntimeError('Structural offset did not meet gradient tolerance')
    return {'theta':theta,'mean':mean,'scale':scale,'columns':columns,'eligible':eligible,'strength':strength,'fit_rows':n,
            'gradient_infinity_norm':float(np.max(np.abs(evaluate(theta)[1]))),'iterations':iteration,'feature_names':[NAMES[i] for i in columns]}


def apply_offset(bundle,base,features):
    base=np.asarray(base,dtype=float);features=np.asarray(features,dtype=float)[:,bundle['columns']]
    x=np.column_stack([np.ones(len(base)),np.clip((features-bundle['mean'])/bundle['scale'],-5,5)]);shift=x@bundle['theta'];score=base.copy();eligible=bundle['eligible']
    p=base[:,eligible];updated=expit(logit(np.clip(p,1e-8,1-1e-8))+shift[:,None]);updated[p==0]=0;updated[p==1]=1;score[:,eligible]=updated
    assert np.isfinite(score).all();return score
