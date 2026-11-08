"""Leakage-aware benchmark split construction and cheap multilabel calibration."""

import hashlib

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, KFold


def text_groups(texts):
    return np.array([hashlib.sha256(t.encode('utf-8')).hexdigest() for t in texts])


def make_folds(texts, repos, protocol, seed=2026, pr_numbers=None):
    groups=text_groups(texts);repos=np.asarray(repos);indices=np.arange(len(texts));folds=[]
    if protocol=='grouped5_pr_components':
        if pr_numbers is None:raise ValueError('PR numbers required for collision-component grouping')
        parent=np.arange(len(texts))
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        for values in (groups,np.asarray(pr_numbers)):
            seen={}
            for i,value in enumerate(values):
                if value in seen:parent[find(i)]=find(seen[value])
                else:seen[value]=i
        groups=np.array([str(find(i)) for i in indices])
    if protocol in ['grouped5_seed2026','grouped5_pr_components']:
        unique=np.unique(groups)
        for k,(_,held) in enumerate(KFold(5,shuffle=True,random_state=seed).split(unique)):
            test=indices[np.isin(groups,unique[held])]
            outer=indices[~np.isin(groups,unique[held])]
            a,b=next(GroupShuffleSplit(1,test_size=.2,random_state=seed+k+1).split(outer,groups=groups[outer]))
            folds.append({'name':f'fold{k}','train':outer[a],'validation':outer[b],'test':test,'purged':0})
    elif protocol=='leave_one_repository_out':
        names=sorted(set(repos))
        for k,name in enumerate(names):
            val_repo=names[(k+1)%len(names)]
            test=indices[repos==name];validation=indices[repos==val_repo]
            train=indices[(repos!=name)&(repos!=val_repo)]
            original=len(train)+len(validation)
            validation=validation[~np.isin(groups[validation],groups[test])]
            train=train[~np.isin(groups[train],np.concatenate([groups[test],groups[validation]]))]
            folds.append({'name':name,'train':train,'validation':validation,'test':test,'purged':original-len(train)-len(validation),'validation_repo':val_repo})
    else:raise ValueError(protocol)
    for f in folds:
        assert all(len(f[k]) for k in ['train','validation','test'])
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            assert not set(f[a])&set(f[b])
            assert not set(groups[f[a]])&set(groups[f[b]])
            if protocol=='grouped5_pr_components':assert not set(np.asarray(pr_numbers)[f[a]])&set(np.asarray(pr_numbers)[f[b]])
    assert sorted(np.concatenate([f['test'] for f in folds]).tolist())==list(indices)
    return folds,groups


def fast_micro(y,pred):
    truth=np.asarray(y,dtype=bool);pred=np.asarray(pred,dtype=bool)
    tp=int((truth&pred).sum());fp=int((~truth&pred).sum());fn=int((truth&~pred).sum())
    return {'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'micro_f1':2*tp/max(2*tp+fp+fn,1)}


def select_threshold(y,scores,grid,baseline):
    records=[]
    for threshold in grid:
        m=fast_micro(y,scores>=threshold)
        feasible=all(m[k]+1e-12>=baseline[k] for k in ['precision','recall','micro_f1'])
        records.append(m|{'threshold':float(threshold),'validation_feasible':feasible})
    feasible=[r for r in records if r['validation_feasible']]
    best=max(feasible or records,key=lambda r:(r['micro_f1'],r['precision'],r['threshold']))
    return best,records


def full_metrics(y,pred):
    from sklearn.metrics import f1_score,hamming_loss,accuracy_score
    result=fast_micro(y,pred)
    supported=y.sum(axis=0)>0
    return result|{'macro_f1_217':float(f1_score(y,pred,average='macro',zero_division=0)),
                   'macro_f1_test_supported':float(f1_score(y[:,supported],pred[:,supported],average='macro',zero_division=0)) if supported.any() else 0.,
                   'samples_f1':float(f1_score(y,pred,average='samples',zero_division=0)),
                   'hamming_loss':float(hamming_loss(y,pred)),'subset_accuracy':float(accuracy_score(y,pred)),
                   'test_supported_labels':int(supported.sum())}


def priors(y_train,repo_train,repo_query,strength=20.):
    mean=y_train.mean(axis=0)
    output=np.tile(mean,(len(repo_query),1))
    for repo in set(repo_train):
        rows=np.asarray(repo_train)==repo
        output[np.asarray(repo_query)==repo]=(y_train[rows].sum(axis=0)+strength*mean)/(rows.sum()+strength)
    return output
