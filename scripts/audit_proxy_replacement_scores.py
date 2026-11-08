"""Independently reconstruct replacement logits from saved model probabilities."""
import json
import joblib
import numpy as np
from scipy.special import expit,logit

from run_channel_residual_capability import base_oof
from run_hierarchy_strength_capability import publication_source
from skillcomp.runtime import ROOT,json_write


def residual(base,channel,repos,expectation,weight,row_weights=None):
    if weight==0:return base.copy()
    center=np.array([expectation['repositories'].get(str(repo),expectation['mean']) for repo in repos]);shift=logit(np.clip(channel,1e-5,1-1e-5))-logit(np.clip(center,1e-5,1-1e-5))
    if row_weights is not None:shift*=row_weights[:,None]
    scores=expit(logit(np.clip(base,1e-5,1-1e-5))+weight*shift);scores[base==0]=0;scores[base==1]=1
    if row_weights is not None:scores[row_weights==0]=base[row_weights==0]
    return scores


def reconstruct(out,cfg,name,context,candidate,weight,base,mode):
    source=(ROOT/'outputs/experiments'/cfg['reused_readout_run']) if '_inner' in context else out;folder=source/'contexts'/context/candidate
    if cfg.get('reuse_partial_inner_heads') and not (folder/'model.joblib').exists():folder=out/'contexts'/context/candidate
    role=np.load(out/'contexts'/context/'roles.npz');fit=role['train'];model=joblib.load(folder/'model.joblib');a=np.load(folder/('test_scores.npz' if mode=='test' else 'scores.npz'));query=a['positions'];data=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True);repos=data['repos'];aux=np.load(ROOT/'outputs/experiments'/cfg['channel_run']/'auxiliary_targets.npz');matched=aux['matched'][fit].astype(bool);global_rate=matched.mean();rates={str(repo):float((matched[repos[fit]==repo].sum()+20*global_rate)/((repos[fit]==repo).sum()+20)) for repo in np.unique(repos[fit])};pick=json.loads((ROOT/'outputs/experiments'/cfg['coverage_run']/'all_training_selections_locked.json').read_text())['selections'][name]['best_policy'];coverage=np.clip(np.array([rates.get(str(repo),global_rate)/global_rate for repo in repos[query]]),0,2);row_weights=(1-pick['mix'])+pick['mix']*coverage;score=residual(base,a['scores'],repos[query],model['expectation'],weight,row_weights);publication,pub_weight=publication_source(cfg.get('publication_source_run',cfg['reference_run']),name)
    if publication is not None:
        mapping=joblib.load(publication/'contexts'/context/'mapping.joblib');a=np.load(publication/'contexts'/context/('test_scores.npz' if mode=='test' else 'scores.npz'));np.testing.assert_array_equal(a['positions'],query);score=residual(score,a['scores'],repos[query],mapping['expectation'],pub_weight)
    return query,score


def audit(out,cfg):
    selected=json.loads((out/'all_training_selections_locked.json').read_text())['selections'];records=[]
    for name,picks in selected.items():
        original,base=base_oof(cfg,name);lookup={int(p):i for i,p in enumerate(original['positions'])};saved=np.load(out/(name+'_oof_scores.npz'))
        for family,pick in picks.items():
            if pick['candidate']=='reference':continue
            for index in range(3):
                context=f'{name}_inner{index}';query=np.load(out/'contexts'/context/'roles.npz')['query'];local=np.array([lookup[int(p)] for p in query]);_,score=reconstruct(out,cfg,name,context,pick['candidate'],pick['residual_weight'],base[local],'calibration');np.testing.assert_allclose(score,saved[pick['score_key']][local],rtol=0,atol=1e-13)
            for mode,suffix,base_key,score_key in [('calibration','_calibration.npz','portfolio',family),('test','_assessment.npz','portfolio_scores',family+'_scores')]:
                reference=np.load(ROOT/'outputs/experiments'/cfg['base_run']/(name+suffix));a=np.load(out/(name+suffix));query,score=reconstruct(out,cfg,name,name+'_final',pick['candidate'],pick['residual_weight'],reference[base_key],mode);np.testing.assert_array_equal(query,a['positions']);np.testing.assert_allclose(score,a[score_key],rtol=0,atol=1e-13)
            records.append({'name':name,'family':family,'candidate':pick['candidate'],'residual_weight':pick['residual_weight'],'training_OOF_validation_and_test_logits_independently_reconstructed':True})
    result={'state':'passed','records':records,'coverage_formula_recomputed_from_fit_only_metadata':True,'no_extra_old_auxiliary_residual':True};json_write(out/'replacement_score_reconstruction.json',result);json_write(ROOT/cfg['report_directory']/'replacement_score_reconstruction.json',result);return result
