"""Nonparametric proxy prototypes are saved model assets, not query labels."""
import numpy as np

from skillcomp.annotation_channel import apply_channel
from skillcomp.channel_residual import apply_residual
from skillcomp.evidence_coverage import coverage_weights
from skillcomp.residual_inference import auxiliary_features


def refine_neighbor_scores(component,base_scores,embeddings,repositories,records=None):
    probability=component['neighbors'].predict_proba(embeddings,repositories);channel=apply_channel(component['channel'],probability,repositories);fraction=component['selection']['neighbor_fraction'];centre=component['knn_expectation']
    if not 0<=fraction<=1:raise ValueError('Neighbor mixture must be in the unit interval')
    if fraction<1:
        reference=component['reference_proxy_component'];old=np.asarray(reference['head'].predict_proba(auxiliary_features(reference,records,embeddings,repositories)));counts=reference['positive_counts'];old[:,counts==0]=0;old[:,counts==reference['matched_training_rows']]=1;old=apply_channel(component['channel'],old,repositories);channel=(1-fraction)*old+fraction*channel;prior=reference['expectation'];centre={'mean':(1-fraction)*prior['mean']+fraction*centre['mean'],'repositories':{repo:(1-fraction)*value+fraction*centre['repositories'][repo] for repo,value in prior['repositories'].items()}}
    row=coverage_weights(component['coverage'],repositories,component['selection']['mix']);return apply_residual(base_scores,channel,repositories,centre,component['selection']['residual_weight'],row_weights=row)
