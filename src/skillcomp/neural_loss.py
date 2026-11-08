"""Asymmetric multilabel loss with focal weight detached, mean reduction.

Formula checked against Alibaba-MIIL/ASL's AsymmetricLoss at commit
8c9e0bd8d5d450cf19093363fc08aa7244ad4408. This implementation uses a scoped
no_grad context, preserving the caller's gradient-enabled state.
"""

import torch


def asymmetric_loss(logits,targets,gamma_neg=4.,gamma_pos=1.,clip=.05,eps=1e-8):
    positive=logits.sigmoid();negative=(1-positive+clip).clamp(max=1.) if clip else 1-positive
    log_likelihood=targets*positive.clamp_min(eps).log()+(1-targets)*negative.clamp_min(eps).log()
    with torch.no_grad():
        probability=targets*positive+(1-targets)*negative
        exponent=targets*gamma_pos+(1-targets)*gamma_neg
        weight=(1-probability).pow(exponent)
    return -(log_likelihood*weight).mean()
