"""Minimal low-rank residual on frozen Linear layers, with fp32 adapters."""
import math

import torch
from torch import nn


class LowRankLinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=.05):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError('Linear module required')
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        original = self.base(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            residual = nn.functional.linear(nn.functional.linear(self.dropout(x.float()), self.lora_a), self.lora_b)
        return original + (residual * self.scale).to(original.dtype)


def inject_last_layers(backbone, config):
    first = len(backbone.layers) - config['last_decoder_layers']
    targets = set(config['targets']); replaced = []
    for index, layer in enumerate(backbone.layers):
        if index < first:
            continue
        for name, module in list(layer.named_modules()):
            if isinstance(module, nn.Linear) and name.split('.')[-1] in targets:
                parent_name, _, leaf = name.rpartition('.')
                parent = layer.get_submodule(parent_name) if parent_name else layer
                setattr(parent, leaf, LowRankLinear(module, config['rank'], config['alpha'], config['dropout']))
                replaced.append(f'layers.{index}.{name}')
    if not replaced:
        raise ValueError('No adapter targets matched')
    return replaced
