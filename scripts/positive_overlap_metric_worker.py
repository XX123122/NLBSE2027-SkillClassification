"""GPU-only tiny adapter training on packed fit prototypes, never query targets."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from skillcomp.runtime import ROOT, require_server


def objective(x, u, v, anchor, high, low, gap, cfg):
    indices = torch.cat([anchor, high, low])
    raw = x[indices]
    adapted = F.normalize(raw + (raw @ v) @ u.T, dim=1)
    a, h, l = adapted.chunk(3)
    loss = F.softplus(((a * l).sum(1) - (a * h).sum(1) + cfg['gap_margin'] * gap) / cfg['temperature']).mean()
    displacement = ((adapted - raw) ** 2).sum(1).mean()
    factor_norm = ((u.T @ u) * (v.T @ v).T).sum().clamp_min(0)
    return loss + cfg['displacement_penalty'] * displacement + cfg['factor_penalty'] * factor_norm, loss, displacement


def self_test():
    def run():
        torch.manual_seed(98)
        x = F.normalize(torch.randn(12, 8), dim=1)
        u = torch.nn.Parameter(torch.zeros(8, 2))
        v = torch.nn.Parameter(torch.linalg.qr(torch.randn(8, 2)).Q)
        opt = torch.optim.AdamW([u, v], lr=.001, weight_decay=0)
        cfg = {'gap_margin': .1, 'temperature': .15, 'displacement_penalty': 1., 'factor_penalty': .01}
        a, h, l = torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6, 7, 8])
        np.testing.assert_allclose(F.normalize(x + (x @ v) @ u.T, dim=1).detach(), x, atol=2e-7)
        losses = []
        for step in range(8):
            opt.zero_grad();loss, _, _ = objective(x, u, v, a, h, l, torch.ones(3) * .2, cfg);loss.backward()
            if step == 0:
                assert u.grad.abs().sum() > 0 and v.grad.abs().sum() == 0
            losses.append(float(loss.detach()));opt.step()
        assert losses[-1] < losses[0]
        return u.detach().numpy(), v.detach().numpy()
    a, b = run();c, d = run();np.testing.assert_array_equal(a, c);np.testing.assert_array_equal(b, d)
    print(json.dumps({'self_test': 'passed', 'identity_initialization': True, 'first_factor_gradient': True, 'seed_repeat': True}))


def main(args):
    require_server();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    if args.self_test:
        self_test();return
    out = Path(args.out);job = json.loads((out / args.job).read_text());cfg = json.loads((out / 'config.json').read_text())
    folder = out / 'contexts' / job['context'];folder.mkdir(exist_ok=True)
    torch.manual_seed(job['seed']);np.random.seed(job['seed']);rng = np.random.default_rng(job['seed'])
    packed = np.load(job['training_inputs']);pools = np.load(job['pools'])
    x = torch.as_tensor(packed['vectors'], dtype=torch.float32, device='cuda')
    u = torch.nn.Parameter(torch.zeros(x.shape[1], cfg['rank'], device='cuda'))
    v = torch.nn.Parameter(torch.linalg.qr(torch.randn(x.shape[1], cfg['rank'], device='cuda')).Q)
    opt = torch.optim.AdamW([u, v], lr=cfg['learning_rate'], weight_decay=0)
    anchors = pools['anchors'];jaccard = pools['jaccard'];history = [];started = time.monotonic()
    for epoch in range(cfg['epochs']):
        losses = [];displacements = []
        for step in range(cfg['steps_per_epoch']):
            slots = rng.integers(0, len(anchors), cfg['batch_triplets']);a = anchors[slots]
            low_offsets, high_offsets = pools['low_offsets'], pools['high_offsets']
            lo = low_offsets[slots] + (rng.random(len(slots)) * (low_offsets[slots + 1] - low_offsets[slots])).astype(int)
            hi = high_offsets[slots] + (rng.random(len(slots)) * (high_offsets[slots + 1] - high_offsets[slots])).astype(int)
            l, h = pools['low_indices'][lo], pools['high_indices'][hi]
            gap = jaccard[a, h] - jaccard[a, l]
            assert (gap >= cfg['minimum_jaccard_quartile_gap'] - 1e-6).all()
            tensors = [torch.as_tensor(value, device='cuda') for value in [a, h, l, gap]]
            opt.zero_grad(set_to_none=True)
            loss, ranking, displacement = objective(x, u, v, *tensors, cfg)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite ordinal loss')
            loss.backward();torch.nn.utils.clip_grad_norm_([u, v], 1.);opt.step()
            losses.append(float(ranking.detach()));displacements.append(float(displacement.detach()))
        history.append({'epoch': epoch + 1, 'ranking_loss': float(np.mean(losses)), 'displacement': float(np.mean(displacements))})
        if epoch % 5 == 0:print(json.dumps(history[-1]), flush=True)
        if time.monotonic() - started > cfg['GPU_fit_timeout_seconds']:raise TimeoutError('Fixed per-fit time budget exceeded')
        if torch.cuda.max_memory_allocated() / 1024**3 > cfg['hard_GPU_allocated_GiB']:raise MemoryError('Fixed adapter GPU allocation budget exceeded')
    left, right = u.detach().cpu().numpy(), v.detach().cpu().numpy()
    assert np.isfinite(left).all() and np.isfinite(right).all()
    np.savez_compressed(folder / 'adapter.npz', left=left, right=right)
    result = {'state': 'complete', 'seed': job['seed'], 'context': job['source_context'], 'known_proxy_fit_rows': len(x),
              'trainable_parameters': int(u.numel() + v.numel()), 'epochs': cfg['epochs'], 'history': history,
              'runtime_seconds': time.monotonic() - started, 'peak_GPU_GiB': torch.cuda.max_memory_allocated() / 1024**3,
              'query_rows_or_targets_loaded': False, 'encoder_fits': 0}
    (folder / 'fit.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--job');p.add_argument('--self-test', action='store_true');main(p.parse_args())
