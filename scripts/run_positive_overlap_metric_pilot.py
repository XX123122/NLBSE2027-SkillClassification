"""Three-seed bounded ordinal metric screen with shared GPU reservations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from run_channel_residual_capability import base_oof
from run_retrieval_plateau_diagnostics import context_inputs
from run_qwen06_split_lora import reservation_entries
from skillcomp.annotation_channel import apply_channel
from skillcomp.benchmark import fast_micro
from skillcomp.capability_search import threshold_choice
from skillcomp.channel_residual import fit_expectation, apply_residual
from skillcomp.evidence_coverage import fit_coverage, coverage_weights
from skillcomp.gpu_policy import snapshot, permitted, reserved_free
from skillcomp.positive_overlap_metric import PositiveOverlapMetricNeighbors
from skillcomp.retrieval_evidence import ExcludedChannelStatistics
from skillcomp.runtime import ROOT, require_server, provenance, json_write, status, encoder_python


def gpu_queue(out, jobs, cfg):
    pending = list(jobs);running = {};records = [];started = time.monotonic();last = 0
    registry = ROOT / 'cache/gpu_dispatch_reservations.json'
    while pending or running:
        for pid, (process, job, stream, gpu) in list(running.items()):
            code = process.poll()
            if code is not None:
                stream.close();records.append({'context': job['context'], 'GPU': gpu, 'PID': pid, 'exit_code': code});del running[pid]
                json_write(out / 'GPU_queue_records.json', records)
        failed = any(r['exit_code'] for r in records)
        if failed and not running:raise RuntimeError('Ordinal worker failure; logs preserved')
        while pending and not failed and len(running) < cfg['maximum_parallel_GPU_workers']:
            with (ROOT / 'cache/gpu_dispatch.lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX);gpus, processes, policy = snapshot();claims = reservation_entries(registry);options = []
                for gpu, row in permitted(gpus).items():
                    local = [c for c in claims if c['UUID'] == row['UUID']]
                    if row['UUID'] in cfg['avoid_GPU_UUIDs'] or len(local) >= policy.get('max_workers_per_GPU_per_queue', 3):continue
                    free = reserved_free(row, processes, local)
                    if free >= cfg['minimum_free_GPU_MiB']:options.append((-len(local), free, gpu))
                if not options:json_write(registry, claims);break
                _, _, gpu = max(options);job = pending.pop(0)
                env = os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT / 'src'), CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false')
                stream = (out / 'contexts' / job['context'] / 'fit.log').open('x')
                child = subprocess.Popen([encoder_python(), '-u', str(ROOT / cfg.get('worker_script', 'scripts/positive_overlap_metric_worker.py')), '--out', str(out), '--job', job['context'] + '_job.json'], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                running[child.pid] = (child, job, stream, gpu)
                claims.append({'PID': child.pid, 'UUID': gpus[gpu]['UUID'], 'reservation_MiB': cfg['minimum_free_GPU_MiB'], 'run_id': out.name, 'mode': 'positive_overlap_metric'})
                json_write(registry, claims)
        if pending and not running and time.monotonic() - started > cfg['GPU_queue_wait_seconds']:raise TimeoutError('No allowed GPU within queue wait budget')
        if time.monotonic() - last > 10:
            status(out, 'running', stage=cfg.get('queue_stage','ordinal_metric_GPU_queue'), completed=len(records), total=len(jobs), pending=len(pending),
                   active=[{'PID': pid, 'GPU': gpu, 'context': job['context']} for pid, (_, job, _, gpu) in running.items()]);last = time.monotonic()
        if pending or running:time.sleep(2)
    with (ROOT / 'cache/gpu_dispatch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX);json_write(registry, reservation_entries(registry))
    return records


def neighbor_quality(model, x, query_targets, repositories, fit_targets, weights):
    normalized = model._transform(x) if hasattr(model, '_transform') else x
    normalized = normalized / np.linalg.norm(normalized, axis=1, keepdims=True)
    similarity = normalized @ model.vectors_.T
    qualities = []
    for i, repo in enumerate(repositories):
        same = model.repositories_ == repo
        eligible = same if same.any() else np.ones(len(fit_targets), bool)
        s = similarity[i].copy();s[~eligible] = -np.inf
        neighbors = np.argsort(-s, kind='stable')[:min(20, int(eligible.sum()))]
        intersection = (fit_targets[neighbors] * query_targets[i] * weights).sum(1)
        union = ((fit_targets[neighbors] > 0) | (query_targets[i] > 0)) @ weights
        qualities.append(float(np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0).mean()))
    return np.asarray(qualities)


def evaluate(job):
    threadpool_limits(1);cfg, out, context = job['config'], Path(job['out']), job['source_context'];folder = out / 'contexts' / job['context']
    fit, query, known, data, aux, groups, x, old, _, pub_weight, mapping, published = context_inputs(cfg, context)
    adapter = np.load(folder / 'adapter.npz')
    model = PositiveOverlapMetricNeighbors(adapter['left'], adapter['right'], blend_original=job['blend'], **cfg['neighbors']).fit(x[known], aux['targets'][known], data['repos'][known], groups[known])
    stats = ExcludedChannelStatistics(data['y'][fit], aux['targets'][fit], aux['matched'][fit], data['repos'][fit], groups[fit])
    centre = fit_expectation(stats.map_fit_centre(model.predict_fit_centre(x[fit], data['repos'][fit], groups[fit]), data['repos'][fit], groups[fit]), data['repos'][fit])
    mapped = apply_channel(old['channel'], model.predict_proba(x[query], data['repos'][query]), data['repos'][query])
    original, main = base_oof(cfg, 'fold0');lookup = {int(v): i for i, v in enumerate(original['positions'])};local = np.array([lookup[int(p)] for p in query])
    coverage_pick = json.loads((ROOT / 'outputs/experiments' / cfg['coverage_run'] / 'all_training_selections_locked.json').read_text())['selections']['fold0']['best_policy']
    rows = coverage_weights(fit_coverage(aux['matched'][fit], data['repos'][fit], 20), data['repos'][query], coverage_pick['mix'])
    scores = apply_residual(main[local], mapped, data['repos'][query], centre, cfg['residual_weight'], row_weights=rows)
    scores = apply_residual(scores, published, data['repos'][query], mapping['expectation'], pub_weight)
    observed = query[aux['matched'][query].astype(bool)]
    diagnostic = ROOT / 'outputs/experiments' / cfg['diagnostics_run'] / 'contexts' / context
    baseline = joblib.load(diagnostic / 'model.joblib')['neighbors']
    feasibility = ROOT / 'outputs/experiments/20261024_positive_overlap_feasibility_v1' / 'contexts' / context
    pools = joblib.load(feasibility / 'pools.joblib');weights = pools['label_weights']
    quality = neighbor_quality(model, x[observed], aux['targets'][observed], data['repos'][observed], aux['targets'][known], weights)
    control_quality = neighbor_quality(baseline, x[observed], aux['targets'][observed], data['repos'][observed], aux['targets'][known], weights)
    tag = 'blend' if job['blend'] else 'primary'
    np.savez_compressed(folder / (tag + '_scores.npz'), positions=query, scores=scores, observed_positions=observed, neighbor_quality=quality, control_quality=control_quality)
    joblib.dump({'neighbors': model, 'expectation': centre, 'channel': old['channel']}, folder / (tag + '_model.joblib'))
    return {'context': job['context'], 'blend': job['blend'], 'seed': job['seed'], 'query_rows': len(query),
            'quality_delta': float((quality - control_quality).mean()), 'observed_query_rows': len(observed)}


def summarize(out, cfg, tag):
    original, _ = base_oof(cfg, 'fold0');positions = original['positions'];lookup = {int(p): i for i, p in enumerate(positions)}
    a = np.load(ROOT / 'outputs/experiments' / cfg['diagnostics_run'] / 'training_scores.npz');np.testing.assert_array_equal(positions, a['positions'])
    control = threshold_choice(original['y_true'], a['retrieval_1_prior_1'], original['rf_prediction']);rf = fast_micro(original['y_true'], original['rf_prediction'])
    cases = []
    for seed in cfg['seeds']:
        scores = np.zeros(original['y_true'].shape);seen = np.zeros(len(scores), int);quality = [];old_quality = []
        for i in range(3):
            a = np.load(out / 'contexts' / f'fold0_inner{i}_s{seed}' / (tag + '_scores.npz'));local = np.array([lookup[int(p)] for p in a['positions']]);scores[local] = a['scores'];seen[local] += 1
            quality.append(a['neighbor_quality']);old_quality.append(a['control_quality'])
        assert (seen == 1).all()
        choice = threshold_choice(original['y_true'], scores, original['rf_prediction'])
        cases.append(choice | {'seed': seed, 'training_F1_delta': choice['micro_f1'] - control['micro_f1'],
                                'proxy_neighbor_Jaccard_delta': float((np.concatenate(quality) - np.concatenate(old_quality)).mean()),
                                'RF_gate': all(choice[k] >= rf[k] for k in rf) and not choice['fallback_rf']})
        np.savez_compressed(out / (tag + '_seed' + str(seed) + '_training.npz'), positions=positions, scores=scores)
    mean_gain = float(np.mean([c['training_F1_delta'] for c in cases]))
    admitted = mean_gain >= cfg['minimum_mean_training_F1_gain'] and all(c['training_F1_delta'] > 0 and c['RF_gate'] for c in cases)
    return {'control': control, 'seeds': cases, 'mean_training_F1_delta': mean_gain, 'mean_proxy_neighbor_Jaccard_delta': float(np.mean([c['proxy_neighbor_Jaccard_delta'] for c in cases])), 'admitted': bool(admitted)}


def run(args, out):
    started = time.monotonic();cfg = json.loads((ROOT / args.config).read_text());provenance(out, cfg)
    feasibility = ROOT / 'outputs/experiments/20261024_positive_overlap_feasibility_v1';receipt = json.loads((feasibility / 'feasibility.json').read_text())
    assert receipt['metric_pilot_admitted'] and receipt['config_sha256'] == digest(ROOT / args.config)
    jobs = []
    for record in receipt['contexts']:
        context = record['context'];source = feasibility / 'contexts' / context
        assert digest(source / 'training_inputs.npz') == record['training_inputs_sha256'] and digest(source / 'pools.joblib') == record['pools_sha256']
        pools = joblib.load(source / 'pools.joblib');folder = out / 'contexts' / context;folder.mkdir(parents=True)
        low_offsets = np.r_[0, np.cumsum([len(v) for v in pools['low']])];high_offsets = np.r_[0, np.cumsum([len(v) for v in pools['high']])]
        pool_path = folder / 'ordinal_pools.npz'
        np.savez_compressed(pool_path, anchors=pools['anchors'], low_offsets=low_offsets, high_offsets=high_offsets,
                            low_indices=np.concatenate(pools['low']), high_indices=np.concatenate(pools['high']), jaccard=pools['weighted_jaccard'].astype(np.float32))
        for seed in cfg['seeds']:
            name = context + '_s' + str(seed);(out / 'contexts' / name).mkdir()
            job = {'context': name, 'source_context': context, 'seed': seed, 'training_inputs': str(source / 'training_inputs.npz'), 'pools': str(pool_path)}
            json_write(out / (name + '_job.json'), job);jobs.append(job)
    json_write(out / 'selection_policy_locked.json', {'primary': 'all3 seed gains positive and mean>=.00025, RF P/R/F1 gates',
        'secondary': 'one half-original/half-adapted cosine only if primary rejected and mean held observed-proxy neighbor Jaccard gain>0; same three-seed gate',
        'test_or_outer_validation_used': False, 'feasibility_config_sha256': receipt['config_sha256']})
    gpu_records = gpu_queue(out, jobs, cfg)
    status(out, 'running', stage='evaluate_frozen_ordinal_metric_on_training_only_queries')
    common = [{'out': str(out), 'config': cfg, **job, 'blend': False} for job in jobs]
    with ProcessPoolExecutor(max_workers=cfg['CPU_workers']) as pool:records = list(pool.map(evaluate, common))
    primary = summarize(out, cfg, 'primary');secondary = None
    if not primary['admitted'] and primary['mean_proxy_neighbor_Jaccard_delta'] > 0:
        status(out, 'running', stage='registered_single_half_cosine_secondary')
        with ProcessPoolExecutor(max_workers=cfg['CPU_workers']) as pool:records += list(pool.map(evaluate, [dict(job, blend=True) for job in common]))
        secondary = summarize(out, cfg, 'blend')
    result = {'state': 'complete', 'training_only': True, 'primary': primary, 'secondary': secondary,
              'full_five_fold_admitted': primary['admitted'] or bool(secondary and secondary['admitted']), 'GPU_records': gpu_records, 'evaluation_records': records,
              'new_encoder_fits': 0, 'new_adapter_fits': len(jobs), 'outer_validation_or_test_predictions_created': False, 'benchmark_score_claimed': False,
              'runtime_seconds': time.monotonic() - started, 'limits': cfg['hypothesis']}
    json_write(out / 'training_only_pilot.json', result);report = ROOT / cfg['report_directory'];report.mkdir(exist_ok=True);json_write(report / 'training_only_pilot.json', result)
    status(out, 'complete', mean_training_F1_delta=primary['mean_training_F1_delta'], secondary_mean_training_F1_delta=secondary['mean_training_F1_delta'] if secondary else None,
           full_five_fold_admitted=result['full_five_fold_admitted'], benchmark_score_claimed=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--run-id', required=True);p.add_argument('--config', default='configs/positive_overlap_metric_pilot_v1.json');p.add_argument('--launch', action='store_true');args = p.parse_args();require_server()
    if args.launch:
        assert not (ROOT / 'outputs/experiments' / args.run_id).exists()
        env = os.environ.copy();env['PYTHONPATH'] = str(ROOT / 'src') + ':' + str(ROOT / 'scripts')
        for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:env[key] = '1'
        with (ROOT / 'logs' / (args.run_id + '.log')).open('x') as stream:
            child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', str(Path(__file__).resolve()), '--run-id', args.run_id, '--config', args.config], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({'state': 'launched', 'PID': child.pid, 'run_id': args.run_id}))
    else:
        out = ROOT / 'outputs/experiments' / args.run_id;out.mkdir()
        try:run(args, out)
        except BaseException:(out / 'error.txt').write_text(traceback.format_exc());status(out, 'failed', error=traceback.format_exc());raise
