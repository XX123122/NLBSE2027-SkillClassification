"""Training-only falsification and component-excluded centre control."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from availability_channel_workers import inputs as availability_inputs
from capability_search_workers import digest
from run_auxiliary_lexical_residual_capability import _features_without_repository, reference_oof
from run_channel_residual_capability import base_oof
from run_hierarchy_strength_capability import publication_source
from skillcomp.annotation_channel import fit_channel, apply_channel
from skillcomp.availability_calibration import calibrate_availability
from skillcomp.benchmark import fast_micro
from skillcomp.capability_search import threshold_choice
from skillcomp.channel_residual import fit_expectation, apply_residual
from skillcomp.evidence_coverage import fit_coverage, coverage_weights
from skillcomp.retrieval_evidence import EvidenceNeighbors, ExcludedChannelStatistics
from skillcomp.runtime import ROOT, require_server, provenance, json_write, status


def context_inputs(cfg, context):
    root = ROOT / 'outputs/experiments'
    original = root / cfg['neighbor_run'] / 'contexts' / context
    roles = np.load(original / 'roles.npz')
    fit, query = roles['train'], roles['query']
    data = np.load(ROOT / 'cache/workshop_v1/dataset.npz', allow_pickle=True)
    aux = np.load(root / cfg['channel_run'] / 'auxiliary_targets.npz')
    groups = np.load(ROOT / 'cache/workshop_v1/key_component_groups.npy')
    assert not np.intersect1d(groups[fit], groups[query]).size
    known = fit[aux['matched'][fit].astype(bool)]
    np.testing.assert_array_equal(roles['known'], known)
    source_cfg = json.loads((root / cfg['proxy_source_run'] / 'config.json').read_text())
    x = _features_without_repository(source_cfg, context, 'codebert_dense', fit)
    x = x.toarray() if hasattr(x, 'toarray') else np.asarray(x)
    old = joblib.load(original / 'model.joblib')
    publication, weight = publication_source(cfg['publication_source_run'], 'fold0')
    mapping = joblib.load(publication / 'contexts' / context / 'mapping.joblib')
    np.testing.assert_array_equal(mapping['train'], fit)
    ps = np.load(publication / 'contexts' / context / 'scores.npz')
    np.testing.assert_array_equal(ps['positions'], query)
    return fit, query, known, data, aux, groups, x, old, publication, weight, mapping, ps['scores']


def prepare(job):
    threadpool_limits(1)
    started = time.monotonic()
    cfg, context, out = job['config'], job['context'], Path(job['out'])
    fit, query, known, data, aux, groups, x, old, pub, pub_weight, mapping, published = context_inputs(cfg, context)
    y, repos = data['y'], data['repos']
    original, main = base_oof(cfg, 'fold0')
    lookup = {int(pos): i for i, pos in enumerate(original['positions'])}
    local = np.array([lookup[int(pos)] for pos in query])
    coverage_pick = json.loads((ROOT / 'outputs/experiments' / cfg['coverage_run'] / 'all_training_selections_locked.json').read_text())['selections']['fold0']['best_policy']
    row_weights = coverage_weights(fit_coverage(aux['matched'][fit], repos[fit], 20), repos[query], coverage_pick['mix'])
    stats = ExcludedChannelStatistics(y[fit], aux['targets'][fit], aux['matched'][fit], repos[fit], groups[fit])
    folder = out / 'contexts' / context
    folder.mkdir(parents=True)
    cases = [(f'retrieval_{int(same)}_prior_{int(prior)}', same, prior, None) for same in [True, False] for prior in [True, False]]
    cases += [('shuffle_' + str(seed), True, True, seed) for seed in cfg['shuffle_seeds']]
    target = aux['targets'][known]
    results = {}; audits = []
    for case, same, prior, seed in cases:
        targets = target.copy()
        if seed is not None:
            rng = np.random.default_rng(seed)
            cardinality = targets.sum(1)
            for repo in np.unique(repos[known]):
                repo_rows = np.flatnonzero(repos[known] == repo)
                bins = np.searchsorted(np.quantile(cardinality[repo_rows], [.25, .5, .75]), cardinality[repo_rows], side='right')
                for b in np.unique(bins):
                    indices = repo_rows[bins == b]
                    targets[indices] = target[rng.permutation(indices)]
                np.testing.assert_array_equal(targets[repo_rows].sum(0), target[repo_rows].sum(0))
        model = EvidenceNeighbors(same_repository=same, repository_prior=prior, **cfg['neighbors']).fit(x[known], targets, repos[known], groups[known])
        fit_evidence = model.fit_centre_evidence(x[fit], repos[fit], groups[fit])
        mapped_fit = stats.map_fit_centre(fit_evidence['probability'], repos[fit], groups[fit])
        centre = fit_expectation(mapped_fit, repos[fit])
        evidence = model.predict_evidence(x[query], repos[query])
        mapped = apply_channel(old['channel'], evidence['probability'], repos[query])
        corrected = apply_residual(main[local], mapped, repos[query], centre, cfg['residual_weight'], row_weights=row_weights)
        results[case] = apply_residual(corrected, published, repos[query], mapping['expectation'], pub_weight)
        if case == 'retrieval_1_prior_1':
            # Real data checks compare analytic removal with literal deletion.
            for group in np.unique(groups[fit])[:5]:
                q = fit[groups[fit] == group][:2]
                kept = known[groups[known] != group]
                brute = EvidenceNeighbors(**cfg['neighbors']).fit(x[kept], aux['targets'][kept], repos[kept], groups[kept])
                np.testing.assert_allclose(model.predict_fit_centre(x[q], repos[q], groups[q]), brute.predict_proba(x[q], repos[q]), atol=2e-6)
                keep = groups[fit] != group
                brute_channel = fit_channel(y[fit][keep], aux['targets'][fit][keep], aux['matched'][fit][keep], repos[fit][keep])
                analytic = stats.excluding(group)
                np.testing.assert_allclose(analytic['global_rates'], brute_channel['global_rates'], atol=1e-12)
                for repo, value in brute_channel['repository_rates'].items():
                    np.testing.assert_allclose(analytic['repository_rates'][repo], value, atol=1e-12)
                audits.append({'group': int(group), 'literal_delete_neighbor_and_mapping_match': True})
            np.savez_compressed(folder / 'query_evidence.npz', positions=query, **evidence)
            np.savez_compressed(folder / 'centre_sufficient_statistics.npz', fit=fit, known=known, query=query,
                                mapped_component_excluded_fit=mapped_fit, global_mean=centre['mean'])
            joblib.dump({'neighbors': model, 'expectation': centre, 'channel': old['channel']}, folder / 'model.joblib')
        del model
    pub_cfg = json.loads((pub / 'config.json').read_text())
    gate = joblib.load(ROOT / mapping['gate_source_path'])
    assert digest(ROOT / mapping['gate_source_path']) == mapping['gate_sha256']
    np.testing.assert_array_equal(gate['fit'], fit)
    avx, _ = availability_inputs(pub_cfg, context, fit)
    av = gate['model'].predict_proba(avx[query])[:, 1]
    if mapping['probability_calibration'] is not None:
        av = calibrate_availability(mapping['probability_calibration'], av, repos[query])
    # Safe-negative support is a feasibility test, never an optimization target.
    proxy_intersection = target.astype(np.int32) @ target.astype(np.int32).T
    pub_intersection = y[known].astype(np.int32) @ y[known].astype(np.int32).T
    different_group = groups[known, None] != groups[known][None, :]
    positive = (proxy_intersection > 0) & different_group
    safe = (proxy_intersection == 0) & (pub_intersection == 0) & different_group
    feasibility = {'observed_anchors': len(known), 'with_positive_pair': int(positive.any(1).sum()),
                   'with_safe_negative': int(safe.any(1).sum()), 'with_both': int((positive.any(1) & safe.any(1)).sum()),
                   'safe_negative_pairs_directed': int(safe.sum()), 'safe_negative_definition': 'no shared observed proxy positive AND no shared published positive, different components'}
    np.savez_compressed(folder / 'training_diagnostic_scores.npz', positions=query, availability=av, **results)
    result = {'context': context, 'fit_rows': len(fit), 'known_rows': len(known), 'query_rows': len(query),
              'bruteforce_audits': audits, 'metric_pair_feasibility': feasibility,
              'source_model_sha256': digest(ROOT / 'outputs/experiments' / cfg['neighbor_run'] / 'contexts' / context / 'model.joblib'),
              'fit_and_query_groups_disjoint': True, 'runtime_seconds': time.monotonic() - started}
    json_write(folder / 'audit.json', result)
    return result


def run(args, out):
    started = time.monotonic()
    cfg = json.loads((ROOT / args.config).read_text())
    provenance(out, cfg)
    assert digest(ROOT / 'cache/workshop_v1/dataset.npz') == '07cb687e2536248bc8c99de8fd7f0a30847c88ecca467b54529a9cf9d5862c42'
    status(out, 'running', stage='component_excluded_centres_and_fixed_counterfactuals')
    with ProcessPoolExecutor(max_workers=cfg['CPU_workers']) as pool:
        records = list(pool.map(prepare, [{'config': cfg, 'context': f'fold0_inner{i}', 'out': str(out)} for i in range(3)]))
    original, _ = base_oof(cfg, 'fold0')
    positions = original['positions'];lookup = {int(v): i for i, v in enumerate(positions)}
    keys = ['retrieval_1_prior_1', 'retrieval_1_prior_0', 'retrieval_0_prior_1', 'retrieval_0_prior_0'] + ['shuffle_' + str(s) for s in cfg['shuffle_seeds']]
    scores = {key: np.zeros(original['y_true'].shape) for key in keys};seen = np.zeros(len(positions), int);av = np.zeros(len(positions))
    for i in range(3):
        a = np.load(out / 'contexts' / f'fold0_inner{i}' / 'training_diagnostic_scores.npz')
        local = np.array([lookup[int(v)] for v in a['positions']]);seen[local] += 1;av[local] = a['availability']
        for key in keys:scores[key][local] = a[key]
    assert (seen == 1).all()
    truth, rf = original['y_true'], original['rf_prediction']
    choices = {key: threshold_choice(truth, value, rf) for key, value in scores.items()}
    control = choices['retrieval_1_prior_1'];threshold = control['threshold']
    fixed = {key: fast_micro(truth, value >= threshold) for key, value in scores.items()}
    old, _ = reference_oof(cfg['neighbor_run'], 'fold0', positions)
    old_choice = threshold_choice(truth, old, rf)
    data = np.load(ROOT / 'cache/workshop_v1/dataset.npz', allow_pickle=True);repos = data['repos'];g = np.load(ROOT / 'cache/workshop_v1/key_component_groups.npy')
    aux = np.load(ROOT / 'outputs/experiments' / cfg['channel_run'] / 'auxiliary_targets.npz')
    cross = {v: len(set(repos[g == v])) > 1 for v in np.unique(g)}
    component_size = {v: int((g == v).sum()) for v in np.unique(g)}
    _, inverse, counts = np.unique(data['y'], axis=0, return_inverse=True, return_counts=True)
    strata = {'component_cross_repository': np.array([cross[v] for v in g[positions]]),
              'component_size_gt1': np.array([component_size[v] > 1 for v in g[positions]]),
              'published_vector_duplicate_gt1': counts[inverse[positions]] > 1,
              'observed_proxy': aux['matched'][positions].astype(bool)}
    av_bin = np.zeros(len(positions), int)
    for repo in np.unique(repos[positions]):
        rows = repos[positions] == repo
        av_bin[rows] = np.searchsorted(np.quantile(av[rows], [1/3, 2/3]), av[rows], side='right')
    for i in range(3):strata['within_repo_predicted_availability_tertile_' + str(i)] = av_bin == i
    subgroup = {}
    for key, mask in strata.items():
        subgroup[key] = {'rows': int(mask.sum())}
        if mask.any():
            subgroup[key]['real'] = fast_micro(truth[mask], scores['retrieval_1_prior_1'][mask] >= threshold)
            subgroup[key]['mean_shuffle_F1'] = float(np.mean([fast_micro(truth[mask], scores['shuffle_' + str(s)][mask] >= threshold)['micro_f1'] for s in cfg['shuffle_seeds']]))
    shuffle_deltas = [fixed['retrieval_1_prior_1']['micro_f1'] - fixed['shuffle_' + str(s)]['micro_f1'] for s in cfg['shuffle_seeds']]
    reconciliation = json.loads((ROOT / 'reports/followup_v1/population_reconciliation.json').read_text())
    reconciliation.update(reused_existing_audit=True, unchanged_internal_population=True, dataset_sha256=digest(ROOT / 'cache/workshop_v1/dataset.npz'),
                          upstream_revision=json.loads((ROOT / 'reports/source_update_check_20261024.json').read_text())['commit']['sha'])
    json_write(out / 'dataset_reconciliation.json', reconciliation)
    result = {'state': 'complete', 'training_only': True, 'original_control': old_choice, 'component_excluded_centre_control': control,
              'centre_change_training_F1_delta': control['micro_f1'] - old_choice['micro_f1'],
              'own_RF_gated_training_choices': choices, 'fixed_new_control_threshold_metrics': fixed,
              'real_minus_shuffle_fixed_threshold_F1_deltas': shuffle_deltas,
              'semantic_correspondence_positive_all_seeds': all(d > 0 for d in shuffle_deltas),
              'strata_fixed_threshold_descriptive_only': subgroup, 'contexts': records,
              'metric_pair_feasibility_gate': all(r['metric_pair_feasibility']['with_both'] >= .5 * r['known_rows'] for r in records),
              'outer_validation_or_test_predictions_created': False, 'new_encoder_fits': 0, 'benchmark_score_claimed': False,
              'centre_limit': 'Per-centre-row component removed from neighbor priors/constants and mapping counts; average these fit-only predictions. Encoder not refitted per centre row; centre mean is not an independent unbiased estimate.',
              'runtime_seconds': time.monotonic() - started}
    np.savez_compressed(out / 'training_scores.npz', positions=positions, **scores)
    json_write(out / 'plateau_diagnostics.json', result)
    report = ROOT / cfg['report_directory'];report.mkdir(exist_ok=True)
    for name in ['plateau_diagnostics.json', 'dataset_reconciliation.json']:json_write(report / name, json.loads((out / name).read_text()))
    status(out, 'complete', control_training_F1=control['micro_f1'], semantic_correspondence_positive_all_seeds=result['semantic_correspondence_positive_all_seeds'], metric_pair_feasibility_gate=result['metric_pair_feasibility_gate'], benchmark_score_claimed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--run-id', required=True);parser.add_argument('--config', default='configs/retrieval_plateau_diagnostics_v1.json');parser.add_argument('--launch', action='store_true');args = parser.parse_args();require_server()
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
