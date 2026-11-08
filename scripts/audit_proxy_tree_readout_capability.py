"""Independently refit80 forests, reconstruct every policy and recompute metrics."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json

import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from audit_proxy_replacement_scores import residual
from capability_search_workers import digest
from run_channel_residual_capability import base_oof
from run_hierarchy_strength_capability import publication_source
from run_proxy_tree_readout_capability import input_values
from skillcomp.annotation_channel import apply_channel
from skillcomp.capability_search import threshold_choice
from skillcomp.channel_residual import fit_expectation
from skillcomp.proxy_tree_readout import ProxyTreeReadout
from skillcomp.retrieval_evidence import ExcludedChannelStatistics
from skillcomp.runtime import ROOT, require_server, json_write


def compare_forests(a, b):
    assert a.repository_vocabulary_ == b.repository_vocabulary_
    np.testing.assert_array_equal(a.projection_.components_, b.projection_.components_)
    np.testing.assert_array_equal(a.positive_counts_, b.positive_counts_)
    assert a.known_rows_ == b.known_rows_ and len(a.head_.estimators_) == len(b.head_.estimators_) == 256
    for first, second in zip(a.head_.estimators_, b.head_.estimators_):
        for key in ['threshold', 'feature', 'children_left', 'children_right', 'value']:
            np.testing.assert_array_equal(getattr(first.tree_, key), getattr(second.tree_, key))


def audit_context(job):
    threadpool_limits(1);out, cfg, context = ROOT / 'outputs/experiments' / job['run_id'], job['config'], job['context'];folder = out / 'contexts' / context
    fit, query, known, data, aux, groups, x, old = input_values(cfg, context);repos = data['repos']
    saved = joblib.load(folder / 'model.joblib');np.testing.assert_array_equal(saved['projection_fit'], fit)
    model = saved['head'];template = ProxyTreeReadout(**cfg['tree']).fit_projection(x[fit], repos[fit])
    np.testing.assert_array_equal(template.projection_.components_, model.projection_.components_)
    full = copy.deepcopy(template).fit_head(x[known], aux['targets'][known], repos[known]);compare_forests(full, model)
    centre = np.zeros((len(fit), 217));seen = np.zeros(len(fit), int)
    for i, (train, held) in enumerate(GroupKFold(3).split(fit, groups=groups[fit])):
        role = np.load(folder / ('sub' + str(i)) / 'roles.npz');observed = fit[train][aux['matched'][fit[train]].astype(bool)]
        np.testing.assert_array_equal(role['train'], fit[train]);np.testing.assert_array_equal(role['query'], fit[held]);np.testing.assert_array_equal(role['known'], observed)
        assert not np.intersect1d(groups[observed], groups[fit[held]]).size
        sub = joblib.load(folder / ('sub' + str(i)) / 'model.joblib')
        refit = copy.deepcopy(template).fit_head(x[observed], aux['targets'][observed], repos[observed]);compare_forests(refit, sub)
        centre[held] = sub.predict_proba(x[fit[held]], repos[fit[held]]);seen[held] += 1
    assert (seen == 1).all()
    a = np.load(folder / 'raw_tree_scores.npz');np.testing.assert_array_equal(query, a['positions']);np.testing.assert_allclose(centre, a['centre_proxy'], atol=1e-12)
    np.testing.assert_allclose(model.predict_proba(x[query], repos[query]), a['proxy'], atol=1e-12)
    stats = ExcludedChannelStatistics(data['y'][fit], aux['targets'][fit], aux['matched'][fit], repos[fit], groups[fit])
    expectation = fit_expectation(stats.map_fit_centre(centre, repos[fit], groups[fit]), repos[fit])
    for key, value in expectation['repositories'].items():np.testing.assert_allclose(saved['expectation']['repositories'][key], value, atol=1e-12)
    np.testing.assert_allclose(saved['expectation']['mean'], expectation['mean'], atol=1e-12)
    neighbor = joblib.load(folder / 'neighbor_component.joblib');nn = neighbor['neighbors']
    np.testing.assert_array_equal(nn.targets_, aux['targets'][known]);np.testing.assert_array_equal(nn.fit_groups_, groups[known]);np.testing.assert_array_equal(nn.repositories_, repos[known].astype(str))
    np.testing.assert_allclose(nn.vectors_, x[known] / np.linalg.norm(x[known], axis=1, keepdims=True), atol=1e-7, rtol=0)
    assert joblib.hash(neighbor['channel']) == joblib.hash(old['channel'])
    nc = fit_expectation(stats.map_fit_centre(nn.predict_fit_centre(x[fit], repos[fit], groups[fit]), repos[fit], groups[fit]), repos[fit])
    np.testing.assert_allclose(nc['mean'], neighbor['expectation']['mean'], atol=1e-12)
    for key, value in nc['repositories'].items():np.testing.assert_allclose(neighbor['expectation']['repositories'][key], value, atol=1e-12)
    np.testing.assert_allclose(apply_channel(neighbor['channel'], a['proxy'], repos[query]), a['mapped'], atol=1e-12)
    n = np.load(folder / 'raw_neighbor_scores.npz');np.testing.assert_array_equal(n['positions'], query)
    np.testing.assert_allclose(apply_channel(neighbor['channel'], nn.predict_proba(x[query], repos[query]), repos[query]), n['mapped'], atol=1e-12)
    if context.endswith('_final'):
        test = np.load(ROOT / 'cache/workshop_v1/grouped5_pr_components' / (context.split('_')[0] + '.npz'))['test']
        assert not np.intersect1d(groups[fit], groups[test]).size
        t = np.load(folder / 'raw_tree_test_scores.npz');n = np.load(folder / 'raw_neighbor_test_scores.npz')
        np.testing.assert_array_equal(t['positions'], test);np.testing.assert_array_equal(n['positions'], test)
        proxy = model.predict_proba(x[test], repos[test]);np.testing.assert_allclose(proxy, t['proxy'], atol=1e-12)
        np.testing.assert_allclose(apply_channel(neighbor['channel'], proxy, repos[test]), t['mapped'], atol=1e-12)
        np.testing.assert_allclose(apply_channel(neighbor['channel'], nn.predict_proba(x[test], repos[test]), repos[test]), n['mapped'], atol=1e-12)
    return {'context': context, 'projection_and_four_forests_refit_from_original_fit_inputs_targets': True,
            'every_tree_structure_threshold_and_leaf_value_bitwise_equal': True, 'centre_subrole_and_whole_query_isolation': True,
            'all_tree_neighbor_probabilities_and_centres_recomputed': True, 'tree_model_sha256': digest(folder / 'model.joblib')}


def reconstruct(out, cfg, context, base, test=False):
    folder = out / 'contexts' / context;name = context.split('_')[0];fit = np.load(folder / 'roles.npz')['train']
    data = np.load(ROOT / 'cache/workshop_v1/dataset.npz', allow_pickle=True);repos = data['repos'];aux = np.load(ROOT / 'outputs/experiments' / cfg['channel_run'] / 'auxiliary_targets.npz')
    tree = joblib.load(folder / 'model.joblib');nn = joblib.load(folder / 'neighbor_component.joblib')
    t = np.load(folder / ('raw_tree_test_scores.npz' if test else 'raw_tree_scores.npz'));n = np.load(folder / ('raw_neighbor_test_scores.npz' if test else 'raw_neighbor_scores.npz'));query = t['positions']
    matched = aux['matched'][fit].astype(bool);mean = matched.mean()
    rates = {str(repo): (matched[repos[fit] == repo].sum() + 20 * mean) / ((repos[fit] == repo).sum() + 20) / mean for repo in np.unique(repos[fit])}
    pick = json.loads((ROOT / 'outputs/experiments' / cfg['coverage_run'] / 'all_training_selections_locked.json').read_text())['selections'][name]['best_policy']
    rows = 1 - pick['mix'] + pick['mix'] * np.clip(np.array([rates.get(str(repo), 1) for repo in repos[query]]), 0, 2)
    pub, weight = publication_source(cfg['publication_source_run'], name);mapped = np.load(pub / 'contexts' / context / ('test_scores.npz' if test else 'scores.npz'));np.testing.assert_array_equal(mapped['positions'], query)
    mapping = joblib.load(pub / 'contexts' / context / 'mapping.joblib');values = {}
    for fraction in [0.] + cfg['tree_fractions']:
        tc, nc = tree['expectation'], nn['expectation']
        centre = {'mean': (1 - fraction) * nc['mean'] + fraction * tc['mean'],
                  'repositories': {repo: (1 - fraction) * value + fraction * tc['repositories'][repo] for repo, value in nc['repositories'].items()}}
        mixed = (1 - fraction) * n['mapped'] + fraction * t['mapped']
        value = residual(base, mixed, repos[query], centre, cfg['residual_weight'], rows)
        values[str(fraction)] = residual(value, mapped['scores'], repos[query], mapping['expectation'], weight)
    return query, values


def metrics(truth, prediction):
    p, r, f, _ = precision_recall_fscore_support(truth, prediction, average='micro', zero_division=0)
    return {'precision': p, 'recall': r, 'micro_f1': f}


def main(args):
    require_server();threadpool_limits(1);out = ROOT / 'outputs/experiments' / args.run_id;cfg = json.loads((out / 'config.json').read_text());result = json.loads((out / 'results.json').read_text())
    assert json.loads((out / 'status.json').read_text())['state'] == 'complete'
    with ProcessPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(audit_context, [{'run_id': args.run_id, 'config': cfg, 'context': f'fold{i}_{suffix}'} for i in range(5) for suffix in ['inner0', 'inner1', 'inner2', 'final']]))
    json_write(out / 'tree_component_audit.json', {'state': 'passed', 'contexts': records, 'independent_forests_refit': 80})
    lock = json.loads((out / 'all_calibrations_locked.json').read_text());training = json.loads((out / 'all_training_selections_locked.json').read_text())
    assert lock['all_five_contexts_locked'] and not lock['test_used'] and not training['validation_used'] and not training['test_used']
    assert (out / 'all_training_selections_locked.json').stat().st_mtime <= (out / 'all_calibrations_locked.json').stat().st_mtime
    data = np.load(ROOT / 'cache/workshop_v1/dataset.npz', allow_pickle=True);truth = data['y'].astype(bool);pooled = np.load(out / 'pooled_predictions.npz');np.testing.assert_array_equal(truth, pooled['y_true'])
    reference = ROOT / 'outputs/experiments' / cfg['reference_run'];original_pooled = np.load(reference / 'pooled_predictions.npz')
    np.testing.assert_array_equal(pooled['RF'], original_pooled['RF']);np.testing.assert_array_equal(pooled['previous_best'], original_pooled['best_policy'])
    seen = np.zeros(len(truth), int);decisions = []
    for name, choices in training['selections'].items():
        original, base = base_oof(cfg, name);lookup = {int(p): i for i, p in enumerate(original['positions'])};a = np.load(out / (name + '_oof_scores.npz'))
        for i in range(3):
            context = name + '_inner' + str(i);query = np.load(out / 'contexts' / context / 'roles.npz')['query'];local = np.array([lookup[int(p)] for p in query])
            _, cases = reconstruct(out, cfg, context, base[local])
            for key, value in cases.items():np.testing.assert_allclose(a[key][local], value, atol=1e-13, rtol=0)
        for method, pick in choices.items():
            if pick['candidate'] == 'reference':continue
            replay = threshold_choice(original['y_true'], a[str(pick['tree_fraction'])], original['rf_prediction'])
            for key in ['precision', 'recall', 'micro_f1', 'threshold', 'fallback_rf']:assert replay[key] == pick[key]
        candidates = json.loads((out / (name + '_OOF_search.json')).read_text());key = lambda row: (row['micro_f1'], row['precision'], row['candidate'] == 'reference', -row['tree_fraction'])
        assert choices['new_policy'] == max([r for r in candidates if r['candidate'] == 'tree'], key=key)
        assert choices['best_policy'] == max([r for r in candidates if r['candidate'] in ['tree', 'reference']], key=key)
        for suffix, base_key, test in [('_calibration.npz', 'portfolio', False), ('_assessment.npz', 'portfolio_scores', True)]:
            current = np.load(out / (name + suffix));main = np.load(ROOT / 'outputs/experiments' / cfg['base_run'] / (name + suffix));prior = np.load(reference / (name + suffix))
            query, cases = reconstruct(out, cfg, name + '_final', main[base_key], test)
            np.testing.assert_array_equal(query, current['positions']);np.testing.assert_array_equal(current['y_true'], truth[query])
            if test:
                role = np.load(ROOT / 'cache/workshop_v1/grouped5_pr_components' / (name + '.npz'));np.testing.assert_array_equal(role['test'], query);seen[query] += 1
                assert (out / 'contexts' / (name + '_final') / 'raw_tree_test_scores.npz').stat().st_mtime >= (out / 'all_calibrations_locked.json').stat().st_mtime
            for method, pick in choices.items():
                value = prior['best_policy_scores' if test else 'best_policy'] if pick['candidate'] == 'reference' else cases[str(pick['tree_fraction'])]
                np.testing.assert_allclose(current[method + '_scores' if test else method], value, atol=1e-13, rtol=0)
                choice = lock['choices'][name][method]
                if not test:
                    replay = threshold_choice(truth[query], value, current['rf_prediction'])
                    assert replay == choice
                else:
                    prediction = current['RF_prediction'] if choice['fallback_rf'] else value >= choice['threshold']
                    np.testing.assert_array_equal(prediction, current[method + '_prediction']);np.testing.assert_array_equal(prediction, pooled[method][query])
        decisions.append({'name': name, 'all_training_choices_thresholds_and_three_score_roles_reconstructed': True})
    assert (seen == 1).all();checked = {key: metrics(truth, pooled[key]) for key in result['metrics']}
    for method, value in checked.items():
        for key in value:assert abs(value[key] - result['metrics'][method][key]) < 1e-14
        assert result['RF_point_gates'][method] == all(value[k] >= checked['RF'][k] for k in value)
    groups = np.load(ROOT / 'cache/workshop_v1/key_component_groups.npy');differences = {}
    for method in ['new_policy', 'best_policy']:
        prediction, old = pooled[method], pooled['previous_best'];tallies = []
        for group in np.unique(groups):
            mask = groups == group;tallies.append([[(p[mask] & truth[mask]).sum(), p[mask].sum(), truth[mask].sum()] for p in [prediction, old]])
        tally = np.asarray(tallies, dtype=np.int64);draw = np.random.RandomState(9231).randint(len(tally), size=(2000, len(tally)));counts = tally[draw].sum(1)
        values = 2 * counts[:, :, 0] / np.maximum(counts[:, :, 1] + counts[:, :, 2], 1)
        differences[method] = {'TP_delta': int((prediction & truth).sum() - (old & truth).sum()), 'FP_delta': int((prediction & ~truth).sum() - (old & ~truth).sum()),
                               'changed_cells': int((prediction != old).sum()), 'paired_component_F1_delta_95pct_descriptive': np.quantile(values[:, 0] - values[:, 1], [.025, .975]).tolist()}
    deployment = ROOT / 'outputs/deployment/20261024_full_neighbor_candidate_v1/model.joblib';assert digest(deployment) == '40ed37a1c4bacc9a79d72e0266c245f93d1d28e54df3951c9c1a721b9582adf2'
    receipt = {'state': 'passed', 'sklearn_metrics': checked, 'all_rows_tested_once': True, 'decisions_replayed': True, 'twenty_contexts_refit_and_reconstructed': True,
               'independent_forests_refit': 80, 'differences': differences, 'independent_confirmation': False, 'source_deployment_unchanged': True}
    json_write(out / 'verification.json', receipt);report = ROOT / cfg['report_directory']
    json_write(report / 'verification.json', receipt);json_write(report / 'tree_component_audit.json', json.loads((out / 'tree_component_audit.json').read_text()))
    print(json.dumps({'state': 'passed', 'contexts': 20, 'forests_refit': 80, 'metrics': checked}))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--run-id', required=True);main(p.parse_args())
