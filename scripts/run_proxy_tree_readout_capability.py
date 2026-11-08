"""Frozen fivefold tree/NN blend; every test follows all-five calibration locks."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from run_auxiliary_lexical_residual_capability import _features_without_repository, reference_oof
from run_channel_residual_capability import base_oof
from run_hierarchy_strength_capability import publication_source
from skillcomp.annotation_channel import apply_channel
from skillcomp.benchmark import fast_micro
from skillcomp.capability_search import threshold_choice
from skillcomp.channel_residual import fit_expectation, apply_residual
from skillcomp.evidence_coverage import fit_coverage, coverage_weights
from skillcomp.proxy_tree_readout import ProxyTreeReadout
from skillcomp.retrieval_evidence import EvidenceNeighbors, ExcludedChannelStatistics
from skillcomp.runtime import ROOT, require_server, provenance, json_write, status


def input_values(cfg, context):
    source = ROOT / 'outputs/experiments' / cfg['neighbor_run'] / 'contexts' / context
    role = np.load(source / 'roles.npz');fit, query, known = role['train'], role['query'], role['known']
    data = np.load(ROOT / 'cache/workshop_v1/dataset.npz', allow_pickle=True)
    aux = np.load(ROOT / 'outputs/experiments' / cfg['channel_run'] / 'auxiliary_targets.npz')
    groups = np.load(ROOT / 'cache/workshop_v1/key_component_groups.npy')
    np.testing.assert_array_equal(known, fit[aux['matched'][fit].astype(bool)])
    assert not np.intersect1d(groups[fit], groups[query]).size
    source_cfg = json.loads((ROOT / 'outputs/experiments' / cfg['proxy_source_run'] / 'config.json').read_text())
    x = _features_without_repository(source_cfg, context, 'codebert_dense', fit)
    x = x.toarray() if hasattr(x, 'toarray') else np.asarray(x)
    old = joblib.load(source / 'model.joblib')
    return fit, query, known, data, aux, groups, x, old


def prepare(job):
    threadpool_limits(1);out, cfg, context = Path(job['out']), job['config'], job['context'];started = time.monotonic()
    folder = out / 'contexts' / context
    fit, query, known, data, aux, groups, x, old = input_values(cfg, context);repos = data['repos']
    stats = ExcludedChannelStatistics(data['y'][fit], aux['targets'][fit], aux['matched'][fit], repos[fit], groups[fit])
    if not job['reused']:
        folder.mkdir(parents=True);np.savez_compressed(folder / 'roles.npz', train=fit, query=query, known=known)
        template = ProxyTreeReadout(**cfg['tree']).fit_projection(x[fit], repos[fit]);centre_proxy = np.zeros((len(fit), 217));seen = np.zeros(len(fit), int)
        for i, (train, held) in enumerate(GroupKFold(3).split(fit, groups=groups[fit])):
            observed = fit[train][aux['matched'][fit[train]].astype(bool)]
            assert not np.intersect1d(groups[observed], groups[fit[held]]).size
            head = copy.deepcopy(template).fit_head(x[observed], aux['targets'][observed], repos[observed])
            centre_proxy[held] = head.predict_proba(x[fit[held]], repos[fit[held]]);seen[held] += 1
            sub = folder / ('sub' + str(i));sub.mkdir();joblib.dump(head, sub / 'model.joblib')
            np.savez_compressed(sub / 'roles.npz', train=fit[train], query=fit[held], known=observed)
        assert (seen == 1).all()
        head = copy.deepcopy(template).fit_head(x[known], aux['targets'][known], repos[known])
        tree_centre = fit_expectation(stats.map_fit_centre(centre_proxy, repos[fit], groups[fit]), repos[fit])
        tree_proxy = head.predict_proba(x[query], repos[query]);mapped = apply_channel(old['channel'], tree_proxy, repos[query])
        joblib.dump({'head': head, 'expectation': tree_centre, 'projection_fit': fit}, folder / 'model.joblib')
        np.savez_compressed(folder / 'raw_tree_scores.npz', positions=query, proxy=tree_proxy, centre_proxy=centre_proxy, mapped=mapped)
    else:
        role = np.load(folder / 'roles.npz');np.testing.assert_array_equal(role['train'], fit);np.testing.assert_array_equal(role['query'], query)
        original = np.load(folder / 'scores.npz')
        np.savez_compressed(folder / 'raw_tree_scores.npz', positions=query, proxy=original['proxy'], centre_proxy=original['centre_proxy'], mapped=original['mapped'])
    nn = EvidenceNeighbors(**cfg['neighbors']).fit(x[known], aux['targets'][known], repos[known], groups[known])
    nn_centre = fit_expectation(stats.map_fit_centre(nn.predict_fit_centre(x[fit], repos[fit], groups[fit]), repos[fit], groups[fit]), repos[fit])
    joblib.dump({'neighbors': nn, 'expectation': nn_centre, 'channel': old['channel']}, folder / 'neighbor_component.joblib')
    nn_query = apply_channel(old['channel'], nn.predict_proba(x[query], repos[query]), repos[query])
    np.savez_compressed(folder / 'raw_neighbor_scores.npz', positions=query, mapped=nn_query)
    record = {'context': context, 'fit_rows': len(fit), 'known_rows': len(known), 'query_rows': len(query),
              'reused_pilot_forests': job['reused'], 'new_forests': 0 if job['reused'] else 4,
              'tree_model_sha256': digest(folder / 'model.joblib'), 'neighbor_model_sha256': digest(folder / 'neighbor_component.joblib'),
              'source_neighbor_sha256': digest(ROOT / 'outputs/experiments' / cfg['neighbor_run'] / 'contexts' / context / 'model.joblib'),
              'query_groups_absent_from_fit': True, 'runtime_seconds': time.monotonic() - started}
    json_write(folder / 'component_fit.json', record);return record


def case_scores(out, cfg, context, base, test=False):
    folder = Path(out) / 'contexts' / context;name = context.split('_')[0]
    role = np.load(folder / 'roles.npz');fit = role['train']
    data = np.load(ROOT / 'cache/workshop_v1/dataset.npz', allow_pickle=True);repos = data['repos']
    aux = np.load(ROOT / 'outputs/experiments' / cfg['channel_run'] / 'auxiliary_targets.npz')
    tree = joblib.load(folder / 'model.joblib');neighbor = joblib.load(folder / 'neighbor_component.joblib')
    if test:
        assert (Path(out) / 'all_calibrations_locked.json').exists()
        query = np.load(ROOT / 'cache/workshop_v1/grouped5_pr_components' / (name + '.npz'))['test']
        _, _, _, _, _, _, x, _ = input_values(cfg, context)
        tree_proxy = tree['head'].predict_proba(x[query], repos[query])
        t = apply_channel(neighbor['channel'], tree_proxy, repos[query])
        n = apply_channel(neighbor['channel'], neighbor['neighbors'].predict_proba(x[query], repos[query]), repos[query])
        np.savez_compressed(folder / 'raw_tree_test_scores.npz', positions=query, proxy=tree_proxy, mapped=t)
        np.savez_compressed(folder / 'raw_neighbor_test_scores.npz', positions=query, mapped=n)
    else:
        a = np.load(folder / 'raw_tree_scores.npz');b = np.load(folder / 'raw_neighbor_scores.npz');query = a['positions'];np.testing.assert_array_equal(query, b['positions'])
        t, n = a['mapped'], b['mapped']
    coverage_pick = json.loads((ROOT / 'outputs/experiments' / cfg['coverage_run'] / 'all_training_selections_locked.json').read_text())['selections'][name]['best_policy']
    rows = coverage_weights(fit_coverage(aux['matched'][fit], repos[fit], 20), repos[query], coverage_pick['mix'])
    pub, weight = publication_source(cfg['publication_source_run'], name)
    pub_scores = np.load(pub / 'contexts' / context / ('test_scores.npz' if test else 'scores.npz'))
    np.testing.assert_array_equal(pub_scores['positions'], query)
    mapping = joblib.load(pub / 'contexts' / context / 'mapping.joblib');np.testing.assert_array_equal(mapping['train'], fit)
    values = {}
    for fraction in [0.] + cfg['tree_fractions']:
        tc, nc = tree['expectation'], neighbor['expectation']
        centre = {'mean': (1 - fraction) * nc['mean'] + fraction * tc['mean'],
                  'repositories': {repo: (1 - fraction) * value + fraction * tc['repositories'][repo] for repo, value in nc['repositories'].items()}, 'fit_rows': len(fit)}
        score = apply_residual(base, (1 - fraction) * n + fraction * t, repos[query], centre, cfg['residual_weight'], row_weights=rows)
        values[str(fraction)] = apply_residual(score, pub_scores['scores'], repos[query], mapping['expectation'], weight)
    return query, values


def choose(job):
    threadpool_limits(1);out, cfg, name = Path(job['out']), job['config'], job['name']
    original, base = base_oof(cfg, name);lookup = {int(p): i for i, p in enumerate(original['positions'])}
    values = {str(f): np.zeros_like(base) for f in [0.] + cfg['tree_fractions']};seen = np.zeros(len(base), int)
    for i in range(3):
        query = np.load(out / 'contexts' / (name + '_inner' + str(i)) / 'roles.npz')['query'];local = np.array([lookup[int(p)] for p in query])
        _, cases = case_scores(out, cfg, name + '_inner' + str(i), base[local]);seen[local] += 1
        for key in values:values[key][local] = cases[key]
    assert (seen == 1).all()
    rows = [threshold_choice(original['y_true'], values[str(f)], original['rf_prediction']) | {'candidate': 'tree', 'tree_fraction': f} for f in cfg['tree_fractions']]
    prior, _ = reference_oof(cfg['reference_run'], name, original['positions'])
    reference = threshold_choice(original['y_true'], prior, original['rf_prediction']) | {'candidate': 'reference', 'tree_fraction': 0.}
    key = lambda row: (row['micro_f1'], row['precision'], row['candidate'] == 'reference', -row['tree_fraction'])
    selected = {'new_policy': max(rows, key=key), 'best_policy': max(rows + [reference], key=key),
                'statistical_control': threshold_choice(original['y_true'], values['0.0'], original['rf_prediction']) | {'candidate': 'statistical_control', 'tree_fraction': 0.}}
    json_write(out / (name + '_OOF_search.json'), rows + [reference, selected['statistical_control']])
    np.savez_compressed(out / (name + '_oof_scores.npz'), positions=original['positions'], **values)
    if name == 'fold0':
        pilot = ROOT / 'outputs/experiments' / cfg['pilot_run'];a = np.load(pilot / 'training_scores.npz');np.testing.assert_array_equal(a['positions'], original['positions'])
        for fraction in cfg['tree_fractions']:np.testing.assert_allclose(a[str(fraction)], values[str(fraction)], atol=1e-12, rtol=0)
    return name, selected


def run(args, out):
    started = time.monotonic();cfg = json.loads((ROOT / args.config).read_text());provenance(out, cfg)
    pilot = ROOT / 'outputs/experiments' / cfg['pilot_run'];screen = json.loads((pilot / 'training_only_pilot.json').read_text())
    assert screen['full_five_fold_admitted'] and json.loads((pilot / 'audit.json').read_text())['state'] == 'passed'
    reference = ROOT / 'outputs/experiments' / cfg['reference_run'];assert json.loads((reference / 'verification.json').read_text())['state'] == 'passed'
    names = [f'fold{i}' for i in range(5)];jobs = [];reused = []
    json_write(out / 'expansion_lock.json', {'pilot_sha256': digest(pilot / 'training_only_pilot.json'), 'pilot_audit_sha256': digest(pilot / 'audit.json'),
               'config_sha256': digest(ROOT / args.config), 'all_hyperparameters_frozen': True, 'new_outer_scores_opened': False})
    for name in names:
        for suffix in ['inner0', 'inner1', 'inner2', 'final']:
            context = name + '_' + suffix;source = pilot / 'contexts' / context;reuse = (source / 'model.joblib').is_file()
            if reuse:
                shutil.copytree(source, out / 'contexts' / context)
                reused.append({'context': context, 'tree_model_sha256': digest(source / 'model.joblib')})
            jobs.append({'out': str(out), 'config': cfg, 'context': context, 'reused': reuse})
    status(out, 'running', stage='prepare_twenty_tree_neighbor_contexts', total=len(jobs))
    with ProcessPoolExecutor(max_workers=cfg['CPU_workers']) as pool:
        records = []
        for future in as_completed([pool.submit(prepare, job) for job in jobs]):
            records.append(future.result());status(out, 'running', stage='prepare_twenty_tree_neighbor_contexts', completed=len(records), total=len(jobs))
        json_write(out / 'fit_records.json', records);json_write(out / 'reused_contexts.json', reused)
        selections = dict(pool.map(choose, [{'out': str(out), 'config': cfg, 'name': name} for name in names]))
    json_write(out / 'all_training_selections_locked.json', {'selections': selections, 'validation_used': False, 'test_used': False})
    data = np.load(ROOT / 'cache/workshop_v1/dataset.npz', allow_pickle=True);y = data['y'];choices = {}
    for name, selected in selections.items():
        base = np.load(ROOT / 'outputs/experiments' / cfg['base_run'] / (name + '_calibration.npz'));prior = np.load(reference / (name + '_calibration.npz'))
        query, cases = case_scores(out, cfg, name + '_final', base['portfolio']);np.testing.assert_array_equal(query, base['positions']);np.testing.assert_array_equal(query, prior['positions'])
        scores = {};choices[name] = {}
        for family, pick in selected.items():
            scores[family] = prior['best_policy'] if pick['candidate'] == 'reference' else cases[str(pick['tree_fraction'])]
            choices[name][family] = threshold_choice(y[query], scores[family], base['rf_prediction'])
        np.savez_compressed(out / (name + '_calibration.npz'), positions=query, y_true=y[query], rf_prediction=base['rf_prediction'], **scores)
    json_write(out / 'all_calibrations_locked.json', {'selections': selections, 'choices': choices, 'test_used': False, 'all_five_contexts_locked': True})
    status(out, 'running', stage='all_five_policies_locked_then_assessment')
    pooled = {key: np.zeros_like(y, bool) for key in ['RF', 'previous_best', 'new_policy', 'best_policy', 'statistical_control']};seen = np.zeros(len(y), int);folds = []
    for name, selected in selections.items():
        base = np.load(ROOT / 'outputs/experiments' / cfg['base_run'] / (name + '_assessment.npz'));prior = np.load(reference / (name + '_assessment.npz'))
        query, cases = case_scores(out, cfg, name + '_final', base['portfolio_scores'], test=True)
        np.testing.assert_array_equal(query, base['positions']);np.testing.assert_array_equal(query, prior['positions']);seen[query] += 1
        pred = {'RF': base['RF_prediction'], 'previous_best': prior['best_policy_prediction']};scores = {}
        for family, pick in selected.items():
            scores[family] = prior['best_policy_scores'] if pick['candidate'] == 'reference' else cases[str(pick['tree_fraction'])]
            choice = choices[name][family];pred[family] = pred['RF'] if choice['fallback_rf'] else scores[family] >= choice['threshold']
        for key, value in pred.items():pooled[key][query] = value
        np.savez_compressed(out / (name + '_assessment.npz'), positions=query, y_true=y[query], **{key + '_prediction': value for key, value in pred.items()}, **{key + '_scores': value for key, value in scores.items()})
        folds.append({'name': name, 'selections': selected, 'choices': choices[name], 'metrics': {key: fast_micro(y[query], value) for key, value in pred.items()}})
    assert (seen == 1).all();metrics = {key: fast_micro(y, value) for key, value in pooled.items()}
    gates = {key: all(value[k] >= metrics['RF'][k] for k in value) for key, value in metrics.items()}
    np.savez_compressed(out / 'pooled_predictions.npz', y_true=y, **pooled)
    result = {'run_id': args.run_id, 'protocol': cfg['protocol'], 'rows': len(y), 'labels': y.shape[1], 'metrics': metrics, 'RF_point_gates': gates, 'folds': folds,
              'all_choices_locked_before_assessment': True, 'new_encoder_fits': 0, 'new_tree_forests': sum(r['new_forests'] for r in records),
              'reused_pilot_forests': 4 * len(reused), 'new_neighbor_models': len(records), 'runtime_seconds': time.monotonic() - started, 'limits': cfg['limits']}
    json_write(out / 'results.json', result);report = ROOT / cfg['report_directory'];report.mkdir(exist_ok=True)
    for file in ['results.json', 'expansion_lock.json', 'all_training_selections_locked.json', 'all_calibrations_locked.json', 'fit_records.json', 'reused_contexts.json']:
        json_write(report / file, json.loads((out / file).read_text()))
    status(out, 'complete', metrics=metrics, RF_point_gates=gates)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--run-id', required=True);p.add_argument('--config', default='configs/proxy_tree_readout_capability_v1.json');p.add_argument('--launch', action='store_true');args = p.parse_args();require_server()
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
