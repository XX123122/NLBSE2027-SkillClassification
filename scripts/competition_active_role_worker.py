"""Extend the verified role with frozen active-base and auxiliary producers."""
import argparse
from collections import Counter
import copy
import json
import os
import runpy
import sys

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

import title_body_workers as base_worker
from skillcomp.runtime import ROOT, require_server, json_write
from skillcomp.residual_inference import fuse_auxiliary_embeddings
from skillcomp.matchability import matchability_features
from skillcomp.availability_inference import semantic_availability_features
from skillcomp.availability_calibration import fit_availability_intercepts, calibrate_availability
from skillcomp.annotation_channel import fit_channel, apply_channel
from skillcomp.peer_publication import peer_observed_union
from skillcomp.conditional_peer_prior import ConditionalPeerPrior
from skillcomp.peer_union_model import union_probability
from skillcomp.retrieval_evidence import EvidenceNeighbors, ExcludedChannelStatistics
from skillcomp.proxy_tree_readout import ProxyTreeReadout
from skillcomp.channel_residual import fit_expectation, apply_residual
from skillcomp.evidence_coverage import fit_coverage, coverage_weights
from skillcomp.benchmark import priors


def fit_experts():
    threadpool_limits(1)
    cfg = json.loads((ROOT / 'configs/active_role_recipe.json').read_text())
    out = ROOT / 'outputs/active_role_experts';out.mkdir()
    with np.load(ROOT / 'cache/workshop_v1/dataset.npz', allow_pickle=True) as data:
        y, repos, positions = data['y'], data['repos'], data['source_positions']
    with np.load(ROOT / 'outputs/experiments/encoder/contexts/fold0_inner0/roles.npz') as role:
        fit, known, query = role['context_train'], role['train'], role['query']
    with np.load(ROOT / 'outputs/experiments/encoder/auxiliary_targets.npz') as aux:
        targets, matched = aux['targets'], aux['matched'].astype(bool)
    groups = np.load(ROOT / 'cache/workshop_v1/key_component_groups.npy')
    with np.load(ROOT / 'outputs/experiments/data_root/candidate/aligned_targets.npz') as lineage:
        pr = lineage['pr_number'][positions]
    observed = matched[fit]
    peer, published = peer_observed_union(targets[fit], observed, repos[fit], pr[fit])
    np.testing.assert_array_equal(published, y[fit])
    own = np.zeros(peer.shape, np.uint8);own[observed] = targets[fit[observed]]
    f0 = ConditionalPeerPrior(20).fit(peer, own, repos[fit], groups[fit], y[fit])
    learned = np.load(ROOT / 'outputs/experiments/features/contexts/fold0_inner0/learned_e8.npy')
    frozen = np.load(ROOT / 'outputs/declared_frozen_Qwen4B.npy', mmap_mode='r')
    x = fuse_auxiliary_embeddings([learned, frozen])
    folder = ROOT / 'outputs/experiments/base/contexts/fold0_inner0'
    lexical = matchability_features(sparse.load_npz(folder / 'title.npz'), sparse.load_npz(folder / 'body.npz'), repos[fit], repos)
    xx = semantic_availability_features(lexical, frozen)
    gate = LogisticRegression(C=1, solver='liblinear', max_iter=2000, random_state=42).fit(xx[fit], observed)
    pfit = gate.predict_proba(xx[fit])[:, 1]
    calibration = fit_availability_intercepts(pfit, observed, repos[fit], 20)
    afit = calibrate_availability(calibration, pfit, repos[fit])
    aq = calibrate_availability(calibration, gate.predict_proba(xx[query])[:, 1], repos[query])
    ac = np.zeros(len(fit));tc_proxy = np.zeros((len(fit), 217));seen = np.zeros(len(fit), int)
    tree_template = ProxyTreeReadout(dimensions=128, estimators=256, depth=12, min_leaf=5, seed=1193, jobs=16).fit_projection(x[fit], repos[fit])
    for k, (a, b) in enumerate(GroupKFold(3).split(fit, groups=groups[fit])):
        av = LogisticRegression(C=1, solver='liblinear', max_iter=2000, random_state=42).fit(xx[fit[a]], observed[a])
        cal = fit_availability_intercepts(av.predict_proba(xx[fit[a]])[:, 1], observed[a], repos[fit[a]], 20)
        ac[b] = calibrate_availability(cal, av.predict_proba(xx[fit[b]])[:, 1], repos[fit[b]])
        train_known = fit[a][observed[a]]
        tree = copy.deepcopy(tree_template).fit_head(x[train_known], targets[train_known], repos[train_known])
        tree.head_.set_params(n_jobs=1)
        tc_proxy[b] = tree.predict_proba(x[fit[b]], repos[fit[b]])
        joblib.dump({'gate': av, 'calibration': cal, 'tree': tree}, out / f'sub{k}.joblib')
        seen[b] += 1
    assert (seen == 1).all()
    tree = copy.deepcopy(tree_template).fit_head(x[known], targets[known], repos[known]);tree.head_.set_params(n_jobs=1)
    tq_proxy = tree.predict_proba(x[query], repos[query])
    neighbors = EvidenceNeighbors(neighbors=20, power=4, pseudocount=1, prior_strength=20).fit(x[known], targets[known], repos[known], groups[known])
    q = neighbors.predict_proba(x[query], repos[query]);qc = neighbors.predict_fit_centre(x[fit], repos[fit], groups[fit])
    fq, counts_q, n_q = f0.predict(repos[query]);fc, counts_c, n_c = f0.fit_centre(repos[fit], groups[fit])
    nq = union_probability(fq, aq, q, counts_q, n_q);nc = union_probability(fc, ac, qc, counts_c, n_c)
    channel = fit_channel(y[fit], targets[fit], observed, repos[fit], 20)
    treeq = apply_channel(channel, tq_proxy, repos[query])
    stats = ExcludedChannelStatistics(y[fit], targets[fit], observed, repos[fit], groups[fit])
    treec = stats.map_fit_centre(tc_proxy, repos[fit], groups[fit])
    te, ne = fit_expectation(treec, repos[fit]), fit_expectation(nc, repos[fit])
    expectation = {'mean': .5 * te['mean'] + .5 * ne['mean'],
                   'repositories': {r: .5 * te['repositories'][r] + .5 * ne['repositories'][r] for r in te['repositories']}}
    # The retained publication expert models the separately observed availability event.
    availability_targets = np.repeat(observed[:, None], 217, axis=1)
    publication_channel = fit_channel(y[fit], availability_targets, np.ones(len(fit), bool), repos[fit], 20)
    pubq = apply_channel(publication_channel, np.repeat(aq[:, None], 217, axis=1), repos[query])
    pubc = apply_channel(publication_channel, np.repeat(afit[:, None], 217, axis=1), repos[fit])
    pe = fit_expectation(pubc, repos[fit])
    coverage = fit_coverage(observed, repos[fit], 20)
    row = coverage_weights(coverage, repos[query], cfg['coverage_mix'])
    np.savez_compressed(out / 'scores.npz', positions=query, fit=fit, mapped_NN=nq, mapped_tree=treeq,
                        centre_NN=nc, centre_tree=treec, availability_query=aq, availability_centre=ac,
                        publication=pubq, row_weights=row, proxy_NN=q, peer_F0=fq)
    joblib.dump({'tree': tree, 'neighbors': neighbors, 'prior': f0, 'channel': channel,
                 'gate': gate, 'calibration': calibration, 'expectation': expectation,
                 'publication_expectation': pe, 'publication_channel': publication_channel,
                 'coverage': coverage}, out / 'components.joblib')
    json_write(out / 'fit.json', {'state': 'complete', 'forests': 4, 'availability_heads': 4,
               'query_rows': len(query), 'fit_rows': len(fit), 'known_rows': len(known),
               'recipe_searched': False, 'declared_frozen_Qwen4B_reused': True})


def fit_active_base():
    threadpool_limits(1)
    out = ROOT / 'outputs/experiments/base_active';out.mkdir()
    source = ROOT / 'outputs/experiments/base'
    import shutil
    shutil.copyfile(source / 'input_fields.json', out / 'input_fields.json')
    config = json.loads((source / 'config.json').read_text());config['embedding_path'] = 'outputs/base_normalized_embeddings.npy'
    json_write(out / 'config.json', config)
    folder = out / 'contexts/fold0_inner0';folder.mkdir(parents=True)
    for name in ['roles.npz', 'title.npz', 'body.npz', 'title_vectorizer.joblib', 'body_vectorizer.joblib']:
        os.link(source / 'contexts/fold0_inner0' / name, folder / name)
    base_worker.initialize(ROOT, out)
    base_worker.fit({'context': 'fold0_inner0', 'parallel_labels': 16,
                     'spec': {'name': 'split_hybrid1_c1', 'title_scale': 1, 'body': True, 'embedding_scale': 1, 'C': 1}})
    with np.load(ROOT / 'cache/workshop_v1/dataset.npz', allow_pickle=True) as data:
        y, repos = data['y'], data['repos']
    with np.load(folder / 'roles.npz') as role:fit, query = role['train'], role['query']
    cfg = json.loads((ROOT / 'configs/active_role_recipe.json').read_text())
    a = np.load(folder / 'split_hybrid1_c1/scores.npz')
    base = cfg['base_text_weight'] * a['scores'] + (1 - cfg['base_text_weight']) * priors(y[fit], repos[fit], repos[query], 20)
    experts = ROOT / 'outputs/active_role_experts'
    scores = np.load(experts / 'scores.npz');model = joblib.load(experts / 'components.joblib')
    mapped = .5 * scores['mapped_tree'] + .5 * scores['mapped_NN']
    candidate = apply_residual(base, mapped, repos[query], model['expectation'], .5, row_weights=scores['row_weights'])
    candidate = apply_residual(candidate, scores['publication'], repos[query], model['publication_expectation'], cfg['publication_weight'])
    np.savez_compressed(ROOT / 'outputs/active_role_candidate.npz', positions=query, base=base, candidate=candidate, mapped=mapped)
    print(json.dumps({'state': 'active_role_candidate_generated', 'rows': len(query), 'accuracy_or_threshold_search': False}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['fit-experts','fit-active-base'],required=True)
    a=p.parse_args()
    if a.stage=='fit-experts':require_server();fit_experts()
    else:require_server();fit_active_base()
