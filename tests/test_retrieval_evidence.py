import inspect

import numpy as np

from skillcomp.annotation_channel import fit_channel, apply_channel
from skillcomp.known_proxy_neighbors import KnownProxyNeighbors
from skillcomp.retrieval_evidence import EvidenceNeighbors, ExcludedChannelStatistics, evidence_features


def fixture():
    rng = np.random.default_rng(34)
    x = rng.normal(size=(13, 6)).astype(np.float32)
    a = rng.integers(0, 2, size=(13, 4));a[:, 3] = 0
    y = np.maximum(a, rng.integers(0, 2, size=a.shape));y[:, 2] = 1
    r = np.array(['a', 'b', 'a', 'b', 'c', 'a', 'a', 'b', 'c', 'c', 'a', 'b', 'a'])
    g = np.array([0, 0, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7])
    return x, a, y, r, g


def test_query_probability_preserves_original_model_and_decomposes_evidence():
    x, a, _, r, g = fixture()
    old = KnownProxyNeighbors().fit(x, a, r, g)
    new = EvidenceNeighbors().fit(x, a, r, g)
    q = np.array(['a', 'b', 'unknown'])
    np.testing.assert_allclose(new.predict_proba(x[:3], q), old.predict_proba(x[:3], q), atol=2e-7)
    e = new.predict_evidence(x[:3], q)
    np.testing.assert_allclose(e['positive_mass'] + e['zero_mass'], np.broadcast_to(e['weight_sum'][:, None], (3, 4)), atol=1e-6)
    assert evidence_features(e, 'A1').shape == (3, 4, 2)
    assert evidence_features(e, 'A2').shape == (3, 4, 8)
    assert list(inspect.signature(new.predict_evidence).parameters) == ['vectors', 'repositories']


def test_component_excluded_priors_neighbors_constants_and_mapping_match_bruteforce():
    x, a, y, r, g = fixture();matched = np.arange(len(x)) % 4 != 1
    model = EvidenceNeighbors().fit(x[matched], a[matched], r[matched], g[matched])
    probabilities = model.predict_fit_centre(x, r, g)
    stats = ExcludedChannelStatistics(y, a, matched, r, g)
    actual = stats.map_fit_centre(probabilities, r, g)
    for group in np.unique(g):
        keep = g != group;query = ~keep;known = keep & matched
        brute = KnownProxyNeighbors().fit(x[known], a[known], r[known], g[known]).predict_proba(x[query], r[query])
        np.testing.assert_allclose(probabilities[query], brute, atol=2e-7)
        channel = fit_channel(y[keep], a[keep], matched[keep], r[keep])
        analytic = stats.excluding(group)
        np.testing.assert_allclose(analytic['global_rates'], channel['global_rates'], atol=1e-12)
        for repo in channel['repository_rates']:
            np.testing.assert_allclose(analytic['repository_rates'][repo], channel['repository_rates'][repo], atol=1e-12)
        np.testing.assert_allclose(actual[query], apply_channel(channel, brute, r[query]), atol=2e-7)


def test_excluded_labels_cannot_change_their_own_component_centre():
    x, a, y, r, g = fixture();matched = np.ones(len(x), bool);query = g == 0
    def predict(a, y):
        m = EvidenceNeighbors().fit(x, a, r, g)
        p = m.predict_fit_centre(x[query], r[query], g[query])
        s = ExcludedChannelStatistics(y, a, matched, r, g)
        return s.map_fit_centre(p, r[query], g[query])
    reference = predict(a, y)
    a2, y2 = a.copy(), y.copy();a2[query] = 1 - a2[query];y2[query] = 1 - y2[query]
    np.testing.assert_allclose(reference, predict(a2, y2), atol=1e-12)
