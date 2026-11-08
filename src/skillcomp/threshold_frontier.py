"""Exact global-threshold frontier with paired cluster influence diagnostics.

The cluster quantity is a delta-method precision-difference standard-error
proxy, not a post-selection coverage guarantee. Taking the maximum over two
one-way cluster partitions is a heuristic, not a multiway covariance estimate.
"""
import numpy as np


def _metrics(truth, pred):
    truth, pred = np.asarray(truth, bool), np.asarray(pred, bool)
    tp = int((truth & pred).sum()); pp = int(pred.sum()); positives = int(truth.sum())
    return {'precision': tp / max(pp, 1), 'recall': tp / max(positives, 1), 'micro_f1': 2 * tp / max(pp + positives, 1)}


def _paired_cluster_se(ordered_y, ordered_groups, baseline_tp, baseline_pp, total_baseline_tp, total_baseline_pp, counts):
    n = len(ordered_y); group_count = len(baseline_pp)
    rf_precision = total_baseline_tp / max(total_baseline_pp, 1)
    influence_rf = (baseline_tp - rf_precision * baseline_pp) / max(total_baseline_pp, 1)
    rf_variance = float(influence_rf @ influence_rf)
    grouped = np.argsort(ordered_groups, kind='stable'); sorted_groups = ordered_groups[grouped]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_groups)) + 1]
    lengths = np.diff(np.r_[starts, n]); cumulative = np.cumsum(ordered_y[grouped], dtype=np.int64)
    offsets = np.r_[0, cumulative[starts[1:] - 1]]
    tp_before = np.empty(n, dtype=np.int64); pp_before = np.empty(n, dtype=np.int64)
    tp_before[grouped] = cumulative - np.repeat(offsets, lengths) - ordered_y[grouped]
    pp_before[grouped] = np.arange(n) - np.repeat(starts, lengths)
    def take(values):
        return np.where(counts > 0, np.cumsum(values, dtype=np.float64)[np.maximum(counts - 1, 0)], 0.)
    tt = take((2 * tp_before + 1) * ordered_y)
    ss = take(2 * pp_before + 1)
    ts = take(tp_before + ordered_y * (pp_before + 1))
    tb = take(ordered_y * influence_rf[ordered_groups]); sb = take(influence_rf[ordered_groups])
    tp = take(ordered_y); denominator = np.maximum(counts, 1); precision = tp / denominator
    first = (tt - 2 * precision * ts + precision ** 2 * ss) / denominator ** 2
    cross = (tb - precision * sb) / denominator
    variance = first - 2 * cross + rf_variance
    tolerance = 64 * np.finfo(float).eps * (np.abs(first) + 2 * np.abs(cross) + rf_variance)
    variance[np.abs(variance) <= tolerance] = 0.
    return np.sqrt(np.maximum(variance, 0.) * group_count / (group_count - 1))


def exact_frontier(truth, scores, rf_prediction, cluster_axes, lower=.05, upper=.95):
    y = np.asarray(truth, dtype=bool); scores = np.asarray(scores, dtype=np.float64); rf = np.asarray(rf_prediction, dtype=bool)
    if y.ndim != 2 or not y.size or y.shape != scores.shape or y.shape != rf.shape:
        raise ValueError('Matching nonempty sample by label arrays required')
    if not np.isfinite(scores).all() or not (0 <= lower < upper <= 1):
        raise ValueError('Finite scores and ordered probability cutoffs required')
    flat = scores.ravel(); targets = y.ravel().astype(np.int64); order = np.argsort(-flat, kind='stable')
    ordered_scores, ordered_y = flat[order], targets[order]
    points = ordered_scores[(ordered_scores >= lower) & (ordered_scores <= upper)]
    thresholds = np.unique(np.r_[points, lower, upper])[::-1]
    counts = np.searchsorted(-ordered_scores, -thresholds, side='right')
    cumulative = np.cumsum(ordered_y, dtype=np.int64)
    tp = np.where(counts > 0, cumulative[np.maximum(counts - 1, 0)], 0)
    positives = int(targets.sum()); baseline_tp = int((rf & y).sum()); baseline_pp = int(rf.sum())
    arrays = {'threshold': thresholds, 'predicted_positives': counts, 'true_positives': tp,
        'precision': tp / np.maximum(counts, 1), 'recall': tp / max(positives, 1),
        'micro_f1': 2 * tp / np.maximum(counts + positives, 1)}
    axis_counts = {}; usable = []; se_values = []
    for name, groups in cluster_axes.items():
        groups = np.asarray(groups)
        if groups.shape != (len(y),):
            raise ValueError('One cluster identifier per sample required')
        _, inverse = np.unique(groups, return_inverse=True); group_count = int(inverse.max() + 1); axis_counts[name] = group_count
        if group_count < 2:
            continue
        cell_groups = np.repeat(inverse, y.shape[1])
        t_group = np.bincount(cell_groups, weights=(rf & y).ravel(), minlength=group_count)
        p_group = np.bincount(cell_groups, weights=rf.ravel(), minlength=group_count)
        se = _paired_cluster_se(ordered_y, cell_groups[order], t_group, p_group, baseline_tp, baseline_pp, counts)
        arrays['se_' + name] = se; usable.append(name); se_values.append(se)
    arrays['precision_se_proxy'] = np.maximum.reduce(se_values) if se_values else np.zeros(len(thresholds))
    assert all(np.isfinite(value).all() for value in arrays.values())
    return {'arrays': arrays, 'baseline': _metrics(y, rf), 'axis_counts': axis_counts, 'usable_axes': usable}


def choose_frontier(frontier, z=0.):
    if z < 0:
        raise ValueError('Nonnegative guard multiplier required')
    arrays = frontier['arrays']; baseline = frontier['baseline']; keys = ['precision', 'recall', 'micro_f1']
    precision_floor = arrays['precision'] - baseline['precision'] - z * arrays['precision_se_proxy']
    eligible = (precision_floor >= -1e-12) & (arrays['recall'] >= baseline['recall'] - 1e-12) & (arrays['micro_f1'] >= baseline['micro_f1'] - 1e-12)
    if z and not frontier['usable_axes']:
        eligible[:] = False
    indices = np.flatnonzero(eligible)
    if len(indices):
        rank = np.lexsort((arrays['threshold'][indices], arrays['precision'][indices], arrays['micro_f1'][indices]))
        index = int(indices[rank[-1]])
        improve = any(arrays[key][index] > baseline[key] + 1e-12 for key in keys)
        if improve:
            return {**{key: float(arrays[key][index]) for key in keys}, 'threshold': float(arrays['threshold'][index]),
                'precision_se_proxy': float(arrays['precision_se_proxy'][index]), 'precision_delta_guard_proxy': float(precision_floor[index]),
                'fallback_rf': False, 'z': z, 'eligible_thresholds': len(indices), 'candidate_thresholds': len(eligible), 'usable_axes': frontier['usable_axes']}
    return baseline | {'threshold': .5, 'precision_se_proxy': 0., 'precision_delta_guard_proxy': 0., 'fallback_rf': True, 'z': z,
        'eligible_thresholds': len(indices), 'candidate_thresholds': len(eligible), 'usable_axes': frontier['usable_axes']}
