"""Align evidence-matched diagnostic targets without inventing negative labels."""
import numpy as np


def align_targets(dataset, companion):
    y = np.asarray(dataset['y']); repos = np.asarray(dataset['repos'], dtype=str); original = np.asarray(companion['original_y']); other_repos = np.asarray(companion['repos'], dtype=str)
    if y.ndim != 2 or original.ndim != 2 or original.shape[1] != y.shape[1]:
        raise ValueError('Matching ordered label columns required')
    if original.shape == y.shape and np.array_equal(original, y) and np.array_equal(other_repos, repos):
        order = np.arange(len(y))
    else:
        order = np.asarray(dataset['source_positions'], dtype=int)
        if order.shape != (len(y),) or (order < 0).any() or (order >= len(original)).any() or not np.array_equal(original[order], y) or not np.array_equal(other_repos[order], repos):
            raise ValueError('Published targets and repositories cannot be exactly aligned')
    diagnostic = np.asarray(companion['y']); mask = np.asarray(companion['matched'])
    if diagnostic.shape != original.shape or mask.shape != (len(original),) or not np.isin(mask, [0, 1]).all():
        raise ValueError('Aligned diagnostic targets and binary evidence mask required')
    matched = mask[order].astype(bool)
    if not np.isin(y, [0, 1]).all() or not np.isin(diagnostic[order][matched], [0, 1]).all():
        raise ValueError('Binary targets required on included rows')
    # Values for missing-evidence rows remain inaccessible to the study roles.
    return {'released': y, 'companion': diagnostic[order], 'matched': matched, 'alignment_order': order,
            'pr_numbers': np.asarray(companion['pr_number'])[order]}


def intersect_roles(split, matched):
    matched = np.asarray(matched, bool); result = {}
    for name in ['train', 'validation', 'test']:
        positions = np.asarray(split[name], dtype=int)
        if positions.ndim != 1 or (positions < 0).any() or (positions >= len(matched)).any() or len(np.unique(positions)) != len(positions):
            raise ValueError('Valid unique role positions required')
        result[name] = positions[matched[positions]]
        if not len(result[name]):
            raise ValueError('Evidence intersection produced an empty role')
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if set(result[a]) & set(result[b]):
            raise ValueError('Role overlap')
    return result
