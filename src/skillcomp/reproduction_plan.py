"""Role and stage contracts. Importing this module never reads a corpus or model."""
import re

ROLES = [f'fold{f}_{part}' for f in range(5) for part in ['inner0', 'inner1', 'inner2', 'final']]
ROLES += ['full_inner0', 'full_inner1', 'full_inner2', 'full_final']
STAGES = ['prepare', 'tokenize', 'encoder-fit', 'base-encode', 'extract', 'base-fit',
          'normalized-encode', 'experts-fit', 'active-base-fit']


def role_contract(role):
    if role not in ROLES:
        raise ValueError('Unknown historical role')
    full = role.startswith('full_')
    inner = '_inner' in role
    return {'role': role, 'family': 'full_training_OOF' if full else 'internal_grouped5',
            'fold': None if full else int(role[4]),
            'inner': int(role[-1]) if inner else None,
            'encoder_epoch': 4 if full else 8,
            'query_role': 'training_in_sample_for_deployment' if role == 'full_final' else
                          'training_OOF' if inner else 'validation',
            'query_is_independent_test': False,
            'historically_rebuilt': role == 'fold0_inner0',
            'full_encoder_schedule_portably_resolved': not full,
            'active_policy_execution_supported': role == 'fold0_inner0'}


def stage_plan(role):
    spec = role_contract(role)
    context = role
    jobs = {
        'prepare': ['CPU_PYTHON', 'scripts/prepare_role.py', '--stage', 'prepare'],
        'tokenize': ['ENCODER_PYTHON', 'scripts/modernbert_full_worker.py', '--out', 'outputs/experiments/encoder', '--mode', 'tokenize'],
        'encoder-fit': ['ENCODER_PYTHON', 'scripts/modernbert_full_worker.py', '--out', 'outputs/experiments/encoder', '--mode', 'fit', '--job', 'fit_job.json'],
        'base-encode': ['ENCODER_PYTHON', 'scripts/encode_frozen_candidate_inputs.py', '--settings', 'configs/base_encoder.json', '--input', 'outputs/original_inputs.json', '--out', 'outputs/base_embeddings.npy'],
        'extract': ['ENCODER_PYTHON', 'scripts/extract_learned_context_features.py', '--out', 'outputs/experiments/features', '--mode', 'extract', '--job', 'extract_job.json'],
        'base-fit': ['CPU_PYTHON', 'scripts/prepare_role.py', '--stage', 'fit-base'],
        'normalized-encode': ['ENCODER_PYTHON', 'scripts/encode_frozen_candidate_inputs.py', '--settings', 'configs/base_encoder_normalized.json', '--input', 'outputs/original_inputs.json', '--out', 'outputs/base_normalized_embeddings.npy'],
        'experts-fit': ['CPU_PYTHON', 'scripts/competition_active_role_worker.py', '--stage', 'fit-experts'],
        'active-base-fit': ['CPU_PYTHON', 'scripts/competition_active_role_worker.py', '--stage', 'fit-active-base'],
    }
    dependencies = {'prepare': [], 'tokenize': ['prepare'], 'encoder-fit': ['tokenize'],
                    'base-encode': ['prepare'], 'extract': ['encoder-fit'],
                    'base-fit': ['base-encode'], 'normalized-encode': ['prepare'],
                    'experts-fit': ['extract'], 'active-base-fit': ['experts-fit', 'normalized-encode']}
    records = []
    for stage in STAGES:
        blocked = None
        if spec['family'] == 'full_training_OOF' and stage != 'prepare':
            blocked = 'Full four-epoch historical scheduler/role recipe must be resolved; no guessed schedule.'
        elif stage in ['experts-fit', 'active-base-fit'] and role != 'fold0_inner0':
            blocked = 'Only fold0_inner0 active policy producer is connected; other selected policies differ.'
        records.append({'stage': stage, 'command': jobs[stage], 'depends_on': dependencies[stage],
                        'execution_blocker': blocked, 'portable_real_execution_this_round': False})
    return {'role': spec, 'stages': records, 'context': context,
            'full_chain_complete': False,
            'excluded_unconnected_phases': ['all-policy selection', 'validation threshold calibration',
                                          'held evaluation', 'full model export']}


def select_role_positions(role, outer, groups):
    """Derive original roles. Tests use synthetic groups; real use requires explicit execution."""
    import numpy as np
    from sklearn.model_selection import GroupKFold
    spec = role_contract(role)
    if spec['fold'] is None:
        pool = np.arange(len(groups))
    else:
        pool = np.asarray(outer['train'])
    if spec['inner'] is not None:
        a, b = list(GroupKFold(3).split(pool, groups=np.asarray(groups)[pool]))[spec['inner']]
        fit, query = pool[a], pool[b]
    elif role == 'full_final':
        fit, query = pool, pool.copy()
    else:
        fit, query = pool, np.asarray(outer['validation'])
    if role != 'full_final' and np.intersect1d(np.asarray(groups)[fit], np.asarray(groups)[query]).size:
        raise ValueError('Fit/query component overlap')
    return fit, query
