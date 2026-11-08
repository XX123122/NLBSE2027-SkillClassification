import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from skillcomp.reproduction_plan import ROLES, role_contract, select_role_positions, stage_plan

ROOT = Path(__file__).resolve().parents[1]


def test_all24_roles_have_explicit_scope_and_commands():
    assert len(ROLES) == 24 and len(set(ROLES)) == 24
    for role in ROLES:
        plan = stage_plan(role)
        assert not plan['full_chain_complete']
        assert plan['role']['encoder_epoch'] == (4 if role.startswith('full') else 8)
        assert len(plan['stages']) == 9
        for stage in plan['stages']:
            assert (ROOT / stage['command'][1]).is_file()
            assert not stage['portable_real_execution_this_round']
        if role.startswith('full'):
            assert all(s['execution_blocker'] for s in plan['stages'] if s['stage'] != 'prepare')


@pytest.mark.parametrize('fold', range(5))
def test_synthetic_inner_groups_disjoint_and_partition_query(fold):
    groups = np.repeat(np.arange(15), 2)
    outer = {'train': np.arange(24), 'validation': np.arange(24, 28)}
    held = []
    for k in range(3):
        fit, query = select_role_positions(f'fold{fold}_inner{k}', outer, groups)
        assert not np.intersect1d(groups[fit], groups[query]).size
        assert set(fit) | set(query) == set(outer['train'])
        held.extend(query)
    assert sorted(held) == list(outer['train'])


def test_synthetic_full_final_overlap_is_named_deployment_only():
    groups = np.arange(12)
    fit, query = select_role_positions('full_final', {}, groups)
    np.testing.assert_array_equal(fit, query)
    assert role_contract('full_final')['query_role'] == 'training_in_sample_for_deployment'
    assert not role_contract('full_final')['query_is_independent_test']


def test_unknown_role_rejected():
    with pytest.raises(ValueError): stage_plan('official_test')


def test_plan_and_blocked_tuning_need_no_assets_or_encoders(tmp_path):
    command = [sys.executable, str(ROOT / 'scripts/reproduce.py'), '--role', 'fold0_inner0']
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['role']['historically_rebuilt']
    failed = subprocess.run(command + ['--phase', 'tune', '--execute'], cwd=tmp_path, capture_output=True, text=True)
    assert failed.returncode != 0 and 'closure is incomplete' in failed.stderr
    assert not list(tmp_path.iterdir())
