"""Check saved aggregate evidence only; no data, weights or new scoring."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    a = json.loads((ROOT / 'results/metrics.json').read_text(encoding='utf-8'))
    expected = {'precision': .6860585657247842, 'recall': .5562402479871845, 'micro_f1': .6143664994611586}
    assert a['methods']['best_policy'] == expected
    assert a['full_training_OOF']['micro_f1'] == .6095585841201916
    assert a['full_weight_file_independent_test'] is None and a['official_qualification'] is None
    evidence = ROOT / 'results/evidence'
    assert json.loads((evidence / 'one_role_comparison.json').read_text(encoding='utf-8'))['full_training_chain_complete'] is False
    active = json.loads((evidence / 'one_role_active_policy.json').read_text(encoding='utf-8'))
    assert active['state'] == 'one_role_active_policy_rebuild_passed'
    assert active['candidate_score_max_abs'] <= 1e-5
    assert json.loads((evidence / 'isolated_inference.json').read_text(encoding='utf-8'))['all217_outputs_cutoff_labels_verified']
    print(json.dumps({'state': 'passed', 'saved_aggregate_consistency_only': True, 'new_scores': False}))


if __name__ == '__main__': main()
