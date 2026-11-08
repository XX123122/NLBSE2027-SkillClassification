import pytest
from skillcomp.frozen_inference import validate_records
from skillcomp.competition_io import validate_label_order


def test_known_unknown_and_blank_inputs_are_valid():
    validate_records([])
    validate_records([{'issue text': '', 'issue description': '', 'Repo Name': ''},
                      {'issue text': 'Synthetic test', 'issue description': 'Fixture only', 'Repo Name': 'never_seen_repo'}])


@pytest.mark.parametrize('key', ['PR #', 'PR ID', 'patch', 'query_labels', 'labels', 'database'])
def test_forbidden_query_information_is_rejected(key):
    with pytest.raises(ValueError):
        validate_records([{'issue text': '', 'issue description': '', 'Repo Name': '', key: 'forbidden'}])


def test_non_string_rejected():
    with pytest.raises(ValueError):
        validate_records([{'issue text': '', 'issue description': None, 'Repo Name': ''}])


def test_217_label_order_is_preserved_and_invalid_bundle_order_rejected():
    labels = [f'synthetic_label_{i}' for i in range(217)]
    assert validate_label_order(labels) is labels
    for invalid in [labels[:216], labels + ['extra'], ['duplicate'] * 217]:
        with pytest.raises(ValueError): validate_label_order(invalid)
