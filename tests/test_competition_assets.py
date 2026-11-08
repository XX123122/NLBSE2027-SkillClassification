import copy
import hashlib
import json

import pytest

from skillcomp.competition_assets import bind_assets, verify_frozen_bundle, REVISIONS, FROZEN_BUNDLE_SHA256


@pytest.fixture
def assets(tmp_path):
    config = {'schema_version': 1, 'bundle_sha256': FROZEN_BUNDLE_SHA256}
    for key in REVISIONS:
        folder = tmp_path / key
        folder.mkdir()
        (folder / 'config.json').write_text('{}')
        (folder / 'model.safetensors').write_bytes(b'synthetic fixture, not model weights')
        config[key] = key
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'synthetic fine-tuned fixture')
    config['codebert_checkpoint'] = checkpoint.name
    fitted = object()
    bundle = {'encoder': {'model': '/old/base', 'revision': REVISIONS['base_encoder']},
              'model': fitted,
              'residual_component': {'encoder': {'kind': 'feature_fusion', 'encoders': [
                  {'kind': 'learned_encoder', 'model': '/old/codebert', 'checkpoint': '/old/checkpoint',
                   'model_revision': REVISIONS['codebert_encoder'],
                   'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
                  {'kind': 'frozen_fields', 'model': '/old/qwen4', 'revision': REVISIONS['qwen4_encoder']}]}}}
    return bundle, config, tmp_path


def test_relocation_preserves_original_and_fitted_parameters(assets):
    bundle, config, root = assets
    original = copy.deepcopy(bundle['residual_component'])
    out = bind_assets(bundle, config, root)
    assert out['model'] is bundle['model']
    assert out['encoder']['model'] == str(root / 'base_encoder')
    assert out['residual_component']['encoder']['encoders'][0]['checkpoint'] == str(root / 'checkpoint.pt')
    assert bundle['residual_component'] == original
    assert bundle['encoder']['model'] == '/old/base'


@pytest.mark.parametrize('change', ['extra_key', 'wrong_bundle', 'missing_weight', 'wrong_checkpoint',
                                  'wrong_revision', 'wrong_order', 'empty_path'])
def test_incompatible_or_damaged_assets_fail_closed(assets, change):
    bundle, config, root = assets
    if change == 'extra_key': config['API labels'] = 'forbidden'
    if change == 'wrong_bundle': config['bundle_sha256'] = '0' * 64
    if change == 'missing_weight': (root / 'qwen4_encoder/model.safetensors').unlink()
    if change == 'wrong_checkpoint': (root / 'checkpoint.pt').write_bytes(b'changed')
    if change == 'wrong_revision': bundle['encoder']['revision'] = 'different'
    if change == 'wrong_order': bundle['residual_component']['encoder']['encoders'].reverse()
    if change == 'empty_path': config['base_encoder'] = ''
    with pytest.raises(ValueError): bind_assets(bundle, config, root)


def test_wrong_bundle_rejected_before_deserialization(tmp_path):
    path = tmp_path / 'untrusted.joblib'
    path.write_bytes(b'not a model')
    with pytest.raises(ValueError): verify_frozen_bundle(path)
