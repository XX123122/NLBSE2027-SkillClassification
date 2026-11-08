"""Relocate encoder assets without changing fitted candidate parameters."""
import hashlib
from pathlib import Path

FROZEN_BUNDLE_SHA256 = '44e367a74fd23c1a5538f655c18238fc6e085a6f9a96415552e8a08a2bb02acf'
REVISIONS = {
    'base_encoder': '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3',
    'codebert_encoder': '3b0952feddeffad0063f274080e3c23d75e7eb39',
    'qwen4_encoder': '5cf2132abc99cad020ac570b19d031efec650f2b',
}
ASSET_KEYS = {*REVISIONS, 'codebert_checkpoint'}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_frozen_bundle(path):
    # Authenticate the fixed local artifact before joblib deserialization.
    if sha256(path) != FROZEN_BUNDLE_SHA256:
        raise ValueError('Bundle does not match the frozen competition candidate')


def bind_assets(bundle, config, config_directory):
    expected = ASSET_KEYS | {'schema_version', 'bundle_sha256'}
    if set(config) != expected or config['schema_version'] != 1:
        raise ValueError('Asset map schema/keys differ from the candidate contract')
    if config['bundle_sha256'] != FROZEN_BUNDLE_SHA256:
        raise ValueError('Asset map names a different candidate bundle')
    auxiliary = bundle.get('residual_component', {}).get('encoder', {})
    members = auxiliary.get('encoders', [])
    if auxiliary.get('kind') != 'feature_fusion' or len(members) != 2:
        raise ValueError('Exactly the frozen CodeBERT and Qwen4 fusion is required')
    if (bundle['encoder'].get('revision') != REVISIONS['base_encoder']
            or members[0].get('model_revision') != REVISIONS['codebert_encoder']
            or members[0].get('kind') != 'learned_encoder'
            or members[1].get('revision') != REVISIONS['qwen4_encoder']
            or members[1].get('kind') != 'frozen_fields'):
        raise ValueError('Encoder identity/order differs from the frozen candidate')
    paths = {}
    for key in ASSET_KEYS:
        value = config[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Each asset path must be a nonempty string')
        path = Path(value)
        paths[key] = (Path(config_directory) / path).resolve() if not path.is_absolute() else path.resolve()
    for key in REVISIONS:
        folder = paths[key]
        if (not folder.is_dir() or not (folder / 'config.json').is_file()
                or not any(folder.glob('*.safetensors'))):
            raise ValueError('Missing local config or safetensors for ' + key)
    checkpoint = paths['codebert_checkpoint']
    if not checkpoint.is_file() or sha256(checkpoint) != members[0]['checkpoint_sha256']:
        raise ValueError('Fine-tuned CodeBERT checkpoint hash mismatch')
    # Copy only configuration dictionaries. All fitted estimators/arrays stay identical.
    out = dict(bundle)
    out['encoder'] = dict(bundle['encoder'], model=str(paths['base_encoder']))
    relocated = [dict(members[0], model=str(paths['codebert_encoder']), checkpoint=str(checkpoint)),
                 dict(members[1], model=str(paths['qwen4_encoder']))]
    out['residual_component'] = dict(bundle['residual_component'],
                                     encoder=dict(auxiliary, encoders=relocated))
    return out
