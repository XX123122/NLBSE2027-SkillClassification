import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/manage_assets.py'
spec = importlib.util.spec_from_file_location('manage_assets', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_archive(tmp_path, unsafe=False):
    data = b'synthetic asset, never a real model'
    archive = tmp_path / 'fixture.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('root/assets/model.joblib', data)
        if unsafe: z.writestr('../escape', b'bad')
    manifest = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'archive_prefix': 'root/', 'assets': [{'files': [{'path': 'assets/model.joblib',
                    'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}]}]}
    return archive, manifest, data


def test_synthetic_verify_and_extract_without_deserialization(tmp_path):
    archive, manifest, data = fixture_archive(tmp_path)
    result = module.verify(archive, manifest, tmp_path / 'assets')
    assert result['state'] == 'passed' and not result['deserialized_or_forwarded']
    assert (tmp_path / 'assets/model.joblib').read_bytes() == data
    with pytest.raises(ValueError): module.verify(archive, manifest, tmp_path / 'assets')


def test_corruption_and_traversal_fail_closed(tmp_path):
    archive, manifest, _ = fixture_archive(tmp_path, unsafe=True)
    with pytest.raises(ValueError): module.verify(archive, manifest)
    manifest['archive_sha256'] = '0' * 64
    with pytest.raises(ValueError): module.verify(archive, manifest)
