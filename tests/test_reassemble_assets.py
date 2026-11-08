import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('reassemble_release_assets', Path(__file__).resolve().parents[1] / 'scripts/reassemble_release_assets.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def pieces(root):
    data = [b'only synthetic ', b'asset bytes, ', b'no executable weights']
    rows = []
    for i, raw in enumerate(data):
        name = f'asset.zip.part{i+1:03d}'; (root / name).write_bytes(raw)
        rows.append({'filename': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    whole = b''.join(data)
    return {'parts': rows, 'archive_bytes': len(whole), 'archive_sha256': hashlib.sha256(whole).hexdigest()}, whole


def test_parts_reassemble_byte_exact_and_refuse_overwrite(tmp_path):
    manifest, whole = pieces(tmp_path); dest = tmp_path / 'combined.zip'
    result = module.assemble(tmp_path, dest, manifest)
    assert result['state'] == 'passed' and dest.read_bytes() == whole
    assert not result['model_loaded_or_forwarded']
    with pytest.raises(FileExistsError): module.assemble(tmp_path, dest, manifest)


def test_corrupt_part_is_rejected_before_output(tmp_path):
    manifest, whole = pieces(tmp_path); part = tmp_path / manifest['parts'][0]['filename']; part.write_bytes(b'x' * part.stat().st_size)
    with pytest.raises(ValueError): module.assemble(tmp_path, tmp_path / 'combined.zip', manifest)
    assert not (tmp_path / 'combined.zip').exists() and not (tmp_path / 'combined.zip.part').exists()


def test_reordered_or_unsafe_parts_fail(tmp_path):
    manifest, whole = pieces(tmp_path); manifest['parts'].reverse()
    with pytest.raises(ValueError): module.assemble(tmp_path, tmp_path / 'combined.zip', manifest)
    assert not (tmp_path / 'combined.zip').exists()
    manifest['parts'][0]['filename'] = '../outside'
    with pytest.raises(ValueError): module.assemble(tmp_path, tmp_path / 'unsafe.zip', manifest)
