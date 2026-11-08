"""Verify/extract only declared assets from a private local ZIP; no model loading."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def verify(archive, manifest, destination=None):
    with Path(archive).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != manifest['archive_sha256']:
        raise ValueError('Archive SHA256 differs from attachment manifest')
    expected = {f['path']: f for asset in manifest['assets'] for f in asset['files']}
    prefix = manifest['archive_prefix']
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        for name in names:
            q = PurePosixPath(name)
            if q.is_absolute() or '..' in q.parts or '\\' in name:
                raise ValueError('Unsafe archive path')
        for member, record in expected.items():
            full = prefix + member
            if z.getinfo(full).file_size != record['bytes']:
                raise ValueError('Asset size mismatch: ' + member)
            with z.open(full) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != record['sha256']:
                raise ValueError('Asset hash mismatch: ' + member)
        if destination is not None:
            # Verify all members first. Refuse replacement of any existing file.
            root = Path(destination).resolve()
            targets = {member: root / PurePosixPath(member).relative_to('assets') for member in expected}
            for target in targets.values():
                if not target.resolve().is_relative_to(root) or target.exists():
                    raise ValueError('Extraction target escapes asset root or already exists')
            for member, target in targets.items():
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(prefix + member) as source, target.open('xb') as dest:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        dest.write(block)
                with target.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != expected[member]['sha256']:
                        raise ValueError('Extracted asset hash mismatch')
    return {'state': 'passed', 'archive_sha256': actual, 'asset_files': len(expected),
            'asset_member_SHA256_and_CRC': True, 'deserialized_or_forwarded': False,
            'extracted': destination is not None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True)
    p.add_argument('--manifest', default=str(ROOT / 'assets/attachment_manifest.json'),
                   help='Legacy handoff or private asset-only Release manifest')
    p.add_argument('--extract-to', help='Optional empty asset directory; existing files are never overwritten')
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    print(json.dumps(verify(args.archive, manifest, args.extract_to), indent=2))


if __name__ == '__main__': main()
