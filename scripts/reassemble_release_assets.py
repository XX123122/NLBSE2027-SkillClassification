"""Reassemble verified private Release parts without executing/deserializing models."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def assemble(parts_dir, output, manifest):
    source = Path(parts_dir).resolve(); target = Path(output).resolve()
    if target.exists(): raise FileExistsError('Refusing to replace an existing output')
    partial = target.with_name(target.name + '.part')
    if partial.exists(): raise FileExistsError('Incomplete output exists; inspect it before retrying')
    rows = manifest['parts']
    if not rows or len({row['filename'] for row in rows}) != len(rows): raise ValueError('Invalid part list')
    for row in rows:
        name = row['filename']
        if Path(name).name != name or '/' in name or '\\' in name: raise ValueError('Unsafe part filename')
        file = source / name
        if not file.is_file() or file.stat().st_size != row['bytes']: raise ValueError('Missing or wrong-sized part: ' + name)
        with file.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']: raise ValueError('Part hash mismatch: ' + name)
    combined = hashlib.sha256(); size = 0
    with partial.open('xb') as dest:
        for row in rows:
            with (source / row['filename']).open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    dest.write(block); combined.update(block); size += len(block)
    if combined.hexdigest() != manifest['archive_sha256'] or size != manifest['archive_bytes']:
        raise ValueError('Reassembled archive hash/size differs; partial retained for inspection')
    partial.rename(target)
    return {'state': 'passed', 'parts': len(rows), 'archive_sha256': combined.hexdigest(),
            'archive_bytes': size, 'model_loaded_or_forwarded': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts-dir', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', default=str(ROOT / 'assets/private_release_manifest.json'))
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    print(json.dumps(assemble(args.parts_dir, args.output, manifest), indent=2))


if __name__ == '__main__': main()
