"""Static/local-file review only; never imports producer workers or loads models."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect(root=ROOT):
    errors = []
    python_files = list(root.rglob('*.py'))
    for file in python_files:
        try:
            tree = ast.parse(file.read_text(encoding='utf-8-sig'), filename=str(file))
            compile(tree, str(file), 'exec')
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f'Python syntax: {file.relative_to(root)}: {exc}')
    checked_links = 0
    def anchors(path):
        out = set()
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if line.startswith('#'):
                title = re.sub(r'^#+\s*', '', line).strip().lower()
                out.add(re.sub(r'[^\w\- ]', '', title).replace(' ', '-'))
        return out
    for file in root.rglob('*.md'):
        text = file.read_text(encoding='utf-8-sig')
        for raw in re.findall(r'\[[^\]\n]*\]\(([^\s)]+)(?:\s+[^)]*)?\)', text):
            url = urlsplit(raw.strip('<>'))
            if url.scheme or raw.startswith('//'): continue
            target = (file.parent / unquote(url.path)).resolve() if url.path else file.resolve()
            checked_links += 1
            if not target.is_relative_to(root.resolve()) or not target.exists():
                errors.append(f'Broken internal link: {file.relative_to(root)} -> {raw}')
            elif url.fragment and target.suffix == '.md' and unquote(url.fragment).lower() not in anchors(target):
                errors.append(f'Missing Markdown anchor: {file.relative_to(root)} -> {raw}')
    forbidden_suffixes = {'.joblib', '.pt', '.safetensors', '.npy', '.npz', '.db', '.zip', '.log'}
    private = re.compile(r'(?:(?<!\w)/home/[^\s\x22\x27]+|(?<!\w)/data/[^\s\x22\x27]+|[A-Za-z]:[\\/]project[\\/]|(?:\d{1,3}\.){3}\d{1,3}|GPU-[a-f0-9]{8}-)', re.I)
    # Do not make the scanner's own literal pattern a self-failure.
    for file in root.rglob('*'):
        if not file.is_file(): continue
        rel = file.relative_to(root).as_posix()
        if any(part in ['__pycache__', '.pytest_cache', '.git'] for part in file.parts): continue
        if file.suffix in forbidden_suffixes:
            errors.append('Unexpected data/model/archive/log: ' + rel)
        if file.suffix in {'.py', '.json', '.md', '.txt', '.tex'} and file.name != 'verify_repository.py':
            if private.search(file.read_text(encoding='utf-8-sig')):
                errors.append('Private execution identity: ' + rel)
    graph_file = root / 'docs/source_graph/python_dependencies.json'
    if graph_file.exists():
        graph = json.loads(graph_file.read_text(encoding='utf-8'))
        for edge in graph['local_import_edges']:
            if not (root / edge['target']).is_file(): errors.append('Missing local import: ' + str(edge))
        errors.extend(graph['unresolved_local_imports'])
    manifest = root / 'FILES_SHA256.json'
    if manifest.exists():
        for name, expected in json.loads(manifest.read_text(encoding='utf-8')).items():
            path = root / name
            if not path.is_file() or digest(path) != expected:
                errors.append('File manifest mismatch: ' + name)
    if errors:
        raise ValueError('\n'.join(errors))
    return {'state': 'passed', 'python_files_parsed_and_compiled': len(python_files),
            'internal_markdown_links_checked': checked_links,
            'private_path_and_asset_exclusion_scan': True,
            'real_models_imported_or_loaded': False}


if __name__ == '__main__':
    print(json.dumps(inspect(), indent=2))
