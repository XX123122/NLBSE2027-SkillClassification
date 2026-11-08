"""Rebuild data/lineage/component folds from pinned SQLite, then compare arrays."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import traceback
from urllib.parse import unquote, urlparse, parse_qs

import numpy as np
import pandas as pd

from skillcomp.benchmark import make_folds
from skillcomp.component_targets import align_targets
from skillcomp.data import quote_identifier
from skillcomp.runtime import ROOT, require_server, official_module, json_write, status

ACTIVE_REBUILD = False


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rebuild(args, out):
    global ACTIVE_REBUILD
    started = time.monotonic()
    source = ROOT / 'data/raw/skillscope_data.db'
    expected_source = 'ed47eb3c2173eb137c6146df78e750c261ff2a9f42f631c8e0186fa4c94022ea'
    assert sha(source) == expected_source
    official = official_module()
    inputs = set()
    denied = []

    def guard(event, values):
        if not ACTIVE_REBUILD:
            return
        if event.startswith('subprocess.') or event in ['socket.connect', 'os.system']:
            raise PermissionError('No network/subprocess during data-root rebuild')
        if event == 'sqlite3.connect':
            name = str(values[0])
            if name == ':memory:':
                return
            parsed = urlparse(name)
            assert Path(unquote(parsed.path)).resolve() == source.resolve() and parse_qs(parsed.query).get('mode') == ['ro']
            inputs.add('data/raw/skillscope_data.db')
        if event != 'open' or isinstance(values[0], int):
            return
        name, mode, flags = values[:3]
        path = Path(os.fsdecode(name)).resolve()
        write = isinstance(mode, str) and any(c in mode for c in 'wax+')
        write |= isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if path.is_relative_to(out):
            return
        if write:
            raise PermissionError('Rebuild may only write its unique output directory')
        if path.is_relative_to(ROOT):
            relative = path.relative_to(ROOT).as_posix()
            if path == source or relative.startswith(('src/', 'scripts/', 'upstream/')) and path.suffix in ['.py', '.pyc']:
                inputs.add(relative)
            else:
                denied.append(relative)
                raise PermissionError('Historical cache/model/metadata denied before root generation: ' + relative)

    sys.addaudithook(guard)
    ACTIVE_REBUILD = True
    try:
        np.load(ROOT / 'cache/workshop_v1/dataset.npz')
    except PermissionError:
        pass
    else:
        raise AssertionError('Historical-cache guard self-check failed')
    expected_self_check = denied.pop()
    candidate = out / 'candidate'
    candidate.mkdir()
    # Same SELECT and preparation as the reference loader, with explicit read-only access.
    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as connection:
        raw = pd.read_sql_query('SELECT * FROM ' + quote_identifier(official.COMPETITION_TABLE), connection)
    prepared, labels = official.prepare_dataset(raw, None)
    y = prepared[labels].to_numpy(dtype=np.int8)
    repos = raw.loc[prepared.index, 'Repo Name'].to_numpy(dtype=str)
    positions = prepared.index.to_numpy()
    texts = (prepared['issue text'] + ' ' + prepared['issue description']).to_numpy()
    json_write(candidate / 'labels.json', labels)
    np.savez_compressed(candidate / 'dataset.npz', y=y, repos=repos, source_positions=positions)
    folds, groups = make_folds(texts, repos, 'grouped5_pr_components', pr_numbers=raw['PR #'].to_numpy())
    np.save(candidate / 'key_component_groups.npy', groups)
    for fold in folds:
        np.savez_compressed(candidate / (fold['name'] + '.npz'), **{k: fold[k] for k in ['train', 'validation', 'test']})
    # Reconstruct the repository-complete companion using only the pinned DB.
    con = sqlite3.connect(':memory:', uri=True)
    try:
        con.execute('ATTACH DATABASE ? AS official', (source.as_uri() + '?mode=ro',))
        helper = con.execute("SELECT sql FROM official.sqlite_master WHERE name='vw_classification_helper'").fetchone()[0]
        assert 'a.pullNumber = b.pullNumber' in helper and 'a.repoNum = b.repoNum' not in helper
        changed = helper.replace('CREATE VIEW vw_classification_helper AS', 'CREATE TEMP VIEW root_rebuild_helper AS', 1)
        changed = changed.replace('a.pullNumber = b.pullNumber', 'a.pullNumber = b.pullNumber AND a.repoNum = b.repoNum AND b.repoNum = c.repoNum', 1)
        con.execute(changed)
        select = ','.join(f'SUM({quote_identifier(label)}) AS {quote_identifier(label)}' for label in labels)
        con.execute('CREATE TABLE companion_counts AS SELECT repoNum,pullNumber,' + select + ' FROM root_rebuild_helper GROUP BY repoNum,pullNumber')
        corrected = pd.read_sql_query('SELECT r.repo_name AS "Repo Name",c.* FROM companion_counts c JOIN official.repositories r ON c.repoNum=r.repoNum', con).rename(columns={'pullNumber': 'PR #'})
    finally:
        con.close()
    aligned = raw[['Repo Name', 'PR #']].merge(corrected, on=['Repo Name', 'PR #'], how='left', validate='one_to_one', indicator=True)
    matched = aligned['_merge'].eq('both').to_numpy()
    original_y = (raw[labels].to_numpy() > 0).astype(np.int8)
    companion = (aligned[labels].fillna(0).to_numpy() > 0).astype(np.int8)
    np.savez_compressed(candidate / 'aligned_targets.npz', y=companion, matched=matched, original_y=original_y,
                        repos=raw['Repo Name'].to_numpy(dtype=str), pr_number=raw['PR #'].to_numpy())
    with np.load(candidate / 'dataset.npz', allow_pickle=True) as data, np.load(candidate / 'aligned_targets.npz') as line:
        targets = align_targets(data, line)
        auxiliary = np.zeros_like(y)
        auxiliary[targets['matched']] = targets['companion'][targets['matched']]
        np.savez_compressed(candidate / 'auxiliary_targets.npz', targets=auxiliary, matched=targets['matched'])
    assert not denied
    generation_seconds = time.monotonic() - started
    manifest = {'generation_completed_before_reference_arrays_opened': True,
                'pinned_SQLite_sha256': expected_source, 'historical_cache_denial_self_check': expected_self_check,
                'declared_reads_during_generation': sorted(inputs),
                'candidate_files_sha256': {f.name: sha(f) for f in candidate.iterdir() if f.is_file()},
                'generation_seconds': generation_seconds}
    json_write(out / 'generation_locked.json', manifest)
    ACTIVE_REBUILD = False
    return manifest
