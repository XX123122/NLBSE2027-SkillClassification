"""Parameterized role preparation, derived from the historically verified single-role producer.
Only synthetic/static migration validation was performed for this adapter.
"""
import argparse
import os
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

import isolated_data_root
import title_body_workers as lexical
from skillcomp.runtime import ROOT, require_server, official_module, json_write
from skillcomp.technical_text import normalize_raw
from skillcomp.reproduction_plan import role_contract, select_role_positions

ROLE = os.environ.get("SKILLCOMP_ROLE", "fold0_inner0")
SPEC = role_contract(ROLE)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare():
    out = ROOT / 'outputs/experiments/data_root'
    out.mkdir()
    isolated_data_root.rebuild(None, out)
    generated = out / 'candidate'
    cache = ROOT / 'cache/workshop_v1'
    cache.mkdir(parents=True)
    for name in ['dataset.npz', 'labels.json', 'key_component_groups.npy']:
        shutil.copyfile(generated / name, cache / name)
    with np.load(cache / 'dataset.npz', allow_pickle=True) as data:
        y, repos, positions = data['y'], data['repos'], data['source_positions']
    groups = np.load(cache / 'key_component_groups.npy')
    if SPEC['fold'] is None:
        split_values = {}
    else:
        with np.load(generated / ('fold' + str(SPEC['fold']) + '.npz')) as split:
            split_values = {key: split[key] for key in split.files}
    fit, query = select_role_positions(ROLE, split_values, groups)
    with np.load(generated / 'auxiliary_targets.npz') as aux:
        known = fit[aux['matched'][fit].astype(bool)]
    if ROLE == "fold0_inner0":
        assert (len(fit), len(known), len(query)) == (3094, 1317, 1546)
    if ROLE != "full_final":
        assert not np.intersect1d(groups[fit], groups[query]).size
    role = cache / 'fixed_inner.npz'
    np.savez_compressed(role, train=fit, held=query)
    db = ROOT / 'data/raw/skillscope_data.db'
    official = official_module()
    with sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True) as con:
        raw = pd.read_sql_query('SELECT * FROM "' + official.COMPETITION_TABLE + '"', con).iloc[positions]
    original = {key: raw[column].tolist() for key, column in [('title', 'issue text'), ('body', 'issue description')]}
    normalized = {key: [normalize_raw(v) for v in values] for key, values in original.items()}
    records = [{'issue text': t, 'issue description': d, 'Repo Name': str(r)}
               for t, d, r in zip(original['title'], original['body'], repos)]
    json_write(ROOT / 'outputs/original_inputs.json', records)
    encoder = ROOT / 'outputs/experiments/encoder'
    encoder.mkdir()
    json_write(encoder / 'input_fields.json', normalized)
    shutil.copyfile(generated / 'auxiliary_targets.npz', encoder / 'auxiliary_targets.npz')
    cfg = json.loads((ROOT / 'configs/encoder_recipe.json').read_text())
    cfg['fields_path'] = 'outputs/experiments/encoder/input_fields.json'
    json_write(encoder / 'config.json', cfg)
    folder = encoder / ('contexts/' + ROLE)
    folder.mkdir(parents=True)
    np.savez_compressed(folder / 'roles.npz', train=known, context_train=fit, query=query)
    json_write(encoder / 'fit_job.json', {'context': ROLE, 'inner': SPEC['inner'] is not None, 'candidate': 'modern_e' + str(SPEC['encoder_epoch'])})
    base = ROOT / 'outputs/experiments/base'
    base.mkdir()
    config = json.loads((ROOT / 'configs/base_recipe.json').read_text())
    json_write(base / 'config.json', config)
    json_write(base / 'input_fields.json', normalized)
    # Existing worker code, no feature-cache reuse.
    lexical.DATA = {'root': ROOT, 'out': base, 'title': np.array(normalized['title']),
                    'body': np.array(normalized['body'])}
    lexical.prepare({'context': ROLE, 'role_path': 'cache/workshop_v1/fixed_inner.npz',
                     'query_key': 'held', 'config': config})
    roles = ROOT / ('outputs/experiments/features/contexts/' + ROLE)
    roles.mkdir(parents=True)
    shutil.copyfile(folder / 'roles.npz', roles / 'roles.npz')
    features = ROOT / 'outputs/experiments/features'
    json_write(features / 'config.json', {'model': cfg['model'], 'context_embedding_dimensions': 768,
               'context_embedding_pattern': 'learned_e{epoch}.npy'})
    json_write(features / 'extract_job.json', {'context': ROLE + '_encode', 'feature_context': ROLE,
               'epoch': SPEC['encoder_epoch'], 'checkpoint': 'outputs/experiments/encoder/contexts/' + ROLE + '/modern_e' + str(SPEC['encoder_epoch']) + '/model.pt',
               'encoder_roles': 'outputs/experiments/encoder/contexts/' + ROLE + '/roles.npz',
               'token_source': 'outputs/experiments/encoder'})
    paths = ['cache/workshop_v1/dataset.npz', 'cache/workshop_v1/key_component_groups.npy',
             'cache/workshop_v1/fixed_inner.npz', 'outputs/experiments/encoder/contexts/' + ROLE + '/roles.npz',
             'outputs/experiments/encoder/auxiliary_targets.npz', 'outputs/original_inputs.json',
             'outputs/experiments/encoder/input_fields.json', 'outputs/experiments/base/contexts/' + ROLE + '/title.npz',
             'outputs/experiments/base/contexts/' + ROLE + '/body.npz']
    json_write(ROOT / 'outputs/preparation_locked.json', {'state': 'prepared', 'role': ROLE,
               'fit_rows': len(fit), 'known_encoder_fit_rows': len(known), 'query_rows': len(query),
               'all_inputs_regenerated_from_pinned_database': True,
               'files_sha256': {p: sha(ROOT / p) for p in paths}})
    print(json.dumps({'state': 'prepared', 'fit_rows': len(fit), 'encoder_fit_rows': len(known), 'query_rows': len(query)}))


def fit_base():
    threadpool_limits(1)
    out = ROOT / 'outputs/experiments/base'
    lexical.initialize(ROOT, out)
    spec = {'name': 'split_hybrid1_c1', 'title_scale': 1, 'body': True, 'embedding_scale': 1, 'C': 1}
    record = lexical.fit({'context': ROLE, 'spec': spec, 'parallel_labels': 16})
    json_write(ROOT / 'outputs/base_fit_complete.json', record)
    print(json.dumps(record))


if __name__ == '__main__':
    require_server()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['prepare', 'fit-base'], required=True)
    args = p.parse_args()
    prepare() if args.stage == 'prepare' else fit_base()
