"""Single-thread server workers for joint lexical/semantic/repository heads."""
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from skillcomp.capability_search import repo_features, joint_features
from skillcomp.models import estimator, continuous_scores
from skillcomp.runtime import json_write
from skillcomp.technical_text import preservation_vectorizer

DATA = None


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def initialize(root, manifest):
    global DATA
    threadpool_limits(1)
    root = Path(root)
    data = np.load(root / 'cache/workshop_v1/dataset.npz', allow_pickle=True)
    feature = json.loads((root / manifest).read_text())['primary_archive']
    DATA = {'root': root, 'y': data['y'], 'repos': data['repos'],
            'embeddings': np.load(root / feature['path'], mmap_mode='r')}


def prepare(job):
    root = DATA['root']; folder = root / job['folder']
    if (folder / 'prepared.json').is_file():
        record = json.loads((folder / 'prepared.json').read_text())
        assert all(record[key] == value for key, value in job.items())
        assert digest(root / job['role_path']) == record['role_sha256']
        return record
    folder.mkdir(parents=True)
    role = np.load(root / job['role_path'])
    train = role['train']; query = role[job['query_key']]
    texts = np.array(json.loads((root / 'cache/input_preservation_v1/raw_texts.json').read_text()))
    if job['inner']:
        source = root / job['vectorizer_source']; saved = joblib.load(source)
        assert np.array_equal(saved['training_positions'], train)
        vec = saved['vectorizer']; reference = {'path': job['vectorizer_source'], 'sha256': digest(source)}
    else:
        vec = preservation_vectorizer('F3_raw_word_char'); vec.fit(texts[train]); reference = None
    sparse.save_npz(folder / 'lexical.npz', vec.transform(texts).tocsr())
    joblib.dump(vec, folder / 'vectorizer.joblib')
    np.savez_compressed(folder / 'roles.npz', train=train, query=query)
    record = job | {'rows_fit': len(train), 'rows_query': len(query), 'role_sha256': digest(root / job['role_path']),
                    'vectorizer_reference': reference, 'vocabulary_fit_only': True}
    json_write(folder / 'prepared.json', record)
    return record


def matrix(folder, spec, train):
    lexical = sparse.load_npz(folder / 'lexical.npz') if spec['lexical'] else None
    repository = repo_features(DATA['repos'][train], DATA['repos'])
    return joint_features(lexical, DATA['embeddings'], repository, spec)


def fit(job):
    started = time.monotonic(); root = DATA['root']; folder = root / job['folder']
    roles = np.load(folder / 'roles.npz'); train, query = roles['train'], roles['query']
    destination = folder / job['spec']['name']
    if (destination / 'fit.json').is_file():
        record = json.loads((destination / 'fit.json').read_text())
        assert digest(destination / 'model.joblib') == record['model_sha256'] and digest(destination / 'scores.npz') == record['score_sha256']
        return record
    destination.mkdir()
    x = matrix(folder, job['spec'], train)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'Label .* present in all training examples\.', category=UserWarning)
        label_jobs = job.get('parallel_labels', 1)
        with joblib.parallel_backend('threading', n_jobs=label_jobs):
            model = estimator('lr', c=job['spec']['C'], jobs=label_jobs); model.fit(x[train], DATA['y'][train])
        model.set_params(n_jobs=1)
    scores = continuous_scores(model, x[query], DATA['y'][train], 'lr')
    np.savez_compressed(destination / 'scores.npz', positions=query, scores=scores)
    joblib.dump({'model': model, 'training_positions': train, 'spec': job['spec']}, destination / 'model.joblib')
    result = {'context': job['context'], 'candidate': job['spec']['name'], 'folder': str(destination.relative_to(root)),
              'model_sha256': digest(destination / 'model.joblib'), 'score_sha256': digest(destination / 'scores.npz'),
              'dimensions': x.shape[1], 'fit_rows': len(train), 'query_rows': len(query),
              'runtime_seconds': time.monotonic() - started, 'parallel_independent_label_fits': label_jobs,
              'numeric_library_threads_per_label': 1}
    json_write(destination / 'fit.json', result)
    return result


def predict(job):
    root = DATA['root']; folder = root / job['folder']; train = np.load(folder / 'roles.npz')['train']
    saved = joblib.load(folder / job['spec']['name'] / 'model.joblib')
    assert np.array_equal(saved['training_positions'], train)
    roles = np.load(root / job['outer_role']); test = roles['test']
    x = matrix(folder, job['spec'], train)
    scores = continuous_scores(saved['model'], x[test], DATA['y'][train], 'lr')
    path = folder / job['spec']['name'] / 'test_scores.npz'
    np.savez_compressed(path, positions=test, scores=scores)
    return {'context': job['context'], 'path': str(path.relative_to(root)), 'sha256': digest(path)}
