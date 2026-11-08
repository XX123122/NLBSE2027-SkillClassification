"""Bind context-specific supervised embeddings to their own fit role and epoch."""
import hashlib
import json
from pathlib import Path
import numpy as np


def load_context_embeddings(folder,settings,epoch,rows):
    folder=Path(folder);path=folder/settings['context_embedding_pattern'].format(epoch=epoch)
    metadata=json.loads(path.with_suffix('.json').read_text());role=np.load(folder/'roles.npz')
    if metadata['epoch']!=epoch or metadata['training_positions_sha256']!=hashlib.sha256(role['train'].tobytes()).hexdigest():raise ValueError('Embedding encoder fit role/epoch mismatch')
    with path.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=metadata['feature_sha256']:raise ValueError('Learned feature content digest mismatch')
    values=np.load(path,mmap_mode='r')
    if values.shape!=(rows,settings['context_embedding_dimensions']) or not np.isfinite(values).all():raise ValueError('Invalid learned feature array')
    return values
