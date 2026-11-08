"""Predict proxy skills on new text using a dedicated full-data encoder/head."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from modernbert_full_worker import Classifier
from skillcomp.hierarchical_loss import hierarchy_marginals
from skillcomp.runtime import require_server


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--settings',required=True);parser.add_argument('--input',required=True);parser.add_argument('--out',required=True);args=parser.parse_args();require_server();cfg=json.loads(Path(args.settings).read_text());records=json.loads(Path(args.input).read_text());assert records
    checkpoint=Path(cfg['checkpoint'])
    with checkpoint.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==cfg['checkpoint_sha256']
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=Classifier(cfg,np.full(217,.5));model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False).to('cuda');tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True,trust_remote_code=False,padding_side='right')
    normalize=lambda text:' '.join(text.split()) if cfg['whitespace']=='normalized' else text;texts=[f"Title: {normalize(row['issue text'])}\nDescription: {normalize(row['issue description'])}" for row in records];parts=[]
    with torch.inference_mode():
        for start in range(0,len(texts),16):
            tokens=tokenizer(texts[start:start+16],padding='max_length',truncation=True,max_length=cfg['max_length'],return_tensors='pt').to('cuda');conditional=model(input_ids=tokens['input_ids'],attention_mask=tokens['attention_mask']).float().sigmoid();probability=hierarchy_marginals(conditional,cfg['hierarchy_parents']) if 'hierarchy_parents' in cfg else conditional;parts.append(probability.cpu().numpy())
    values=np.concatenate(parts);assert values.shape==(len(records),217) and np.isfinite(values).all();np.save(args.out,values)


if __name__=='__main__':main()
