"""Encode new issue text with the full-data supervised auxiliary encoder."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from skillcomp.runtime import require_server


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--settings', required=True); parser.add_argument('--input', required=True); parser.add_argument('--out', required=True); args = parser.parse_args(); require_server()
    cfg = json.loads(Path(args.settings).read_text()); records = json.loads(Path(args.input).read_text()); assert records
    checkpoint = Path(cfg['checkpoint'])
    with checkpoint.open('rb') as stream: assert hashlib.file_digest(stream, 'sha256').hexdigest() == cfg['checkpoint_sha256']
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    config = AutoConfig.from_pretrained(cfg['model'], local_files_only=True, trust_remote_code=False); config.reference_compile = False
    model = AutoModel.from_pretrained(cfg['model'], config=config, local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.float32, attn_implementation='sdpa')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True); model.load_state_dict({k[len('encoder.'):]: v for k, v in state.items() if k.startswith('encoder.')}, strict=True); del state
    model.eval().requires_grad_(False).to('cuda'); tokenizer = AutoTokenizer.from_pretrained(cfg['model'], local_files_only=True, trust_remote_code=False, padding_side='right')
    normalize = lambda text: ' '.join(text.split())
    texts = [f"Title: {normalize(row['issue text'])}\nDescription: {normalize(row['issue description'])}" for row in records]
    parts = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            tokens = tokenizer(texts[start:start + 32], padding='max_length', truncation=True, max_length=512, return_tensors='pt').to('cuda'); hidden = model(**tokens).last_hidden_state.float(); weights = tokens['attention_mask'].unsqueeze(-1).float()
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1); parts.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
    values = np.concatenate(parts); assert values.shape == (len(records), 768) and np.isfinite(values).all(); np.save(args.out, values)


if __name__ == '__main__': main()
