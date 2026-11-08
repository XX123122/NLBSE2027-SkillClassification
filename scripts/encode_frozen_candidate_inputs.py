"""Stateless online field encoding for a frozen skill classifier bundle."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer

from skillcomp.runtime import require_server


def main():
    p=argparse.ArgumentParser();p.add_argument('--settings',required=True);p.add_argument('--input',required=True);p.add_argument('--out',required=True);args=p.parse_args();require_server()
    cfg=json.loads(Path(args.settings).read_text());records=json.loads(Path(args.input).read_text());assert records
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True,trust_remote_code=False,padding_side='left')
    dtype=torch.float32 if cfg['dtype']=='float32' else torch.bfloat16;model=AutoModel.from_pretrained(cfg['model'],local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=dtype,attn_implementation='sdpa').eval().requires_grad_(False).to('cuda')
    arrays=[]
    with torch.inference_mode():
        for field in ['issue text','issue description']:
            parts=[]
            for start in range(0,len(records),cfg['batch_size']):
                texts=[row[field] for row in records[start:start+cfg['batch_size']]]
                if cfg['whitespace']=='normalized':texts=[' '.join(text.split()) for text in texts]
                tokens=tokenizer([cfg['instruction']+text for text in texts],truncation=True,max_length=512,padding='max_length',return_tensors='pt').to('cuda');positions=tokens['attention_mask'].cumsum(-1)-1;positions.masked_fill_(tokens['attention_mask']==0,0);tokens['position_ids']=positions
                hidden=model(**tokens,use_cache=False).last_hidden_state[:,-1].float();parts.append(torch.nn.functional.normalize(hidden,dim=1).cpu().numpy())
            arrays.append(np.concatenate(parts))
    values=np.concatenate(arrays,axis=1)/np.float32(np.sqrt(2));assert len(values)==len(records) and np.isfinite(values).all();np.save(args.out,values)


if __name__=='__main__':main()
