"""Stateless encoding with a supervised encoder trained on exactly this context."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoConfig,AutoModel

from skillcomp.runtime import ROOT,require_server,json_write


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--mode',required=True);p.add_argument('--job',required=True);args=p.parse_args();require_server()
    out=ROOT/args.out;cfg=json.loads((out/'config.json').read_text());job=json.loads((out/args.job).read_text());folder=out/'contexts'/job['feature_context'];own=np.load(folder/'roles.npz');source=np.load(ROOT/job['encoder_roles'])
    np.testing.assert_array_equal(own['train'],source['train']);np.testing.assert_array_equal(own['query'],source['query'])
    checkpoint=ROOT/job['checkpoint'];record=json.loads(checkpoint.with_name('fit.json').read_text());assert record['candidate']==f'modern_e{job["epoch"]}'
    assert record.get('fit_rows',record.get('training_rows'))==len(own['train'])
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    config=AutoConfig.from_pretrained(ROOT/cfg['model'],local_files_only=True,trust_remote_code=False);config.reference_compile=False
    model=AutoModel.from_pretrained(ROOT/cfg['model'],config=config,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.float32,attn_implementation='sdpa')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);encoder={key[len('encoder.'):]:value for key,value in state.items() if key.startswith('encoder.')};model.load_state_dict(encoder,strict=True);del state,encoder
    model.eval().requires_grad_(False).to('cuda');tokens=ROOT/job['token_source'];ids=np.load(tokens/'token_ids.npy',mmap_mode='r');masks=np.load(tokens/'token_masks.npy',mmap_mode='r');parts=[];started=time.monotonic()
    with torch.inference_mode():
        for start in range(0,len(ids),32):
            token_ids=torch.tensor(ids[start:start+32],dtype=torch.long,device='cuda');mask=torch.tensor(masks[start:start+32],dtype=torch.long,device='cuda')
            hidden=model(input_ids=token_ids,attention_mask=mask).last_hidden_state.float();weights=mask.unsqueeze(-1).float();pooled=(hidden*weights).sum(1)/weights.sum(1).clamp_min(1)
            parts.append(torch.nn.functional.normalize(pooled,dim=1).cpu().numpy())
    values=np.concatenate(parts);assert values.shape==(7154,cfg['context_embedding_dimensions']) and np.isfinite(values).all();np.testing.assert_allclose(np.linalg.norm(values,axis=1),1,atol=2e-6)
    path=folder/cfg['context_embedding_pattern'].format(epoch=job['epoch']);assert not path.exists();np.save(path,values)
    metadata={'epoch':job['epoch'],'shape':list(values.shape),'dtype':str(values.dtype),'checkpoint':job['checkpoint'],'checkpoint_sha256':digest(checkpoint),
              'encoder_roles':job['encoder_roles'],'training_positions_sha256':hashlib.sha256(own['train'].tobytes()).hexdigest(),'feature_sha256':digest(path),
              'pooling':'masked mean, L2','encoder_dtype':'float32','rows_encoded':len(values),'new_labels_read':False,'encoder_updated_during_extraction':False,
              'runtime_seconds':time.monotonic()-started,'peak_GPU_GiB':torch.cuda.max_memory_allocated()/1024**3}
    json_write(path.with_suffix('.json'),metadata);print(json.dumps(metadata))


if __name__=='__main__':main()
