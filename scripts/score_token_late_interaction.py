"""Read-only pinned Qwen token features and pair scores; no target files read."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer

from skillcomp.token_late_interaction import document_frequencies,idf_values
from skillcomp.runtime import require_server


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main(args):
    require_server();out=Path(args.out);cfg=json.loads((out/'config.json').read_text());job=json.loads((out/args.job).read_text());folder=out/'contexts'/job['context'];texts=json.loads((out/'encoder_inputs.json').read_text());fit=np.load(job['idf_fit_positions'])['positions'];pairs=np.load(job['pair_positions']);queries,documents=pairs['queries'],pairs['documents'];replay=job.get('replay_only',False);selected=np.asarray(job['replay_positions'],int) if replay else np.unique(np.r_[queries,documents]);source=Path(cfg['model']);verified=json.loads((source/'source_verified.json').read_text());assert verified['all_files_verified'] and verified['revision']=='5cf2132abc99cad020ac570b19d031efec650f2b';assert len(texts)==7154;started=time.monotonic();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False,padding_side='left');special=set(tokenizer.all_special_ids);encoded={}
    for pos in np.unique(np.r_[fit,selected]):encoded[int(pos)]=tokenizer.encode(texts[int(pos)],add_special_tokens=True,truncation=True,max_length=cfg['max_length'])
    counts=document_frequencies([encoded[int(pos)] for pos in fit]);(folder/'idf.json').write_text(json.dumps({'documents':len(fit),'counts':{str(k):v for k,v in sorted(counts.items())}},sort_keys=True)+'\n');eligible={}
    for ids in encoded.values():
        for token in ids:
            if token not in eligible:eligible[token]=token not in special and any(c.isalnum() for c in tokenizer.decode([token],skip_special_tokens=False))
    ids=[np.asarray([token for token in encoded[int(pos)] if eligible[token]],np.int64) for pos in selected];offsets=np.r_[0,np.cumsum([len(a) for a in ids])];all_ids=np.concatenate(ids);weights=idf_values(all_ids,counts,len(fit));np.savez_compressed(folder/'token_index.npz',positions=selected,offsets=offsets,ids=all_ids,weights=weights);model,loading=AutoModel.from_pretrained(source,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True);assert not loading['missing_keys'] and not loading['unexpected_keys'];model=model.eval().requires_grad_(False).to('cuda');dimensions=model.config.hidden_size;cache=np.lib.format.open_memmap(folder/'token_vectors.npy',mode='w+',dtype=np.float32,shape=(int(offsets[-1]),dimensions));maximum_repeat=0.
    def check_budget():
        if time.monotonic()-started>cfg['maximum_inference_seconds']:raise TimeoutError('Fixed token encoding+scoring time budget')
        if torch.cuda.max_memory_allocated()/1024**3>cfg['hard_GPU_allocated_GiB']:raise MemoryError('Fixed token feature memory budget')
    def encode(pos):
        tokens=encoded[int(pos)];pad=cfg['max_length']-len(tokens);input_ids=torch.tensor([[tokenizer.pad_token_id]*pad+tokens],device='cuda');mask=torch.tensor([[0]*pad+[1]*len(tokens)],device='cuda');positions=mask.cumsum(-1)-1;positions.masked_fill_(mask==0,0);hidden=model(input_ids=input_ids,attention_mask=mask,position_ids=positions,use_cache=False).last_hidden_state[0,pad:];active=torch.tensor([eligible[token] for token in tokens],device='cuda',dtype=torch.bool);return torch.nn.functional.normalize(hidden[active].float(),dim=1).cpu().numpy()
    with torch.inference_mode():
        for i,pos in enumerate(selected):
            values=encode(pos);assert values.shape==(len(ids[i]),dimensions) and np.isfinite(values).all();cache[offsets[i]:offsets[i+1]]=values
            if i<4:
                repeated=encode(pos);maximum_repeat=max(maximum_repeat,float(np.abs(values-repeated).max()) if len(values) else 0.);assert maximum_repeat==0
            if i%128==0:cache.flush();print(json.dumps({'encoded_documents':i+1,'total':len(selected),'seconds':time.monotonic()-started}),flush=True)
            check_budget()
        cache.flush();lookup={int(pos):i for i,pos in enumerate(selected)};scores=[];valid=[]
        if not replay:
            current=None;q=None;qw=None
            for i,(query,doc) in enumerate(zip(queries,documents)):
                if int(query)!=current:
                    j=lookup[int(query)];a,b=offsets[j:j+2];q=torch.tensor(np.asarray(cache[a:b]),device='cuda');qw=torch.tensor(weights[a:b],device='cuda',dtype=torch.float32);current=int(query)
                j=lookup[int(doc)];a,b=offsets[j:j+2];d=torch.tensor(np.asarray(cache[a:b]),device='cuda');ok=bool(len(q) and len(d));value=float(((q@d.T).max(1).values*qw).sum().item()/qw.sum().item()) if ok else 0.;scores.append(value);valid.append(ok)
                if i%512==0:print(json.dumps({'scored_pairs':i+1,'total':len(queries),'seconds':time.monotonic()-started}),flush=True)
                check_budget()
            np.savez_compressed(folder/'pair_scores.npz',queries=queries,documents=documents,token_score=np.asarray(scores),valid=np.asarray(valid,bool))
    metadata={'state':'complete','replay_only':replay,'documents':len(selected),'contextual_tokens':int(offsets[-1]),'dimensions':dimensions,'actual_parameters':sum(p.numel() for p in model.parameters()),'pairs_scored':len(scores),'all_weights_frozen':True,'strict_safe_loading':True,'targets_loaded':False,'idf_fit_documents':len(fit),'first4_repeat_max_abs':maximum_repeat,'padding':'fixed512left','position_ids':'maskcumulative','dtype':'BF16backbone/FP32normalizedtokens/FP32GPUdot','runtime_seconds':time.monotonic()-started,'peak_GPU_GiB':torch.cuda.max_memory_allocated()/1024**3,'token_index_sha256':sha(folder/'token_index.npz'),'idf_sha256':sha(folder/'idf.json')};(folder/'fit.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--job',required=True);main(p.parse_args())
