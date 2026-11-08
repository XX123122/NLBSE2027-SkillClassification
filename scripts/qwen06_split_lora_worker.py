"""Own multi-label LoRA on separately encoded issue title and body, with warm heads."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoModel,AutoTokenizer

from skillcomp.compact_lora import inject_last_layers
from skillcomp.runtime import ROOT,require_server,json_write


class SplitClassifier(nn.Module):
    def __init__(self,cfg,initial):
        super().__init__()
        dtype=torch.bfloat16 if cfg.get('backbone_dtype')=='bfloat16' else torch.float32;self.sequential_fields=cfg.get('sequential_field_encoding',False)
        self.backbone=AutoModel.from_pretrained(ROOT/cfg['model'],local_files_only=True,trust_remote_code=False,
                      use_safetensors=True,dtype=dtype,attn_implementation='sdpa').requires_grad_(False)
        self.backbone.config.use_cache=False;self.targets=inject_last_layers(self.backbone,cfg['lora'])
        self.head=warm_head(initial)

    def forward(self,tokens):
        if self.sequential_fields:hidden=torch.cat([self.backbone(**{key:value[index:index+1] for key,value in tokens.items()},use_cache=False).last_hidden_state[:,-1].float() for index in range(len(tokens['input_ids']))])
        else:hidden=self.backbone(**tokens,use_cache=False).last_hidden_state[:,-1].float()
        normalized=nn.functional.normalize(hidden,dim=1);n=len(normalized)//2
        values=nn.functional.normalize(torch.cat([normalized[:n],normalized[n:]],dim=1),dim=1)
        return self.head(values)

    def adaptable(self):
        return {key:value.detach().cpu() for key,value in self.named_parameters() if value.requires_grad}

    def load_adaptable(self,state):
        parameters={key:value for key,value in self.named_parameters() if value.requires_grad}
        assert set(parameters)==set(state)
        with torch.no_grad():
            for key,value in state.items():parameters[key].copy_(value.to(parameters[key].device))


def warm_head(initial):
    head=nn.Linear(initial['coef'].shape[1],initial['coef'].shape[0])
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(initial['coef']).float());head.bias.copy_(torch.from_numpy(initial['intercept']).float())
    return head


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--mode',choices=['tokenize','fit','predict','smoke'],required=True);p.add_argument('--job');args=p.parse_args();require_server()
    out=ROOT/args.out;cfg=json.loads((out/'config.json').read_text());torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    if args.mode=='tokenize':
        fields=json.loads((ROOT/cfg['fields_path']).read_text());manifest=json.loads((ROOT/cfg['feature_manifest']).read_text())
        instruction=cfg.get('instruction') or manifest['archives']['title']['metadata']['instruction'];tokenizer=AutoTokenizer.from_pretrained(ROOT/cfg['model'],local_files_only=True,trust_remote_code=False,padding_side='left')
        all_ids=[];all_masks=[]
        for field in ['title','body']:
            tokenized=tokenizer([instruction+text for text in fields[field]],max_length=cfg['max_length'],truncation=True,padding='max_length',return_tensors='np')
            all_ids.append(tokenized['input_ids'].astype(np.int32));all_masks.append(tokenized['attention_mask'].astype(np.uint8))
        np.save(out/'token_ids.npy',np.stack(all_ids,axis=1));np.save(out/'token_masks.npy',np.stack(all_masks,axis=1))
        json_write(out/'tokenization.json',{'rows':len(fields['title']),'fields':2,'length_each':cfg['max_length'],'labels_used':False});return
    job=json.loads((out/args.job).read_text());folder=out/'contexts'/job['context'];roles=np.load(folder/'roles.npz');train,query=roles['train'],roles['query']
    if args.mode=='predict':
        assert (out/'all_calibrations_locked.json').is_file();query=roles['test']
    if args.mode=='smoke':train,query=train[:4],query[:4]
    if cfg.get('primary_targets'):
        source=np.load(out/cfg['primary_targets']);y=source['targets'];assert source['matched'][train].astype(bool).all() and np.isin(y[train],[0,1]).all()
    else:y=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['y']
    initial=np.load(folder/'initial_head.npz')
    np.testing.assert_array_equal(initial['train'],roles['train'])
    ids=np.load(out/'token_ids.npy',mmap_mode='r');masks=np.load(out/'token_masks.npy',mmap_mode='r')
    embeddings=torch.tensor(np.load(ROOT/cfg['embedding_path']),dtype=torch.float32,device='cuda');embeddings=nn.functional.normalize(embeddings,dim=1)
    target=torch.tensor(y[train],dtype=torch.float32,device='cuda');zero=initial['absent'];one=initial['universal']
    def batch(positions):
        values=np.concatenate([ids[positions,0],ids[positions,1]]);attention=np.concatenate([masks[positions,0],masks[positions,1]])
        token_ids=torch.tensor(values,dtype=torch.long,device='cuda');mask=torch.tensor(attention,dtype=torch.long,device='cuda')
        positions=mask.cumsum(-1)-1;positions.masked_fill_(mask==0,0)
        return {'input_ids':token_ids,'attention_mask':mask,'position_ids':positions}
    def predict(model,positions,adapter):
        model.eval();parts=[]
        with torch.inference_mode():
            for start in range(0,len(positions),cfg['micro_batch_pairs']):
                chosen=positions[start:start+cfg['micro_batch_pairs']]
                logits=model(batch(chosen)) if adapter else model(embeddings[chosen])
                parts.append(logits.float().sigmoid().cpu().numpy())
        score=np.concatenate(parts);score[:,zero]=0;score[:,one]=1
        assert np.isfinite(score).all();return score
    job_status=folder/('worker_'+args.mode+'_status.json');started=time.monotonic();json_write(job_status,{'state':'running','PID':os.getpid(),'mode':args.mode})
    kinds=['adapter'] if cfg.get('auxiliary_only_training') else ['head','adapter'] if job.get('inner',False) else [job['candidate'].split('_')[0]]
    if args.mode=='smoke':kinds=['adapter']
    records=[]
    for kind in kinds:
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);adapter=kind=='adapter'
        model=(SplitClassifier(cfg,initial) if adapter else warm_head(initial)).to('cuda')
        if args.mode=='predict':
            state=torch.load(folder/job['candidate']/'model.pt',map_location='cpu',weights_only=True)
            if adapter:model.load_adaptable(state)
            else:model.load_state_dict(state)
            np.savez_compressed(folder/job['candidate']/'test_scores.npz',positions=query,scores=predict(model,query,adapter));continue
        if adapter:
            model.eval();probe=query[:min(4,len(query))]
            baseline=warm_head(initial).to('cuda');reference=predict(baseline,probe,False);actual=predict(model,probe,True);difference=float(np.abs(reference-actual).max())
            json_write(folder/'initial_score_check.json',{'max_absolute_difference':difference,'tolerance':cfg['initial_score_max_abs_tolerance'],'passed':difference<=cfg['initial_score_max_abs_tolerance']})
            if difference>cfg['initial_score_max_abs_tolerance']:raise RuntimeError('Warm-start input/score mismatch')
            del baseline
        head=model.head if adapter else model;anchor=head.weight.detach().clone()
        groups=[{'params':head.parameters(),'lr':cfg['head_learning_rate']}]
        if adapter:groups.append({'params':[value for name,value in model.named_parameters() if 'lora_' in name],'lr':cfg['lora']['learning_rate']})
        optimizer=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay']);epochs=cfg['epochs'] if job.get('inner',False) else [int(job['candidate'].split('_e')[1])]
        if args.mode=='smoke':epochs=[1]
        history=[]
        for epoch in range(1,max(epochs)+1):
            model.train();order=np.random.RandomState(cfg['seed']+epoch).permutation(len(train));loss_sum=0.
            for start in range(0,len(train),cfg['effective_batch_pairs']):
                macro=order[start:start+cfg['effective_batch_pairs']];optimizer.zero_grad(set_to_none=True)
                for begin in range(0,len(macro),cfg['micro_batch_pairs']):
                    local=macro[begin:begin+cfg['micro_batch_pairs']];positions=train[local]
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(batch(positions)) if adapter else model(embeddings[positions])
                        loss=nn.functional.binary_cross_entropy_with_logits(logits.float(),target[local])
                        loss=loss+cfg['head_anchor']*.5*(head.weight-anchor).square().sum()/(len(train)*y.shape[1])
                    if not torch.isfinite(loss):raise ValueError('Nonfinite supervised loss')
                    (loss*len(local)/len(macro)).backward();loss_sum+=float(loss.detach())*len(local)
                nn.utils.clip_grad_norm_([value for value in model.parameters() if value.requires_grad],1.);optimizer.step()
                if adapter and start%256==0:
                    json_write(job_status,{'state':'running','PID':os.getpid(),'mode':args.mode,'kind':kind,'epoch':epoch,'rows_seen':start+len(macro),'training_rows':len(train),'elapsed_seconds':time.monotonic()-started})
            history.append({'epoch':epoch,'loss':loss_sum/len(train)})
            if epoch in epochs:
                name=f'{kind}_e{epoch}';destination=folder/('smoke' if args.mode=='smoke' else name);destination.mkdir()
                state=model.adaptable() if adapter else {key:value.detach().cpu() for key,value in model.state_dict().items()};torch.save(state,destination/'model.pt')
                np.savez_compressed(destination/'scores.npz',positions=query,scores=predict(model,query,adapter))
                if cfg.get('save_fit_reference_scores') and args.mode!='smoke':
                    reference=roles['context_train'];assert np.isin(train,reference).all() and not np.intersect1d(reference,query).size
                    np.savez_compressed(destination/'fit_reference_scores.npz',positions=reference,scores=predict(model,reference,adapter))
                record={'context':job['context'],'candidate':name,'training_rows':len(train),'query_rows':len(query),'history':list(history),
                        'trainable_parameters':sum(value.numel() for value in model.parameters() if value.requires_grad),'peak_GPU_GiB':torch.cuda.max_memory_allocated()/1024**3}
                json_write(destination/'fit.json',record);records.append(record)
                print(json.dumps(record),flush=True)
        del model,optimizer;gc.collect();torch.cuda.empty_cache()
    json_write(job_status,{'state':'complete','PID':os.getpid(),'mode':args.mode,'records':records,'runtime_seconds':time.monotonic()-started})


if __name__=='__main__':main()
