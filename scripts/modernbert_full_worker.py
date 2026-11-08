"""Full supervised training of a small bidirectional encoder on issue text."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig,AutoModel,AutoTokenizer

from skillcomp.runtime import ROOT,require_server,json_write
from skillcomp.neural_loss import asymmetric_loss
from skillcomp.auxiliary_loss import masked_auxiliary_bce,proxy_distillation_losses
from skillcomp.hierarchical_loss import parents_from_labels,conditional_prevalence,conditional_hierarchy_bce,hierarchy_marginals


class Classifier(nn.Module):
    def __init__(self,cfg,prevalence,auxiliary_prevalence=None):
        super().__init__();source=ROOT/cfg['model'];config=AutoConfig.from_pretrained(source,local_files_only=True,trust_remote_code=False);config.reference_compile=False
        self.encoder=AutoModel.from_pretrained(source,config=config,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.float32,attn_implementation='sdpa')
        self.dropout=nn.Dropout(cfg['dropout']);self.head=nn.Linear(config.hidden_size,len(prevalence))
        with torch.no_grad():
            self.head.weight.mul_(.01);self.head.bias.copy_(torch.tensor(np.log(prevalence/(1-prevalence)),dtype=torch.float32))
        self.aux_head=None
        if auxiliary_prevalence is not None:
            self.aux_head=nn.Linear(config.hidden_size,len(auxiliary_prevalence))
            with torch.no_grad():
                self.aux_head.weight.mul_(.01);self.aux_head.bias.copy_(torch.tensor(np.log(auxiliary_prevalence/(1-auxiliary_prevalence)),dtype=torch.float32))

    def forward(self,input_ids,attention_mask,return_auxiliary=False):
        hidden=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state.float();mask=attention_mask.unsqueeze(-1).float()
        pooled=self.dropout((hidden*mask).sum(1)/mask.sum(1).clamp_min(1));primary=self.head(pooled)
        return (primary,self.aux_head(pooled)) if return_auxiliary else primary


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--mode',choices=['tokenize','fit','predict'],required=True);p.add_argument('--job');args=p.parse_args();require_server()
    out=ROOT/args.out;cfg=json.loads((out/'config.json').read_text());torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    if args.mode=='tokenize':
        fields=json.loads((ROOT/cfg['fields_path']).read_text());texts=[f"Title: {title}\nDescription: {body}" for title,body in zip(fields['title'],fields['body'])]
        if cfg.get('repository_prefix'):
            repositories=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];texts=[f'Repository: {repo}\n{text}' for repo,text in zip(repositories,texts)]
        tokenizer=AutoTokenizer.from_pretrained(ROOT/cfg['model'],local_files_only=True,trust_remote_code=False,padding_side='right')
        tokens=tokenizer(texts,padding='max_length',truncation=True,max_length=cfg['max_length'],return_tensors='np')
        np.save(out/'token_ids.npy',tokens['input_ids'].astype(np.int32));np.save(out/'token_masks.npy',tokens['attention_mask'].astype(np.uint8))
        json_write(out/'tokenization.json',{'rows':len(texts),'max_length':cfg['max_length'],'labels_used':False});return
    job=json.loads((out/args.job).read_text());folder=out/'contexts'/job['context'];role=np.load(folder/'roles.npz');train,query=role['train'],role['query']
    if job.get('gradient_smoke'):
        assert args.mode=='fit';known_mask=np.load(out/'auxiliary_targets.npz')['matched'][train].astype(bool)
        if cfg.get('prediction_head')=='auxiliary' or cfg.get('teacher',{}).get('target')=='unknown_proxy_probability':train=np.concatenate([train[known_mask][:2],train[~known_mask][:2]])
        else:assert known_mask.all();train=train[:4]
        assert len(train)==4;query=query[:4];cfg=dict(cfg,epochs=[1],save_fit_reference_scores=False)
    if args.mode=='predict':assert (out/'all_calibrations_locked.json').is_file();query=role['test']
    y=np.load(out/cfg['primary_targets'])['targets'] if cfg.get('primary_targets') else np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['y']
    proxy_distillation=cfg.get('teacher',{}).get('target')=='unknown_proxy_probability';primary_known=None
    if cfg.get('primary_targets'):
        target_source=np.load(out/cfg['primary_targets']);primary_known=target_source['matched'].astype(bool)
        if not proxy_distillation:assert primary_known[train].all()
        assert np.isin(y[train[primary_known[train]]],[0,1]).all()
    observed=train[primary_known[train]] if proxy_distillation else train;assert len(observed)>0;counts=y[observed].sum(0);zero=counts==0;one=counts==len(observed);prevalence=(counts+.5)/(len(observed)+1)
    hierarchy=None
    if cfg.get('loss')=='hierarchical_bce':
        assert cfg.get('primary_targets') and not cfg.get('auxiliary');hierarchy=parents_from_labels(json.loads((ROOT/'cache/workshop_v1/labels.json').read_text()));prevalence=conditional_prevalence(y[train],hierarchy)
    auxiliary=None;aux_prevalence=None
    if cfg.get('auxiliary'):
        auxiliary=np.load(out/'auxiliary_targets.npz');known=auxiliary['matched'][train].astype(bool);aux_counts=auxiliary['targets'][train[known]].sum(0)
        aux_prevalence=(aux_counts+.5)/(int(known.sum())+1)
    if cfg.get('prediction_head')=='auxiliary':
        assert auxiliary is not None and cfg.get('primary_targets')=='availability_targets.npz' and y.shape[1]==1
        zero=aux_counts==0;one=aux_counts==int(known.sum())
    model=Classifier(cfg,prevalence,aux_prevalence).to('cuda');ids=np.load(out/'token_ids.npy',mmap_mode='r');masks=np.load(out/'token_masks.npy',mmap_mode='r')
    teacher=None
    if cfg.get('teacher') and args.mode=='fit':
        saved=np.load(folder/'teacher_targets.npz');expected_train=role['train'] if job.get('gradient_smoke') else train;np.testing.assert_array_equal(saved['positions'],expected_train)
        teacher=np.zeros(y.shape,dtype=np.float32);teacher[expected_train]=saved['scores']
    def batch(positions):return {'input_ids':torch.tensor(ids[positions],dtype=torch.long,device='cuda'),'attention_mask':torch.tensor(masks[positions],dtype=torch.long,device='cuda')}
    def predict(positions):
        model.eval();parts=[]
        with torch.inference_mode():
            for start in range(0,len(positions),cfg['prediction_batch_size']):
                chosen=positions[start:start+cfg['prediction_batch_size']];logits=model(**batch(chosen),return_auxiliary=True)[1] if cfg.get('prediction_head')=='auxiliary' else model(**batch(chosen));conditional=logits.float().sigmoid();probability=hierarchy_marginals(conditional,hierarchy) if hierarchy is not None else conditional;parts.append(probability.cpu().numpy())
        score=np.concatenate(parts);score[:,zero]=0;score[:,one]=1;assert np.isfinite(score).all();return score
    progress=folder/('worker_'+args.mode+'_status.json');started=time.monotonic();json_write(progress,{'state':'running','PID':os.getpid(),'mode':args.mode})
    if args.mode=='predict':
        model.load_state_dict(torch.load(folder/job['candidate']/'model.pt',map_location='cpu',weights_only=True));np.savez_compressed(folder/job['candidate']/'test_scores.npz',positions=query,scores=predict(query))
        json_write(progress,{'state':'complete','PID':os.getpid(),'mode':args.mode,'rows':len(query)});return
    heads=list(model.head.parameters())+([] if model.aux_head is None else list(model.aux_head.parameters()))
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':cfg['encoder_learning_rate']},{'params':heads,'lr':cfg['head_learning_rate']}],weight_decay=cfg['weight_decay'])
    steps_per_epoch=math.ceil(len(train)/cfg['effective_batch_size']);total_steps=steps_per_epoch*max(cfg['epochs']);warmup=max(1,int(total_steps*cfg['warmup_fraction']))
    def rate(step):
        if step<warmup:return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(total_steps-warmup,1))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate);epochs=cfg['epochs'] if job['inner'] else [int(job['candidate'].split('_e')[1])];history=[]
    if job.get('save_epochs'):
        assert out.parent==ROOT/'outputs/deployment' and job['context']=='full_final' and not job['inner'];epochs=job['save_epochs'];assert all(epoch in cfg['epochs'] for epoch in epochs)
    for epoch in range(1,max(epochs)+1):
        model.train();order=np.random.RandomState(cfg['seed']+epoch).permutation(train);loss_sum=0.;aux_sum=0.;aux_rows=0;proxy_rows=0;pseudo_rows=0;proxy_loss_sum=0.;pseudo_loss_sum=0.
        for start in range(0,len(train),cfg['effective_batch_size']):
            macro=order[start:start+cfg['effective_batch_size']];optimizer.zero_grad(set_to_none=True)
            macro_known=int(auxiliary['matched'][macro].sum()) if auxiliary is not None else 0
            macro_proxy_known=int(primary_known[macro].sum()) if proxy_distillation else 0
            for begin in range(0,len(macro),cfg['micro_batch_size']):
                positions=macro[begin:begin+cfg['micro_batch_size']];target=torch.tensor(y[positions],dtype=torch.float32,device='cuda')
                if teacher is not None and not proxy_distillation:
                    weight=cfg['teacher']['weight'];target=(1-weight)*target+weight*torch.tensor(teacher[positions],dtype=torch.float32,device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    if auxiliary is not None:logits,aux_logits=model(**batch(positions),return_auxiliary=True);logits=logits.float()
                    else:logits=model(**batch(positions)).float()
                    if proxy_distillation:
                        assert auxiliary is None and hierarchy is None;soft=torch.tensor(teacher[positions],dtype=torch.float32,device='cuda');loss,count,soft_loss,soft_count=proxy_distillation_losses(logits,target,primary_known[positions],soft);backward_loss=loss*count/max(macro_proxy_known,1)+cfg['teacher']['weight']*soft_loss*soft_count/max(len(macro)-macro_proxy_known,1);proxy_rows+=count;pseudo_rows+=soft_count;proxy_loss_sum+=float(loss.detach())*count;pseudo_loss_sum+=float(soft_loss.detach())*soft_count
                    else:
                        loss=conditional_hierarchy_bce(logits,target,hierarchy) if hierarchy is not None else asymmetric_loss(logits,target,**cfg['asymmetric_loss']) if cfg.get('loss','bce')=='asl' else nn.functional.binary_cross_entropy_with_logits(logits,target)
                        backward_loss=cfg.get('primary_loss_weight',1)*loss*len(positions)/len(macro)
                    if auxiliary is not None:
                        aux_target=torch.tensor(auxiliary['targets'][positions],dtype=torch.float32,device='cuda');aux_loss,count=masked_auxiliary_bce(aux_logits,aux_target,auxiliary['matched'][positions])
                        backward_loss=backward_loss+cfg['auxiliary']['weight']*aux_loss*count/max(macro_known,1);aux_sum+=float(aux_loss.detach())*count;aux_rows+=count
                assert torch.isfinite(backward_loss);backward_loss.backward();loss_sum+=float(loss.detach())*len(positions)
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
            if job.get('gradient_smoke'):assert norm>0
            optimizer.step();scheduler.step()
            if start%256==0:json_write(progress,{'state':'running','PID':os.getpid(),'mode':args.mode,'epoch':epoch,'rows_seen':start+len(macro),'training_rows':len(train),'elapsed_seconds':time.monotonic()-started})
        history.append({'epoch':epoch,'loss':loss_sum/len(train),**({'auxiliary_loss':aux_sum/max(aux_rows,1),'auxiliary_rows':aux_rows} if auxiliary is not None else {}),**({'known_proxy_rows':proxy_rows,'teacher_soft_rows':pseudo_rows,'known_proxy_loss':proxy_loss_sum/max(proxy_rows,1),'teacher_soft_loss':pseudo_loss_sum/max(pseudo_rows,1)} if proxy_distillation else {})})
        if epoch in epochs:
            name=f'modern_e{epoch}';destination=folder/name;destination.mkdir();torch.save({key:value.detach().cpu() for key,value in model.state_dict().items()},destination/'model.pt')
            score=predict(query);np.savez_compressed(destination/'scores.npz',positions=query,scores=score)
            if cfg.get('save_fit_reference_scores'):
                reference_positions=role['context_train'];assert np.isin(train,reference_positions).all()
                overlap=np.intersect1d(reference_positions,query).size
                if overlap:assert cfg.get('allow_full_deployment_fit_query_overlap') and out.parent==ROOT/'outputs/deployment' and job['context']=='full_final' and not job['inner']
                np.savez_compressed(destination/'fit_reference_scores.npz',positions=reference_positions,scores=predict(reference_positions))
            record={'context':job['context'],'candidate':name,'loss':cfg.get('loss','bce'),'teacher_weight':cfg.get('teacher',{}).get('weight',0),'fit_rows':len(train),'query_rows':len(query),'history':list(history),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                    'peak_GPU_GiB':torch.cuda.max_memory_allocated()/1024**3,'runtime_seconds':time.monotonic()-started}
            json_write(destination/'fit.json',record);print(json.dumps(record),flush=True)
    json_write(progress,{'state':'complete','PID':os.getpid(),'mode':args.mode,'runtime_seconds':time.monotonic()-started})


if __name__=='__main__':main()
