"""Conditional tree bundle inference; legacy code and source bundles untouched."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import joblib
import numpy as np

from skillcomp.frozen_inference import validate_records,predict_with_embeddings
from skillcomp.residual_inference import refine_scores,fuse_auxiliary_embeddings
from skillcomp.availability_inference import refine_availability
from skillcomp.score_ensemble import refine_policy_ensemble
from skillcomp.neighbor_inference import refine_neighbor_scores
from skillcomp.conditional_tree_inference import refine_conditional_tree_union
from skillcomp.gpu_policy import snapshot,permitted,permitted_for_models,inference_devices
from skillcomp.runtime import ROOT,require_server,json_write,official_module,encoder_python


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--gpu',default='auto');p.add_argument('--gpu-wait-seconds',type=float,default=300);p.add_argument('--scores');args=p.parse_args();require_server()
    bundle=joblib.load(args.bundle);assert bundle.get('conditional_tree_component'), 'Conditional tree bundle required';records=json.loads(Path(args.input).read_text());validate_records(records)
    if not records:json_write(args.output,{'label_order':bundle['labels'],'predictions':[]});return
    embeddings=np.zeros((len(records),0)); auxiliary_embeddings=None;semantic_embeddings=None
    if not bundle['selection']['fallback_rf'] or bundle.get('residual_component'):
        auxiliary_encoder=bundle.get('residual_component',{}).get('encoder',{});members=auxiliary_encoder.get('encoders',[auxiliary_encoder]);model_names=[bundle['encoder']['model']]+[member['model'] for member in members if 'model' in member]
        if args.gpu=='auto':
            minimum=bundle.get('residual_component',{}).get('encoder',{}).get('minimum_GPU_MiB',6000)
            started=time.monotonic()
            while True:
                gpus,processes,policy=snapshot();allowed=permitted_for_models(inference_devices(gpus,processes),policy,model_names);options=[(row['free_MiB'],index) for index,row in allowed.items() if row['free_MiB']>=minimum and (policy.get('allow_busy_GPU_sharing') or row['utilization']<=20)]
                if options:break
                if time.monotonic()-started>=args.gpu_wait_seconds:raise RuntimeError(f'No GPU with at least{minimum}MiB and utilization<=20 within wait budget')
                time.sleep(5)
            gpu=max(options)[1]
        else:
            gpu=int(args.gpu);gpus,_,policy=snapshot()
            if gpu not in permitted(gpus):raise RuntimeError('Requested GPU is reserved for the other user benchmark')
            if gpu not in permitted_for_models(gpus,policy,model_names):raise RuntimeError('Requested GPU has observed transient video memory pressure for this Qwen workload')
        with tempfile.TemporaryDirectory(prefix='skill_prediction_',dir=ROOT/'outputs') as temporary:
            folder=Path(temporary);settings=folder/'settings.json';json_write(settings,bundle['encoder']);destination=folder/'embeddings.npy';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT/'src'),CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false')
            subprocess.run([encoder_python(),str(ROOT/'scripts/encode_frozen_candidate_inputs.py'),'--settings',str(settings),'--input',str(Path(args.input).resolve()),'--out',str(destination)],cwd=ROOT,env=env,check=True)
            embeddings=np.load(destination)
            if bundle.get('residual_component'):
                encoder=bundle['residual_component']['encoder'];members=encoder['encoders'] if encoder.get('kind')=='feature_fusion' else [encoder];arrays=[]
                for index,member in enumerate(members):
                    settings=folder/f'auxiliary_settings_{index}.json';json_write(settings,member);destination=folder/f'auxiliary_embeddings_{index}.npy';encoder_script={'frozen_fields':'encode_frozen_candidate_inputs.py','proxy_head':'encode_proxy_candidate_inputs.py'}.get(member.get('kind'),'encode_auxiliary_candidate_inputs.py')
                    subprocess.run([encoder_python(),str(ROOT/'scripts'/encoder_script),'--settings',str(settings),'--input',str(Path(args.input).resolve()),'--out',str(destination)],cwd=ROOT,env=env,check=True);arrays.append(np.load(destination))
                auxiliary_embeddings=fuse_auxiliary_embeddings(arrays) if encoder.get('kind')=='feature_fusion' else arrays[0]
                if bundle.get('availability_component',{}).get('semantic_feature_dimensions'):
                    availability=bundle['availability_component'];member_index=availability['semantic_encoder_member_index'];assert members[member_index]['model']==availability['semantic_encoder_model'];semantic_embeddings=arrays[member_index]
    prediction,scores=predict_with_embeddings(bundle,records,embeddings,official_module().clean_text)
    initial_scores=scores.copy()
    component=bundle['conditional_tree_component'];repositories=[row['Repo Name'] for row in records];scores=refine_conditional_tree_union(bundle,component,initial_scores,auxiliary_embeddings,repositories,records,semantic_embeddings);scores=refine_availability(bundle,bundle['availability_component'],scores,records,semantic_embeddings=semantic_embeddings);choice=component['selection']
    if choice['fallback_rf']:prediction,_=predict_with_embeddings(dict(bundle,selection={'fallback_rf':True}),records,embeddings,official_module().clean_text)
    else:prediction=scores>=choice['threshold']
    json_write(args.output,{'label_order':bundle['labels'],'predictions':[{'vector':row.astype(int).tolist(),'labels':[name for name,present in zip(bundle['labels'],row) if present]} for row in prediction],
                           'model_training_rows':bundle['training_rows'],'deployment_independent_score':None})
    if args.scores:np.save(args.scores,scores)


if __name__=='__main__':main()
