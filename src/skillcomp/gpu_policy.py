"""Shared-GPU dispatch policy, protecting the user-reserved benchmark device."""
import json
import os
from pathlib import Path
import subprocess

from skillcomp.runtime import ROOT


def policy():
    path=ROOT/'configs/gpu_resource_policy.json'
    return json.loads(path.read_text()) if path.exists() else {}


def snapshot():
    import pwd
    settings=policy();result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True);gpus={}
    for row in result.stdout.strip().splitlines():
        index,uuid,free,util=[v.strip() for v in row.split(',')];gpus[int(index)]={'UUID':uuid,'free_MiB':int(free),'utilization':int(util),'excluded':uuid in settings.get('excluded_GPU_UUIDs',[])}
    result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True);processes={};by_uuid={v['UUID']:v for v in gpus.values()}
    for row in result.stdout.strip().splitlines():
        uuid,pid,used=[v.strip() for v in row.split(',')]
        try:owner=pwd.getpwuid(os.stat('/proc/'+pid).st_uid).pw_name
        except (FileNotFoundError,ProcessLookupError):continue
        if owner in settings.get('excluded_owners',[]) and uuid in by_uuid:by_uuid[uuid]['excluded']=True
        try:memory=int(used)
        except ValueError:memory=0
        processes[int(pid)]={'UUID':uuid,'used_MiB':memory,'owner':owner}
    return gpus,processes,settings


def permitted(gpus):
    return {index:row for index,row in gpus.items() if not row['excluded']}


def assert_visible_devices(visible,gpus):
    devices={value.strip() for value in visible.split(',')}
    if any(row['excluded'] and (str(index) in devices or row['UUID'] in devices) for index,row in gpus.items()):raise RuntimeError('CUDA device is reserved for the other user benchmark')


def reserved_free(row,processes,claims):
    return row['free_MiB']-sum(max(0,entry['reservation_MiB']-processes.get(entry['PID'],{}).get('used_MiB',0)) for entry in claims if entry['UUID']==row['UUID'])


def inference_devices(gpus,processes):
    path=ROOT/'cache/gpu_dispatch_reservations.json';claims=json.loads(path.read_text()) if path.exists() else [];claims=[row for row in claims if Path(f'/proc/{row["PID"]}/cmdline').exists()]
    return {index:dict(row,free_MiB=reserved_free(row,processes,claims)) for index,row in permitted(gpus).items()}


def permitted_for_models(gpus,settings,models):
    excluded=set()
    for model in models:
        name=str(model).lower()
        if 'qwen3-embedding-0.6b' in name:excluded.update(settings.get('Qwen06_excluded_GPU_UUIDs',[]))
        if 'qwen3-embedding-4b' in name:excluded.update(settings.get('Qwen4_excluded_GPU_UUIDs',[]))
    return {index:row for index,row in permitted(gpus).items() if row['UUID'] not in excluded}
