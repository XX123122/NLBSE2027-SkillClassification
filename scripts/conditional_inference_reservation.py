"""Short-lived shared GPU reservation for raw conditional-tree bundle inference."""
import fcntl
import os
from pathlib import Path
import subprocess
import time
from run_qwen06_split_lora import reservation_entries
from skillcomp.gpu_policy import snapshot,permitted,reserved_free,permitted_for_models
from skillcomp.runtime import ROOT,json_write


def run_reserved(command,stream,model_names,env=None,wait_seconds=1800):
    registry=ROOT/'cache/gpu_dispatch_reservations.json';pid=os.getpid();started=time.monotonic();chosen=None
    while chosen is None:
        with (ROOT/'cache/gpu_dispatch.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);gpus,processes,policy=snapshot();claims=reservation_entries(registry);allowed=permitted_for_models(permitted(gpus),policy,model_names);options=[]
            for gpu,row in allowed.items():
                if row['UUID'] in set(policy.get('excluded_GPU_UUIDs', [])):continue
                own=[c for c in claims if c['UUID']==row['UUID']];free=reserved_free(row,processes,own)
                if free>=11000 and row['utilization']<=20:options.append((free,gpu))
            if options:
                _,chosen=max(options);claims.append({'PID':pid,'UUID':gpus[chosen]['UUID'],'reservation_MiB':11000,'run_id':'conditional_tree_raw_inference','mode':'conditional_tree_raw_inference'});json_write(registry,claims)
        if chosen is None:
            if time.monotonic()-started>wait_seconds:raise TimeoutError('No allowed free GPU for conditional-tree inference within fixed wait budget')
            time.sleep(5)
    try:subprocess.run(command+['--gpu',str(chosen)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    finally:
        with (ROOT/'cache/gpu_dispatch.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);claims=reservation_entries(registry);json_write(registry,[c for c in claims if not(c['PID']==pid and c.get('mode')=='conditional_tree_raw_inference')])
    return chosen
