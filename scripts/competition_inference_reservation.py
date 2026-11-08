"""Inference reservation usable from a relocated candidate source directory."""
import os
from pathlib import Path
import subprocess
import time

from skillcomp.gpu_policy import snapshot, permitted, permitted_for_models, reserved_free
from skillcomp.runtime import ROOT, json_write


def live_claims(path):
    import json
    rows = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    kept = []
    for row in rows:
        try:
            if Path(f'/proc/{row["PID"]}/cmdline').read_bytes():
                kept.append(row)
        except (FileNotFoundError, ProcessLookupError):
            pass
    return kept


def run_reserved(command, stream, models, wait_seconds=1800, env=None):
    import fcntl
    shared = Path(os.environ.get('SKILLCOMP_GPU_REGISTRY_ROOT', str(ROOT / 'cache')))
    shared.mkdir(parents=True, exist_ok=True)
    registry = shared / 'gpu_dispatch_reservations.json'
    lock_path = shared / 'gpu_dispatch.lock'
    started, chosen, pid = time.monotonic(), None, os.getpid()
    mode = 'competition_candidate_inference'
    while chosen is None:
        with lock_path.open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            gpus, processes, policy = snapshot()
            claims = live_claims(registry)
            allowed = permitted_for_models(permitted(gpus), policy, models)
            options = []
            for gpu, row in allowed.items():
                if row['UUID'] in set(policy.get('excluded_GPU_UUIDs', [])):
                    continue
                free = reserved_free(row, processes, [c for c in claims if c['UUID'] == row['UUID']])
                if free >= 11000 and row['utilization'] <= 20:
                    options.append((free, gpu))
            if options:
                _, chosen = max(options)
                claims.append({'PID': pid, 'UUID': gpus[chosen]['UUID'], 'reservation_MiB': 11000,
                               'run_id': mode, 'mode': mode})
                json_write(registry, claims)
        if chosen is None:
            if time.monotonic() - started >= wait_seconds:
                raise TimeoutError('No allowed GPU with 11000MiB available in the wait budget')
            time.sleep(5)
    try:
        subprocess.run(command + ['--gpu', str(chosen)], cwd=ROOT, env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    finally:
        with lock_path.open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            json_write(registry, [c for c in live_claims(registry)
                                  if not (c['PID'] == pid and c.get('mode') == mode)])
    return chosen
