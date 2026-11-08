"""Fresh production token forward with explicit supervised-file access denial."""
import argparse
from pathlib import Path
import numpy as np
from score_token_late_interaction import main


original_load=np.load
def guarded(path,*args,**kwargs):
    if Path(path).name in ['dataset.npz','auxiliary_targets.npz']:raise RuntimeError('Supervised file access forbidden during fresh token forward')
    return original_load(path,*args,**kwargs)
np.load=guarded


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--job',required=True);main(p.parse_args())
