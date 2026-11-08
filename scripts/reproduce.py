"""Plan historical roles safely; real stages require explicit local execution flags."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from skillcomp.reproduction_plan import ROLES, STAGES, stage_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=ROLES, default='fold0_inner0')
    p.add_argument('--phase', choices=['train', 'tune', 'evaluate', 'export'], default='train')
    p.add_argument('--stage', choices=STAGES, default='prepare')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--work-dir')
    p.add_argument('--asset-map', default='asset-map.json')
    p.add_argument('--database')
    p.add_argument('--encoder-python', default=os.environ.get('SKILLCOMP_ENCODER_PYTHON'))
    p.add_argument('--frozen-qwen4-features')
    p.add_argument('--frozen-qwen4-sha256')
    p.add_argument('--gpu', type=int)
    args = p.parse_args()
    plan = stage_plan(args.role)
    if args.phase != 'train':
        plan.update(requested_phase=args.phase, executable=False,
                    blocker='Historical source is included, but full dynamic asset/selection closure is incomplete. See docs/reproduction_gaps.json.')
    if not args.execute:
        print(json.dumps(plan, indent=2)); return
    if args.phase != 'train':
        p.error(plan['blocker'])
    stage = next(s for s in plan['stages'] if s['stage'] == args.stage)
    if stage['execution_blocker']:
        p.error(stage['execution_blocker'])
    if sys.platform != 'linux' or not args.work_dir:
        p.error('Explicit Linux local work directory required for real stages')
    work = Path(args.work_dir).resolve()
    if work == ROOT or work.is_relative_to(ROOT):
        p.error('Use a fresh work directory outside the source repository')
    if not args.database:
        p.error('--database must name the pinned SQLite; nothing is downloaded automatically')
    database = Path(args.database).resolve()
    def digest(path):
        with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest(database) != 'ed47eb3c2173eb137c6146df78e750c261ff2a9f42f631c8e0186fa4c94022ea':
        p.error('Database identity differs from the historical recipe')
    asset_map_path = Path(args.asset_map).resolve()
    assets = json.loads(asset_map_path.read_text(encoding='utf-8'))
    def asset(key):
        q = Path(assets[key]); return (asset_map_path.parent / q).resolve() if not q.is_absolute() else q
    if args.stage == 'prepare':
        if work.exists(): p.error('Work directory already exists; never reuse historical fitted caches')
        work.mkdir(parents=True)
        for folder in ['src', 'scripts', 'configs', 'upstream']:
            shutil.copytree(ROOT / folder, work / folder, ignore=shutil.ignore_patterns('__pycache__'))
        for folder in ['outputs/experiments', 'cache', 'data/raw']:
            (work / folder).mkdir(parents=True, exist_ok=True)
        (work / 'data/raw/skillscope_data.db').symlink_to(database)
        (work / '.skillcomp-server').touch()
        (work / 'role_contract.json').write_text(json.dumps(plan['role'], indent=2))
        for name, key in [('encoder_recipe', 'codebert_encoder'), ('base_encoder', 'base_encoder')]:
            path = work / 'configs' / (name + '.json')
            cfg = json.loads(path.read_text(encoding='utf-8')); cfg['model'] = str(asset(key))
            path.write_text(json.dumps(cfg, indent=2))
        normalized = json.loads((work / 'configs/base_encoder.json').read_text(encoding='utf-8')); normalized['whitespace'] = 'normalized'
        (work / 'configs/base_encoder_normalized.json').write_text(json.dumps(normalized, indent=2))
        (work / 'configs/active_role_recipe.json').write_text(json.dumps({'base_text_weight': .3, 'coverage_mix': .5, 'publication_weight': .5}))
    elif not (work / 'role_contract.json').is_file() or json.loads((work / 'role_contract.json').read_text(encoding='utf-8')) != plan['role']:
        p.error('Run prepare for exactly this role in a fresh directory first')
    if args.stage == 'experts-fit':
        if not args.frozen_qwen4_features or not args.frozen_qwen4_sha256:
            p.error('Declare a pure input-only Qwen4 transform and its expected hash; see gap manifest')
        path = Path(args.frozen_qwen4_features).resolve()
        if digest(path) != args.frozen_qwen4_sha256: p.error('Frozen input-only features hash mismatch')
        dest = work / 'outputs/declared_frozen_Qwen4B.npy'
        if not dest.exists(): dest.symlink_to(path)
        if digest(dest) != args.frozen_qwen4_sha256: p.error('Existing frozen feature hash mismatch')
    command = stage['command'].copy()
    encoder = command[0] == 'ENCODER_PYTHON'
    if encoder and not args.encoder_python: p.error('Separate --encoder-python required')
    if args.stage in ['encoder-fit', 'base-encode', 'extract', 'normalized-encode'] and args.gpu is None:
        p.error('Explicit --gpu required; reserve an allowed device locally before executing')
    command[0] = str(Path(args.encoder_python).resolve()) if encoder else sys.executable
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(work / 'src'), str(work / 'scripts')]),
               SKILLCOMP_ROLE=args.role, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if args.gpu is not None: env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    subprocess.run(command, cwd=work, env=env, check=True)


if __name__ == '__main__': main()
