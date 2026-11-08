"""Frozen candidate CLI with explicit relocatable assets and shared GPU claims."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--asset-map', required=True)
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--scores')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--gpu-wait-seconds', type=float, default=1800)
    args = parser.parse_args()
    if not args.check_only and (not args.input or not args.output):
        parser.error('--input and --output are required for prediction')
    import joblib
    from skillcomp.competition_assets import verify_frozen_bundle, bind_assets, FROZEN_BUNDLE_SHA256
    from skillcomp.frozen_inference import validate_records
    from skillcomp.runtime import ROOT, require_server, json_write
    from competition_inference_reservation import run_reserved
    require_server()
    records = json.loads(Path(args.input).read_text(encoding='utf-8')) if args.input else None
    if records is not None:
        validate_records(records)
    verify_frozen_bundle(args.bundle)
    bundle = joblib.load(args.bundle)
    asset_map = Path(args.asset_map).resolve()
    bundle = bind_assets(bundle, json.loads(asset_map.read_text(encoding='utf-8')), asset_map.parent)
    from skillcomp.competition_io import validate_label_order
    validate_label_order(bundle['labels'])
    if args.check_only:
        print(json.dumps({'state': 'passed', 'bundle_sha256': FROZEN_BUNDLE_SHA256,
                          'labels': len(bundle['labels']), 'training_rows': bundle['training_rows'],
                          'encoder_paths_relocated_in_memory': True, 'GPU_forward': False}))
        return
    if not records:
        json_write(args.output, {'label_order': bundle['labels'], 'predictions': []})
        if args.scores:
            import numpy as np
            np.save(args.scores, np.empty((0, len(bundle['labels']))))
        return
    command = [sys.executable, str(ROOT / 'scripts/predict_competition_candidate.py'),
               '--bundle', str(Path(args.bundle).resolve()), '--asset-map', str(asset_map),
               '--input', str(Path(args.input).resolve()), '--output', str(Path(args.output).resolve())]
    if args.scores:
        command += ['--scores', str(Path(args.scores).resolve())]
    models = [bundle['encoder']['model']] + [m['model'] for m in bundle['residual_component']['encoder']['encoders']]
    del bundle
    log = Path(str(args.output) + '.inference.log')
    with log.open('a', encoding='utf-8') as stream:
        gpu = run_reserved(command, stream, models, args.gpu_wait_seconds)
    print(json.dumps({'state': 'complete', 'output': args.output, 'GPU': gpu,
                      'bundle_sha256': FROZEN_BUNDLE_SHA256, 'independent_score': None}))


if __name__ == '__main__':
    main()
