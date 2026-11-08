"""Add explicit technical-text evidence to auxiliary skill classification."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import warnings

import joblib
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from capability_search_workers import digest
from run_annotation_channel_capability import feature_array
from run_channel_residual_capability import base_oof
from skillcomp.annotation_channel import fit_channel, apply_channel
from skillcomp.benchmark import fast_micro
from skillcomp.capability_search import threshold_choice,repo_features
from skillcomp.channel_residual import fit_expectation, apply_residual
from skillcomp.models import estimator, continuous_scores
from skillcomp.paired_dimensions import truncate_paired_embeddings
from skillcomp.learned_features import load_context_embeddings
from skillcomp.hierarchical_readout import ConditionalHierarchyClassifier
from skillcomp.runtime import ROOT, require_server, provenance, json_write, status


def features(cfg,context,candidate,train,positions=None):
    values=_features_without_repository(cfg,context,candidate,train,positions)
    if cfg.get('auxiliary_repository_scale'):
        from skillcomp.repository_conditioning import append_repository_indicators
        repos=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];values=append_repository_indicators(values,repos[train],repos,cfg['auxiliary_repository_scale'])
    if cfg.get('auxiliary_repository_interaction_scale'):
        from skillcomp.repository_interactions import append_repository_interactions
        repos=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];dense=_features_without_repository(cfg,context,'codebert_dense',train,positions);values=append_repository_interactions(values,dense,repos[train],repos,cfg['auxiliary_repository_interaction_scale'])
    return values


def _features_without_repository(cfg, context, candidate, train, positions=None):
    if candidate.endswith('_tree'):candidate=candidate[:-5]
    if cfg.get('contextual_auxiliary_features_run'):
        source=ROOT/'outputs/experiments'/cfg['contextual_auxiliary_features_run'];folder=source/'contexts'/context;role=np.load(folder/'roles.npz');np.testing.assert_array_equal(role['context_train'],train);aux=np.load(ROOT/'outputs/experiments'/cfg['channel_run']/'auxiliary_targets.npz');known_train=train[aux['matched'][train].astype(bool)]
        if cfg.get('context_encoder_supervision') in ['availability_and_masked_proxy','known_proxy_and_teacher_soft_targets']:
            np.testing.assert_array_equal(role['train'],train);np.testing.assert_array_equal(role['known_proxy_train'],known_train)
        else:np.testing.assert_array_equal(role['train'],known_train)
        epoch=json.loads((source/'source_epochs.json').read_text())[context.split('_')[0]];settings={'context_embedding_pattern':'learned_e{epoch}.npy','context_embedding_dimensions':cfg.get('contextual_embedding_dimensions',768)};values=load_context_embeddings(folder,settings,epoch,7154)
        if cfg.get('additional_contextual_auxiliary'):
            parts=[sparse.csr_matrix(values)]
            for item in cfg['additional_contextual_auxiliary']:
                other=ROOT/'outputs/experiments'/item['run'];other_folder=other/'contexts'/context;other_role=np.load(other_folder/'roles.npz');np.testing.assert_array_equal(other_role['train'],role['train']);np.testing.assert_array_equal(other_role['context_train'],train);other_epoch=json.loads((other/'source_epochs.json').read_text())[context.split('_')[0]];parts.append(sparse.csr_matrix(load_context_embeddings(other_folder,{'context_embedding_pattern':'learned_e{epoch}.npy','context_embedding_dimensions':item['dimensions']},other_epoch,7154)))
            values=normalize(sparse.hstack(parts,format='csr'),norm='l2')
        if cfg.get('frozen_auxiliary_fusion'):
            parts=[sparse.csr_matrix(values),sparse.csr_matrix(np.load(ROOT/cfg['frozen_auxiliary_fusion']['embedding_path'],mmap_mode='r'))]
            parts.extend(sparse.csr_matrix(np.load(ROOT/item['embedding_path'],mmap_mode='r')) for item in cfg.get('additional_frozen_auxiliary',[]));values=normalize(sparse.hstack(parts,format='csr'),norm='l2')
        if candidate=='codebert_dense':return values
        assert candidate=='codebert_hybrid';lexical=ROOT/'outputs/experiments'/cfg['lexical_run']/'contexts'/context;np.testing.assert_array_equal(np.load(lexical/'roles.npz')['train'],train)
        return normalize(sparse.hstack([sparse.load_npz(lexical/'title.npz'),sparse.load_npz(lexical/'body.npz'),sparse.csr_matrix(values)],format='csr'),norm='l2')
    if cfg.get('map_kind')=='cross_skill':
        source=ROOT/'outputs/experiments'/cfg['channel_run'];ccfg=json.loads((source/'config.json').read_text());epoch=json.loads((source/'input_identity.json').read_text())['source_epochs'][context.split('_')[0]];folder=source/'contexts'/context/'learned_selected';bundle=joblib.load(folder/'model.joblib');np.testing.assert_array_equal(bundle['train'],train)
        assert digest(folder/'model.joblib')==json.loads((folder/'fit.json').read_text())['model_sha256'];x=feature_array(source,ccfg,context,'learned_selected',epoch);positions=np.arange(len(x)) if positions is None else np.asarray(positions);aux=np.zeros((len(x),217));values=np.asarray(bundle['head'].predict_proba(x[positions]));counts=bundle['auxiliary_positive_counts'];values[:,counts==0]=0;values[:,counts==bundle['auxiliary_fit_rows']]=1;aux[positions]=values
        repos=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];return sparse.hstack([sparse.csr_matrix(aux),repo_features(repos[train],repos)],format='csr')
    if cfg.get('auxiliary_embedding_path'):
        values=np.load(ROOT/cfg['auxiliary_embedding_path'],mmap_mode='r')
        if candidate=='qwen4_mrl':return sparse.csr_matrix(truncate_paired_embeddings(values,1024))
        if candidate=='qwen4':return sparse.csr_matrix(values)
        assert candidate=='qwen4_hybrid';source=ROOT/'outputs/experiments'/cfg['lexical_run']/'contexts'/context;np.testing.assert_array_equal(np.load(source/'roles.npz')['train'],train)
        return normalize(sparse.hstack([sparse.load_npz(source/'title.npz'),sparse.load_npz(source/'body.npz'),sparse.csr_matrix(values)],format='csr'),norm='l2')
    source = ROOT / 'outputs/experiments' / cfg['lexical_run'] / 'contexts' / context; role = np.load(source / 'roles.npz'); np.testing.assert_array_equal(role['train'], train)
    parts = [sparse.load_npz(source / 'title.npz'), sparse.load_npz(source / 'body.npz')]
    if candidate == 'hybrid':
        channel = ROOT / 'outputs/experiments' / cfg['channel_run']; ccfg = json.loads((channel / 'config.json').read_text()); epoch = json.loads((channel / 'input_identity.json').read_text())['source_epochs'][context.split('_')[0]]
        np.testing.assert_array_equal(np.load(channel / 'contexts' / context / 'roles.npz')['train'], train)
        parts.append(sparse.csr_matrix(feature_array(channel, ccfg, context, 'learned_selected', epoch)))
    else: assert candidate == 'lexical'
    return normalize(sparse.hstack(parts, format='csr'), norm='l2')


def fit_case(job):
    threadpool_limits(1); out = Path(job['out']); cfg = job['config']; context = job['context']; role = np.load(out / 'contexts' / context / 'roles.npz'); train, query = role['train'], role['query']; data = np.load(ROOT / 'cache/workshop_v1/dataset.npz', allow_pickle=True); y, repos = data['y'], data['repos']; aux = np.load(ROOT / 'outputs/experiments' / cfg['channel_run'] / 'auxiliary_targets.npz'); known = aux['matched'][train].astype(bool); target = aux['targets'][train[known]]
    x = features(cfg, context, job['candidate'], train,positions=np.union1d(train,query));fit_x=x[train[known]]
    if cfg.get('map_kind')=='cross_skill':
        fit_x=sparse.hstack([sparse.csr_matrix(target),repo_features(repos[train],repos[train[known]])],format='csr');target=y[train[known]]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'Label .* present in all training examples\.', category=UserWarning); c=cfg.get('candidate_C',{}).get(job['candidate'],cfg['auxiliary_head_C']);conditional=cfg.get('map_kind')=='conditional_hierarchy' or job['candidate'] in cfg.get('conditional_candidates',[]);head=ConditionalHierarchyClassifier(json.loads((ROOT/'cache/workshop_v1/labels.json').read_text()),c=c) if conditional else estimator('lr',c=c,jobs=1);head.fit(fit_x, target)
    channel = {'global_rates':np.stack([np.zeros(y.shape[1]),np.ones(y.shape[1])]),'repository_rates':{},'mean':y[train].mean(0)} if cfg.get('map_kind')=='cross_skill' else fit_channel(y[train], aux['targets'][train], known, repos[train], cfg['channel_strength'])
    expectation = fit_expectation(apply_channel(channel, continuous_scores(head, x[train], target, 'lr'), repos[train]), repos[train]); scores = apply_channel(channel, continuous_scores(head, x[query], target, 'lr'), repos[query])
    folder = out / 'contexts' / context / job['candidate']; folder.mkdir();bundle={'head': head, 'channel': channel, 'expectation': expectation, 'train': train, 'positive_counts': target.sum(0), 'known_rows': len(target)}
    if cfg.get('auxiliary_repository_scale'):bundle.update(repository_vocabulary=sorted(set(str(name) for name in repos[train])),repository_scale=cfg['auxiliary_repository_scale'])
    if cfg.get('auxiliary_repository_interaction_scale'):bundle.update(repository_interaction_scale=cfg['auxiliary_repository_interaction_scale'],repository_interaction_dense_dimensions=5888)
    joblib.dump(bundle, folder / 'model.joblib'); np.savez_compressed(folder / 'scores.npz', positions=query, scores=scores)
    record = {'context': context, 'candidate': job['candidate'], 'fit_rows': len(train), 'known_proxy_rows': len(target), 'feature_dimensions': x.shape[1], 'model_sha256': digest(folder / 'model.joblib'),'teacher_forced_auxiliary_fit_features':cfg.get('map_kind')=='cross_skill','ground_truth_auxiliary_used_at_query':False}; json_write(folder / 'fit.json', record); return record


def reference_oof(run,name,positions,method='best_policy'):
    root=ROOT/'outputs/experiments'/run;selection=json.loads((root/'all_training_selections_locked.json').read_text())['selections'][name][method];config=json.loads((root/'config.json').read_text())
    if config.get('study_id')=='proxy_tree_readout_capability_v1' and selection.get('candidate')!='reference':
        a=np.load(root/(name+'_oof_scores.npz'));np.testing.assert_array_equal(a['positions'],positions);return a[str(selection['tree_fraction'])],config['residual_weight']
    if config.get('channel_kind')=='availability':
        original,_=base_oof(config,name);np.testing.assert_array_equal(original['positions'],positions);base,_=reference_oof(config.get('correction_base_run',config['reference_run']),name,positions);repos=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];lookup={int(p):i for i,p in enumerate(positions)};scores=np.zeros_like(base);seen=np.zeros(len(base),int)
        if selection['candidate']=='reference':return reference_oof(config['reference_run'],name,positions)
        for index in range(3):
            folder=root/'contexts'/f'{name}_inner{index}';a=np.load(folder/'scores.npz');query=a['positions'];local=np.array([lookup[int(p)] for p in query]);seen[local]+=1;mapping=joblib.load(folder/'mapping.joblib');scores[local]=apply_residual(base[local],a['scores'],repos[query],mapping['expectation'],selection['residual_weight'])
        assert (seen==1).all();return scores,selection['residual_weight']
    if config.get('study_id')=='evidence_coverage_gate_capability_v1':
        from run_evidence_coverage_gate_capability import case
        original,base=base_oof(config,name);lookup={int(p):i for i,p in enumerate(original['positions'])};repos=np.load(ROOT/'cache/workshop_v1/dataset.npz',allow_pickle=True)['repos'];scores=np.zeros_like(base);seen=np.zeros(len(base),int)
        from skillcomp.evidence_coverage import coverage_weights
        for index in range(3):
            query,channel,center,coverage,_=case(root,config,name,f'{name}_inner{index}','calibration');local=np.array([lookup[int(p)] for p in query]);seen[local]+=1;scores[local]=apply_residual(base[local],channel,repos[query],center,selection['residual_weight'],row_weights=coverage_weights(coverage,repos[query],selection['mix']))
        np.testing.assert_array_equal(original['positions'],positions);assert (seen==1).all();return scores,selection['residual_weight']
    if 'fraction' in selection and 'source_runs' in config:
        left,_=reference_oof(config['source_runs'][0],name,positions);right,_=reference_oof(config['source_runs'][1],name,positions);fraction=selection['fraction'];return (1-fraction)*left+fraction*right,1
    if selection.get('candidate')=='reference':return reference_oof(config['reference_run'],name,positions,config.get('reference_method','best_policy'))
    a=np.load(root/(name+'_oof_scores.npz'));np.testing.assert_array_equal(a['positions'],positions);key=selection.get('score_key') or (selection['candidate']+'_w' if 'candidate' in selection else 'w')+str(selection['residual_weight']);return a[key],selection['residual_weight']


def choose(job):
    threadpool_limits(1); out = Path(job['out']); cfg = job['config']; name = job['name']; original, base = base_oof(cfg, name); lookup = {int(p): i for i, p in enumerate(original['positions'])}; repos = np.load(ROOT / 'cache/workshop_v1/dataset.npz', allow_pickle=True)['repos']; records = []; saved = {}
    if cfg.get('correction_base_run'):base,_=reference_oof(cfg['correction_base_run'],name,original['positions'])
    for candidate in cfg['candidates']:
        scores = {str(w): np.zeros_like(base) for w in cfg['residual_weights']}; seen = np.zeros(len(base), int)
        for index in range(3):
            folder = out / 'contexts' / f'{name}_inner{index}'; query = np.load(folder / 'roles.npz')['query']; local = np.array([lookup[int(p)] for p in query]); seen[local] += 1; a = np.load(folder / candidate / 'scores.npz'); np.testing.assert_array_equal(a['positions'], query); model = joblib.load(folder / candidate / 'model.joblib')
            for w in cfg['residual_weights']: scores[str(w)][local] = apply_residual(base[local], a['scores'], repos[query], model['expectation'], w)
        assert (seen == 1).all()
        for w, score in scores.items():
            key = candidate + '_w' + w; saved[key] = score; records.append(threshold_choice(original['y_true'], score, original['rf_prediction']) | {'candidate': candidate, 'residual_weight': float(w), 'score_key': key})
    ref_scores,ref_weight=reference_oof(cfg['reference_run'],name,original['positions'],cfg.get('reference_method','best_policy'));control = threshold_choice(original['y_true'], ref_scores, original['rf_prediction']) | {'candidate': 'reference', 'residual_weight': ref_weight}
    key = lambda row: (row['micro_f1'], row['precision'], -row['residual_weight'], row['candidate']); new = max([r for r in records if r['residual_weight'] > 0], key=key); best = max([*records, control], key=key)
    json_write(out / (name + '_OOF_search.json'), records + [control]); np.savez_compressed(out / (name + '_oof_scores.npz'), positions=original['positions'], **saved); return name, {'new_policy': new, 'best_policy': best}


def test_case(job):
    threadpool_limits(1); out = Path(job['out']); cfg = job['config']; name = job['name']; candidate = job['candidate']; context = name + '_final'; assert (out / 'all_calibrations_locked.json').exists(); role = np.load(ROOT / 'cache/workshop_v1/grouped5_pr_components' / (name + '.npz')); folder = out / 'contexts' / context / candidate; model = joblib.load(folder / 'model.joblib'); np.testing.assert_array_equal(model['train'], role['train']); x = features(cfg, context, candidate, role['train'],positions=role['test']); aux = np.asarray(model['head'].predict_proba(x[role['test']])); counts = model['positive_counts']; aux[:, counts == 0] = 0; aux[:, counts == model['known_rows']] = 1; repos = np.load(ROOT / 'cache/workshop_v1/dataset.npz', allow_pickle=True)['repos']
    np.savez_compressed(folder / 'test_scores.npz', positions=role['test'], scores=apply_channel(model['channel'], aux, repos[role['test']])); return {'name': name, 'candidate': candidate}


def pool(executor, function, jobs, out, stage):
    records = []
    for future in as_completed([executor.submit(function, job) for job in jobs]): records.append(future.result()); status(out, 'running', stage=stage, completed=len(records), total=len(jobs))
    return records


def run(args, out):
    started = time.monotonic(); cfg = json.loads((ROOT / args.config).read_text()); provenance(out, cfg); status(out, 'running', stage='prepare_cached_own_fit_features'); source = ROOT / 'outputs/experiments' / cfg['channel_run']; ref = ROOT / 'outputs/experiments' / cfg['reference_run']; assert json.loads((ref / 'verification.json').read_text())['state'] == 'passed'; data = np.load(ROOT / 'cache/workshop_v1/dataset.npz', allow_pickle=True); y, repos = data['y'], data['repos']; assert digest(ROOT / 'cache/workshop_v1/dataset.npz') == json.loads((source / 'input_identity.json').read_text())['dataset_sha256']; jobs = []; roles = {}
    if cfg.get('auxiliary_embedding_path'):
        manifest=json.loads((ROOT/cfg['auxiliary_embedding_manifest']).read_text());assert manifest['combined_sha256']==digest(ROOT/cfg['auxiliary_embedding_path']) and manifest['rows']==len(y) and not manifest['labels_used_for_encoding'] and not manifest['encoder_finetuned']
        assert digest(ROOT/'outputs/experiments/20261023_markdown_semantic_features_v1/original_fields.json')==manifest['encoded_original_fields_sha256'];assert np.load(ROOT/cfg['auxiliary_embedding_path'],mmap_mode='r').shape==(len(y),manifest['dimensions']);json_write(out/'auxiliary_feature_identity.json',manifest)
    if cfg.get('frozen_auxiliary_fusion'):
        settings=cfg['frozen_auxiliary_fusion'];manifest=json.loads((ROOT/settings['manifest']).read_text());assert digest(ROOT/settings['embedding_path'])==manifest['combined_sha256'] and not manifest['encoder_finetuned'] and not manifest['labels_used_for_encoding'];assert digest(ROOT/'outputs/experiments/20261023_markdown_semantic_features_v1/original_fields.json')==manifest['encoded_original_fields_sha256'];assert np.load(ROOT/settings['embedding_path'],mmap_mode='r').shape==(len(y),manifest['dimensions']);json_write(out/'frozen_fusion_identity.json',manifest)
    for index,settings in enumerate(cfg.get('additional_frozen_auxiliary',[])):
        manifest=json.loads((ROOT/settings['manifest']).read_text());assert digest(ROOT/settings['embedding_path'])==manifest['combined_sha256'] and not manifest['encoder_finetuned'] and not manifest['labels_used_for_encoding'];assert digest(ROOT/'outputs/experiments/20261023_markdown_semantic_features_v1/original_fields.json')==manifest['encoded_original_fields_sha256'];assert np.load(ROOT/settings['embedding_path'],mmap_mode='r').shape==(len(y),manifest['dimensions']);json_write(out/f'additional_frozen_fusion_{index}.json',manifest)
    for fold in range(5):
        name = f'fold{fold}'; role = np.load(ROOT / 'cache/workshop_v1/grouped5_pr_components' / (name + '.npz')); roles[name] = {k: role[k] for k in ['train', 'validation', 'test']}
        for suffix in ['inner0', 'inner1', 'inner2', 'final']:
            context = name + '_' + suffix; folder = out / 'contexts' / context; folder.mkdir(parents=True); shutil_source = source / 'contexts' / context / 'roles.npz'; original = np.load(shutil_source); np.savez_compressed(folder / 'roles.npz', train=original['train'], query=original['query'])
            for candidate in cfg['candidates']: jobs.append({'out': str(out), 'config': cfg, 'context': context, 'candidate': candidate})
    with ProcessPoolExecutor(max_workers=cfg['CPU_workers']) as executor:
        json_write(out / 'inner_fit_records.json', pool(executor, fit_case, [job for job in jobs if not job['context'].endswith('_final')], out, 'auxiliary_lexical_inner_fits'))
        selected = dict(pool(executor, choose, [{'out': str(out), 'config': cfg, 'name': name} for name in roles], out, 'training_selection')); json_write(out / 'all_training_selections_locked.json', {'selections': selected, 'validation_used': False, 'test_used': False})
        finals = [{'out': str(out), 'config': cfg, 'context': name + '_final', 'candidate': candidate} for name, selection in selected.items() for candidate in sorted({p['candidate'] for p in selection.values()} - {'reference'})]; json_write(out / 'final_fit_records.json', pool(executor, fit_case, finals, out, 'auxiliary_lexical_final_fits')); choices = {}
        for name, selection in selected.items():
            query = roles[name]['validation']; base = np.load(ROOT / 'outputs/experiments' / cfg['base_run'] / (name + '_calibration.npz')); reference = np.load(ref / (name + '_calibration.npz')); np.testing.assert_array_equal(base['positions'], query); np.testing.assert_array_equal(reference['positions'], query); choices[name] = {}; scores = {}
            for family, pick in selection.items():
                if pick['candidate'] == 'reference': scores[family] = reference[cfg.get('reference_method','best_policy')]
                else:
                    folder = out / 'contexts' / (name + '_final') / pick['candidate']; a = np.load(folder / 'scores.npz'); np.testing.assert_array_equal(a['positions'], query); model = joblib.load(folder / 'model.joblib'); baseline=np.load(ROOT/'outputs/experiments'/cfg['correction_base_run']/(name+'_calibration.npz'))['best_policy'] if cfg.get('correction_base_run') else base['portfolio']; scores[family] = apply_residual(baseline, a['scores'], repos[query], model['expectation'], pick['residual_weight'])
                choices[name][family] = threshold_choice(y[query], scores[family], base['rf_prediction'])
            np.savez_compressed(out / (name + '_calibration.npz'), positions=query, y_true=y[query], rf_prediction=base['rf_prediction'], **scores)
        json_write(out / 'all_calibrations_locked.json', {'selections': selected, 'choices': choices, 'test_used': False, 'all_five_contexts_locked': True}); pool(executor, test_case, [job | {'name': job['context'].split('_')[0]} for job in finals], out, 'locked_assessment_predictions')
    methods = ['RF', 'previous_best', 'new_policy', 'best_policy']; pooled = {k: np.zeros_like(y, dtype=bool) for k in methods}; seen = np.zeros(len(y), int); folds = []
    for name, role in roles.items():
        test = role['test']; seen[test] += 1; base = np.load(ROOT / 'outputs/experiments' / cfg['base_run'] / (name + '_assessment.npz')); reference = np.load(ref / (name + '_assessment.npz')); np.testing.assert_array_equal(base['positions'], test); np.testing.assert_array_equal(reference['positions'], test); pred = {'RF': base['RF_prediction'], 'previous_best': reference[cfg.get('reference_method','best_policy')+'_prediction']}; scores = {}
        for family, pick in selected[name].items():
            if pick['candidate'] == 'reference': scores[family] = reference[cfg.get('reference_method','best_policy')+'_scores']
            else:
                folder = out / 'contexts' / (name + '_final') / pick['candidate']; a = np.load(folder / 'test_scores.npz'); np.testing.assert_array_equal(a['positions'], test); model = joblib.load(folder / 'model.joblib'); baseline=np.load(ROOT/'outputs/experiments'/cfg['correction_base_run']/(name+'_assessment.npz'))['best_policy_scores'] if cfg.get('correction_base_run') else base['portfolio_scores']; scores[family] = apply_residual(baseline, a['scores'], repos[test], model['expectation'], pick['residual_weight'])
            choice = choices[name][family]; pred[family] = pred['RF'] if choice['fallback_rf'] else scores[family] >= choice['threshold']
        for method in methods: pooled[method][test] = pred[method]
        np.savez_compressed(out / (name + '_assessment.npz'), positions=test, y_true=y[test], **{k + '_prediction': v for k, v in pred.items()}, **{k + '_scores': v for k, v in scores.items()}); folds.append({'name': name, 'selections': selected[name], 'choices': choices[name], 'metrics': {k: fast_micro(y[test], v) for k, v in pred.items()}})
    assert (seen == 1).all(); metrics = {k: fast_micro(y, v) for k, v in pooled.items()}; gates = {k: all(v[m] >= metrics['RF'][m] for m in metrics['RF']) for k, v in metrics.items()}; np.savez_compressed(out / 'pooled_predictions.npz', y_true=y, **pooled); result = {'run_id': args.run_id, 'protocol': cfg['protocol'], 'rows': len(y), 'metrics': metrics, 'RF_point_gates': gates, 'folds': folds, 'all_choices_locked_before_assessment': True, 'new_encoder_fits': 0, 'inner_auxiliary_fits': 15*len(cfg['candidates']), 'runtime_seconds': time.monotonic() - started, 'limits': cfg['limits']}; json_write(out / 'results.json', result); report = ROOT / cfg['report_directory']; report.mkdir(exist_ok=True)
    for filename in ['results.json', 'inner_fit_records.json', 'final_fit_records.json', 'all_training_selections_locked.json', 'all_calibrations_locked.json']: json_write(report / filename, json.loads((out / filename).read_text()))
    status(out, 'complete', metrics=metrics, RF_point_gates=gates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True); parser.add_argument('--config',default='configs/auxiliary_lexical_residual_capability_v1.json');parser.add_argument('--launch', action='store_true'); args = parser.parse_args(); require_server()
    if args.launch:
        assert not (ROOT / 'outputs/experiments' / args.run_id).exists(); env = os.environ.copy(); env['PYTHONPATH'] = str(ROOT / 'src')
        for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']: env[key] = '1'
        with (ROOT / 'logs' / (args.run_id + '.log')).open('x') as stream: child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', str(Path(__file__).resolve()), '--run-id', args.run_id,'--config',args.config], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({'state': 'launched', 'PID': child.pid, 'run_id': args.run_id}))
    else:
        out = ROOT / 'outputs/experiments' / args.run_id; out.mkdir()
        try: run(args, out)
        except BaseException:
            (out / 'error.txt').write_text(traceback.format_exc()); status(out, 'failed', error=traceback.format_exc()); raise
