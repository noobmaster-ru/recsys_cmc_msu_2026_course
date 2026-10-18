"""Preregistered iALS search on train/validation only, with fixed CF comparison."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data_utils import FitData
from evaluation import (EvaluationContext, EvaluationResult, evaluate_model, history_slices,
                        popularity_slices, paired_bootstrap)
from models.ials import IALS
from scripts.prepare_data import deterministic_gzip, sha256


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + '\n', encoding='utf-8')


def load_result(local, prefix, metrics, context):
    frame = pd.read_csv(local / f'{prefix}_per_user.csv.gz', dtype={'user_id': 'string'}).set_index('user_id')
    frame.index = pd.Index(frame.index.to_numpy(dtype=object), dtype=object, name='user_id')
    rankings = np.load(local / f'{prefix}_recommendation_indices.npy', allow_pickle=False)
    if not np.array_equal(frame.index, context.user_ids) or rankings.shape != (len(frame), context.k):
        raise ValueError('Cached users/ranking shape differ from the common cohort')
    return EvaluationResult(metrics, frame, rankings, context.fingerprint, context.k)


def prepare(root):
    protocol = json.loads((root / 'reports/evaluation_protocol.json').read_text())
    config = json.loads((root / 'configs/ials_search.json').read_text())
    if protocol['k'] != 10 or protocol['relevance']['rating_at_least'] != 4:
        raise ValueError('iALS search requires the frozen K=10, rating>=4 protocol')
    frames, hashes = {}, {}
    for name in ('train', 'validation'):
        spec = protocol['splits'][name]
        path = root / spec['path']
        hashes[name] = sha256(path)
        if hashes[name] != spec['sha256']:
            raise ValueError(f'Split checksum mismatch: {name}')
        frames[name] = pd.read_csv(path, dtype={'user_id': 'string', 'item_id': 'string',
                                               'rating': 'int8', 'timestamp': 'int64'})
    data = FitData.from_frame(frames['train'])
    timestamp = int(pd.Timestamp(protocol['validation_start_utc']).timestamp())
    context = EvaluationContext.from_frames(frames['train'], frames['validation'], fit_data=data,
        k=10, query_timestamp=timestamp)
    cf_path = root / 'reports/item_knn_validation/summary.json'
    cf = json.loads(cf_path.read_text())
    if cf['cohort_fingerprint'] != context.fingerprint or cf['input_hashes'] != hashes:
        raise ValueError('Saved CF experiment has different users, labels, candidates or input files')
    for path, digest in cf['local_artifacts'].items():
        if sha256(root / path) != digest:
            raise ValueError('Saved CF per-user metrics or rankings changed')
    local_cf = root / 'data/validation/item_knn'
    comparator = load_result(local_cf, 'best', cf['best_validation_metrics'], context)
    popularity = load_result(local_cf, 'popularity', cf['popularity_validation_metrics'], context)
    for key in ('evaluation_users', 'catalog_items', 'relevant_interactions'):
        if context.diagnostics[key] != protocol['evaluation_diagnostics']['validation'][key]:
            raise ValueError(f'Validation cohort changed: {key}')
    files = ['models/ials.py', 'data_utils.py', 'evaluation.py', 'scripts/tune_ials.py',
             'configs/ials_search.json', 'requirements.txt']
    plan = {'configuration': config, 'input_hashes': hashes,
        'source_hashes': {path: sha256(root / path) for path in files},
        'protocol_sha256': sha256(root / 'reports/evaluation_protocol.json'),
        'cohort_fingerprint': context.fingerprint, 'cohort': context.diagnostics,
        'query_timestamp': timestamp, 'cf_summary_sha256': sha256(cf_path),
        'cf_best_params': cf['best_params'], 'test_evaluated': False,
        'training_data': 'train only; no train+validation refit yet'}
    return protocol, config, data, context, comparator, popularity, cf, plan


def run_validation_search(root=ROOT, *, reuse_completed=False, batch_size=128):
    root = Path(root)
    protocol, config, data, context, cf_result, popularity, cf_summary, plan = prepare(root)
    plan['batch_size'] = batch_size
    output = root / 'reports/ials_validation'
    local = root / 'data/validation/ials'
    output.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    summary_path = output / 'summary.json'
    if reuse_completed and summary_path.exists():
        cached = json.loads(summary_path.read_text())
        same_plan = all(cached.get(key) == value for key, value in plan.items())
        intact = all((root/path).is_file() and sha256(root/path) == digest
                     for path, digest in cached.get('artifact_hashes', {}).items())
        if same_plan and intact and cached.get('completed_trials') == config['budget']:
            print('Loaded verified iALS results: identical code, configuration, inputs and cohort.', flush=True)
            best = load_result(local, 'best', cached['best_validation_metrics'], context)
            return assemble(output, cached, best, context, cf_result, popularity, True)
        print('iALS cache differs from current code/inputs: running the fixed search again.', flush=True)
    grid = list(product(config['grid']['n_factors'], config['grid']['alpha'], config['grid']['regularization']))
    if len(grid) != config['budget']:
        raise ValueError('Grid does not match the preregistered budget')
    write_json(output / 'search_plan.json', plan)  # before any real-data iALS fit
    records, curve = [], []
    best_key, best, best_params, best_factors = None, None, None, None
    best_fit_seconds = None
    for trial, (dimensions, alpha, regularization) in enumerate(grid, 1):
        params = {'n_factors': dimensions, 'alpha': alpha, 'regularization': regularization,
                  'n_iterations': config['n_iterations'], 'seed': config['seed'],
                  'solver_batch_size': config['solver_batch_size']}

        def progress(iteration, objective, seconds):
            curve.append({'trial': trial, 'n_factors': dimensions, 'alpha': alpha,
                'regularization': regularization, 'iteration': iteration,
                'objective': objective, 'epoch_seconds': seconds})
            pd.DataFrame(curve).sort_values(['trial','iteration']).to_csv(output/'training_curve.csv', index=False)

        started = perf_counter()
        model = IALS(**params).fit(data, progress_callback=progress)
        fit_seconds = perf_counter() - started
        curve.append({'trial': trial, 'n_factors': dimensions, 'alpha': alpha,
            'regularization': regularization, 'iteration': 0,
            'objective': model.loss_history_[0], 'epoch_seconds': 0.})
        result = evaluate_model(model, context, batch_size=batch_size)
        record = {'trial': trial, **params, **result.metrics, 'fit_seconds': fit_seconds,
                  'initial_objective': model.loss_history_[0], 'final_objective': model.loss_history_[-1],
                  'last_relative_loss_decrease': (model.loss_history_[-2]-model.loss_history_[-1])/model.loss_history_[-2]}
        records.append(record)
        pd.DataFrame(records).to_csv(output/'search_results.csv', index=False)
        key = (-result.metrics['ndcg_at_10'], dimensions, alpha, -regularization)
        if best_key is None or key < best_key:
            best_key, best, best_params, best_fit_seconds = key, result, params, fit_seconds
            best_factors = (model.user_factors_.copy(), model.item_factors_.copy())
        print(f'[{trial:02d}/{len(grid)}] factors={dimensions}, alpha={alpha:g}, lambda={regularization:g}: '
              f'NDCG@10={result.metrics["ndcg_at_10"]:.6f}, Recall@10={result.metrics["recall_at_10"]:.6f}', flush=True)
        del model
    curve_table = pd.DataFrame(curve).sort_values(['trial','iteration']).reset_index(drop=True)
    curve_table.to_csv(output/'training_curve.csv', index=False)
    comparison = pd.DataFrame([
        {'model':'Popularity', **popularity.metrics, 'fit_seconds':0.},
        {'model':'ItemKNN', **cf_result.metrics, 'fit_seconds':pd.read_csv(root/'reports/item_knn_validation/comparison.csv').iloc[1].fit_seconds},
        {'model':'iALS', **best.metrics, 'fit_seconds':best_fit_seconds}])
    comparison.to_csv(output/'comparison.csv', index=False)
    results = {'Popularity': popularity, 'ItemKNN': cf_result, 'iALS': best}
    pd.concat({name:history_slices(value) for name,value in results.items()}, names=['model']).to_csv(output/'history_slices.csv')
    pd.concat({name:popularity_slices(value,context) for name,value in results.items()}, names=['model']).to_csv(output/'popularity_slices.csv')
    boot = protocol['confidence_intervals']
    intervals = paired_bootstrap(best, cf_result, resamples=boot['resamples'], seed=boot['seed'],
        confidence=boot['confidence'], result_name='iALS', baseline_name='ItemKNN')
    intervals.to_csv(output/'bootstrap_vs_cf.csv', index=False)
    with deterministic_gzip(local/'best_per_user.csv.gz') as stream:
        best.per_user.to_csv(stream, lineterminator='\n')
    np.save(local/'best_recommendation_indices.npy', best.recommendations)
    np.savez(local/'best_factors.npz', user_factors=best_factors[0], item_factors=best_factors[1])
    summary = {**plan, 'completed_trials':len(records), 'best_params':best_params,
        'best_validation_metrics':best.metrics, 'cf_validation_metrics':cf_result.metrics,
        'bootstrap_intervals':intervals.to_dict(orient='records'),
        'interval_interpretation':'validation after hyperparameter selection; not an independent test claim',
        'cf_timing_note':'CF timing reused from its prior recorded run; input/indexing excluded from model fit times',
        'artifact_hashes':{str(path.relative_to(root)):sha256(path) for path in
            [*sorted(local.iterdir()), *sorted(output.glob('*.csv'))] if path.is_file()}}
    write_json(summary_path, summary)
    return assemble(output, summary, best, context, cf_result, popularity, False)


def assemble(output, summary, best, context, comparator, popularity, cached):
    return {'summary':summary, 'search':pd.read_csv(output/'search_results.csv'),
        'comparison':pd.read_csv(output/'comparison.csv'),
        'history_slices':pd.read_csv(output/'history_slices.csv').set_index(['model','history_slice']),
        'popularity_slices':pd.read_csv(output/'popularity_slices.csv').set_index(['model','item_slice']),
        'intervals':pd.read_csv(output/'bootstrap_vs_cf.csv'),
        'curve':pd.read_csv(output/'training_curve.csv'), 'best':best, 'context':context,
        'cf':comparator, 'popularity':popularity, 'reused_cache':cached}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-completed', action='store_true')
    arguments = parser.parse_args()
    results = run_validation_search(reuse_completed=arguments.reuse_completed)
    print(results['comparison'].to_string(index=False))
