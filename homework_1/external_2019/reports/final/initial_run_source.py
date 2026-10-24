"""Fixed-v2 external evaluation: freeze, refit all five, then unseal 2019 labels."""
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HW = ROOT.parent
sys.path.insert(0, str(HW))
from data_utils import FitData
from evaluation import (EvaluationContext, evaluate_model, evaluate_popularity,
                        history_slices, popularity_slices, paired_bootstrap)
from scripts.prepare_data import sha256, deterministic_gzip
from scripts.run_final_test import refit_all, restore_model
from scripts.final_config import MODEL_FOLDERS


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def immutable_json(path, obj):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != obj:
            raise ValueError('Frozen protocol differs: ' + str(path))
    else:
        write_json(path, obj)


class TestGate:
    def __init__(self, root=ROOT, hw=HW):
        self.root = Path(root).resolve()
        self.old_data = (Path(hw) / 'data').resolve()
        self.allowed = False
        self.active = True
        self.label_open_operations = 0
        self.forbidden_attempts = []
        self.sealed = {(self.root / 'data' / name).resolve() for name in
                       ['Video_Games_0core_2023.csv.gz', 'test_2019_all.csv.gz', 'test_2019_selected.csv.gz']}

    def __call__(self, event, args):
        if not self.active or event != 'open' or not isinstance(args[0], (str, bytes, Path)):
            return
        value = args[0].decode() if isinstance(args[0], bytes) else args[0]
        path = Path(value).resolve()
        if self.old_data == path or self.old_data in path.parents:
            self.forbidden_attempts.append(str(path))
            raise RuntimeError('Old v2 data/weights are not inputs to the external run')
        if path in self.sealed:
            if not self.allowed:
                self.forbidden_attempts.append(str(path))
                raise RuntimeError('2019 labels/source sealed until all five refits complete')
            self.label_open_operations += 1


def preserved_v2():
    protected = json.loads((ROOT / 'reports/selection_plan.json').read_text())['protected_v2_files']
    for rel, digest in protected.items():
        if sha256(HW / rel) != digest:
            raise ValueError('Protected v2 evidence modified: ' + rel)
    return len(protected)


def freeze_protocol(lock, feasibility, source_hashes):
    return {
        'experiment': 'amazon2023_external_2019_fixed_v2',
        'selection_plan_sha256': sha256(ROOT / 'reports/selection_plan.json'),
        'feasibility_sha256': sha256(ROOT / 'reports/feasibility.json'),
        'original_v2_lock_sha256': sha256(HW / 'configs/final_models.json'),
        'models': lock['models'], 'fm_preprocessing': lock['fm_preprocessing'],
        'configuration_policy': 'exact v2 hyperparameters, preprocessing constants, seeds, epochs; no new HPO or early stopping',
        'fit': 'prepared Amazon Reviews 2023 fit-only 5-core, all timestamps before 2019-01-01 UTC; fresh weights',
        'query_time_utc': '2019-01-01T00:00:00+00:00', 'query_timestamp': 1546300800,
        'test_end_utc_exclusive': '2020-01-01T00:00:00+00:00',
        'test_file': 'data/test_2019_all.csv.gz',
        'input_hashes': {key: feasibility['artifacts'][key] for key in
                         ['data/fit_before_2019.csv.gz', 'data/test_2019_all.csv.gz', 'data/test_2019_selected.csv.gz']},
        'cohort': 'known fit users, at least one unseen fit-catalog test item rated >=4, at least 10 candidates; identical for all models',
        'candidates': 'full selected fit catalog minus every observed fit item, including ratings 1-3; no candidate sampling',
        'relevance_threshold': 4, 'k': 10, 'ties': 'lexicographic parent_asin/item_id',
        'fallback': 'same shared evaluator: whole-user positive-fit popularity if any candidate score nonfinite; no user dropping',
        'history_policy': 'frozen at query for the whole 2019 window; no future event dates passed to scorers',
        'primary_metric': 'macro NDCG@10', 'secondary_metric': 'macro Recall@10',
        'diagnostics': ['catalog coverage', 'fallback', 'history 1-4/5-9/10+', 'head top10% / tail by fit-positive count', 'training curves'],
        'validation_reference': lock['validation_reference'],
        'reference_policy': 'EASE selected on original v2 validation; no new-test winner selection for reference',
        'evaluation_batch_size': lock['evaluation_batch_size'],
        'bootstrap': {'resamples': 2000, 'seed': 42, 'confidence': .95,
                      'scope': 'paired identical users; fixed models; no seed uncertainty or multiple-comparison adjustment'},
        'test_history': 'IDs/timestamps and cohort feasibility counts inspected previously; test ratings not analyzed; no model test metrics previously computed',
        'independence_scope': 'previously unused external temporal holdout, not an IID sample or historical snapshot guarantee',
        'post_test_policy': 'no changes to models, epochs, features, cohort, thresholds or protocol from 2019 results; no test ablation fits',
        'source_hashes': source_hashes,
    }


def main():
    output = ROOT / 'reports/final'
    local = ROOT / 'data/final'
    if output.exists():
        raise RuntimeError('Existing final directory; refusing to overwrite or silently rerun test')
    feasibility = json.loads((ROOT / 'reports/feasibility.json').read_text())
    if feasibility['models_trained'] or feasibility['test_labels_analyzed']:
        raise ValueError('Expected pre-evaluation data audit')
    protected = preserved_v2()
    lock = json.loads((HW / 'configs/final_models.json').read_text())
    if lock['experiment'] != 'temporal_metadata_v2' or lock['validation_reference'] != 'EASE':
        raise ValueError('Wrong original lock')
    files = ['models/item_knn.py', 'models/ials.py', 'models/ease.py', 'models/fm.py', 'models/timesvdpp.py',
             'data_utils.py', 'evaluation.py', 'scripts/run_final_test.py', 'scripts/final_config.py',
             'scripts/prepare_data.py', 'scripts/tune_ials.py', 'requirements.txt',
             'external_2019/run_final.py']
    sources = {rel: sha256(HW / rel) for rel in files}
    protocol = freeze_protocol(lock, feasibility, sources)
    immutable_json(ROOT / 'reports/protocol.json', protocol)
    output.mkdir()
    local.mkdir(parents=True)

    def journal(event, **kwargs):
        row = {'event': event, 'utc': datetime.now(timezone.utc).isoformat(), **kwargs}
        with (output / 'execution_journal.jsonl').open('a') as out:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

    journal('protocol_frozen_test_labels_sealed', protocol_sha256=sha256(ROOT / 'reports/protocol.json'))
    gate = TestGate()
    sys.addaudithook(gate)
    try:
        fit_path = ROOT / 'data/fit_before_2019.csv.gz'
        if sha256(fit_path) != protocol['input_hashes']['data/fit_before_2019.csv.gz']:
            raise ValueError('Fit hash changed')
        frame = pd.read_csv(fit_path, dtype={'user_id': 'string', 'item_id': 'string', 'rating': 'int8', 'timestamp': 'int64'})
        data = FitData.from_frame(frame)
        if data.max_timestamp >= protocol['query_timestamp']:
            raise ValueError('Fit reaches the query boundary')
        if (len(frame), len(data.user_ids), len(data.item_ids)) != (520585, 59204, 15650):
            raise ValueError('Prepared cohort changed')
        # Reuse the original generic refit/restore helpers without modifying v2.
        # Only the query timestamp is changed in a LOCAL lock copy; all model params stay exact.
        external_lock = {**lock, 'query_time_utc': protocol['query_time_utc']}
        journal('new_fit_verified', events=len(frame), users=len(data.user_ids), items=len(data.item_ids), fit_fingerprint=data.fingerprint)
        manifest = refit_all(data, frame, {}, external_lock, output, local,
                             event=lambda event, **kw: journal(event, **kw))
        gc.collect()
        if len(manifest) != 5 or gate.label_open_operations or gate.forbidden_attempts:
            raise RuntimeError('Five fresh fits required before test access')
        journal('all_five_refits_complete_test_still_sealed')
        gate.allowed = True
        journal('test_unsealed_for_fixed_evaluation')
        test_path = ROOT / protocol['test_file']
        if sha256(test_path) != protocol['input_hashes'][protocol['test_file']]:
            raise ValueError('Test bytes changed')
        test = pd.read_csv(test_path, dtype={'user_id': 'string', 'item_id': 'string', 'rating': 'float64', 'timestamp': 'int64'})
        if not test.rating.isin([1., 2., 3., 4., 5.]).all():
            raise ValueError('Invalid test ratings')
        test['rating'] = test.rating.astype('int8')
        if not ((test.timestamp >= protocol['query_timestamp']) & (test.timestamp < 1577836800)).all():
            raise ValueError('Test outside 2019')
        journal('test_read', rows=len(test), sha256=protocol['input_hashes'][protocol['test_file']])
        context = EvaluationContext.from_frames(frame, test, fit_data=data, k=10,
                                                relevance_threshold=4, query_timestamp=protocol['query_timestamp'])
        results = {}
        rows = []
        for name in ['Popularity', *MODEL_FOLDERS]:
            print('Evaluating fixed external test:', name, flush=True)
            if name == 'Popularity':
                result = evaluate_popularity(context, batch_size=protocol['evaluation_batch_size'])
                fit_seconds = 0.
            else:
                model = restore_model(name, data, frame, external_lock, local / MODEL_FOLDERS[name])
                result = evaluate_model(model, context, batch_size=protocol['evaluation_batch_size'])
                fit_seconds = manifest[name]['fit_seconds']
                del model
                gc.collect()
            results[name] = result
            rows.append({'model': name, **result.metrics, 'fit_seconds': fit_seconds})
            folder = 'popularity' if name == 'Popularity' else MODEL_FOLDERS[name]
            directory = local / folder
            directory.mkdir(exist_ok=True)
            deterministic_gzip(directory / 'test_per_user.csv.gz', result.per_user.to_csv(index=True).encode())
            np.save(directory / 'test_recommendation_indices.npy', result.recommendations, allow_pickle=False)
            journal('test_evaluated', model=name, cohort_fingerprint=result.cohort_fingerprint)
        comparison = pd.DataFrame(rows)
        comparison.to_csv(output / 'comparison.csv', index=False)
        reference = protocol['validation_reference']
        intervals = []
        for name, result in results.items():
            table = paired_bootstrap(result, results[reference], resamples=2000, seed=42, confidence=.95,
                                     result_name=name, baseline_name=reference)
            table['comparison'] = name + ' - ' + reference
            intervals.append(table)
        pd.concat(intervals, ignore_index=True).to_csv(output / 'bootstrap_vs_EASE.csv', index=False)
        hs, ps = [], []
        for name, result in results.items():
            h = history_slices(result).reset_index(); h.insert(0, 'model', name); hs.append(h)
            p = popularity_slices(result, context).reset_index(); p.insert(0, 'model', name); ps.append(p)
        pd.concat(hs, ignore_index=True).to_csv(output / 'history_slices.csv', index=False)
        pd.concat(ps, ignore_index=True).to_csv(output / 'popularity_slices.csv', index=False)
        all_positive = int((test.rating >= 4).sum())
        known_positive = test[(test.rating >= 4) & test.user_id.isin(data.user_index) & test.item_id.isin(data.item_index)]
        if len(known_positive) != context.diagnostics['relevant_interactions']:
            raise ValueError('Unexpected seen relevance/candidate exclusions')
        fit_info = {'events': len(frame), 'users': len(data.user_ids), 'items': len(data.item_ids),
                    'max_timestamp': data.max_timestamp}
        coverage = {'all_test_events': len(test), 'all_positive_test_events': all_positive,
                    'available_relevant_events': context.diagnostics['relevant_interactions'],
                    'positive_event_coverage': context.diagnostics['relevant_interactions'] / all_positive,
                    'selected_all_rating_events_from_feasibility': 10092,
                    'users_with_any_selected_test_event_from_feasibility': 5536}
        if preserved_v2() != protected or gate.forbidden_attempts:
            raise ValueError('Unexpected protected evidence or IO change')
        journal('final_evaluation_complete_no_choices_changed', refits=5, model_evaluations=5, controls=1)
        artifacts = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
        artifacts.update({str(p.relative_to(ROOT)): sha256(p) for p in sorted(local.rglob('*')) if p.is_file()})
        summary = {'experiment': protocol['experiment'], 'protocol_sha256': sha256(ROOT / 'reports/protocol.json'),
                   'models': lock['models'], 'completed_refits': 5, 'completed_test_evaluations': 6,
                   'new_hyperparameter_trials': 0, 'test_evaluated': True, 'query_timestamp': protocol['query_timestamp'],
                   'fit': fit_info, 'fit_fingerprint': data.fingerprint, 'cohort': context.diagnostics,
                   'cohort_fingerprint': context.fingerprint, 'test_scope': coverage,
                   'validation_reference': reference, 'results': {n: r.metrics for n, r in results.items()},
                   'refit_manifest': manifest, 'protected_v2_files_verified_unchanged': protected,
                   'source_hashes': sources, 'artifact_hashes': artifacts,
                   'io_audit': {'label_open_operations_after_refits': gate.label_open_operations,
                                'forbidden_attempts': gate.forbidden_attempts}}
        write_json(output / 'summary.json', summary)
        print(comparison.to_string(index=False), flush=True)
        print(json.dumps({'cohort': context.diagnostics, 'scope': coverage}, indent=2), flush=True)
    finally:
        gate.active = False


if __name__ == '__main__':
    main()
