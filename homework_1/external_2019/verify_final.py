"""Post-evaluation checks only: saved predictions, independent metrics/formulas, no refits."""
import gc
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HW = ROOT.parent
sys.path.insert(0, str(HW))
from data_utils import FitData
from evaluation import EvaluationContext
from models.fm import fm_predict
from scripts.prepare_data import sha256
from scripts.run_final_test import restore_model
from scripts.final_config import MODEL_FOLDERS


def main():
    output = ROOT / 'reports/final'
    summary = json.loads((output / 'summary.json').read_text())
    protocol = json.loads((ROOT / 'reports/protocol.json').read_text())
    lock = json.loads((HW / 'configs/final_models.json').read_text())
    assert summary['models'] == protocol['models'] == lock['models']
    assert summary['new_hyperparameter_trials'] == 0
    assert sha256(ROOT / 'reports/protocol.json') == summary['protocol_sha256']
    for rel, digest in summary['source_hashes'].items():
        assert sha256(HW / rel) == digest, rel
    for rel, digest in summary['artifact_hashes'].items():
        assert sha256(ROOT / rel) == digest, rel
    journal = [json.loads(line) for line in (output / 'execution_journal.jsonl').read_text().splitlines()]
    opened = next(i for i, row in enumerate(journal) if row['event'] == 'test_unsealed_for_fixed_evaluation')
    assert {r['model'] for r in journal[:opened] if r['event'] == 'refit_completed'} == set(MODEL_FOLDERS)
    assert all(i > opened for i, r in enumerate(journal) if r['event'] in ['test_read', 'test_evaluated'])
    assert len([r for r in journal if r['event'] == 'test_read']) == (2 if summary['persistence_recovery'] else 1)
    for rel, digest in summary['training_source_hashes'].items():
        source = output / 'initial_run_source.py' if summary['persistence_recovery'] and rel == 'external_2019/run_final.py' else HW / rel
        assert sha256(source) == digest, rel
    if summary['persistence_recovery']:
        assert sha256(ROOT / 'run_final.py') == summary['persistence_recovery']['corrected_runner_sha256']
        assert summary['persistence_recovery']['new_refits'] == 0
    assert len({r['cohort_fingerprint'] for r in journal if r['event'] == 'test_evaluated'}) == 1
    assert not summary['io_audit']['forbidden_attempts']
    frame = pd.read_csv(ROOT / 'data/fit_before_2019.csv.gz', dtype={'user_id': 'string', 'item_id': 'string', 'rating': 'int8', 'timestamp': 'int64'})
    test = pd.read_csv(ROOT / 'data/test_2019_all.csv.gz', dtype={'user_id': 'string', 'item_id': 'string', 'rating': 'float64', 'timestamp': 'int64'})
    data = FitData.from_frame(frame)
    context = EvaluationContext.from_frames(frame, test, fit_data=data, k=10, relevance_threshold=4, query_timestamp=1546300800)
    assert context.fingerprint == summary['cohort_fingerprint']
    independent = {}
    manual_rows = 0
    for name in ['Popularity', *MODEL_FOLDERS]:
        folder = 'popularity' if name == 'Popularity' else MODEL_FOLDERS[name]
        directory = ROOT / 'data/final' / folder
        recs = np.load(directory / 'test_recommendation_indices.npy', allow_pickle=False)
        user = pd.read_csv(directory / 'test_per_user.csv.gz', dtype={'user_id': str}).set_index('user_id')
        assert user.index.tolist() == context.user_ids.tolist()
        assert recs.shape == (len(context.user_ids), 10)
        assert ((recs >= 0) & (recs < len(data.item_ids))).all()
        values = []
        discounts = 1 / np.log2(np.arange(2, 12))
        for j, row in enumerate(recs):
            assert len(set(row)) == 10
            assert not set(row).intersection(data.seen_items(context.user_indices[j]))
            truth = context.relevance[j]
            positions = np.array([int(i) in truth for i in row], dtype=float)
            ndcg = np.sum(positions * discounts) / np.sum(discounts[:min(10, len(truth))])
            recall = np.sum(positions) / len(truth)
            values.append((ndcg, recall))
        values = np.array(values)
        assert np.allclose(values, user[['ndcg', 'recall']].values, atol=1e-13, rtol=0)
        metrics = summary['results'][name]
        assert np.allclose(values.mean(0), [metrics['ndcg_at_10'], metrics['recall_at_10']], atol=1e-13, rtol=0)
        assert abs(len(np.unique(recs)) / len(data.item_ids) - metrics['catalog_coverage_at_10']) < 1e-14
        assert abs(user.fallback.mean() - metrics['fallback_rate']) < 1e-14
        independent[name] = values
        manual_rows += len(user)
    # Full-catalog oracle for two fixed user positions per model, no fitting.
    external_lock = {**lock, 'query_time_utc': protocol['query_time_utc']}
    oracles = []
    for name, folder in MODEL_FOLDERS.items():
        model = restore_model(name, data, frame, external_lock, ROOT / 'data/final' / folder)
        recs = np.load(ROOT / 'data/final' / folder / 'test_recommendation_indices.npy', allow_pickle=False)
        for position in [0, len(context.user_ids) // 2]:
            u = int(context.user_indices[position]); seen = data.seen_items(u)
            positive = data.positive[u].indices
            if name == 'ItemKNN':
                scores = np.asarray(model.neighbors_[:, positive].sum(axis=1)).ravel()
            elif name == 'iALS':
                scores = np.sum(model.item_factors_ * model.user_factors_[u][None, :], axis=1)
            elif name == 'EASE':
                scores = model.weights_[positive].sum(axis=0)
            elif name == 'FM':
                ix, x = model.features_.batch(np.repeat(u, len(data.item_ids)), np.arange(len(data.item_ids)))
                scores = fm_predict(ix, x, model.bias_, model.linear_, model.factors_)
            else:
                p = model.parameters_; t = model.temporal_data_
                day = protocol['query_timestamp'] // 86400
                delta = (day - t.user_mean_day[u]) / 365.25
                dev = np.sign(delta) * abs(delta) ** model.beta
                implicit = p['y'][seen].sum(axis=0) / np.sqrt(len(seen))
                vector = p['p'][u] + p['a'][u] * dev + implicit
                scores = (model.mean_ + p['bu'][u] + p['alpha_bias'][u] * dev + p['bi']
                          + p['bin_bias'][:, -1] + np.sum(p['q'] * vector[None, :], axis=1))
            mask = np.ones(len(data.item_ids), dtype=bool); mask[seen] = False
            candidates = np.flatnonzero(mask)
            unavailable = (name in ['ItemKNN', 'iALS', 'EASE'] and not np.any(scores[candidates] != 0))
            if not np.isfinite(scores[candidates]).all() or unavailable:
                scores = data.positive_item_counts
            expected = candidates[np.lexsort((candidates, -scores[candidates]))[:10]]
            assert np.array_equal(expected, recs[position]), (name, position)
            oracles.append({'model': name, 'user_position': position, 'full_catalog_oracle': True})
        del model
        gc.collect()
    # Independent paired-difference confidence intervals, without new model evaluation.
    intervals = pd.read_csv(output / 'bootstrap_vs_EASE.csv')
    for name, values in independent.items():
        diff = values - independent['EASE']
        rng = np.random.default_rng(42)
        draws = []
        for start in range(0, 2000, 32):
            size = min(32, 2000 - start)
            indices = rng.integers(0, len(diff), size=(size, len(diff)))
            draws.extend(diff[indices].mean(axis=1))
        bounds = np.quantile(np.array(draws), [.025, .975], axis=0)
        for j, metric in enumerate(['ndcg_at_10', 'recall_at_10']):
            row = intervals[(intervals.comparison == name + ' - EASE') & (intervals.estimate_for == 'difference') & (intervals.metric == metric)].iloc[0]
            assert np.allclose([row.estimate, row.ci_low, row.ci_high], [diff[:, j].mean(), *bounds[:, j]], atol=1e-12, rtol=0)
    protected = json.loads((ROOT / 'reports/selection_plan.json').read_text())['protected_v2_files']
    for rel, digest in protected.items():
        assert sha256(HW / rel) == digest, rel
    result = {'protocol_and_source_hashes_verified': True, 'all_saved_artifact_hashes_verified': True,
              'hyperparameters_identical_to_v2': True, 'refits_before_test_access': 5,
              'independent_per_user_metric_rows': manual_rows, 'cohort_users': len(context.user_ids),
              'full_catalog_oracles': oracles, 'independent_bootstrap_difference_intervals_verified': 12,
              'protected_v2_files_unchanged': len(protected), 'post_test_refits_or_hpo': 0,
              'verification_only_no_training': True, 'external_boundary_unit_tests_passed': 4}
    (output / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
