"""Render committed aggregate evidence without training or reading private artifacts.

This is a presentation path. Full artifact verification (including excluded weights)
remains in the original training pipelines; this loader verifies committed evidence.
"""
import hashlib
import json
from pathlib import Path
import pandas as pd


def verify_reports(root):
    root = Path(root)
    manifest = json.loads((root/'reports/submission_manifest.json').read_text())
    for relative, expected in manifest.items():
        path = root/relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Changed submission evidence: {relative}')
    # Check original experiment provenance for every committed source/report.
    for path in sorted((root/'reports').glob('*_validation/summary.json')) + [root/'reports/final_test/summary.json']:
        summary = json.loads(path.read_text())
        for group in ['source_hashes', 'artifact_hashes']:
            for relative, expected in summary.get(group, {}).items():
                if relative.startswith('data/'):
                    continue  # Deliberately excluded weights and personal outputs.
                if hashlib.sha256((root/relative).read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Changed experiment evidence: {relative}')
    return manifest


def load_validation_report(root, model):
    root = Path(root)
    verify_reports(root)
    directory = root/f'reports/{model}_validation'
    summary = json.loads((directory/'summary.json').read_text())
    if summary['test_evaluated'] is not False:
        raise ValueError('Expected validation-only evidence')
    interval_file = {'item_knn':'bootstrap.csv','ials':'bootstrap_vs_cf.csv'}.get(model,'bootstrap_vs_baselines.csv')
    result = {'summary': summary}
    for key, file in {'search':'search_results.csv','comparison':'comparison.csv',
                      'intervals':interval_file,'history_slices':'history_slices.csv',
                      'popularity_slices':'popularity_slices.csv'}.items():
        result[key] = pd.read_csv(directory/file)
    result['history_slices'] = result['history_slices'].set_index(['model','history_slice'])
    result['popularity_slices'] = result['popularity_slices'].set_index(['model','item_slice'])
    if (directory/'training_curve.csv').exists():
        result['curve'] = pd.read_csv(directory/'training_curve.csv')
    ablation = {'fm':'metadata_ablation.csv','timesvdpp':'temporal_ablation.csv'}.get(model)
    if ablation:
        result['ablation'] = pd.read_csv(directory/ablation)
    return result


def load_final_report(root):
    root = Path(root)
    verify_reports(root)
    directory = root/'reports/final_test'
    s = json.loads((directory/'summary.json').read_text())
    lock_path = root/'configs/final_models.json'
    if hashlib.sha256(lock_path.read_bytes()).hexdigest() != s['locked_config_sha256']:
        raise ValueError('Changed final lock')
    if s['completed_refits'] != 5 or s['completed_test_evaluations'] != 6 or not s['test_evaluated']:
        raise ValueError('Incomplete final evaluation')
    for name, relative in s['validation_reports'].items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest() != s['validation_summary_hashes'][name]:
            raise ValueError('Changed validation selection evidence')
    result = {'summary':s}
    for key, file in {'comparison':'comparison.csv','intervals':'bootstrap_vs_validation_reference.csv',
                      'history_slices':'history_slices.csv','popularity_slices':'popularity_slices.csv',
                      'validation_vs_test':'validation_vs_test.csv'}.items():
        result[key] = pd.read_csv(directory/file)
    result['history_slices'] = result['history_slices'].set_index(['model','history_slice'])
    result['popularity_slices'] = result['popularity_slices'].set_index(['model','item_slice'])
    result['training_curves'] = {name:pd.read_csv(directory/f'training_{folder}.csv')
                                for name,folder in [('iALS','ials'),('FM','fm'),('timeSVD++','timesvdpp')]}
    return result
