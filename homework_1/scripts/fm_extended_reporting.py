"""Read only completed extended-validation evidence; never read final/test reports."""
from pathlib import Path
import json
import sys
import pandas as pd
from scripts.tune_fm_extended import ValidationOnlyGuard
from scripts.prepare_data import sha256


def load_extended_report(root):
    root=Path(root).resolve();guard=ValidationOnlyGuard(root);sys.addaudithook(guard)
    try:
        directory=root/'reports/fm_extended_validation'
        summary=json.loads((directory/'summary.json').read_text())
        if summary['test_evaluated'] or summary['train_validation_refit']:raise ValueError('Expected validation only')
        for relative,digest in summary['source_hashes'].items():
            if sha256(root/relative)!=digest:raise ValueError('Changed extended search source: '+relative)
        for relative,digest in summary['artifact_hashes'].items():
            if relative.startswith('data/'):continue
            if sha256(root/relative)!=digest:raise ValueError('Changed extended validation report: '+relative)
        files={'search':'search_results.csv','curves':'training_curves.csv','sampling':'sampling_history.csv',
               'comparison':'comparison.csv','intervals':'bootstrap.csv','history':'history_slices.csv','popularity':'popularity_slices.csv'}
        return {'summary':summary,**{key:pd.read_csv(directory/name) for key,name in files.items()}}
    finally:guard.active=False
