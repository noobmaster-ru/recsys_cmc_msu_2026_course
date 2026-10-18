"""Validation report rendering must not require data or read frozen final reports."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.fm_extended_reporting import load_extended_report
from scripts.prepare_data import sha256


class ExtendedReportingTests(unittest.TestCase):
    def fixture(self,root):
        directory=root/'reports/fm_extended_validation';directory.mkdir(parents=True)
        names=['search_results.csv','training_curves.csv','sampling_history.csv','comparison.csv','bootstrap.csv','history_slices.csv','popularity_slices.csv']
        for name in names:(directory/name).write_text('value\n1\n')
        summary={'test_evaluated':False,'train_validation_refit':False,'source_hashes':{},
                 'artifact_hashes':{str((directory/name).relative_to(root)):sha256(directory/name) for name in names}}
        (directory/'summary.json').write_text(json.dumps(summary))
        return directory,summary

    def test_load_without_data_and_reject_report_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);directory,summary=self.fixture(root)
            self.assertFalse((root/'data').exists())
            self.assertEqual(len(load_extended_report(root)['comparison']),1)
            (directory/'comparison.csv').write_text('value\n999\n')
            with self.assertRaisesRegex(ValueError,'Changed extended validation report'):load_extended_report(root)

    def test_injected_final_artifact_cannot_be_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);directory,summary=self.fixture(root)
            summary['artifact_hashes']={'reports/final_test/summary.json':'not-a-real-digest'}
            (directory/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(RuntimeError,'Validation-only'):load_extended_report(root)


if __name__=='__main__':unittest.main()
