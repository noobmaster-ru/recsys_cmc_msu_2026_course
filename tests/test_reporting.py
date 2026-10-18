"""A published report must render without excluded datasets and model weights."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from scripts.reporting import load_final_report, load_validation_report

ROOT = Path(__file__).resolve().parents[1]


class ReportingTests(unittest.TestCase):
    def test_aggregates_load_without_any_data_directory_and_detect_tampering(self):
        with tempfile.TemporaryDirectory(prefix='report without data ') as temp:
            root = Path(temp)
            manifest = json.loads((ROOT/'reports/submission_manifest.json').read_text())
            # Copy only published evidence; never even probe excluded weights.
            for relative in [*manifest, 'reports/submission_manifest.json']:
                target = root/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(ROOT/relative,target)
            self.assertFalse((root/'data').exists())
            for model, trials in [('item_knn',12),('ials',12),('ease',7),('fm',8),('timesvdpp',8)]:
                report = load_validation_report(root,model)
                self.assertEqual(len(report['search']),trials)
                self.assertFalse(report['summary']['test_evaluated'])
            final = load_final_report(root)
            self.assertEqual(len(final['comparison']),6)
            self.assertEqual(final['summary']['cohort']['evaluation_users'],9521)
            path = root/'reports/final_test/comparison.csv'
            path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'Changed submission evidence'):
                load_final_report(root)

    def test_reproduction_refuses_existing_output_without_changing_it(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp)/'keep.txt'
            marker.write_text('keep')
            run = subprocess.run([sys.executable,str(ROOT/'scripts/reproduce.py'),'--output',temp],
                                 capture_output=True,text=True,timeout=10)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('new directory',run.stderr)
            self.assertEqual(marker.read_text(),'keep')


if __name__ == '__main__':
    unittest.main()
