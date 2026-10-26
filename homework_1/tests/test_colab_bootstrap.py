"""Colab setup must be repeatable and preserve published evidence and the kernel."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.colab_bootstrap import (COLAB_BASE_COMMIT, ensure_colab_workspace,
                                    local_homework_root, preserve_or_create_report)


class ColabStartupTests(unittest.TestCase):
    def fixture(self, base):
        checkout = base / ('recsys_hw1_' + COLAB_BASE_COMMIT[:8])
        root = checkout / 'homework_1'
        (root / 'reports').mkdir(parents=True)
        (root / 'scripts').mkdir()
        (root / 'scripts/prepare_data.py').write_text('# fixture')
        payload = b'prepared fixture data'
        digest = hashlib.sha256(payload).hexdigest()
        (root / 'reports/data_audit.json').write_text(json.dumps({'outputs': {'data/fixture.gz': {'sha256': digest}}}))
        return checkout, root, payload

    def test_clean_setup_downloads_once_and_reuses_workspace(self):
        before = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp); calls = []
                root = base / ('recsys_hw1_' + COLAB_BASE_COMMIT[:8]) / 'homework_1'
                def run(args, **kwargs):
                    calls.append(args)
                    if args[:2] == ['git', 'clone']:
                        self.fixture(base)
                    elif '--download' in args:
                        (root / 'data').mkdir()
                        (root / 'data/fixture.gz').write_bytes(b'prepared fixture data')
                with patch('scripts.colab_bootstrap.subprocess.run', side_effect=run), \
                     patch('scripts.colab_bootstrap.subprocess.check_output', return_value=COLAB_BASE_COMMIT+'\n'), \
                     patch('scripts.colab_bootstrap.importlib.metadata.version', return_value='installed'):
                    self.assertEqual(ensure_colab_workspace(base), root)
                    self.assertEqual(ensure_colab_workspace(base), root)
                self.assertEqual(sum(args[:2] == ['git','clone'] for args in calls), 1)
                self.assertEqual(sum('--download' in args for args in calls), 1)
                self.assertFalse(any('reset' in args for args in calls))
                self.assertEqual(local_homework_root().resolve(), root.resolve())
        finally:
            os.chdir(before)

    def test_existing_other_revision_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(Path(tmp))
            with patch('scripts.colab_bootstrap.subprocess.check_output', return_value='another-commit\n'), \
                 patch('scripts.colab_bootstrap.subprocess.run') as run:
                with self.assertRaisesRegex(RuntimeError, 'another revision'):
                    ensure_colab_workspace(Path(tmp))
                run.assert_not_called()

    def test_only_missing_dependency_is_installed(self):
        import importlib.metadata
        before = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp); _, root, payload = self.fixture(base)
                (root / 'data').mkdir(); (root / 'data/fixture.gz').write_bytes(payload)
                def version(package):
                    if package == 'nbclient':
                        raise importlib.metadata.PackageNotFoundError(package)
                    return 'preloaded'
                with patch('scripts.colab_bootstrap.importlib.metadata.version', side_effect=version), \
                     patch('scripts.colab_bootstrap.subprocess.check_output', return_value=COLAB_BASE_COMMIT+'\n'), \
                     patch('scripts.colab_bootstrap.subprocess.run') as run:
                    ensure_colab_workspace(base)
                args = run.call_args.args[0]
                self.assertEqual(args[-1], 'nbclient>=0.10,<1')
                self.assertFalse(any('numpy' in x or 'ipykernel' in x for x in args))
        finally:
            os.chdir(before)

    def test_published_report_keeps_runtime_versions_and_rejects_changed_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'eda_summary.json'
            preserve_or_create_report(path, {'events': 123, 'library_versions': {'numpy': 'old'}})
            original = path.read_bytes()
            preserve_or_create_report(path, {'events': 123, 'library_versions': {'numpy': 'colab'}})
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, 'differ from the published report'):
                preserve_or_create_report(path, {'events': 124, 'library_versions': {'numpy': 'colab'}})
            self.assertEqual(path.read_bytes(), original)

    def test_existing_protocol_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'evaluation_protocol.json'
            preserve_or_create_report(path, {'query': '2017-01-01'})
            with self.assertRaises(ValueError):
                preserve_or_create_report(path, {'query': '2019-01-01'})


if __name__ == '__main__':
    unittest.main()
