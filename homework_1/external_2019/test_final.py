"""Boundary/immutability checks before external test access; synthetic files only."""
import copy
import tempfile
import unittest
from pathlib import Path
from run_final import TestGate, immutable_json


class ExternalFinalTests(unittest.TestCase):
    def test_all_test_aliases_and_old_data_blocked_until_refits(self):
        with tempfile.TemporaryDirectory() as directory:
            hw = Path(directory); root = hw / 'external'
            gate = TestGate(root, hw)
            for name in ['test_2019_all.csv.gz', 'test_2019_selected.csv.gz', 'Video_Games_0core_2023.csv.gz']:
                with self.assertRaisesRegex(RuntimeError, 'sealed'):
                    gate('open', (str(root / 'data' / name), 'r', 0))
            gate('open', (str(root / 'data/fit_before_2019.csv.gz'), 'r', 0))
            gate.allowed = True
            gate('open', (str(root / 'data/test_2019_all.csv.gz'), 'r', 0))
            self.assertEqual(gate.label_open_operations, 1)
            with self.assertRaisesRegex(RuntimeError, 'Old v2'):
                gate('open', (str(hw / 'data/final_test/fm/parameters.npz'), 'r', 0))

    def test_frozen_protocol_rejects_changed_choice_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'protocol.json'
            choice = {'models': {'FM': {'n_epochs': 12}}, 'query': '2019-01-01'}
            immutable_json(path, choice)
            original = path.read_bytes()
            immutable_json(path, copy.deepcopy(choice))
            choice['models']['FM']['n_epochs'] = 4
            with self.assertRaisesRegex(ValueError, 'Frozen protocol differs'):
                immutable_json(path, choice)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
