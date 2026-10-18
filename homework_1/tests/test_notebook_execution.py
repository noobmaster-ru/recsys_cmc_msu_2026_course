"""The validation-only notebook mode must skip old cells and block test IO."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import nbformat


class NotebookExecutionTests(unittest.TestCase):
    def test_fm_extended_section_preserves_all_historical_results(self):
        self.check_section('fm_extended')

    def test_ials_section_preserves_old_outputs_and_blocks_test_file(self):
        self.check_section('ials')

    def test_ease_section_preserves_old_outputs_and_blocks_test_file(self):
        self.check_section('ease')

    def test_fm_section_preserves_old_outputs_and_blocks_test_file(self):
        self.check_section('fm')

    def test_timesvdpp_section_preserves_old_outputs_and_blocks_test_file(self):
        self.check_section('timesvdpp')

    def test_final_section_preserves_old_outputs_and_blocks_raw_test_file(self):
        self.check_section('final')

    def check_section(self, section):
        source_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='ials notebook ') as directory:
            root = Path(directory)
            (root/'scripts').mkdir()
            (root/'data/splits').mkdir(parents=True)
            fake_test = root/'data/splits/test.csv.gz'
            fake_test.write_text('DO NOT READ', encoding='utf-8')
            shutil.copy(source_root/'scripts/execute_notebook.py', root/'scripts/execute_notebook.py')
            old = nbformat.v4.new_code_cell("raise AssertionError('Historical EDA must not run')")
            old.execution_count = 19
            old.outputs = [nbformat.v4.new_output('stream', name='stdout', text='preserved output\n')]
            setup = nbformat.v4.new_code_cell('shared_value = 123', metadata={'tags':['shared_setup']})
            selected = nbformat.v4.new_code_cell(
                'assert shared_value == 123\nblocked = False\ntry:\n'
                f'    open({str(fake_test)!r}).read()\n'
                'except RuntimeError:\n    blocked = True\nassert blocked\nprint("test IO blocked")',
                metadata={'tags':[section]})
            notebook = nbformat.v4.new_notebook(cells=[setup, old, selected])
            nbformat.write(notebook,root/'experiments.ipynb')
            run = subprocess.run([sys.executable,str(root/'scripts/execute_notebook.py'),'--section',section],
                capture_output=True,text=True,timeout=45)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            executed = nbformat.read(root/'experiments.ipynb',as_version=4)
            self.assertEqual(executed.cells[1],old)
            self.assertIsNone(executed.cells[0].execution_count)
            self.assertIsNotNone(executed.cells[2].execution_count)
            self.assertIn('test IO blocked',executed.cells[2].outputs[0].text)


if __name__=='__main__':
    unittest.main()
