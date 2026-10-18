"""Recompute the homework in a NEW directory; preserve published test evidence."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def evidence_manifest(root):
    import hashlib
    paths = []
    for directory in ['models','configs','scripts','tests','reports']:
        paths.extend(p for p in (root/directory).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix in {'.py','.json','.csv','.jsonl'})
    paths.extend(root/p for p in ['data_utils.py','evaluation.py','requirements.txt'])
    target = root/'reports/submission_manifest.json'
    manifest = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(paths) if p != target}
    target.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New directory outside the source repository')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    target = args.output.expanduser().resolve()
    # A homework lives inside a multi-homework Git repository. Keep generated
    # data/weights outside that repository, including sibling homework folders.
    repository = next((p for p in [source, *source.parents] if (p/'.git').exists()), source)
    if target == repository or repository in target.parents or target.exists():
        parser.error('Output must be a new directory outside the source repository')
    target.mkdir(parents=True)
    for folder in ['models','scripts','configs']:
        shutil.copytree(source/folder,target/folder,ignore=shutil.ignore_patterns('__pycache__','final_models.json'))
    for name in ['data_utils.py','evaluation.py','requirements.txt','experiments.ipynb']:
        shutil.copy2(source/name,target/name)
    (target/'reports').mkdir()
    shutil.copy2(source/'reports/evaluation_protocol.json',target/'reports/evaluation_protocol.json')
    shutil.copy2(source/'reports/presentation_diagnostics.json',target/'reports/presentation_diagnostics.json')
    env = dict(os.environ)
    env['JUPYTER_PATH'] = str(Path(sys.prefix)/'share/jupyter')+os.pathsep+env.get('JUPYTER_PATH','')
    def run(*args):
        subprocess.run([sys.executable,*args],cwd=target,env=env,check=True)
    run('scripts/prepare_data.py','--download')
    # The first 32 cells contain only EDA, split creation and protocol definition.
    run('-c', "import nbformat; from nbclient import NotebookClient; from pathlib import Path; "
        "p=Path('experiments.ipynb'); n=nbformat.read(p,as_version=4); "
        "e=nbformat.v4.new_notebook(cells=n.cells[:32],metadata=n.metadata); "
        "NotebookClient(e,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(Path.cwd())}}).execute(); "
        "n.cells[:32]=e.cells; nbformat.write(n,p)")
    # Separate processes release dense EASE matrices before the next stage.
    for model in ['item_knn','ials','ease','fm','timesvdpp']:
        run(f'scripts/tune_{model}.py')
    run('scripts/run_final_test.py')
    evidence_manifest(target)
    run('scripts/execute_notebook.py','--section','reports')
    print(f'Completed independent reproduction: {target}',flush=True)


if __name__ == '__main__':
    main()
