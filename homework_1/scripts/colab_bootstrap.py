"""Notebook startup for Colab; no model training or changes to published evidence."""
from pathlib import Path
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys

COLAB_REPOSITORY = 'https://github.com/noobmaster-ru/recsys_cmc_msu_2026_course.git'
COLAB_BASE_COMMIT = '3cb3e93bed3fb789ed01c9035a36643d2fbc4076'


def in_google_colab():
    if 'google.colab' in sys.modules or os.environ.get('COLAB_RELEASE_TAG'):
        return True
    try:
        return importlib.util.find_spec('google.colab') is not None
    except (ModuleNotFoundError, ValueError):
        return False


def local_homework_root(working=None):
    working = Path.cwd() if working is None else Path(working)
    return next((p for p in [working, working / 'homework_1', working / 'recsys' / 'homework_1']
                 if (p / 'scripts/prepare_data.py').is_file()), None)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_colab_workspace(base=Path('/content')):
    """Use a separate pinned checkout; never reset or overwrite another project."""
    base = Path(base)
    checkout = base / ('recsys_hw1_' + COLAB_BASE_COMMIT[:8])
    if not checkout.exists():
        base.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', COLAB_REPOSITORY, str(checkout)], check=True)
        subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', COLAB_BASE_COMMIT], check=True)
    actual = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COLAB_BASE_COMMIT:
        raise RuntimeError('The dedicated Colab checkout has another revision; use a fresh runtime.')
    # Keep the Colab kernel and preloaded numerical libraries. Install only missing packages.
    requirements = {'numpy': 'numpy>=1.26,<3', 'scipy': 'scipy>=1.11,<2',
                    'pandas': 'pandas>=2.2,<3', 'matplotlib': 'matplotlib>=3.8,<4',
                    'nbformat': 'nbformat>=5.10,<6', 'nbclient': 'nbclient>=0.10,<1',
                    'ipykernel': 'ipykernel>=6,<8'}
    missing = []
    for package, requirement in requirements.items():
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(requirement)
    if missing:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *missing], check=True)
    root = checkout / 'homework_1'
    audit = json.loads((root / 'reports/data_audit.json').read_text())
    ready = all((root / rel).is_file() and file_sha256(root / rel) == info['sha256']
                for rel, info in audit['outputs'].items())
    if not ready:
        print('Downloading and preparing Amazon Reviews 2018 (about 200 MB)...', flush=True)
        subprocess.run([sys.executable, str(root / 'scripts/prepare_data.py'), '--download'], cwd=root, check=True)
    for rel, info in audit['outputs'].items():
        if file_sha256(root / rel) != info['sha256']:
            raise ValueError('Prepared data differ from the published dataset: ' + rel)
    os.chdir(root)
    print('Colab ready: EDA and published model results. Full retraining is a separate command.', flush=True)
    return root


def preserve_or_create_report(path, payload):
    """Validate existing data statistics, retain published runtime-version metadata."""
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text())
        current = json.loads(json.dumps(payload, ensure_ascii=False))
        # EDA statistics must match; package versions describe different runtimes.
        if path.name == 'eda_summary.json':
            saved = {k: v for k, v in saved.items() if k != 'library_versions'}
            current = {k: v for k, v in current.items() if k != 'library_versions'}
        if saved != current:
            raise ValueError('Recomputed data/protocol differ from the published report: ' + path.name)
    else:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
