"""Check approved README/notebook compatibility edits against frozen baselines."""
import json


def verify_preserved_files(root, hw, protected, sha256):
    revisions = {}
    allowed = {'documentation_revision.json': 'README.md',
               'notebook_revision.json': 'experiments.ipynb'}
    for filename, relative in allowed.items():
        path = root / 'reports' / filename
        if not path.exists():
            continue
        entry = json.loads(path.read_text())
        if (relative not in protected
                or entry['protected_path'] != relative
                or entry['repository_path'] != 'homework_1/' + relative
                or entry['original_sha256'] != protected.get(relative)):
            raise ValueError('Only the documented README/notebook revision is allowed')
        revisions[relative] = entry['current_sha256']
    for relative, original in protected.items():
        if sha256(hw / relative) != revisions.get(relative, original):
            raise ValueError('Protected file changed: ' + relative)
    return len(protected) - len(revisions), sorted(revisions)
