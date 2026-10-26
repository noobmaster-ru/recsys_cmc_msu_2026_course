"""Allow the documented README edit without changing frozen experiment hashes."""
import json


def verify_preserved_files(root, hw, protected, sha256):
    revisions = {}
    path = root / 'reports/documentation_revision.json'
    if path.exists():
        entry = json.loads(path.read_text())
        if (entry['protected_path'] != 'README.md'
                or entry['repository_path'] != 'homework_1/README.md'
                or entry['original_sha256'] != protected.get('README.md')):
            raise ValueError('Only the documented README revision is allowed')
        revisions['README.md'] = entry['current_sha256']
    for relative, original in protected.items():
        if sha256(hw / relative) != revisions.get(relative, original):
            raise ValueError('Protected file changed: ' + relative)
    return len(protected) - len(revisions), sorted(revisions)
