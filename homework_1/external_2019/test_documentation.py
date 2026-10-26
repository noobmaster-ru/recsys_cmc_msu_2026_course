"""A README revision must never permit changes to scientific artifacts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from documentation_policy import verify_preserved_files


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DocumentationRevisionTests(unittest.TestCase):
    def fixture(self, directory):
        hw=Path(directory); root=hw/'external_2019'
        (root/'reports').mkdir(parents=True)
        (hw/'README.md').write_text('original docs')
        (hw/'model.py').write_text('original science')
        protected={p:sha(hw/p) for p in ['README.md','model.py']}
        (hw/'README.md').write_text('concise docs')
        revision={'protected_path':'README.md','repository_path':'homework_1/README.md',
                  'original_sha256':protected['README.md'],'current_sha256':sha(hw/'README.md')}
        (root/'reports/documentation_revision.json').write_text(json.dumps(revision))
        return root,hw,protected,revision

    def test_readme_revision_allowed_but_model_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root,hw,protected,_=self.fixture(temp)
            self.assertEqual(verify_preserved_files(root,hw,protected,sha),(1,['README.md']))
            (hw/'model.py').write_text('changed science')
            with self.assertRaisesRegex(ValueError,'Protected file changed'):
                verify_preserved_files(root,hw,protected,sha)

    def test_revision_cannot_whitelist_another_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root,hw,protected,revision=self.fixture(temp)
            revision['protected_path']='model.py'
            (root/'reports/documentation_revision.json').write_text(json.dumps(revision))
            with self.assertRaisesRegex(ValueError,'Only the documented README'):
                verify_preserved_files(root,hw,protected,sha)


if __name__=='__main__':
    unittest.main()
