"""Packaging contracts only; no new scientific evaluation."""
import hashlib,json,unittest
from pathlib import Path
from submission.bootstrap import ROOT,ORIGINAL

class SubmissionPortabilityTests(unittest.TestCase):
    def test_original_project_access_is_blocked(self):
        with self.assertRaisesRegex(RuntimeError,'refuses access'):
            Path(ORIGINAL,'README.md').read_bytes()
    def test_frozen_metadata_is_unchanged_but_paths_resolve_inside_copy(self):
        name='outputs/bloom_events/run_03/training.json'
        raw=(ROOT/name).read_bytes()
        expected=json.loads((ROOT/'.submission/source_file_identity.json').read_text())[name]['sha256']
        self.assertEqual(hashlib.sha256(raw).hexdigest(),expected)
        self.assertIn(ORIGINAL.encode(),raw)
        loaded=json.loads(raw)
        self.assertEqual(Path(loaded['dataset_dir']),ROOT/'data/processed/bloom_events_v2/dataset_02')
    def test_all_aliases_are_internal(self):
        for path in ROOT.rglob('*'):
            if path.is_symlink():
                with self.subTest(path=str(path.relative_to(ROOT))):
                    self.assertTrue(path.resolve().is_relative_to(ROOT))
                    self.assertTrue(path.exists())

if __name__=='__main__':unittest.main()
