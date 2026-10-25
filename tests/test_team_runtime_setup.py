"""Release setup contracts; these fixtures are not scientific performance data."""
import hashlib,importlib.util,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('team_runtime_setup',Path(__file__).resolve().parents[1]/'scripts/setup_runtime.py')
setup=importlib.util.module_from_spec(spec);spec.loader.exec_module(setup)

class TeamRuntimeSetupTests(unittest.TestCase):
    def fixture(self,root,entries):
        archive=root/'test.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for name,body in entries.items():z.writestr('bundle/'+name,body)
        (root/'.submission').mkdir();(root/'.submission/timestamps.json').write_text('{}')
        return archive,{'archive_root':'bundle','sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    def test_restores_runtime_without_overwriting_working_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();(root/'frontend').mkdir();(root/'frontend/app.js').write_text('working code');(root/'data').mkdir();(root/'data/README.md').write_text('working docs')
            archive,manifest=self.fixture(root,{'data/input.bin':b'input','outputs/model.json':b'{}','frontend/app.js':b'old code','data/README.md':b'old docs'})
            with patch.object(setup,'ROOT',root):self.assertEqual(setup.restore(archive,manifest),2)
            self.assertEqual((root/'frontend/app.js').read_text(),'working code')
            self.assertEqual((root/'data/README.md').read_text(),'working docs')
            self.assertEqual((root/'data/input.bin').read_bytes(),b'input')
    def test_rejects_wrong_checksum_before_restoring_anything(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();archive,manifest=self.fixture(root,{'data/input.bin':b'input'});manifest['sha256']='0'*64
            with patch.object(setup,'ROOT',root),self.assertRaisesRegex(ValueError,'checksum'):setup.restore(archive,manifest)
            self.assertFalse((root/'data').exists())
    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();archive,manifest=self.fixture(root,{'../outside.txt':b'bad'})
            with patch.object(setup,'ROOT',root),self.assertRaisesRegex(ValueError,'unsafe'):setup.restore(archive,manifest)
    def test_existing_directory_alias_cannot_redirect_into_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();archive,manifest=self.fixture(root,{'data/alias/app.js':b'bad'})
            (root/'frontend').mkdir();(root/'frontend/app.js').write_text('working code');(root/'data').mkdir();(root/'data/alias').symlink_to('../frontend',target_is_directory=True)
            with patch.object(setup,'ROOT',root),self.assertRaisesRegex(ValueError,'working code'):setup.restore(archive,manifest)
            self.assertEqual((root/'frontend/app.js').read_text(),'working code')

if __name__=='__main__':unittest.main()
