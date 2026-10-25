"""Restore frozen release data/checkpoints into a clone without replacing code."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,os,shutil,stat,tempfile,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'docs/release_manifest.json'

def request(url,token):
 headers={'User-Agent':'DeepLearrningProject-runtime-setup','Accept':'application/octet-stream'}
 if token:headers['Authorization']='Bearer '+token
 return urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120)

def allowed(name):
 parts=PurePosixPath(name).parts
 return name!='data/README.md' and bool(parts) and (parts[0] in ('data','outputs') or (parts[0]=='models' and (len(parts)==2 and parts[-1].endswith(('.keras','.h5')))))

def restore(archive,manifest):
 digest=hashlib.sha256()
 with archive.open('rb') as handle:
  for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
 if digest.hexdigest()!=manifest['sha256']:raise ValueError('Release ZIP checksum does not match the versioned manifest.')
 links=[];restored=0
 with zipfile.ZipFile(archive) as zipped:
  for item in zipped.infolist():
   parts=PurePosixPath(item.filename).parts
   if not parts or parts[0]!=manifest['archive_root'] or '..' in parts or item.filename.startswith('/'):
    raise ValueError('Unexpected or unsafe ZIP member: '+item.filename)
   relative=PurePosixPath(*parts[1:])
   if not allowed(str(relative)):continue
   destination=ROOT/Path(*relative.parts)
   if not destination.resolve(strict=False).is_relative_to(ROOT):raise ValueError('Runtime path escapes repository.')
   expected_base=ROOT/relative.parts[0]
   if item.is_dir():
    if not destination.resolve(strict=False).is_relative_to(expected_base):raise ValueError('Runtime directory redirects into working code.')
    destination.mkdir(parents=True,exist_ok=True);continue
   if not destination.parent.resolve(strict=False).is_relative_to(expected_base):raise ValueError('Runtime parent redirects into working code.')
   destination.parent.mkdir(parents=True,exist_ok=True)
   if stat.S_ISLNK(item.external_attr>>16):
    link=zipped.read(item).decode()
    resolved=(destination.parent/link).resolve(strict=False)
    if not resolved.is_relative_to(ROOT):raise ValueError('Unsafe runtime alias.')
    links.append((destination,link));continue
   if destination.is_symlink():destination.unlink()
   with zipped.open(item) as source,tempfile.NamedTemporaryFile(dir=destination.parent,delete=False) as temporary:
    shutil.copyfileobj(source,temporary,length=1024*1024);temporary_path=Path(temporary.name)
   os.replace(temporary_path,destination);restored+=1
  for destination,link in links:
   if destination.is_symlink():destination.unlink()
   elif destination.exists():continue
   destination.symlink_to(link)
 # Required nanoseconds are restored before immutable source-fingerprint checks.
 # Restore no values/statistics and do not touch any training/test decisions.
 times=json.loads((ROOT/'.submission/timestamps.json').read_text())
 for name,ns in times.items():
  path=ROOT/name
  if path.is_file():os.utime(path,ns=(path.stat().st_atime_ns,ns))
 return restored

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--archive',type=Path,help='Use an already downloaded release ZIP.')
 args=parser.parse_args();manifest=json.loads(MANIFEST.read_text())
 if args.archive:
  archive=args.archive.expanduser().resolve()
  count=restore(archive,manifest)
 else:
  token=os.getenv('GH_TOKEN') or os.getenv('GITHUB_TOKEN')
  with tempfile.TemporaryDirectory(prefix='bloom-runtime-') as temporary:
   archive=Path(temporary)/manifest['asset_name']
   print('Downloading versioned runtime bundle; approximately 622 MiB.',flush=True)
   with request(manifest['download_url'],token) as source,archive.open('wb') as output:shutil.copyfileobj(source,output,length=1024*1024)
   count=restore(archive,manifest)
 print(f'Restored {count} runtime files. Working source files were not replaced.')
 print('Run: python -m submission serve --port 8020')

if __name__=='__main__':main()
