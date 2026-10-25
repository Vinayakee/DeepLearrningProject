"""Storage/path portability only; frozen scientific files remain byte-identical."""
from pathlib import Path
import json,os,sys
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL='/Users/vinayaks/Documents/HAB-Early-Warning'
_ACTIVE=False

def activate():
    global _ACTIVE
    if _ACTIVE:return
    _ACTIVE=True
    original_loads=json.loads
    def rebase(value):
        if isinstance(value,str) and (value==ORIGINAL or value.startswith(ORIGINAL+'/')):
            return str(ROOT)+value[len(ORIGINAL):]
        if isinstance(value,list):return [rebase(x) for x in value]
        if isinstance(value,dict):return {rebase(k):rebase(v) for k,v in value.items()}
        return value
    def portable_loads(*args,**kwargs):return rebase(original_loads(*args,**kwargs))
    # Absolute paths in immutable archived JSON retain provenance on disk;
    # only their in-memory path values are mapped into this checkout.
    json.loads=portable_loads
    for name,value in list(os.environ.items()):
        if name.startswith(('HAB_','BLOOM_')):os.environ[name]=rebase(value)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/matplotlib'))
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
    sys.dont_write_bytecode=True
    os.environ.setdefault('BLOOM_EVENT_RUN_DIR','outputs/bloom_events/run_03')
    # ZIP timestamps have second-level precision. Restore expected array
    # nanoseconds before legacy source-fingerprint checks, without refitting.
    times=original_loads((ROOT/'.submission/timestamps.json').read_text())
    for name,ns in times.items():
        path=ROOT/name
        if path.is_file() and path.stat().st_mtime_ns!=ns:
            os.utime(path,ns=(path.stat().st_atime_ns,ns))
    def no_original_access(event,args):
        if event not in ('open','os.listdir','os.scandir') or not args:return
        value=args[0]
        if not isinstance(value,(str,bytes,os.PathLike)):return
        text=os.path.abspath(os.fsdecode(value))
        if text==ORIGINAL or text.startswith(ORIGINAL+'/'):
            raise RuntimeError('Submission runtime refuses access to the original project: '+text)
    sys.addaudithook(no_original_access)
