"""Portable entry point; no original checkout or Git repository required."""
import argparse,os,subprocess,sys,unittest
from pathlib import Path
from .bootstrap import ROOT

def check():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    if not result.wasSuccessful():return 1
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    for script in ('script.js','hab.js','bloom.js'):
        subprocess.run(['node','--check','frontend/'+script],cwd=ROOT,check=True)
    for script in ('bloom_dashboard.mjs','hab_dashboard.mjs','dashboard_race.mjs','dashboard_interaction.mjs'):
        subprocess.run(['node','tests/'+script],cwd=ROOT,check=True)
    print('All Python, dependency and JavaScript checks passed. Git history is intentionally not bundled.')
    return 0

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('serve','check','audit'))
    parser.add_argument('--port',type=int,default=8020)
    args=parser.parse_args();os.chdir(ROOT)
    if args.action=='serve':
        import uvicorn
        uvicorn.run('backend.app:app',host='127.0.0.1',port=args.port)
    elif args.action=='check':sys.exit(check())
    else:
        from operations.baseline_audit import audit
        import json
        print(json.dumps(audit(),indent=2))
if __name__=='__main__':main()
