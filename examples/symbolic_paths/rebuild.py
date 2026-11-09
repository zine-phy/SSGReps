"""Rebuild exact constant matrices for six lines and their four endpoints.

Usage: python rebuild.py --gap /path/to/gap [--cases GX GM GR XM MR XR G X M R]
"""
import argparse,json,shutil,subprocess,sys,tempfile,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--gap',default=shutil.which('gap'))
p.add_argument('--cases',nargs='+',default=['GX','GM','GR','XM','MR','XR','G','X','M','R'])
p.add_argument('--jobs',type=int,default=2)
p.add_argument('--skip-prepare',action='store_true')
a=p.parse_args()
if not a.gap:p.error('Provide --gap or install GAP on PATH.')
if not a.skip_prepare:subprocess.run([sys.executable,str(ROOT/'prepare_paths.py')],check=True)

def solve(name):
    started=time.monotonic();case=ROOT/name
    with tempfile.TemporaryDirectory(prefix='ssg_path_'+name+'_') as td:
        td=Path(td);shutil.copy2(case/'group_data.g',td/'group_data.g')
        proc=subprocess.run([a.gap,'-q','-A','-b','--quitonbreak',str(ROOT/'solve_exact.g')],cwd=td,capture_output=True,text=True)
        (case/'solve.log').write_text(proc.stdout+proc.stderr)
        if proc.returncode!=0 or not (td/'representations.json').exists():
            raise RuntimeError(name+': GAP failed; see '+str(case/'solve.log'))
        result=json.loads((td/'representations.json').read_text())
        assert all(r['irreducibility_indicator']==1 for r in result['representations'])
        for file in ['representations.json','characters.g']:shutil.copy2(td/file,case/file)
    return name,[r['dimension'] for r in result['representations']],round(time.monotonic()-started,2)

with ThreadPoolExecutor(max_workers=a.jobs) as pool:
    futures=[pool.submit(solve,name) for name in a.cases]
    for f in as_completed(futures):print('PASS',*f.result(),flush=True)
