"""Readable output for the certified 16D example (R point of SSG 221.8.6.3).

python demo.py --rep r3 --ops 2 193 --convention bloch
All formatting functions in ssg_output.py also work with other exact datasets
and with original numeric SSGReps LittleGroup objects.
"""
import argparse,json,sys
from functools import lru_cache
from pathlib import Path
import sympy as sp
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent/'SSGReps'))
sys.path.insert(0,str(ROOT.parent/'symbolic_output'))
from symbolic_numbers import parse_gap
from ssg_output import report_text,report_html

ROOT=Path(__file__).resolve().parent
@lru_cache(None)
def exact(s):return sp.simplify(sp.expand_complex(parse_gap(s)))

def load_example():
    raw=json.loads((ROOT.parent/'symbolic_paths/R/representations.json').read_text())
    geometry=json.loads((ROOT.parent/'symbolic_paths/R/geometry.json').read_text())
    parent=json.loads((ROOT.parent/'symbolic_paths/parent_group.json').read_text())
    operations=[]
    for i,g in enumerate(geometry['global_indices_0based']):
        operations.append(dict(id=g+1,anti=geometry['antiunitary'][i],phase=sp.I**(-sum(parent['translations'][g])),
                               R=parent['space_rotations'][g],spin=parent['spin_rotations'][g],tau=parent['translations'][g]))
    reps=[dict(label=r['label'],dimension=r['dimension'],characters=[exact(x) for x in r['character']],
               matrices=[sp.ImmutableMatrix([[exact(x) for x in row] for row in m]) for m in r['matrices']]) for r in raw['representations']]
    return dict(ssg='221.8.6.3',k=['1/2','1/2','1/2'],operations=operations,representations=reps,exact=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rep',choices=['r1','r2','r3'],default='r3')
    p.add_argument('--ops',type=int,nargs='+',default=[2,193])
    p.add_argument('--convention',choices=['bloch','projective'],default='bloch')
    p.add_argument('--characters',choices=['compact','full'],default='compact')
    p.add_argument('--outdir',type=Path,default=ROOT)
    a=p.parse_args();data=load_example()
    if not set(a.ops)<={o['id'] for o in data['operations']}:p.error('Invalid operation ID.')
    a.outdir.mkdir(parents=True,exist_ok=True)
    text=report_text(data,a.rep,a.ops,a.convention,compress=a.characters=='compact')
    (a.outdir/'representation_report.txt').write_text(text)
    (a.outdir/'representation_report.html').write_text(report_html(data,a.rep,a.ops,a.convention,compress=a.characters=='compact'))
    print(text)
