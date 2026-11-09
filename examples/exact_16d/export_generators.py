"""Export a compact, readable specification of the exact corepresentation."""
import json
from pathlib import Path
import sympy as sp
from exact_ssg import ExactSSG

root=Path(__file__).resolve().parent
r=ExactSSG()
e=r.geometry['cocycle_exponents_mod2'];mt=r.geometry['mul_0based'];a=192
assert all(mt[a][i]==mt[i][a] and e[a][i]==e[i][a] for i in range(192))
assert all(m[8:,8:]==m[:8,:8].conjugate() for m in r.matrices[:192])
assert r.matrices[a]==sp.Matrix.vstack(sp.Matrix.hstack(sp.zeros(8),sp.eye(8)),
                                     sp.Matrix.hstack(sp.eye(8),sp.zeros(8)))
lines=['SSG 221.8.6.3, double-valued; k=(1/2,1/2,1/2) in pure-translation reciprocal basis.',
       'Pure translations: 2 Z^3. k=(1/4,1/4,1/4) in conventional reciprocal coordinates.',
       'Indices below are one based and follow SSGReps operation order.',
       'For all unitary h: M(h)=diag(D(h),conjugate(D(h))).',
       'a=g193=[-I_spin || I_space | (1,1,1)] is antiunitary.',
       'M(a)=[[0,I8],[I8,0]], so the operator is M(a) K.',
       'M(ah)=(-1)^ce[a,h] M(a) conjugate(M(h)).',
       'The five 8x8 matrices below generate D on the 192-element unitary quotient.',
       'Products obey D(g)D(h)=(-1)^ce[g,h]D(gh), not ordinary multiplication.',
       'The full 384 matrices and cocycle table are also included.','']
out=[]
for gid in r.result['generator_indices_1based']:
    i=gid-1;d=r.matrices[i][:8,:8]
    op=dict(index_1based=gid,spin=r.geometry['spin_rotations'][i],
        space_rotation=r.geometry['space_rotations'][i],translation=r.geometry['translations'][i],
        D8=[[str(x) for x in row] for row in d.tolist()])
    out.append(op)
    lines+=['g'+str(gid),'spin = '+str(op['spin']),'R = '+str(op['space_rotation']),
            'tau = '+str(op['translation']),'D8 =']
    lines+=['[ '+', '.join(row)+' ]' for row in op['D8']]
    lines+=['']
(root/'generators.txt').write_text('\n'.join(lines))
(root/'generators_exact.json').write_text(json.dumps(out,indent=2))
print('Wrote generators.txt and generators_exact.json')
