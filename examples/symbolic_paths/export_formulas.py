"""Export exact parameter-dependent generators and one compact worked example."""
import json
import sympy as sp
from symbolic_paths import ROOT,PathFamily

parent=json.loads((ROOT/'parent_group.json').read_text())
lines=['All coordinates refer to the reciprocal lattice of pure translations 2 Z^3.',
       '0<t<1/2 on every open line. All matrix entries below are exact.',
       'Operator = exp(-i*pi*b*t) * C * K^anti; K is complex conjugation.',
       'The constant offset Bloch phase is already included in C.',
       'Global operation indices below are one based, matching the original database.',
       'Labels r1,r2,... are local computed labels, not standardized Bilbao labels.','']
formulas=[]
for name in ['GX','GM','GR','XM','MR','XR']:
    f=PathFamily(name);g=f.geometry
    lines+=['PATH '+name+'  k(t)='+str(list(f.k))]
    indices=f.result['generator_indices_0based']+[f.result['anti_generator_index_0based']]
    for rep in f.result['representations']:
        lines+=['  '+rep['label']+' dimension='+str(rep['dimension'])]
        for i in indices:
            global_i=g['global_indices_0based'][i]
            b=2*sp.Rational(g['phase_slope'][i])
            alpha=sp.simplify(sp.exp(-2*sp.pi*sp.I*sp.Rational(g['phase_offset'][i])))
            C=alpha*f.constant_matrix(rep['label'],i)
            entry=dict(path=name,label=rep['label'],dimension=rep['dimension'],
                       local_index_0based=i,global_index_1based=global_i+1,
                       antiunitary=g['antiunitary'][i],exponent_b=str(b),
                       spin=parent['spin_rotations'][global_i],R=parent['space_rotations'][global_i],
                       tau=parent['translations'][global_i],C=[[str(x) for x in row] for row in C.tolist()])
            formulas.append(entry)
            lines+=['    g'+str(global_i+1)+' : exp(-i*pi*('+str(b)+')*t) * C; anti='+str(entry['antiunitary']),
                    '    spin='+str(entry['spin'])+'; R='+str(entry['R'])+'; tau='+str(entry['tau']),
                    '    C='+str(C.tolist())]
    lines+=['']
(ROOT/'analytic_generators.txt').write_text('\n'.join(lines)+'\n')
(ROOT/'analytic_generators.json').write_text(json.dumps(formulas,indent=2))
f=PathFamily('MR');i=f.geometry['global_indices_0based'].index(3)
M,anti=f.operation('r1',i)
H=sp.Matrix([[1,-1],[-1,-1]])/sp.sqrt(2)
expected=sp.exp(-sp.I*sp.pi*f.t)*sp.kronecker_product(sp.eye(2),H)
assert not anti and sp.simplify(M-expected)==sp.zeros(4)
assert sp.simplify(M*M-sp.exp(-2*sp.pi*sp.I*f.t)*sp.eye(4))==sp.zeros(4)
example='''Worked example: MR, k(t)=(1/2,1/2,t), branch r1 (dimension 4).
g4 = [C2y_spin || I_space | (1,0,1)], a unitary spin-decorated translation.
D_t(g4) = exp(-i*pi*t)/sqrt(2) *
 [ [1,-1,0,0], [-1,-1,0,0], [0,0,1,-1], [0,0,-1,-1] ].
Eigenvalues: +exp(-i*pi*t) and -exp(-i*pi*t), each twice.
D_t(g4)^2 = exp(-2*i*pi*t) I4.
This is one generator of the full irreducible 4D family, whose other generators
and antiunitary operation are in analytic_generators.txt.
The root and phase formula above was checked symbolically, not fitted.
'''
(ROOT/'worked_example.txt').write_text(example)
print('Exported all parameter-dependent generators and verified the compact MR example.')
