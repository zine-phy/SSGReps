"""Read the exact 16D example in Python; retain conjugation for antiunitary ops.

    rep = ExactSSG()
    matrix, anti = rep.operation(192)  # M(a), where operator is M(a) K
    numeric = rep.as_ssgreps_dict()   # matrix-consumer fields; one selected corep

Matrix indices are zero based, in SSGReps' original operation order.
The default matrices are projective co-group matrices. Set bloch=True for
the small-representation matrix including exp(-i k.tau), as in the paper.
"""
import json
from pathlib import Path
import sympy as sp


class ExactSSG:
    def __init__(self, directory=None):
        directory=Path(directory or Path(__file__).resolve().parent)
        self.geometry=json.loads((directory/'group_data.json').read_text())
        self.result=json.loads((directory/'exact_representation.json').read_text())
        def parse(x):
            return sp.simplify(sp.expand_complex(sp.sympify(x.replace('^','**'),
                locals={'E':lambda n:sp.exp(2*sp.pi*sp.I/n)})))
        self.entries={x:parse(x) for x in self.result['entries']}
        self.matrices=[sp.ImmutableMatrix([[self.entries[x] for x in row] for row in mat])
                       for mat in self.result['matrices']]
        self.anti=self.geometry['antiunitary']

    def operation(self,index,bloch=False):
        m=self.matrices[index]
        if bloch:
            # k_conv=(pi/2,pi/2,pi/2) in conventional direct coordinates.
            m=sp.I**(-sum(self.geometry['translations'][index]))*m
        return m,self.anti[index]

    def as_ssgreps_dict(self):
        """Numeric adapter for existing consumers; exact data remain available."""
        import numpy as np
        matrices=np.array([np.array(m.evalf(),dtype=complex) for m in self.matrices])
        return dict(ssgNum=self.geometry['ssg'],kvec=np.array([.5,.5,.5]),
            rotC=np.array(self.geometry['space_rotations']),
            spin=np.array(self.geometry['spin_rotations']),
            tauC=np.array(self.geometry['translations']),
            timeReversal=np.array([-1 if a else 1 for a in self.anti]),
            character=np.array([[np.trace(m) for m in matrices[:192]]]),
            repMatrix=np.array([matrices]),repDegree=[16],torsion=[2])


if __name__=='__main__':
    rep=ExactSSG()
    a,anti=rep.operation(192)
    print('SSG:',rep.geometry['ssg'],'k=(1/2,1/2,1/2), double-valued')
    print('Dimension:',a.rows,'; antiunitary generator: M(193) K')
    print('A squared = I:',a*a.conjugate()==sp.eye(16))
    print('All distinct exact entries:',sorted(map(str,set(rep.entries.values()))))
    print('GAP exact product checks:',rep.result['exact_products_checked'])
