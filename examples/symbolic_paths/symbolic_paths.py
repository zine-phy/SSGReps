"""Exact parametric small corepresentations along six high-symmetry lines.

    family=PathFamily('MR')
    t=family.t
    D,anti=family.operation('r1',0) # local zero-based operation index
    # Operator is D*K when anti=True. D includes its Bloch phase.
    value=D.subs(t,sp.sqrt(2)/5)

k coordinates use the reciprocal lattice of the pure translations 2 Z^3.
An endpoint substitution is a representation of the line subgroup; the full
endpoint group has additional operations and its own endpoint irreps.
"""
import json
from pathlib import Path
from functools import lru_cache
import sympy as sp

ROOT=Path(__file__).resolve().parent

@lru_cache(None)
def cyclotomic(s):
    # Only trusted locally generated GAP expressions are loaded here.
    return sp.simplify(sp.expand_complex(sp.sympify(s.replace('^','**'),
        locals={'E':lambda n:sp.exp(2*sp.pi*sp.I/n)})))

class PathFamily:
    def __init__(self,name,directory=None):
        self.root=Path(directory or ROOT);self.name=name
        self.geometry=json.loads((self.root/name/'geometry.json').read_text())
        self.result=json.loads((self.root/name/'representations.json').read_text())
        self.t=sp.Symbol('t',real=True)
        self.labels=[r['label'] for r in self.result['representations']]
        self.representations={r['label']:r for r in self.result['representations']}
        self.u=sp.Matrix(list(map(sp.Rational,self.geometry['u'])))
        self.v=sp.Matrix(list(map(sp.Rational,self.geometry['v'])))
        self.k=self.u+self.t*self.v

    def constant_matrix(self,label,index):
        data=self.representations[label]['matrices'][index]
        return sp.ImmutableMatrix([[cyclotomic(x) for x in row] for row in data])

    def phase(self,index):
        p=sp.Rational(self.geometry['phase_offset'][index])
        q=sp.Rational(self.geometry['phase_slope'][index])
        return sp.simplify(sp.exp(-2*sp.pi*sp.I*p))*sp.exp(-2*sp.pi*sp.I*q*self.t)

    def operation(self,label,index,bloch=True):
        m=self.constant_matrix(label,index)
        return (self.phase(index)*m if bloch else m),self.geometry['antiunitary'][index]

    def translation(self,primitive_lattice_coefficients):
        """Scalar for pure translation 2*(n1,n2,n3)."""
        return sp.exp(-2*sp.pi*sp.I*(self.k.dot(sp.Matrix(primitive_lattice_coefficients))))

    def numeric(self,label,t,bloch=True):
        import numpy as np
        m=np.array([np.array(self.constant_matrix(label,i).evalf(),complex)
                    for i in range(self.geometry['order'])])
        if bloch:
            p=np.array([float(sp.Rational(x)) for x in self.geometry['phase_offset']])
            q=np.array([float(sp.Rational(x)) for x in self.geometry['phase_slope']])
            m=np.exp(-2j*np.pi*(p+float(t)*q))[:,None,None]*m
        return m

if __name__=='__main__':
    for name in ['GX','GM','GR','XM','MR','XR']:
        f=PathFamily(name)
        print(name,'k(t)=',list(f.k),'dimensions=',[r['dimension'] for r in f.result['representations']])
    f=PathFamily('MR')
    # Choose a unitary generator with a nonconstant Bloch phase.
    i=next(i for i in f.result['generator_indices_0based'] if sp.Rational(f.geometry['phase_slope'][i])!=0)
    print('MR example, local operation',i,'global operation',f.geometry['global_indices_0based'][i]+1)
    print(f.operation('r1',i)[0])
