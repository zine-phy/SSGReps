"""Extract and certify case 221.8.6.3 from an existing SSGReps checkout.

Usage: python prepare_case.py --repo /absolute/path/to/SSGReps
This adapter deliberately handles this integer, signed-permutation example.
It rejects other data instead of guessing arbitrary algebraic numbers.
The representation solver in construct.g uses the resulting exact tables.
"""
import sys,json,pickle,argparse
from pathlib import Path
import numpy as np
import sympy as sp
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
OUT=args.output.resolve()
OUT.mkdir(exist_ok=True,parents=True)
sys.path.insert(0,str(args.repo.resolve()/'SSGReps'))
import SSGReps
try:
    from ssg_database import load_database
    record=load_database()['221.8.6.3']
except ImportError:
    with (args.repo/'ssg_data/identify.pkl').open('rb') as f:
        record=next(x for x in pickle.load(f) if x['ssgNum']=='221.8.6.3')
group=SSGReps.ssgGroup('221.8.6.3','double')
group.load_ssg(record)
group.ssg_bz()
d=dict(R=np.asarray(group.rotC),T=np.asarray(group.tauC),S=np.asarray(group.spin),
       U=np.asarray(group.su2s),A=np.asarray(group.pure_T).T,k=np.array([.5,.5,.5]),
       signs=np.asarray(group.time_reversal))
def integral(a):
    b=np.rint(a).astype(int)
    assert np.max(abs(a-b))<1e-10
    return b
R,T,S,A=map(integral,[d['R'],d['T'],d['S'],d['A']])
assert np.array_equal(A,2*np.eye(3))
assert np.array_equal(d['k'],[.5,.5,.5])
n=len(R); h=n//2
assert h==192 and np.all(d['signs'][:h]==1) and np.all(d['signs'][h:]==-1)
assert np.array_equal(S[h:],-S[:h])
assert np.array_equal(R[h:],R[:h])
assert np.array_equal(T[h:],T[:h]+1)
# All 384 operations leave k fixed modulo the reciprocal lattice of 2 Z^3.
assert all(np.all(np.sum(s*r-np.eye(3,dtype=int),axis=0)%2==0)
           for s,r in zip(d['signs'],R))
I=sp.I; sx=sp.Matrix([[0,1],[1,0]]); sy=sp.Matrix([[0,-I],[I,0]]); sz=sp.diag(1,-1)
J=sp.Matrix([[0,-1],[1,0]])
def lift(flat):
    r=sp.Matrix(3,3,flat);tr=sp.trace(r)
    assert r.T*r==sp.eye(3) and r.det()==1
    if tr==-1:
        axis=next((r+sp.eye(3))[:,j] for j in range(3) if (r+sp.eye(3))[:,j]!=sp.zeros(3,1))
        v=axis/sp.sqrt(axis.dot(axis));q=0
    else:
        q=sp.sqrt(1+tr)/2
        v=sp.Matrix([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])/(4*q)
    return (q*sp.eye(2)-I*(v[0]*sx+v[1]*sy+v[2]*sz)).applyfunc(sp.simplify)
cache={}
U=[]
for r in S[:h]:
    key=tuple(r.flatten())
    if key not in cache: cache[key]=lift(key)
    U.append(cache[key])
U+= [J*u.conjugate() for u in U]
assert max(np.linalg.norm(np.array(u.evalf(),complex)-v) for u,v in zip(U,d['U']))<1e-12
def key(r,t,s):return tuple(r.flatten())+tuple(t%2)+tuple(s.flatten())
idx={key(*x):i for i,x in enumerate(zip(R,T,S))}
m=np.empty((n,n),int); e=np.empty((n,n),int); signcache={}
for i in range(n):
    sg=1 if i<h else -1
    for j in range(n):
        p=idx[key(R[i]@R[j],R[i]@T[j]+T[i],S[i]@S[j])]
        m[i,j]=p
        ck=(tuple(U[i]),sg,tuple(U[j]),tuple(U[p]))
        if ck not in signcache:
            product=(U[i]*(U[j] if sg==1 else U[j].conjugate())).applyfunc(sp.simplify)
            plus=(product-U[p]).applyfunc(sp.simplify)==sp.zeros(2)
            minus=(product+U[p]).applyfunc(sp.simplify)==sp.zeros(2)
            assert plus or minus,(i,j,p,product,U[p])
            signcache[ck]=0 if plus else 1
        numerator=-int(np.sum(R[i]@T[j]-sg*T[j]))
        assert numerator%2==0
        e[i,j]=(numerator//2+signcache[ck])%2
for i in range(n):
    assert np.array_equal(m[m[i],:],m[i,m])
    assert np.all((e[i,:,None]+e[m[i],:]-e[i,m]-e)%2==0)
data={'ssg':'221.8.6.3','group_type':'double','k':['1/2']*3,'pure_translation_basis_columns':A.tolist(),
      'space_rotations':R.tolist(),'translations':T.tolist(),'spin_rotations':S.tolist(),
      'antiunitary':[bool(x<0) for x in d['signs']],
      'mul_0based':m.tolist(),'cocycle_exponents_mod2':e.tolist(),
      'source_commit':'97e839509ec1a7f08af68b047f75aad0c71468b7',
      'certification':{'associativity_triples':n**3,'cocycle_triples':n**3,
                       'exact_SU2_pair_signs':n**2,'arithmetic':'integer geometry and symbolic SU(2) lifts'}}
(OUT/'group_data.json').write_text(json.dumps(data,separators=(',',':')))
gap='n := %d;; h := %d;;\n'%(n,h)
gap+='mt := '+str((m+1).tolist())+';;\n'
gap+='ce := '+str(e.tolist())+';;\n'
(OUT/'group_data.g').write_text(gap)
print('Exact integer group closure, all associativity and cocycle triples, and exact SU(2) lift products PASS',flush=True)
