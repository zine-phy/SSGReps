"""Exact affine little-group analysis for SSG 221.8.6.3, in pure-T coordinates.

Input is the certified integer geometry and cocycle from the previous 16D case.
All membership, exceptional-point and phase calculations use exact rationals.
"""
import json,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np

ROOT=Path(__file__).resolve().parent
PARENT=json.loads((ROOT/'parent_group.json').read_text())
R=np.array(PARENT['space_rotations'],int)
T=np.array(PARENT['translations'],int)
SIGNS=np.array([-1 if a else 1 for a in PARENT['antiunitary']],int)
TABLE=np.array(PARENT['mul_0based'],int)
SPIN=np.array(PARENT['cocycle_exponents_mod2'],int)
# Remove the known k_R=(1/2,1/2,1/2) translation cocycle to recover SU(2).
for i in range(len(R)):
    delta=T@R[i].T-SIGNS[i]*T
    numerator=delta.sum(axis=1)
    assert np.all(numerator%2==0)
    SPIN[i]=(SPIN[i]+numerator//2)%2

PATHS=[('GX',[0,0,0],[1,0,0],'G','X'),
       ('GM',[0,0,0],[1,1,0],'G','M'),
       ('GR',[0,0,0],[1,1,1],'G','R'),
       ('XM',[F(1,2),0,0],[0,1,0],'X','M'),
       ('MR',[F(1,2),F(1,2),0],[0,0,1],'M','R'),
       ('XR',[F(1,2),0,0],[0,1,1],'X','R')]
POINTS=[('G',[0,0,0]),('X',[F(1,2),0,0]),
        ('M',[F(1,2),F(1,2),0]),('R',[F(1,2)]*3)]

def integer(x):return F(x).denominator==1
def little_at(k):
    k=np.array(list(map(F,k)),dtype=object)
    return [i for i in range(len(R)) if all(integer(x) for x in k@(SIGNS[i]*R[i]-np.eye(3,dtype=int)))]

def make_case(name,u,v,endpoints=None):
    u=np.array(list(map(F,u)),dtype=object);v=np.array(list(map(F,v)),dtype=object)
    ids=[];exceptional={}
    for i in range(len(R)):
        A=SIGNS[i]*R[i]-np.eye(3,dtype=int)
        offset=u@A;slope=v@A
        if all(x==0 for x in slope):
            if all(integer(x) for x in offset):ids.append(i)
            continue
        # Any extra little-group element on the open line must satisfy these
        # three affine integrality constraints. Enumerate all possible roots.
        j=next(j for j,b in enumerate(slope) if b)
        lo,hi=sorted([offset[j],offset[j]+slope[j]/2])
        for z in range(math.floor(lo)-1,math.ceil(hi)+2):
            t=(z-offset[j])/slope[j]
            if 0<t<F(1,2) and all(integer(a+t*b) for a,b in zip(offset,slope)):
                exceptional.setdefault(str(t),[]).append(i)
    if endpoints:assert not exceptional,(name,exceptional)
    index={g:j for j,g in enumerate(ids)}
    n=len(ids);h=sum(SIGNS[ids]>0)
    mt=np.array([[index[TABLE[i,j]] for j in ids] for i in ids],int)
    ce=SPIN[np.ix_(ids,ids)].copy()
    phase0=[];phase1=[]
    for i in ids:
        p=F(u@T[i])/2;q=F(v@T[i])/2
        phase0.append(str(p));phase1.append(str(q))
    linear_checks=0
    for a,i in enumerate(ids):
        assert np.all(v@(R[i]-SIGNS[i]*np.eye(3,dtype=int))==0)
        for b,j in enumerate(ids):
            delta=R[i]@T[j]-SIGNS[i]*T[j]
            assert F(v@delta)==0   # omega's t coefficient vanishes identically
            phase=F(u@delta)
            assert integer(phase) # this example has only +/-1 cocycles
            ce[a,b]=(ce[a,b]-int(phase))%2
            gprod=TABLE[i,j]
            lattice=T[i]+R[i]@T[j]-T[gprod]
            assert np.all(lattice%2==0)
            # D_t(g)=exp[-2pi i (phase0[g]+t phase1[g])] M(g).
            # The section relation is sigma(g,h) exp[-i k.lattice] D_t(gh).
            lhs0=F(phase0[a])+SIGNS[i]*F(phase0[b])+F(int(ce[a,b]),2)
            rhs0=F(int(SPIN[i,j]),2)+F(u@lattice)/2+F(phase0[mt[a,b]])
            lhs1=F(phase1[a])+SIGNS[i]*F(phase1[b])
            rhs1=F(v@lattice)/2+F(phase1[mt[a,b]])
            # ce/2 represents +pi; in exp[-2pi i p], sign is the same mod Z.
            assert integer(lhs0-rhs0) and lhs1==rhs1
            linear_checks+=1
    for i in range(n):
        assert np.array_equal(mt[mt[i],:],mt[i,mt])
        assert np.all((ce[i,:,None]+ce[mt[i],:]-ce[i,mt]-ce)%2==0)
    out=ROOT/name;out.mkdir(exist_ok=True)
    data=dict(name=name,ssg=PARENT['ssg'],group_type='double',u=list(map(str,u)),v=list(map(str,v)),
              interval=['0','1/2'] if endpoints else None,endpoints=endpoints,global_indices_0based=ids,
              order=n,unitary_order=int(h),antiunitary=[bool(SIGNS[i]<0) for i in ids],
              mt=mt.tolist(),ce=ce.tolist(),spin_ce=SPIN[np.ix_(ids,ids)].tolist(),
              phase_offset=phase0,phase_slope=phase1,interior_exceptional_points=exceptional,
              symbolic_phase_relations_checked=linear_checks,associativity_triples=n**3,
              endpoint_orders=[len(little_at(u)),len(little_at(u+v/2))] if endpoints else [])
    (out/'geometry.json').write_text(json.dumps(data,indent=2))
    (out/'group_data.g').write_text(f'n:={n};; h:={h};;\nmt:={str((mt+1).tolist())};;\nce:={str(ce.tolist())};;\n')
    print(name,n,h,'phase identities',linear_checks,flush=True)
    return data

if __name__=='__main__':
    cases=[make_case(name,u,v,[left,right]) for name,u,v,left,right in PATHS]
    cases +=[make_case(name,u,[0,0,0]) for name,u in POINTS]
    (ROOT/'path_index.json').write_text(json.dumps([{k:d[k] for k in ['name','u','v','interval','endpoints','order','unitary_order','endpoint_orders']} for d in cases],indent=2))
