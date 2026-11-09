"""Independent numerical regular-representation cross-check of the exact result.

This does not construct or certify the exact representation; construct.g does.
It checks compatibility with SSGReps' regular-representation reduction formula.
"""
import json
from pathlib import Path
import numpy as np
from exact_ssg import ExactSSG

rep=ExactSSG()
data=rep.geometry
m=np.asarray(data['mul_0based'],int)
w=(-1.)**np.asarray(data['cocycle_exponents_mod2'],int)
anti=np.asarray(data['antiunitary'])
n=len(anti);h=sum(~anti)
Ms=rep.as_ssgreps_dict()['repMatrix'][0]
max_product=0.
for i in range(n):
    diff=Ms[i]@(Ms.conj() if anti[i] else Ms)-w[i,:,None,None]*Ms[m[i]]
    max_product=max(max_product,float(np.max(np.linalg.norm(diff,axis=(1,2)))))
max_unitary=float(np.max(np.linalg.norm(Ms@Ms.conj().transpose(0,2,1)-np.eye(16),axis=(1,2))))

# Monomial form of the exact same group average used by SSGReps, avoiding
# dense inverses and storing 384 dense regular matrices.
rng=np.random.default_rng(12)
X=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
Q=np.zeros((n,n),complex)
for i in range(n):
    Q[np.ix_(m[i],m[i])]+=w[i,:,None]*(X.conj() if anti[i] else X)*w[i,None,:]
Q+=Q.conj().T
vals,V=np.linalg.eigh(Q)
cuts=np.r_[0,np.flatnonzero(np.diff(vals)>1e-6)+1,n]
characters=[];dimensions=[]
for lo,hi in zip(cuts[:-1],cuts[1:]):
    basis=V[:,lo:hi]
    char=np.array([np.trace(basis[m[i]].conj().T@(w[i,:,None]*basis)) for i in range(h)])
    if not any(np.max(abs(char-c))<1e-7 for c in characters):
        dimensions.append(int(hi-lo));characters.append(char)
exact_char=np.trace(Ms[:h],axis1=1,axis2=2)
character_error=float(min(np.max(abs(exact_char-c)) for c in characters))
assert sorted(dimensions)==[8,8,16]
assert max_product<1e-12 and max_unitary<1e-12 and character_error<1e-10
result=dict(ssg='221.8.6.3',k=[.5,.5,.5],dimensions=sorted(dimensions),
    double_valued=True,numerical_product_frobenius=max_product,
    numerical_unitarity_frobenius=max_unitary,character_match_max=character_error,
    construction='exact central extension, induced module and projector, Wigner pairing',
    note='Numerical validation is secondary; GAP verifies every relation exactly.')
Path(__file__).with_name('numeric_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
