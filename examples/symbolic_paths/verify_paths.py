"""Validate every path at rational and irrational parameters, optionally using SSGReps.

    python verify_paths.py --repo /path/to/SSGReps

The numerical reducer is compared with independently constructed exact data.
Symbolic validity for all real t is proved separately by prepare_paths.py and
solve_exact.g; sampling is only an independent cross-check.
"""
import argparse,contextlib,copy,io,json,pickle,sys,time,warnings
from pathlib import Path
import numpy as np
import sympy as sp
from symbolic_paths import ROOT,PathFamily

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path)
args=p.parse_args()
parent=json.loads((ROOT/'parent_group.json').read_text())
Rs=np.array(parent['space_rotations'],int);Ts=np.array(parent['translations'],int)
sgns=np.array([-1 if a else 1 for a in parent['antiunitary']],int)
samples=[sp.Rational(1,17),sp.Rational(1,8),sp.Rational(1,4),sp.Rational(3,8),sp.Rational(8,17),sp.sqrt(2)/5,sp.sqrt(3)/7]
native=None
if args.repo:
    sys.path.insert(0,str(args.repo.resolve()/'SSGReps'))
    import SSGReps as native
    try:
        from ssg_database import load_database
        record=load_database()[parent['ssg']]
    except ImportError:
        with (args.repo/'ssg_data/identify.pkl').open('rb') as f:
            record=next(d for d in pickle.load(f) if d['ssgNum']==parent['ssg'])
    full=native.ssgGroup(parent['ssg'],'double');full.load_ssg(record);full.ssg_bz()
    native.LittleGroup.solve_factor=lambda self,i:self.correp_matrix[i]

report=dict(ssg=parent['ssg'],parameter_interval='0<t<1/2',samples=list(map(str,samples)),
            native_repository=args.repo.resolve().name if args.repo else None,
            native_stage='get_irreducible_rep (legacy solve_factor disabled)',paths=[])
for name in ['GX','GM','GR','XM','MR','XR']:
    start=time.monotonic();f=PathFamily(name);g=f.geometry
    ids=np.array(g['global_indices_0based']);n=g['order'];h=g['unitary_order'];sign=sgns[ids]
    mt=np.array(g['mt']);ce=np.array(g['ce']);spin=(-1.)**np.array(g['spin_ce'])
    u=np.array(list(map(float,f.u)));v=np.array(list(map(float,f.v)))
    R=Rs[ids];T=Ts[ids]
    lattice=np.array([[T[i]+R[i]@T[j]-T[mt[i,j]] for j in range(n)] for i in range(n)])
    constants={lab:f.numeric(lab,0,bloch=False) for lab in f.labels}
    constant_chars={lab:np.trace(c[:h],axis1=1,axis2=2) for lab,c in constants.items()}
    stats=dict(name=name,dimensions=[r['dimension'] for r in f.result['representations']],
               order=n,endpoint_orders=g['endpoint_orders'],samples=[],
               symbolic_phase_relations=g['symbolic_phase_relations_checked'],
               exact_matrix_products=sum(r['exact_products_checked'] for r in f.result['representations']))
    if native:
        small=copy.copy(full)
        for attr in ['rotC','tauC','spin','su2s','time_reversal']:
            setattr(small,attr,[getattr(full,attr)[i] for i in ids])
        with contextlib.redirect_stdout(io.StringIO()):small.get_table()
        assert np.array_equal(small.mul_table.astype(int)-1,mt)
        assert np.max(abs(small.factor_su2-spin))<1e-12
    for t in samples:
        k=u+float(t)*v
        maxerr=0.;ue=0.;charerr=0.;factorerr=0.
        physical_factor=spin*np.exp(-1j*np.pi*np.einsum('ijc,c->ij',lattice,k))
        phase=np.exp(-1j*np.pi*(T@k))
        for lab,c in constants.items():
            D=phase[:,None,None]*c;d=D.shape[1]
            ue=max(ue,float(np.max(np.linalg.norm(D@D.conj().transpose(0,2,1)-np.eye(d),axis=(1,2)))))
            for i in range(n):
                rhs=D.conj() if sign[i]<0 else D
                diff=D[i]@rhs-physical_factor[i,:,None,None]*D[mt[i]]
                maxerr=max(maxerr,float(np.max(np.linalg.norm(diff,axis=(1,2)))))
        if native:
            # Test membership against the original whole-group k_op, then solve
            # the original restricted regular representation with no new maths.
            lg=native.LittleGroup.__new__(native.LittleGroup)
            lg.ssg=full;lg.kvec=k;lg.antiunitary=False
            lg.k_op();assert lg.oplist==ids.tolist()
            small.get_trans_factor(k)
            factorerr=float(np.max(abs(small.factor_su2*small.factor_trans-(-1.)**ce)))
            lg.ssg=small;lg.ssgNum=parent['ssg'];lg.need_beautify=False
            lg.k_op();lg.get_regular_rep_list();np.random.seed(123)
            with contextlib.redirect_stdout(io.StringIO()),warnings.catch_warnings():
                warnings.simplefilter('ignore');lg.get_irreducible_rep()
            assert sorted(lg.rep_degree)==sorted(stats['dimensions'])
            matches=[]
            for lab,ch in constant_chars.items():
                errs=[float(np.max(abs(ch-np.asarray(other)))) for other in lg.character]
                j=int(np.argmin(errs));matches.append(j);charerr=max(charerr,errs[j])
            assert len(set(matches))==len(matches)
        assert maxerr<1e-11 and ue<1e-11 and charerr<1e-10 and factorerr<1e-12
        stats['samples'].append(dict(t=str(t),k=k.tolist(),physical_product_error=maxerr,
                                     unitarity_error=ue,native_character_error=charerr if native else None,
                                     native_factor_error=factorerr if native else None))
    # Endpoints remain representations of the line subgroup; their full little
    # groups must be read from the separately solved endpoint cases.
    for t,endpoint in zip([0,.5],g['endpoints']):
        ek=u+t*v
        if native:
            lg.ssg=full;lg.kvec=ek;lg.k_op()
            eg=json.loads((ROOT/endpoint/'geometry.json').read_text())
            assert lg.oplist==eg['global_indices_0based']
    stats['seconds']=time.monotonic()-start;report['paths'].append(stats)
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2))
    print('PASS',name,'dims',stats['dimensions'],'samples',len(samples),'seconds',round(stats['seconds'],2),flush=True)
print('All path samples passed. The algebraic certificates prove every t in each open segment.',flush=True)
