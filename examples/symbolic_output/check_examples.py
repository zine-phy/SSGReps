"""Check exact rewrites and demonstrate bounded float recovery and rejection."""
import cmath,json,math
from pathlib import Path
import sympy as sp
from symbolic_numbers import format_exact,parse_gap,recognize_numeric

root=Path(__file__).resolve().parent;t=sp.Symbol('t',real=True)
cases=[0.7071067811865476,0.5+0.8660254037844386j,-0.5+0.8660254037844386j,
       complex(sp.N(sp.sqrt(3)/4*sp.exp(-sp.I*sp.pi/4))),
       0.123456789+0.234567891j,cmath.exp(1j*math.sqrt(2)),0.5+0.866j]
report=[]
for i,z in enumerate(cases):
    r=recognize_numeric(z)
    assert r['status']==('numeric_candidate' if i<4 else 'unrecognized')
    assert not r['certified_exact']
    report.append({'input':[z.real,z.imag],**{k:str(v) if isinstance(v,sp.Basic) else v for k,v in r.items()}})
assert recognize_numeric(0.5+0.866j,atol=5e-4)['status']=='numeric_candidate'
assert not recognize_numeric(1.0)['certified_exact']
assert sp.simplify(parse_gap('E(12)^7')/sp.exp(7*sp.I*sp.pi/6))==1
rewrites=[]
for x in [sp.sqrt(2)/2,(1+sp.sqrt(3)*sp.I)/2,-sp.I/4*sp.exp(-sp.I*sp.pi*t),
          sp.exp(-sp.I*sp.pi*(t+sp.Rational(1,3))),sp.E**sp.Rational(1,2)]:
    for style in ['exponential','radical']:
        result=format_exact(x,style)
        assert sp.simplify(sp.expand_complex(sp.powsimp(x/result)))==1
        rewrites.append(dict(input=str(x),style=style,output=str(result)))
try:format_exact(0.7071067811865476)
except ValueError:pass
else:raise AssertionError('Float accepted as exact')
try:parse_gap('other(3)')
except ValueError:pass
else:raise AssertionError('Unexpected expression accepted')
(root/'numeric_examples.json').write_text(json.dumps(report,indent=2))
(root/'exact_rewrites.json').write_text(json.dumps(rewrites,indent=2))
print('PASS: seven numerical acceptance/rejection cases, ten exact rewrites, input guards.')
