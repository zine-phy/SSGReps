"""Export the exact generators of the six paths in exponential notation and LaTeX."""
import argparse,json
from pathlib import Path
import sympy as sp
from symbolic_numbers import format_exact

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,default=ROOT/'input_generators.json')
p.add_argument('--output',type=Path,default=ROOT)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
items=json.loads(a.source.read_text());t=sp.Symbol('t',real=True)
text=[];latex=[];records=[];verified=0
for item in items:
    # Source is a trusted generated result, with no floating numbers.
    C=sp.Matrix([[sp.sympify(x) for x in row] for row in item['C']])
    M=sp.exp(-sp.I*sp.pi*sp.Rational(item['exponent_b'])*t)*C
    formatted=format_exact(M,'exponential')
    for x,y in zip(M,formatted):
        if x==0 or y==0:
            assert x==y
        else:
            assert sp.simplify(sp.expand_complex(sp.powsimp(x/y)))==1
        verified+=1
    label=f"{item['path']} {item['label']} g{item['global_index_1based']}"
    text.extend([label+'; antiunitary='+str(item['antiunitary']),str(formatted),''])
    latex.extend(['% '+label,sp.latex(formatted)+(r'\mathcal{K}' if item['antiunitary'] else ''),''])
    records.append(dict(path=item['path'],label=item['label'],global_index_1based=item['global_index_1based'],
                        antiunitary=item['antiunitary'],matrices=[[str(x) for x in row] for row in formatted.tolist()],
                        latex=sp.latex(formatted)))
(a.output/'generators_exponential.txt').write_text('\n'.join(text))
(a.output/'generators_latex.tex').write_text('\n'.join(latex))
(a.output/'generators_exponential.json').write_text(json.dumps(records,indent=2))
(a.output/'export_validation.json').write_text(json.dumps(dict(generators=len(items),exact_entry_equalities=verified,
       method='symbolic equality; no floating recognition used for exact results'),indent=2))
print('Exported',len(items),'generators;',verified,'exact entry equalities passed.')
