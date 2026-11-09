"""Compute exact endpoint-to-line compatibility relations using character inner products."""
import json
from pathlib import Path
import sympy as sp
from symbolic_paths import ROOT,cyclotomic

index=json.loads((ROOT/'path_index.json').read_text())
out={}
for line in index:
    if not line['endpoints']:continue
    name=line['name'];geo=json.loads((ROOT/name/'geometry.json').read_text())
    reps=json.loads((ROOT/name/'representations.json').read_text())['representations']
    h=geo['unitary_order'];ids=geo['global_indices_0based'][:h]
    # All line coreps here are Wigner +1, so the restricted H character norm is 1.
    assert all(r['wigner']==1 and r['restriction_norm']==1 for r in reps)
    linechars=[[cyclotomic(x) for x in r['character']] for r in reps]
    out[name]={}
    for endpoint in line['endpoints']:
        eg=json.loads((ROOT/endpoint/'geometry.json').read_text())
        er=json.loads((ROOT/endpoint/'representations.json').read_text())['representations']
        lookup={x:i for i,x in enumerate(eg['global_indices_0based'])}
        local=[lookup[x] for x in ids]
        assert all(eg['ce'][lookup[g]][lookup[k]]==geo['ce'][i][j]
                   for i,g in enumerate(geo['global_indices_0based'])
                   for j,k in enumerate(geo['global_indices_0based']))
        branches=[]
        for r in er:
            ch=[cyclotomic(r['character'][i]) for i in local]
            multiplicities=[]
            for lc in linechars:
                inner=sp.simplify(sum(x*sp.conjugate(y) for x,y in zip(ch,lc))/h)
                assert inner.is_Integer and inner>=0,(name,endpoint,r['label'],inner)
                multiplicities.append(int(inner))
            assert sum(m*r['dimension'] for m,r in zip(multiplicities,reps))==r['dimension']
            branches.append(dict(endpoint_label=r['label'],endpoint_dimension=r['dimension'],multiplicities=multiplicities))
        out[name][endpoint]=dict(line_labels=[r['label'] for r in reps],
                                line_dimensions=[r['dimension'] for r in reps],branches=branches)
(ROOT/'compatibility.json').write_text(json.dumps(out,indent=2))
lines=[]
for name,data in out.items():
    for endpoint,item in data.items():
        lines.append(endpoint+' -> '+name)
        for r in item['branches']:
            terms=[f"{m} x {lab}({d}D)" for m,lab,d in zip(r['multiplicities'],item['line_labels'],item['line_dimensions']) if m]
            lines.append('  '+r['endpoint_label']+'('+str(r['endpoint_dimension'])+'D) = '+' + '.join(terms))
(ROOT/'compatibility.txt').write_text('\n'.join(lines)+'\n')
print('All 12 endpoint/line compatibility tables verified exactly.')
