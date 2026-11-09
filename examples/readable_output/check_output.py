"""Verify scientific display conventions, exact factoring and column alignment."""
import copy,json
from pathlib import Path
import numpy as np
import sympy as sp
from demo import load_example
from ssg_output import matrix_view,matrix_text,character_columns,report_text,scalar

data=load_example();checks=0
for r in data['representations']:
    for i,op in enumerate(data['operations']):
        M=r['matrices'][i]
        if i in [0,1,4,192]:
            for phase in [sp.S.One,op['phase']]:
                view=matrix_view(M,phase)
                assert view['factor_value']*view['body']==phase*M
                assert max(map(len,matrix_text(view).splitlines()))<=110
                checks+=1
        if not op['anti']:
            assert sp.trace(M)==r['characters'][i]
            checks+=1
for convention in ['projective','bloch']:
    columns=character_columns(data,convention,True)
    ids=[g for c in columns for g in c['ids']]
    assert sorted(ids)==list(range(1,193)) and len(ids)==len(set(ids))
    assert len(columns)==6
    for c in columns:
        for gid in c['ids']:
            phase=data['operations'][gid-1]['phase'] if convention=='bloch' else 1
            assert c['values']==[sp.simplify(phase*sp.trace(r['matrices'][gid-1])) for r in data['representations']]
            checks+=1
full=character_columns(data,'bloch',False)
assert len(full)==192
assert scalar(sp.Integer(-4)*sp.I)=='-4i'
assert scalar(complex(-0.,-1e-14))=='0'
# Equal rounded display must NOT be interpreted as exact equality.
fake=dict(operations=[dict(id=1,anti=False,phase=1),dict(id=2,anti=False,phase=1)],
          representations=[dict(characters=[sp.Float(1),sp.Float(1+1e-8)])])
assert len(character_columns(fake,compress=True))==2
# Wide numerical matrices split into labelled column panels without mutation.
rng=np.random.default_rng(31);numeric=rng.normal(size=(16,16))+1j*rng.normal(size=(16,16))
before=numeric.copy();text=matrix_text(matrix_view(numeric))
assert 'Columns 1..' in text and max(map(len,text.splitlines()))<=110
assert np.array_equal(numeric,before)
try:report_text(data,rep_label='does_not_exist')
except ValueError:pass
else:raise AssertionError('Unknown representation accepted')
summary=dict(exact_factoring_and_character_checks=checks,compressed_character_columns=6,
             unitary_operations_accounted_for=192,antiunitary_character_columns=0,
             numeric_nonmutation=True,wide_matrix_pagination=True,maximum_text_width=110)
Path(__file__).with_name('format_validation.json').write_text(json.dumps(summary,indent=2))
print('PASS',summary)
