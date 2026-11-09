"""Readable SSG matrix/character output. Formatting never changes input matrices.

Works with exact SymPy matrices or numeric SSGReps arrays. Character tables
contain unitary operations only. Equal-value columns may be compressed, but
are deliberately not labelled conjugacy classes of a projective co-group.
"""
import html
import math
import re
import textwrap
from collections import OrderedDict
import sympy as sp


def is_exact(x):
    return not sp.sympify(x).has(sp.Float)


def scalar(x,precision=6,zero_tol=1e-10,aliases=None):
    x=sp.sympify(x)
    if x.free_symbols and not is_exact(x):
        # Keep symbolic phases exact while formatting their numerical amplitude.
        constant, dependent = x.as_independent(*x.free_symbols, as_Add=False)
        if constant != 1:
            amplitude = scalar(constant, precision, zero_tol, aliases)
            if amplitude == '0':
                return '0'
            phase = scalar(dependent, precision, zero_tol, aliases)
            if amplitude == '1':
                return phase
            if amplitude == '-1':
                return '-'+phase
            if ' + ' in amplitude or ' - ' in amplitude:
                amplitude = '('+amplitude+')'
            return amplitude+'*'+phase
        return sp.sstr(x).replace('I','i').replace('**','^')
    if is_exact(x):
        text=sp.sstr(x).replace('I','i').replace('**','^')
        text=re.sub(r'(?<=\d)\*i\b','i',text)
        for radical,name in (aliases or {}).items():
            text=text.replace(sp.sstr(radical),name)
            text=text.replace('i*'+name,'i'+name).replace(name+'*i','i'+name)
        return text
    z=complex(x);a=z.real;b=z.imag
    if abs(a)<zero_tol:a=0.
    if abs(b)<zero_tol:b=0.
    def real(v):return format(v,'.'+str(precision)+'g')
    if not b:return real(a)
    imag='i' if real(abs(b))=='1' else real(abs(b))+'i'
    if not a:return ('-' if b<0 else '')+imag
    return real(a)+(' - ' if b<0 else ' + ')+imag


def array_text(values):
    """Format operation metadata with the same clean scalar convention."""
    if isinstance(values,(list,tuple)):
        return '['+', '.join(array_text(x) for x in values)+']'
    return scalar(values)


def matrix_view(matrix,prefactor=1,precision=6,zero_tol=1e-10):
    M=sp.ImmutableMatrix(matrix);exact=all(is_exact(x) for x in M) and is_exact(prefactor)
    common=sp.S.One
    if exact:
        contents=[x.as_content_primitive()[0] for x in M if x!=0]
        if contents:common=sp.gcd_list(contents)
    B=M/common
    radicals=[sp.sqrt(n) for n in [2,3,5,6,7] if sum(x.has(sp.sqrt(n)) for x in B)>=4]
    aliases={r:('r' if len(radicals)==1 else 'r'+str(r**2)) for r in radicals}
    cells=[[scalar(x,precision,zero_tol,aliases) for x in row] for row in B.tolist()]
    d=M.rows;half=d//2;split=None
    if d==M.cols and d>=8 and d%2==0:
        z=sp.zeros(half)
        if (B[:half,half:]==z and B[half:,:half]==z) or (B[:half,:half]==z and B[half:,half:]==z):split=half
    factor=sp.simplify(prefactor*common) if exact else prefactor*common
    return dict(cells=cells,factor=scalar(factor,precision,zero_tol),factor_value=factor,
                body=B,rows=M.rows,cols=M.cols,split=split,exact=exact,
                aliases={name:scalar(r) for r,name in aliases.items()},
                zero_count=sum(x=='0' for row in cells for x in row))


def matrix_text(view,width=110):
    cells=view['cells'];rows=view['rows'];cols=view['cols'];split=view['split']
    widths=[max(len(str(j+1)),*(len(row[j]) for row in cells)) for j in range(cols)]
    rw=len(str(rows))
    def line(values,indices):
        out=''
        for k,(j,v) in enumerate(zip(indices,values)):
            if k:out+=' | ' if j==split else ' '
            out+=v.rjust(widths[j])
        return out
    panels=[];first=0
    while first<cols:
        last=first+1
        while last<cols and rw+6+len(line(['0']*(last-first+1),range(first,last+1)))<=width:last+=1
        panels.append(list(range(first,last)));first=last
    output=[]
    for indices in panels:
        if len(panels)>1:output+=['','Columns '+str(indices[0]+1)+'..'+str(indices[-1]+1)]
        output.append(' '*(rw+3)+line([str(j+1) for j in indices],indices))
        for i,row in enumerate(cells):
            shown=[row[j] for j in indices]
            if i==split:output.append(' '*(rw+1)+'| '+'-'*len(line(shown,indices))+' |')
            output.append(str(i+1).rjust(rw)+' | '+line(shown,indices)+' |')
    if view['aliases']:output+=['  '+', '.join(name+' = '+value for name,value in view['aliases'].items())]
    return '\n'.join(output)


def aligned_table(headers,rows):
    data=[[str(x) for x in row] for row in [headers]+rows]
    widths=[max(len(row[j]) for row in data) for j in range(len(headers))]
    lines=[' | '.join(x.rjust(w) for x,w in zip(data[0],widths)),
           '-+-'.join('-'*w for w in widths)]
    lines+=[' | '.join(x.rjust(w) for x,w in zip(row,widths)) for row in data[1:]]
    return '\n'.join(lines)


def operation_ranges(ids):
    ids=sorted(ids);parts=[];i=0
    while i<len(ids):
        j=i
        while j+1<len(ids) and ids[j+1]==ids[j]+1:j+=1
        parts.append('g%03d'%ids[i]+('..g%03d'%ids[j] if j>i else ''))
        i=j+1
    return ', '.join(parts)


def character_columns(data,convention='bloch',compress=True,precision=6):
    ops=data['operations'];reps=data['representations'];columns=OrderedDict()
    for j,op in enumerate(ops):
        if op['anti']:continue
        phase=op['phase'] if convention=='bloch' else 1
        values=[]
        for r in reps:
            character = r['characters'][j]
            if is_exact(phase) and is_exact(character):
                value = sp.simplify(phase*character)
            elif sp.sympify(phase).free_symbols or sp.sympify(character).free_symbols:
                value = sp.sympify(phase)*sp.sympify(character)
            else:
                value = complex(phase)*complex(character)
            values.append(value)
        # Compression uses exact underlying values, never the rounded display.
        key=tuple(values) if compress else (j,)
        if key not in columns:columns[key]=dict(ids=[],values=values)
        columns[key]['ids'].append(op['id'])
    result=list(columns.values())
    for c in result:c['strings']=[scalar(v,precision) for v in c['values']]
    return result


def character_text(data,convention='bloch',compress=True,width=110):
    columns=character_columns(data,convention,compress)
    chunks=[];start=0
    while start<len(columns):
        end=min(start+10,len(columns))
        while True:
            selected=columns[start:end]
            headers=['rep','dim']+['g%03d'%c['ids'][0] for c in selected]
            rows=[['count','-']+[len(c['ids']) for c in selected]]
            rows += [[r['label'],r['dimension']]+[c['strings'][i] for c in selected] for i,r in enumerate(data['representations'])]
            table=aligned_table(headers,rows)
            if max(map(len,table.splitlines()))<=width or end==start+1:break
            end-=1
        chunks.append(table);start=end
    caption='Characters: unitary operations only.'
    if compress:caption+=' Identical-value columns merged; these are NOT conjugacy classes.'
    text=caption+'\n\n'+'\n\n'.join(chunks)
    if compress:
        text+='\n\nColumn membership:\n'
        for c in columns:
            text+=textwrap.fill('g%03d: '%c['ids'][0]+operation_ranges(c['ids']),width=width,subsequent_indent='      ')+'\n'
    return text


def report_text(data,rep_label=None,operation_ids=None,convention='bloch',output='both',compress=True):
    if convention not in ('bloch','projective'):raise ValueError('Invalid convention')
    ops=data['operations'];reps=data['representations'];selected=reps if rep_label is None else [r for r in reps if r['label']==rep_label]
    if not selected:raise ValueError('Unknown representation: '+str(rep_label))
    title='SSG '+data['ssg']+' | k = ('+', '.join(map(str,data['k']))+')'
    lines=[title,'='*len(title),'Coordinates: reciprocal basis of pure translations.',
           'Convention: '+('D(g) = exp(-i k_cart.tau_g) M(g); antiunitary operator = D(g) K.' if convention=='bloch' else 'Projective M(g); antiunitary operator = M(g) K.'),
           'Operations: '+str(len(ops))+' | unitary: '+str(sum(not o['anti'] for o in ops))+' | dimensions: '+', '.join(str(r['dimension']) for r in reps)]
    if not data.get('exact',False):lines+=['Numeric display: 6 significant digits; components below 1e-10 shown as 0.']
    lines += data.get('notes', [])
    if output in ('both','character'):lines+=['',character_text(data,convention,compress)]
    if output in ('both','rep_matrix'):
        for r in selected:
            for j,op in enumerate(ops):
                if operation_ids is not None and op['id'] not in operation_ids:continue
                view=matrix_view(r['matrices'][j],op['phase'] if convention=='bloch' else 1)
                lines+=['',r['label']+' | dimension '+str(r['dimension'])+' | g%03d'%op['id']+' | '+('antiunitary' if op['anti'] else 'unitary'),
                        'R = '+array_text(op['R'])+'; tau = '+array_text(op['tau']),
                        'spin = '+array_text(op['spin']),
                        ('D' if convention=='bloch' else 'M')+'(g%03d)'%op['id']+' = '+view['factor']+' * matrix'+(' ; operator includes K' if op['anti'] else ''),matrix_text(view)]
    return '\n'.join(lines)+'\n'


def report_html(data,rep_label,operation_ids,convention='bloch',compress=True):
    if convention not in ('bloch','projective'):raise ValueError('Invalid convention')
    E=lambda s:html.escape(str(s));reps=data['representations'];rep=next(r for r in reps if r['label']==rep_label)
    columns=character_columns(data,convention,compress)
    header='<th>Representation</th><th>Dimension</th>'+''.join('<th>g%03d<small>%d operation(s)</small></th>'%(c['ids'][0],len(c['ids'])) for c in columns)
    body=''.join('<tr><th>'+E(r['label'])+'</th><td>'+str(r['dimension'])+'</td>'+''.join('<td>'+E(c['strings'][i])+'</td>' for c in columns)+'</tr>' for i,r in enumerate(reps))
    membership='<details><summary>Show group elements in each column</summary>'+''.join('<p><b>g%03d</b>: '%c['ids'][0]+E(operation_ranges(c['ids']))+'</p>' for c in columns)+'</details>'
    cards=[]
    for j,op in enumerate(data['operations']):
        if op['id'] not in operation_ids:continue
        view=matrix_view(rep['matrices'][j],op['phase'] if convention=='bloch' else 1);split=view['split']
        rowhtml=[]
        for ri,row in enumerate(view['cells']):
            cells=''.join('<td class="'+('zero ' if x=='0' else '')+('splitcol' if ci==split else '')+'">'+E(x)+'</td>' for ci,x in enumerate(row))
            rowhtml.append('<tr'+(' class="splitrow"' if ri==split else '')+'><th>'+str(ri+1)+'</th>'+cells+'</tr>')
        min_width=30+view['cols']*max(50,14+10*max(len(x) for row in view['cells'] for x in row))
        table='<table class="matrix" style="min-width:'+str(min_width)+'px"><thead><tr><th></th>'+''.join('<th>'+str(k+1)+'</th>' for k in range(view['cols']))+'</tr></thead><tbody>'+''.join(rowhtml)+'</tbody></table>'
        zero=100*view['zero_count']/(view['rows']*view['cols'])
        legend='，'.join(name+' = '+value for name,value in view['aliases'].items())
        caption=('D' if convention=='bloch' else 'M')+'(g%03d) = '%op['id']+view['factor']+' × the matrix below'
        if op['anti']:caption+='; the full operator includes K'
        cards.append('<section class="card matrixcard" id="matrix-g%03d">'%op['id']+'<div class="eyebrow">SSG '+E(data['ssg'])+' · k = ('+E(', '.join(map(str,data['k'])))+')</div><h2>'+E(rep_label)+' · '+str(rep['dimension'])+'D representation <span>g%03d · '%op['id']+('antiunitary' if op['anti'] else 'unitary')+'</span></h2><p class="formula">'+E(caption)+'</p><div class="matrix-scroll">'+table+'</div><div class="legend">'+E(legend or ('All entries are exact' if view['exact'] else 'Numeric output: 6 significant digits'))+'<span>'+('Exact zeros: ' if view['exact'] else 'Entries displayed as zero: ')+format(zero,'.2f')+'%</span></div><details><summary>Spatial and spin operations</summary><p>R = '+E(array_text(op['R']))+'</p><p>spin = '+E(array_text(op['spin']))+'</p><p>tau = '+E(array_text(op['tau']))+'</p></details></section>')
    notes = ''.join('<p class="note">'+E(note)+'</p>' for note in data.get('notes', []))
    output_kind = ('Symbolic phases; numerical constant matrices' if data.get('symbolic', False)
                   else 'Exact symbolic output' if data.get('exact', False) else 'Numeric output: 6 significant digits')
    css='''*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#172333;font:15px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1180px;margin:auto;padding:38px 28px}h1{font-size:30px;margin:7px 0 14px}h2{font-size:21px;margin:8px 0 18px}h2 span{float:right;font-size:15px;font-weight:500;color:#536577}.eyebrow{font-size:12px;letter-spacing:.07em;color:#62758a}.intro{color:#536577;line-height:1.8}.card{background:white;border:1px solid #dce3eb;border-radius:12px;padding:24px;margin:22px 0;box-shadow:0 2px 9px #14223105}.tags{display:flex;gap:10px;flex-wrap:wrap}.tags span{background:#e8eef6;color:#354d6c;border-radius:5px;padding:7px 11px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}.characters td,.characters th{text-align:right;padding:12px 16px;border-bottom:1px solid #e5eaf0}.characters th:first-child{text-align:left}small{display:block;font-size:11px;font-weight:400;color:#748294;margin-top:5px}.matrix-scroll{overflow:auto}.matrix{table-layout:fixed;font-family:ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace;font-size:16px}.matrix th{color:#8795a5;font-size:11px;font-weight:400;height:27px}.matrix th:first-child{width:30px}.matrix td{text-align:center;height:29px;color:#183c73}.matrix .zero{color:#a3adb9}.matrix .splitcol{border-left:2px solid #a4b3c6}.matrix .splitrow td{border-top:2px solid #a4b3c6}.formula{background:#f1f5fa;border-radius:5px;padding:12px 14px;font-family:ui-monospace,Menlo,monospace;font-size:15px}.legend{margin-top:16px;color:#536577}.legend span{float:right}details{border-top:1px solid #e7ebf1;margin-top:18px;padding-top:13px;color:#56677a;font-size:13px}summary{cursor:pointer}details p{line-height:1.8;overflow-wrap:anywhere}.note{font-size:13px;color:#697b90;line-height:1.7}@media print{body{background:white}main{padding:0}.card{break-inside:avoid;box-shadow:none}.matrix{font-size:12px}}'''
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SSG '+E(data['ssg'])+' Representation report</title><style>'+css+'</style><main><div class="eyebrow">SPIN SPACE GROUP REPRESENTATIONS</div><h1>SSG '+E(data['ssg'])+' · Representation report</h1><p class="intro">k = ('+E(', '.join(map(str,data['k'])))+'), in the reciprocal basis of pure translations.<br>'+('Matrices and characters include the Bloch phase: D(g) = exp(-i k_cart·tau) M(g).' if convention=='bloch' else 'Matrices and characters use the projective convention M(g).')+'</p><div class="tags"><span>'+str(len(data['operations']))+' group operations</span><span>Irreducible corepresentations: '+E(', '.join(str(r['dimension'])+'D' for r in reps))+'</span><span>'+output_kind+'</span></div>'+notes+'<section class="card"><h2>Character table</h2><p class="note">Unitary operations only. '+('Identical character columns are merged and labelled by a representative element. These groups are not conjugacy classes.' if compress else 'One column per group operation; scroll horizontally to view all columns.')+'</p><div class="matrix-scroll"><table class="characters"><thead><tr>'+header+'</tr></thead><tbody>'+body+'</tbody></table></div>'+membership+'</section>'+''.join(cards)+'<p class="note">Representation names are local labels for this calculation. K denotes complex conjugation; traces of antiunitary matrix parts are not ordinary characters.</p></main></html>'


def from_little_group(lg):
    """Adapt the original numeric SSGReps object without changing its matrices."""
    import numpy as np
    kconv=lg.kvec[0]*lg.b1+lg.kvec[1]*lg.b2+lg.kvec[2]*lg.b3
    ops=[dict(id=i+1,anti=t<0,phase=complex(np.exp(-1j*np.dot(kconv,lg.tauC[i]))),
              R=np.asarray(lg.rotC[i]).tolist(),spin=np.asarray(lg.spin[i]).tolist(),tau=np.asarray(lg.tauC[i]).tolist()) for i,t in enumerate(lg.time_reversal)]
    reps=[dict(label='r'+str(i+1),dimension=int(lg.rep_degree[i]),matrices=ms,
               characters=[complex(np.trace(m)) for m in ms]) for i,ms in enumerate(lg.correp_matrix)]
    return dict(ssg=lg.ssgNum,k=[scalar(x) for x in lg.kvec],operations=ops,representations=reps,exact=False)
