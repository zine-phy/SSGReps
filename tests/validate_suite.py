"""Reproducible cross-SSG, geometry, output, and exact-character validation.

Run from any directory: python /path/to/SSGReps/tests/validate_suite.py
No network or GAP installation is required for this numerical cross-check.
"""
import argparse
import importlib.metadata
import json
from html.parser import HTMLParser
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'SSGReps'))
import SSGReps as native
from ssg_database import load_database
from ssg_numeric import validate
from ssg_output import from_little_group, report_text, report_html, matrix_text, matrix_view, character_columns


class MatrixParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.active = False
        self.in_cell = False
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'table' and attrs.get('class') == 'matrix':
            self.active = True
            self.tables.append([])
        if self.active and tag == 'td':
            self.in_cell = True
            self.tables[-1].append('')
    def handle_endtag(self, tag):
        if tag == 'td':
            self.in_cell = False
        if tag == 'table':
            self.active = False
    def handle_data(self, data):
        if self.active and self.in_cell:
            self.tables[-1][-1] += data


def geometry_check(lg):
    """Recompute little-group membership and product geometry independently."""
    full = lg.ssg
    lattice = np.asarray(full.pure_T).T
    inv_lattice = np.linalg.inv(lattice)
    reciprocal = 2*np.pi*inv_lattice.T
    kcart = reciprocal @ lg.kvec
    expected = []
    for i, (r, sign) in enumerate(zip(full.rotC, full.time_reversal)):
        shift = lg.kvec @ inv_lattice @ (sign*np.asarray(r)-np.eye(3)) @ lattice
        if np.max(abs(shift-np.rint(shift))) < 1e-9:
            expected.append(i)
    assert lg.oplist == expected, 'Incorrect little-group membership'
    assert lg.antiunitary == any(t < 0 for t in lg.time_reversal)
    mt = np.asarray(lg.mul_table)-1
    rotations = np.asarray(lg.rotC)
    translations = np.asarray(lg.tauC)
    spins = np.asarray(lg.spin)
    signs = np.asarray(lg.time_reversal)
    residual = 0.
    for i in range(len(mt)):
        residual = max(residual, float(np.max(abs(rotations[i]@rotations-rotations[mt[i]]))))
        residual = max(residual, float(np.max(abs(spins[i]@spins-spins[mt[i]]))))
        defects = (translations[i]+translations@rotations[i].T-translations[mt[i]]) @ inv_lattice.T
        residual = max(residual, float(np.max(abs(defects-np.rint(defects)))))
        assert np.array_equal(signs[i]*signs, signs[mt[i]])
    assert residual < 1e-7
    return residual


def output_check(lg, example_dir=None):
    data = from_little_group(lg)
    ri = int(np.argmax(lg.rep_degree))
    indices = list(dict.fromkeys([0, min(1, len(lg.time_reversal)-1)]
                + [i for i, t in enumerate(lg.time_reversal) if t < 0][:1]))
    label = 'r'+str(ri+1)
    worst = 0.
    for convention in ['bloch', 'projective']:
        html = report_html(data, label, [i+1 for i in indices], convention)
        parser = MatrixParser()
        parser.feed(html)
        assert len(parser.tables) == len(indices)
        assert '<html lang="en">' in html
        assert not any('\u3400' <= c <= '\u9fff' for c in html)
        for values, index in zip(parser.tables, indices):
            factor = data['operations'][index]['phase'] if convention == 'bloch' else 1
            view = matrix_view(lg.correp_matrix[ri][index], factor)
            parsed = np.array([complex(x.replace(' ', '').replace('i', 'j')) for x in values])
            displayed_factor = complex(view['factor'].replace(' ', '').replace('i', 'j'))
            expected = factor*np.asarray(lg.correp_matrix[ri][index])
            error = float(np.max(abs(parsed.reshape(expected.shape)*displayed_factor-expected)))
            worst = max(worst, error)
            assert error < 1e-5, 'Rendered HTML matrix changed its numerical value'
            assert max(map(len, matrix_text(view).splitlines())) <= 110
        columns = character_columns(data, convention, compress=True)
        ids = [gid for c in columns for gid in c['ids']]
        assert sorted(ids) == [o['id'] for o in data['operations'] if not o['anti']]
        for c in columns:
            for gid in c['ids']:
                phase = data['operations'][gid-1]['phase'] if convention == 'bloch' else 1
                assert np.max(abs(np.array(list(map(complex, c['values'])))-
                    np.array([phase*np.trace(r[gid-1]) for r in lg.correp_matrix]))) < 1e-12
        text = report_text(data, label, [i+1 for i in indices], convention, compress=True)
        assert 'nan' not in text.lower() and 'inf' not in text.lower()
        if example_dir and convention == 'bloch':
            example_dir.mkdir(parents=True, exist_ok=True)
            (example_dir/'report.html').write_text(html, encoding='utf-8')
            (example_dir/'report.txt').write_text(text, encoding='utf-8')
    return worst


def exact_character_check(lg, name):
    directory = ROOT/'examples/symbolic_paths'/name
    geometry = json.loads((directory/'geometry.json').read_text())
    assert lg.oplist == geometry['global_indices_0based']
    raw = json.loads((directory/'representations.json').read_text())
    sys.path.insert(0, str(ROOT/'examples/symbolic_output'))
    from symbolic_numbers import parse_gap
    exact = [np.array([complex(parse_gap(c).evalf()) for c in r['character']]) for r in raw['representations']]
    errors = np.array([[np.max(abs(a-b)) for b in lg.character] for a in exact])
    matches = np.argmin(errors, axis=1)
    assert len(exact) == len(lg.character) and len(set(matches)) == len(matches)
    error = float(np.max(errors[np.arange(len(exact)), matches]))
    assert error < 1e-9
    return error


def cases():
    names = ['1.2.1.3', '1.6.1.3', '1.4.1.5', '63.1.4.67', '194.1.6.1.P',
             '1.1.1.1.L', '221.8.6.3', '194.9.4.19', '194.10.4.17',
             '173.11.1.1', '184.12.2.9', '5.1.2.1', '76.1.2.1',
             '146.1.3.1', '225.1.12.1', '229.1.12.1']
    points = [('Gamma', [0,0,0]), ('boundary', [.5,.5,.5]),
              ('axis_line', [0,0,1/7]), ('diagonal_line', [1/7]*3),
              ('general', [.137,.219,.317])]
    for name in names:
        for label, k in points:
            exact = ('G' if label == 'Gamma' else 'R') if name == '221.8.6.3' and label in ('Gamma','boundary') else None
            yield dict(ssg=name, label=label, k=k, group='double', optimize=False, seed=0, exact=exact)
    # Single-valued and optimized modes exercise antiunitary and large-cell cases.
    for name in names[:11]:
        for group, optimize in [('single', False), ('double', True)]:
            yield dict(ssg=name, label='mode_regression', k=[.5]*3,
                       group=group, optimize=optimize, seed=17, exact=None)
    # Six genuine cubic high-symmetry paths, with rational and irrational parameters.
    paths = dict(GX=([0,0,0],[1,0,0]), GM=([0,0,0],[1,1,0]), GR=([0,0,0],[1,1,1]),
                 XM=([.5,0,0],[0,1,0]), MR=([.5,.5,0],[0,0,1]), XR=([.5,0,0],[0,1,1]))
    for name, (u,v) in paths.items():
        for t in [1/17, 1/4, np.sqrt(2)/5]:
            yield dict(ssg='221.8.6.3', label=name, k=(np.array(u)+t*np.array(v)).tolist(),
                       group='double', optimize=True, seed=31, exact=name, t=float(t))
    for name, k in [('X',[.5,0,0]), ('M',[.5,.5,0])]:
        yield dict(ssg='221.8.6.3', label=name, k=k, group='double', optimize=True, seed=19, exact=name)
    for seed in [1, 29]:
        for name in ['1.2.1.3','1.6.1.3','184.12.2.9']:
            yield dict(ssg=name,label='seed_regression',k=[.137,.219,.317],group='double',optimize=True,seed=seed,exact=None)
    yield dict(ssg='221.8.6.3',label='near_Gamma',k=[0,0,1e-5],group='double',optimize=False,seed=7,exact='GX')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'validation/extended_validation.json')
    parser.add_argument('--quick', action='store_true', help='Run the first three SSGs only')
    args = parser.parse_args()
    database = load_database()
    report = dict(versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','sympy','spglib']},
                  tolerance=1e-7, display_tolerance=1e-5, cases=[])
    for number, case in enumerate(cases()):
        if args.quick and number >= 15:
            break
        started = time.monotonic()
        row = dict(case)
        record = database[case['ssg']]
        row['translation_index'] = int(round(abs(np.linalg.det(record['superCell']))))
        row['spin_dimension'] = int(record['dim'])
        try:
            lg = native.load_little_group(case['ssg'], np.array(case['k']), case['optimize'], case['group'], record, seed=case['seed'])
            row['validation'] = validate(lg, check_group=True)
            row['geometry_max'] = geometry_check(lg)
            directory = ROOT/'validation/examples'/f'{number:03d}' if number in [0,31,40,45,50,110] else None
            row['display_max'] = output_check(lg, directory)
            if case['exact']:
                # GX is the x-axis; the near-Gamma test uses the z-axis instead.
                if case['label'] != 'near_Gamma':
                    row['exact_character_max'] = exact_character_check(lg, case['exact'])
                else:
                    assert len(lg.time_reversal) == 64
            row['dimensions'] = lg.rep_degree
            row['antiunitary'] = lg.antiunitary
            row['status'] = 'pass'
        except Exception as exc:
            row.update(status='fail', error=repr(exc), traceback=traceback.format_exc())
        row['seconds'] = round(time.monotonic()-started, 3)
        report['cases'].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(row['status'].upper(), number, case['ssg'], case['label'], row.get('dimensions', row.get('error')), row['seconds'], flush=True)
    failures = [r for r in report['cases'] if r['status'] != 'pass']
    report['summary'] = dict(cases=len(report['cases']), failed=len(failures),
        ssg_count=len({r['ssg'] for r in report['cases']}),
        max_translation_index=max(r['translation_index'] for r in report['cases']),
        product_checks_per_convention=sum(r.get('validation',{}).get('product_checks',0) for r in report['cases']),
        max_projective_residual=max(r.get('validation',{}).get('projective_product_max',0) for r in report['cases']),
        max_bloch_residual=max(r.get('validation',{}).get('bloch_product_max',0) for r in report['cases']))
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['summary']), flush=True)
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
