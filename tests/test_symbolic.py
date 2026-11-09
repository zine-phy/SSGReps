"""General symbolic paths: geometry, exceptional points, transfer, and reports."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'SSGReps'), str(ROOT/'examples/symbolic_paths')]
import SSGReps as native
from ssg_database import load_database
from ssg_output import character_columns, report_text, report_html, scalar
from ssg_symbolic import SymbolicPath, SymbolicPathError, parse_expression
from symbolic_paths import PathFamily


class SymbolicPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.database = load_database()
        cls.mr = SymbolicPath('221.8.6.3', ['1/2', '1/2', 't'], interval=('0', '1/2'),
                              optimize=True, seed=19, database=cls.database, validation_samples=2)

    def make(self, ssg, k, interval=('0', '1'), **kwargs):
        return SymbolicPath(ssg, k, interval=interval, database=self.database, validation_samples=2, **kwargs)

    def test_safe_exact_parser(self):
        t = sp.Symbol('t', real=True)
        self.assertEqual(parse_expression('0.1 + t/3', t), sp.Rational(1, 10)+t/3)
        self.assertEqual(parse_expression('sin(pi*t)/5', t), sp.sin(sp.pi*t)/5)
        for text in ['__import__("os").system("id")', '(1).__class__', 't[0]', 'u', 'I',
                     'sqrt(-1)', '1/0', 'floor(t)', 't**t', '1e999999999']:
            with self.subTest(expression=text), self.assertRaises(SymbolicPathError):
                parse_expression(text, t)

    def test_invalid_domains(self):
        for k, interval in [(['1/(t-1/2)', '0', '0'], ('0', '1')),
                            (['sqrt(t)', '0', '0'], ('-1', '1')),
                            (['t', '0', '0'], ('1', '0'))]:
            with self.subTest(k=k, interval=interval), self.assertRaises(SymbolicPathError):
                self.make('1.1.1.1.L', k, interval)

    def test_mr_against_independent_exact_characters(self):
        f = self.mr
        exact = PathFamily('MR')
        self.assertEqual(f.reference_group.rep_degree, [4, 4])
        self.assertEqual(f.global_indices, exact.geometry['global_indices_0based'])
        for value in ['1/7', 'sqrt(2)/5', '3/8']:
            t = parse_expression(value)
            unit = np.flatnonzero(np.array(f.reference_group.time_reversal) > 0)
            expected = [np.trace(exact.numeric(label, float(t))[unit], axis1=1, axis2=2) for label in exact.labels]
            for label in f.labels:
                actual = np.trace(f.numeric(label, value)[unit], axis1=1, axis2=2)
                self.assertLess(min(np.max(abs(actual-other)) for other in expected), 1e-10)
            self.assertTrue(f.validate_at(value)['passed'])
        self.assertTrue(f.validation['projective_factor_parameter_independent'])
        self.assertEqual(f.validation['exact_spatial_product_checks'], 64**2)
        self.assertEqual(f.translation([0, 0, 1]), sp.exp(-2*sp.pi*sp.I*f.t))
        for exponent, phase in zip(f.phase_exponents, f.phases):
            self.assertEqual(sp.simplify(phase-sp.exp(-2*sp.pi*sp.I*exponent)), 0)

    def test_unitary_characters_and_mixed_symbolic_format(self):
        f = self.mr
        data = f.report_data()
        columns = character_columns(data)
        self.assertEqual(sum(len(c['ids']) for c in columns), 32)
        self.assertTrue(all(not data['operations'][i-1]['anti'] for c in columns for i in c['ids']))
        text = report_text(data, 'r1', [3])
        html = report_html(data, 'r1', [3])
        self.assertIn('t', text)
        self.assertIn('numerical constant matrices', html)
        self.assertNotIn('Exact symbolic output', html)
        self.assertIn('Endpoints require separate', html)
        t = f.t
        self.assertEqual(scalar((2.0+1e-14j)*sp.exp(-sp.I*sp.pi*t)), '2*exp(-i*pi*t)')
        self.assertEqual(scalar((1e-14+1e-14j)*sp.exp(-sp.I*sp.pi*t)), '0')
        self.assertEqual(scalar(complex(0, 1+1e-14)), 'i')

    def test_centered_and_nonorthogonal_cells(self):
        for ssg, k in [('63.1.4.67', ['0', '0', 't']),
                       ('146.1.3.1', ['t', 't', 't'])]:
            with self.subTest(ssg=ssg):
                f = self.make(ssg, k, ('0', '1/2'))
                self.assertLess(f.geometry_audit['maximum_reconstruction_error'], 1e-10)
                self.assertNotEqual(f.lattice, sp.eye(3))
                self.assertTrue(f.validate_at('sqrt(2)/5')['passed'])

    def test_hexagonal_and_high_translation_index(self):
        for ssg, k in [('194.1.6.1.P', ['1/3', '1/3', 't']),
                       ('184.12.2.9', ['1/7', '2/11', 't'])]:
            with self.subTest(ssg=ssg):
                f = self.make(ssg, k, ('0', '1/2'))
                value = sp.Rational(2, 7)
                actual = native.load_little_group(ssg, f._k_numeric(value), False, 'double', self.database[ssg], seed=27)
                self.assertEqual(actual.oplist, f.global_indices)
                for character in f.reference_group.character:
                    self.assertLess(min(np.max(abs(character-other)) for other in actual.character), 1e-9)
                self.assertTrue(f.validate_at(value)['passed'])

    def test_internal_enhancement_and_endpoint_rejection(self):
        f = self.make('221.8.6.3', ['t', '0', '0'])
        point = next(x for x in f.exceptional_points if x['parameter'] == '1/2')
        self.assertGreater(point['full_little_group_order'], len(f.global_indices))
        self.assertTrue(f.exceptions_fully_enumerated)
        with self.assertRaisesRegex(SymbolicPathError, 'enhanced'):
            f.numeric('r1', '1/2')
        for endpoint in ['0', '1']:
            with self.assertRaisesRegex(SymbolicPathError, 'strictly inside'):
                f.numeric('r1', endpoint)
        self.assertTrue(f.validate_at('3/4')['passed'])

    def test_nonlinear_polynomial_path(self):
        f = self.make('221.8.6.3', ['t**2', '0', '0'])
        roots = [parse_expression(x['parameter']) for x in f.exceptional_points]
        self.assertTrue(any(sp.simplify(x-sp.sqrt(2)/2) == 0 for x in roots))
        self.assertTrue(f.exceptions_fully_enumerated)
        with self.assertRaisesRegex(SymbolicPathError, 'enhanced'):
            f.check_parameter('sqrt(2)/2')
        self.assertTrue(f.validate_at('1/3')['passed'])

    def test_transcendental_path_preserves_exact_exclusions(self):
        f = self.make('63.1.4.67', ['0', '0', 'sin(t)/5'], ('0', '2*pi'))
        self.assertFalse(f.exceptions_fully_enumerated)
        self.assertTrue(any('sin(t)' in item for x in f.exclusions for item in x['integer_conditions']))
        with self.assertRaisesRegex(SymbolicPathError, 'enhanced'):
            f.check_parameter('pi')
        self.assertTrue(f.validate_at('1/3')['passed'])

    def test_constant_path_with_kramers_case_and_single_valued(self):
        f = self.make('1.2.1.3', ['0', '0', '0'])
        self.assertEqual(f.reference_group.rep_degree, [2])
        self.assertEqual(f.reference_group.torsion, [4])
        anti = next(i for i, s in enumerate(f.reference_group.time_reversal) if s < 0)
        b = f.numeric('r1', '1/3')[anti]
        np.testing.assert_allclose(b @ b.conj(), -np.eye(2), atol=1e-12)
        single = self.make('63.1.4.67', ['0', '0', 't'], ('0', '1/2'), group_type='single')
        self.assertTrue(np.all(single.spin_factors == 1))
        self.assertTrue(single.validate_at('1/4')['passed'])

    def test_reproducible_sampling_and_export_contract(self):
        a = self.make('63.1.4.67', ['0', '0', 's'], ('0', '1/2'), parameter='s', seed=4)
        b = self.make('63.1.4.67', ['0', '0', 's'], ('0', '1/2'), parameter='s', seed=4)
        self.assertEqual(a.reference_parameter, b.reference_parameter)
        np.testing.assert_array_equal(a.reference_group.correp_matrix, b.reference_group.correp_matrix)
        exported = a.to_dict()
        self.assertFalse(exported['exactness']['fully_exact'])
        self.assertEqual(exported['kPath'], ['0', '0', 's'])
        self.assertEqual(len(exported['unitaryLocalOperationIds']), len(a.reference_group.character[0]))
        self.assertEqual(len(exported['validation']['samples']), 2)

    def test_cli_symbolic_exports_and_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            command = [sys.executable, str(ROOT/'ssgreps.py'), '--ssgNum', '63.1.4.67',
                       '--kp-symbolic', '0', '0', 't', '--interval', '0', '1/2',
                       '--out', 'both', '--rep', '1', '--ops', '2', '--fileType', 'json',
                       '--export', str(temp/'path.json'), '--html', str(temp/'path.html')]
            result = subprocess.run(command, cwd=temp, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((temp/'path.json').read_text())
            self.assertEqual(data['schema'], 'ssgreps.symbolic_path.v1')
            self.assertIn('t', ''.join(data['blochPhases']))
            for extra in [['--kp', '0', '0', '0'], ['--parameter', 'u']]:
                bad = subprocess.run(command+extra, cwd=temp, capture_output=True, text=True, timeout=120)
                self.assertNotEqual(bad.returncode, 0)
            bad = subprocess.run([sys.executable, str(ROOT/'ssgreps.py'), '--ssgNum', '63.1.4.67',
                                  '--kp', '0', '0', '0', '--interval', '0', '1'],
                                 cwd=temp, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('require --kpath', bad.stderr)


if __name__ == '__main__':
    unittest.main()
