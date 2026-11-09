"""Regression tests for mathematical output and command-line failure modes."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'SSGReps'))
import SSGReps as native
from ssg_database import load_database
from ssg_numeric import validate, RepresentationError
from ssg_output import report_html, from_little_group, matrix_view, matrix_text


class ReleaseRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.database = load_database()

    def solve(self, name, k=(0,0,0), optimize=False):
        return native.load_little_group(name, np.array(k), optimize, 'double', self.database[name], seed=19)

    def test_kramers_square_and_optimization(self):
        for optimize in [False, True]:
            lg = self.solve('1.2.1.3', optimize=optimize)
            self.assertEqual(lg.rep_degree, [2])
            a = lg.correp_matrix[0][1]
            np.testing.assert_allclose(a@a.conj(), -np.eye(2), atol=1e-12)
            self.assertTrue(validate(lg, check_group=True)['passed'])

    def test_corruption_is_rejected(self):
        lg = self.solve('1.2.1.3')
        lg.correp_matrix[0][1] *= 2
        with self.assertRaises(RepresentationError):
            validate(lg)

    def test_known_mixed_corepresentation_failure(self):
        lg = self.solve('1.6.1.3', optimize=True)
        self.assertEqual(lg.rep_degree, [2,2])
        self.assertLess(validate(lg)['projective_product_max'], 1e-12)

    def test_json_and_npy_export_match_final_matrices(self):
        with tempfile.TemporaryDirectory() as temp:
            for kind in ['json','npy']:
                target = Path(temp)/('result.'+kind)
                with contextlib.redirect_stdout(io.StringIO()):
                    lg = native.load_one_ssg_kvec('1.2.1.3', [0,0,0], fileType=kind,
                        optimize=True, export_path=target, html_path=Path(temp)/'report.html')
                if kind == 'json':
                    def hook(obj):
                        return complex(obj['real'],obj['imag']) if set(obj)=={'real','imag'} else obj
                    data = json.loads(target.read_text(), object_hook=hook)
                else:
                    data = np.load(target, allow_pickle=True).item()
                np.testing.assert_allclose(np.array(data['repMatrix']), np.array(lg.correp_matrix))
                self.assertTrue(data['validation']['passed'])
                self.assertEqual(data['matrixConvention'], 'projective')
                for r, chars in zip(data['repMatrix'], data['character']):
                    np.testing.assert_allclose(np.trace(np.array(r)[np.array(data['timeReversal'])>0], axis1=1, axis2=2), chars)

    def test_cli_false_and_failure_exit_codes(self):
        with tempfile.TemporaryDirectory() as temp:
            base = [sys.executable, str(ROOT/'ssgreps.py'), '--ssgNum','1.2.1.3','--kp','0','0','0']
            proc = subprocess.run(base+['--optimize','False','--fileType','json','--export','result.json'],
                                  cwd=temp, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, '')
            self.assertIn('SSG 1.2.1.3', proc.stdout)
            for extra in [['--rep','0'], ['--groupType','3'], ['--optimize','garbage'], ['--ops','999']]:
                proc = subprocess.run(base+extra, cwd=temp, capture_output=True, text=True)
                self.assertNotEqual(proc.returncode, 0)
                self.assertEqual(proc.stdout, '')
                self.assertIn('error', proc.stderr.lower())
            proc = subprocess.run([sys.executable, str(ROOT/'ssgreps.py'), '--ssgNum','1.2.1.3',
                                   '--kp','nan','0','0'],cwd=temp,capture_output=True,text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('finite', proc.stderr)

    def test_numeric_html_never_claims_exactness(self):
        lg = self.solve('1.2.1.3')
        html = report_html(from_little_group(lg), 'r1', [1,2])
        self.assertNotIn('Exact symbolic output', html)
        self.assertIn('Numeric output', html)
        self.assertIn('antiunitary', html)

    def test_wide_matrix_roundtrip_and_nonmutation(self):
        rng = np.random.default_rng(5)
        array = rng.normal(size=(16,16))+1j*rng.normal(size=(16,16))
        before = array.copy()
        text = matrix_text(matrix_view(array))
        self.assertIn('Columns 1..', text)
        self.assertLessEqual(max(map(len,text.splitlines())),110)
        np.testing.assert_array_equal(array,before)


if __name__ == '__main__':
    unittest.main()
