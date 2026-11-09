"""Coefficient conventions, Wigner norms, cached material data, and adapter checks."""
from functools import lru_cache
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/wavecar_mn3sn"
sys.path[:0] = [str(ROOT / "SSGReps"), str(EXAMPLE)]
from ssg_wavecar import (orthonormal_basis, transform_coefficients,
                        representation_matrix, match_unitary_characters,
                        validate_unitary_block)
from replay_validation import replay
from analyze_wavecar import select_blocks


class CoefficientTests(unittest.TestCase):
    def test_forward_translation_phase_and_common_gram_basis(self):
        c = np.array([[1, 0.5], [0, np.sqrt(3)/2]], complex).reshape(1, 2, 2)
        q, stats = orthonormal_basis(c)
        spin = np.diag(np.exp([-1j*np.pi/3, 1j*np.pi/3]))
        b, _ = representation_matrix(q, [[0, 0, 0]], [0.25, 0, 0],
                                     np.eye(3), [0.5, 0, 0], spin)
        expected = q.reshape(2, 2).conj().T @ spin @ q.reshape(2, 2) * np.exp(-1j*np.pi/4)
        np.testing.assert_allclose(b, expected, atol=1e-12, rtol=0)
        self.assertLess(stats["orthonormality_residual"], 1e-12)
        with self.assertRaisesRegex(ValueError, "common orthonormal_basis"):
            representation_matrix(c, [[0, 0, 0]], [0, 0, 0], np.eye(3), [0, 0, 0], np.eye(2))

    def test_16d_real_space_action_and_spinful_products(self):
        g = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        rng = np.random.default_rng(419)
        mix, _ = np.linalg.qr(rng.normal(size=(16, 16)) + 1j*rng.normal(size=(16, 16)))
        c = mix.reshape(8, 2, 16)
        r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        tau = np.array([0.25, 0.125, 0])
        spin = np.diag(np.exp([-1j*np.pi/4, 1j*np.pi/4]))
        time_reversal = np.array([[0, 1], [-1, 0]])
        b, _ = representation_matrix(c, g, [0, 0, 0], r, tau, spin)
        t, _ = representation_matrix(c, g, [0, 0, 0], np.eye(3), [0, 0, 0],
                                     time_reversal, antiunitary=True)
        np.testing.assert_allclose(np.linalg.matrix_power(b, 4), -np.eye(16), atol=1e-12, rtol=0)
        np.testing.assert_allclose(t @ t.conj(), -np.eye(16), atol=1e-12, rtol=0)
        np.testing.assert_allclose(t @ b.conj(), b @ t, atol=1e-12, rtol=0)
        for rotation, shift, v, anti in [(r, tau, spin, False),
                                         (np.eye(3), np.zeros(3), time_reversal, True)]:
            transformed = transform_coefficients(c, g, [0, 0, 0], rotation, shift, v, antiunitary=anti)
            for x in rng.random((7, 3)):
                def evaluate(coeff, position):
                    return np.einsum("g,gsn->sn", np.exp(2j*np.pi*(g @ position)), coeff)
                value = evaluate(c, np.linalg.solve(rotation, x-shift))
                np.testing.assert_allclose(evaluate(transformed, x), v @ (value.conj() if anti else value),
                                           atol=1e-12, rtol=0)
        with self.assertRaisesRegex(ValueError, "Subspace check failed"):
            representation_matrix(c[:, :, :1], g, [0, 0, 0], r, tau, spin)

    def test_invalid_subspaces_and_reciprocal_mapping(self):
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            orthonormal_basis(np.ones((1, 2, 2)))
        c = np.eye(2).reshape(1, 2, 2)
        with self.assertRaisesRegex(ValueError, "does not preserve"):
            transform_coefficients(c, [[0, 0, 0]], [0.137, 0, 0], -np.eye(3), [0, 0, 0], np.eye(2))
        with self.assertRaisesRegex(ValueError, "not closed"):
            transform_coefficients(c, [[1, 0, 0]], [0, 0, 0], -np.eye(3), [0, 0, 0], np.eye(2))


class CharacterTests(unittest.TestCase):
    def test_wigner_norms_and_incomplete_multiplet(self):
        z = np.exp(2j*np.pi/3)
        c3 = np.array([[1, 1, 1], [1, z, z*z], [1, z*z, z]])
        result = match_unitary_characters(c3[0]+2*c3[1], c3)
        self.assertEqual(result["multiplicities"], {"r1": 1, "r2": 2, "r3": 0})
        for reference, norm in [([[2]], 4), ([c3[1]+c3[2]], 2)]:
            result = match_unitary_characters(reference[0], reference, ["branch"])
            self.assertEqual(result["multiplicities"], {"branch": 1})
            self.assertAlmostEqual(result["unitary_restriction_norms"]["branch"], norm)
        with self.assertRaisesRegex(ValueError, "integer"):
            match_unitary_characters([1], [[2]])
        with self.assertRaises(ValueError):
            match_unitary_characters([1, 1, 1.1], c3)

    def test_exact_8d_16d_restrictions(self):
        sys.path.insert(0, str(ROOT / "examples/symbolic_output"))
        from symbolic_numbers import parse_gap
        raw = json.loads((ROOT / "examples/symbolic_paths/R/representations.json").read_text())
        @lru_cache(None)
        def number(value):
            return complex(parse_gap(value).evalf(30))
        refs = np.array([[number(v) for v in r["character"]] for r in raw["representations"]])
        for index, dimension in enumerate([8, 8, 16]):
            result = match_unitary_characters(refs[index], refs)
            self.assertEqual(result["dimension"], dimension)
            self.assertEqual(list(result["multiplicities"].values()), [int(i == index) for i in range(3)])
        self.assertEqual(match_unitary_characters(2*refs[0]+refs[2], refs)["dimension"], 32)

    def test_product_and_leakage_rejections_despite_matching_characters(self):
        # Traces alone miss this false C2 action: trace(U)=0 but U^2=-I.
        matrices = np.array([np.eye(2), np.diag([1j, -1j])])
        refs = np.array([[1, 1], [1, -1]])
        products = np.array([[0, 1], [1, 0]])
        factors = np.ones((2, 2))
        assessment = validate_unitary_block(matrices, refs, products, factors, 0)
        self.assertFalse(assessment["passed"])
        self.assertAlmostEqual(assessment["character_residual"], 0)
        matrices[1] = np.diag([1, -1])
        self.assertTrue(validate_unitary_block(matrices, refs, products, factors, 0)["passed"])
        self.assertFalse(validate_unitary_block(matrices, refs, products, factors, 0.01)["passed"])
        with self.assertRaises(ValueError):
            validate_unitary_block(matrices, refs, products, factors, float("nan"))


class MaterialIntegrationTests(unittest.TestCase):
    def test_cached_real_material_matrix_replay(self):
        audit = replay(EXAMPLE / "results")
        self.assertEqual((audit["blocks_passed"], audit["blocks_total"]), (590, 642))
        self.assertEqual((audit["distinct_blocks_passed"], audit["distinct_blocks_total"]), (530, 582))
        self.assertFalse(audit["all_physical_blocks_passed"])
        self.assertTrue(audit["incomplete_4D_block_rejected"])

    def test_band_boundary_is_never_split(self):
        blocks = select_blocks([-3, -3, -1, -1, 1, 1, 3], 0.002, 3)
        self.assertEqual([list(b) for b in blocks], [[0, 1]])
        with self.assertRaisesRegex(ValueError, "No complete"):
            select_blocks([0, 0, 1], 0.002, 1)

    def test_adapter_cli_with_synthetic_reader(self):
        # Exercise reader-to-report wiring with a G=0 spin doublet at Gamma.
        # This tests the adapter, not the external binary WAVECAR parser.
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            for rel in ["probe/INCAR", "probe/POSCAR", "provenance.json", "execution_validation.json",
                        "symmetry/identified_operations.json", "symmetry/spin_basis_mapping.npz",
                        "symmetry/mapping_validation.json"]:
                target = run / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(EXAMPLE / "results" / rel, target)
            (run / "vaspwfc.py").write_text('''from pathlib import Path
import numpy as np
class vaspwfc:
    def __init__(self, path, lsorbit):
        self._nspin, self._nkpts = 1, 1
        self._kvecs = np.zeros((1, 3))
        split = Path(path).read_text() == "split"
        self._bands = np.array([[[-1., 0. if split else -1., 2.]]])
        lines = Path(path).with_name("POSCAR").read_text().splitlines()
        self._Acell = float(lines[1])*np.array([[float(v) for v in s.split()] for s in lines[2:5]])
    def gvectors(self, ikpt):
        return np.zeros((1, 3), int)
    def readBandCoeff(self, ispin, ikpt, iband, norm):
        return np.eye(2, dtype=complex)[iband-1]
''')
            command = [sys.executable, str(EXAMPLE / "analyze_wavecar.py"), "--run", str(run),
                       "--reader", str(run), "--last-band", "2", "--require-all"]
            for content, exit_code, expected_passed in [("doublet", 0, 1), ("split", 2, 0)]:
                (run / "probe/WAVECAR").write_text(content)
                proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
                self.assertEqual(proc.returncode, exit_code, proc.stdout + proc.stderr)
                report = json.loads((run / "analysis_probe/validation.json").read_text())
                self.assertEqual(report["blocks_passed"], expected_passed)
                self.assertFalse(report["uses_antiunitary_wavefunction_data"])
            incar = run / "probe/INCAR"
            incar.write_text(incar.read_text().replace("LSORBIT = .FALSE.", "LSORBIT = .TRUE."))
            proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("LSORBIT=F", proc.stderr)


if __name__ == "__main__":
    unittest.main()
