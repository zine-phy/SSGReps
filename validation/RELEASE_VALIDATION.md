# Release validation

All 129 tested configurations passed across 16 SSGs. Configuration counts include different k points, single/double-valued modes, optimization settings, and random seeds.

The numerical-solver survey below predates the material adapter. Its all-pass
statement concerns abstract group representations, not every measured VASP band
block. The subsequent [WAVECAR integration audit](WAVECAR_INTEGRATION.md) records
16 passing software regressions at that integration stage and 52 unresolved
Mn3Sn blocks. The [general symbolic-path audit](SYMBOLIC_PATHS.md) records the
current 28-test suite and 16 path specifications across 10 SSGs.

## Coverage

- Database inventory: 183498 records; maximum translation index 12.
- Tested translation indices: 1, 2, 4, 6, 8, 9, 10, 11, 12.
- Primitive, base-centered, body-centered, face-centered, rhombohedral, and hexagonal settings; collinear, coplanar, and noncoplanar records.
- Gamma, boundary points, axis/diagonal lines, generic k points, and a point only 1e-5 from Gamma.
- Six cubic high-symmetry paths: GX, GM, GR, XM, MR, XR, each at rational and irrational parameters.
- 22 comparisons against independently constructed exact character sets (four endpoints and 18 path cases).
- Both display conventions; aligned text and parsed HTML matrix values; explicit antiunitary operators.
- Seven regression tests passed in a new dependency-only environment and again against the packaged archive-only database.
- A separate path cross-check passed all 42 points (seven parameters per path) and all 12 endpoint membership comparisons.

| SSG | Translation index | Configurations | Maximum little-group order | Maximum dimension | Result |
|---|---:|---:|---:|---:|---|
| 1.2.1.3 | 2 | 9 | 2 | 2 | PASS |
| 1.6.1.3 | 6 | 9 | 6 | 2 | PASS |
| 1.4.1.5 | 4 | 7 | 4 | 2 | PASS |
| 63.1.4.67 | 1 | 7 | 8 | 2 | PASS |
| 194.1.6.1.P | 1 | 7 | 48 | 2 | PASS |
| 1.1.1.1.L | 1 | 7 | 4 | 1 | PASS |
| 221.8.6.3 | 8 | 28 | 384 | 16 | PASS |
| 194.9.4.19 | 9 | 7 | 216 | 2 | PASS |
| 194.10.4.17 | 10 | 7 | 240 | 2 | PASS |
| 173.11.1.1 | 11 | 7 | 66 | 1 | PASS |
| 184.12.2.9 | 12 | 9 | 144 | 2 | PASS |
| 5.1.2.1 | 1 | 5 | 2 | 1 | PASS |
| 76.1.2.1 | 1 | 5 | 4 | 2 | PASS |
| 146.1.3.1 | 1 | 5 | 3 | 1 | PASS |
| 225.1.12.1 | 1 | 5 | 48 | 3 | PASS |
| 229.1.12.1 | 1 | 5 | 48 | 3 | PASS |

## Residuals

Every pair of group elements was checked for each computed corepresentation: **20,398,105 projective products and 20,398,105 Bloch products**. These are full checks, not pair sampling.

| Check | Maximum absolute residual |
|---|---:|
| projective_product_max | 1.223987e-14 |
| bloch_product_max | 8.564839e-15 |
| unitarity_max | 1.491495e-14 |
| irreducibility_max | 4.884981e-15 |
| inequivalence_max | 5.121964e-16 |
| completeness_error | 1.705303e-13 |
| cocycle_max | 4.042572e-15 |
| geometry_max | 1.998401e-15 |
| display_max | 1.055737e-06 |
| exact_character_max | 8.882026e-15 |

The acceptance tolerance for mathematical checks is 1e-7. Matrix-product and unitarity residuals use the Frobenius norm. The 1e-5 display tolerance accounts for six-significant-digit formatting; exported matrices retain full numerical precision.

## Corrected defects

1. Legacy factor postprocessing violated antiunitary multiplication and unitarity. Baseline witnesses at Gamma gave errors 4.242640687 and 49.497474683 for SSG 1.2.1.3 and 1.6.1.3. The release replaces entry rescaling with unitary basis transformations and rejects invalid final matrices.
2. `--optimize False` previously evaluated as true because of boolean argument parsing. Explicit false now disables optimization.
3. A 1e-3 little-group phase tolerance incorrectly admitted extra symmetries near special points. The tightened tolerance correctly leaves a 64-element line little group at k=(0,0,1e-5) for SSG 221.8.6.3.
4. Failure exit codes, finite-coordinate checks, character consistency, export conventions, and direct archive loading are covered by regression tests.

## Installation and packaging

- Fresh Python 3.9.6 environment on macOS arm64, with only requirements.txt and its transitive dependencies.
- NumPy 1.26.4, SciPy 1.13.1, SymPy 1.14.0, spglib 2.7.0; no Pymatgen installation.
- Generic 12-fold-cell and 16D CLI examples ran successfully; JSON matrices passed internal validation before export.
- The packaged repository contains only the compressed database; no local workspace or manually extracted pickle is needed.
- English-only source comments, documentation, and reports; a SHA-256 manifest accompanies the complete release.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py'
python tests/validate_suite.py
python examples/readable_output/check_output.py
python examples/symbolic_paths/verify_paths.py --repo .
python verify_release.py
```

Run checksum verification before commands that regenerate reports. Such commands intentionally update output files.

## Scope

This is a tested release candidate, not an exhaustive proof for all database records, arbitrary k points, or every platform. The generic backend is numerical; exact symbolic construction remains limited to the documented examples. The legacy k-dot-p helper routines are outside this validation scope.

Raw case details are in extended_validation.json; baseline defects, installation metadata, test logs, and representative HTML/TXT reports are included in this directory.
