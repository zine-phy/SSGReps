# General symbolic-path validation

The general path interface combines exact symbolic parameter dependence with
validated numerical constant matrices. This audit is distinct from the cached
fully algebraic path-family checks.

## Results

- All 28 software regressions pass, including 12 tests dedicated to the new path
  input, symbolic/numerical report formatting, and domain handling.
- 16 path specifications across 10 SSGs pass the general-path comparison suite.
- 32 independent numerical point solves, including rational and irrational
  parameters, agree with the transferred path character sets. The maximum
  character-set residual is `8.8818e-15`.
- Each path also passes three seeded checks of all semilinear products at
  additional generic points. The maximum transferred Bloch-product residual in
  these checks is `7.6358e-15`.
- Symbolic constant factors agree with their numerical reference values to at
  most `2.3887e-15`.
- The cases include centered, rhombohedral, hexagonal, face-centered, and
  body-centered cells; translation indices 9, 10, 11, and 12; a generic curved
  path; quadratic and trigonometric parametrizations; and a constant path with
  8D, 8D, and 16D corepresentations.
- Regressions independently compare the general MR family with the cached exact
  MR family at three parameters. They also cover the doubled Wigner case,
  single-valued representations, endpoint rejection, and an interior enhancement
  at `sqrt(2)/2` on a quadratic path.

The spatial geometry is rationally reconstructed within the stated tolerance,
then checked exactly for lattice preservation and products modulo pure
translations. Factor constancy is checked symbolically. The underlying
numerical matrices retain their residual-based accuracy; none of these tests
turns their coefficients into certified exact algebraic numbers.

For general expressions whose exceptional sets are not fully enumerated, the
export retains exact integer conditions and marks the finite point list
incomplete. The trigonometric test explicitly rejects the enhanced point `t=pi`.
The checked evaluation methods verify the group at each requested parameter;
manual SymPy substitution does not perform that check.

## Reproduce

```bash
python -m unittest discover -s tests -p 'test_*.py'
python tests/validate_symbolic_paths.py
```

The recorded environment is Python 3.9.6 with the pinned core dependencies on
macOS arm64. Machine-readable case details and residuals are in
[symbolic_paths_general.json](symbolic_paths_general.json). The
[usage guide](../docs/symbolic_paths.md) documents input syntax and limitations;
[generated reports](../examples/general_symbolic_paths/README.md) demonstrate
both a cubic and a hexagonal family.

The suite covers selected families, not all SSG database entries or every
allowed expression. Original numerical-point formats and the WAVECAR acceptance
records are preserved. The full regression suite includes replay of the
previously measured 642 block decisions, including their unresolved cases.
