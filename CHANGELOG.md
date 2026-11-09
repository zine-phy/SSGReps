# Release changes

## v0.2.0 — 2026-11-09

This release adds general symbolic-path input, validated numerical
corepresentations, exact examples, readable reports, and the documented
WAVECAR workflow. The sections below describe its capabilities and scope.
It also includes a Linux regression CI template for Python 3.9–3.12, full reference author
lists, and portable provenance identifiers in the bundled material results.
The CI template is not enabled in this release; local validation remains on
Python 3.9.6/macOS arm64.

## General symbolic path input

- Added `--kpath` / `--kp-symbolic`, a real parameter name, and an open interval
  as alternatives to numerical `--kp` input.
- Added `ssg_symbolic.SymbolicPath`: seeded generic sampling, rational spatial
  geometry checks, symbolic membership and factor-system constancy, numerical
  constant corepresentations, and explicit parameter-dependent Bloch phases.
- Reports and exports retain symbols, unitary-only characters, operation-index
  mappings, enhanced-point exclusions, and separate endpoint metadata. Checked
  evaluation rejects enhanced points and endpoints. Non-enumerated exceptional
  sets remain explicit integer conditions.
- Added general-path regressions and independent point comparisons covering
  nonlinear expressions, nonorthogonal cells, high translation indices, and
  antiunitary representations. Numerical constants are not labelled exact.

## WAVECAR interface and Mn3Sn validation

- Added `SSGReps/ssg_wavecar.py`: forward spinor plane-wave action, common Gram
  orthonormalization, unitary-character decomposition with Wigner restriction
  norms, and leakage/unitarity/product checks. The binary reader is optional
  and external. Antiunitary wavefunction traces are never matching data.
- Integrated portable Mn3Sn identification/spin-frame alignment, WAVECAR
  analysis, and plotting examples. The tested route is NSOC SSG 194.1.6.1.P with
  identity cell/origin mapping; unsupported assumptions are rejected.
- Bundled measured matrices, theory, labels, plots, inputs, convergence evidence,
  and controls. At 14 distinct k points 530/582 blocks pass; 52 remain explicitly
  unresolved. An independent general-point repeat gives 590/642 total. All 19
  tested A-point blocks pass, including the 4D bands 53-56 example.
- Added offline replay of all recorded matrix decisions and regression tests
  for coefficient conventions, Wigner norms, incomplete multiplets, and adapter
  execution. A replay pass reproduces both accepted and unresolved cases.
- Replaced obsolete documentation saying no real WAVECAR had been tested.
  Documented PAW assumptions, custom-VASP provenance, the excluded unconverged
  SCF trial, and the limits of the available material validation.

## Correctness

- Reproduced invalid final antiunitary matrices in the legacy `solve_factor`
  postprocessing: SSG 1.2.1.3 at Gamma had Frobenius errors about 4.24; SSG
  1.6.1.3 had errors about 49.50. Baseline witnesses are recorded in validation.
- Replaced independent entry rescaling with unitary basis changes. The legacy
  `solve_factor` method remains as an identity compatibility wrapper.
- Retained the group-averaging reduction method, using the monomial regular
  action directly, without allocating all dense regular matrices in the main solve path.
  Added deterministic local seeds and bounded retries with
  explicit failure. Character ordering no longer overwrites tied sort slots.
- Recompute characters from final matrices. Validate all projective and Bloch
  products, unitarity, irreducibility, inequivalence, and completeness before output.
- Operation lookup uses rounded keys only as an accelerator, with geometric and
  SU(2) checks. The little-group phase tolerance is tightened from 1e-3 to 1e-8,
  preventing nearby generic points from being treated as high-symmetry points.

## Output and execution

- English, aligned text and offline HTML reports; common factors and radicals
  retained for exact input; explicit antiunitary K and Bloch conventions.
- Matrix selection, optional character-column compression, and report paths.
- Correct boolean parsing for `--optimize False`; nonzero error exits; finite
  coordinate checks; reproducible seeds; explicit JSON/NPY export paths.
- Read the bundled database directly from its archive. Removed unused imports
  of Pymatgen, Turtle/Tk, and tqdm from the representation entry point.
- Include pinned dependencies, a repository-root CLI, example backends,
  regression tests, validation records, and checksums.

## Compatibility

The numerical matrices can differ from earlier output by a valid basis choice;
local labels and ordering can differ as well. Compare characters or equivalence,
not entrywise equality across unrelated bases. Existing JSON/NPY fields are
retained and augmented with conventions, lattice bases, and validation metadata.
The exact example matrices and their certified mathematical content are unchanged.
