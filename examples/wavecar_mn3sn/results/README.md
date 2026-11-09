# Mn3Sn WAVECAR representation validation

The material adapter identifies SSG **194.1.6.1.P**, calculates the theoretical little-group corepresentations, and assigns wavefunction band blocks using **unitary characters only**. The tested route is noncollinear **without SOC**.

**Result: 590/642 tested band blocks pass; 52 remain UNRESOLVED under the stated thresholds.** This is a material validation example, not certification of a general-purpose WAVECAR interface. No unresolved block is silently assigned a label.

## Results

- [A-point character table, band assignments, and 4D matrices](A_point.md).
- [Probe-point report](analysis_probe/report.txt) and [machine-readable validation](analysis_probe/validation.json).
- [High-symmetry-line and generic-point report](analysis_bands/report.txt) and [machine-readable validation](analysis_bands/validation.json).
- [Band plot](Mn3Sn_bands.png), [vector PDF](Mn3Sn_bands.pdf), and [SVG](Mn3Sn_bands.svg).
- [Band energies](band_energies.csv), [probe labels](analysis_probe/band_labels.csv), and [line labels](analysis_bands/band_labels.csv).
- [Summary](summary.json), [coordinate/spin audit](symmetry/mapping_validation.json), and [incomplete-block rejection](incomplete_block_test.json).

| Point | Fractional k | Unitary order | Corep dimensions | Passed/blocks |
| --- | --- | --- | --- | --- |
| Gamma | 0.000000, 0.000000, 0.000000 | 24 | 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2 | 38/38 |
| M | 0.500000, 0.000000, 0.000000 | 8 | 1, 1, 1, 1, 1, 1, 1, 1 | 52/60 |
| K | 0.333333, 0.333333, 0.000000 | 12 | 1, 1, 1, 1, 2, 2 | 41/41 |
| A | 0.000000, 0.000000, 0.500000 | 24 | 2, 2, 4 | 19/19 |
| L | 0.500000, 0.000000, 0.500000 | 8 | 2, 2 | 24/30 |
| H | 0.333333, 0.333333, 0.500000 | 12 | 2, 2, 2 | 21/29 |
| Q | 0.137000, 0.219000, 0.317000 | 1 | 1 | 60/60 |
| path #8 | 0.233333, 0.000000, 0.000000 | 4 | 1, 1, 1, 1 | 53/59 |
| path #23 | 0.422222, 0.155556, 0.000000 | 4 | 1, 1, 1, 1 | 54/60 |
| path #38 | 0.177778, 0.177778, 0.000000 | 4 | 1, 1, 1, 1 | 54/58 |
| path #53 | 0.000000, 0.000000, 0.233333 | 12 | 1, 1, 1, 1, 2, 2 | 37/38 |
| path #68 | 0.233333, 0.000000, 0.500000 | 4 | 2, 2 | 21/30 |
| path #83 | 0.422222, 0.155556, 0.500000 | 4 | 2 | 28/30 |
| path #98 | 0.177778, 0.177778, 0.500000 | 4 | 2 | 28/30 |
| path #107 | 0.137000, 0.219000, 0.317000 | 1 | 1 | 60/60 |

The 15 evaluations include the generic point twice in independent WAVECAR files; there are 14 distinct tested k points. The full path contains 106 points on Gamma-M-K-Gamma-A-L-H-A, with a generic point appended to the calculation but excluded from the connected plot. The saved WAVECAR has 96 bands; matching uses complete candidate blocks within the first 60 bands. A block crossing that boundary is omitted in full. Labels r1, r2, etc. are local to each k point and lift convention, not band-connectivity labels.

## What was checked

1. Match database spatial operations to the material cell and origin, then solve for one common spin-basis rotation. Independently check position and magnetic-moment covariance.
2. Construct full theoretical corepresentations with SSGReps, including antiunitary operations on the theory side. Validate its full semilinear products, irreducibility, inequivalence, and completeness.
3. Read full spinor plane-wave coefficients and G vectors with the same WAVECAR reader. Check exact reciprocal-index closure for every extracted operation.
4. Group candidate bands using a 2 meV adjacent-energy gap, orthonormalize their pseudo-state Gram matrix, and apply the forward unitary Bloch action including fractional translations.
5. Match full unrounded character vectors to the restrictions of the theoretical corepresentations, accounting for norms 1, 2, or 4. Check nonnegative integer multiplicities and full-vector reconstruction.
6. Check band-subspace leakage, unitarity, and all products in the unitary subgroup. No antiunitary wavefunction matrices or traces are used.

Acceptance thresholds are absolute character error 0.002, relative subspace leakage 0.005, and matrix/product Frobenius residual 0.002. These are numerical acceptance criteria, not exact-arithmetic proofs. Theoretical matrices and complete measured unitary matrices are stored in the per-k JSON/NPZ files. Display rounding never enters matching.

Pseudo-state Gram orthonormalization supplies a consistent representation basis; it does not reconstruct all-electron overlaps or spin observables. The symmetry-equivalence argument assumes the PAW map intertwines the actual symmetry.

## Physical inputs and limits

PBE PAW Mn/Sn, ENCUT=500 eV, noncollinear spinors, LSORBIT=F, SAXIS=(0,0,1), LASPH=F, GGA_COMPAT=F, no Hubbard U, and the original crystal geometry were used. The reference SCF uses a Gamma-centered 7x7x7 mesh, EDIFF=1e-6, LMAXMIX=2, and ISMEAR=-5. Its Fermi level is 6.192000 eV. The band calculations use ICHARG=11, ISYM=-1, ISMEAR=0, SIGMA=0.05, and EDIFF=1e-10. The density and physical model are preserved between the two band-count controls.

The rounded input moment component 2.5981 was replaced by 3*sqrt(3)/2, a maximum change of 2.38e-5 mu_B, preserving the intended 120-degree texture and SSG. Both FFT meshes were audited: 42x42x36 and 84x84x72.

The selected pre-existing SCF density is independently converged and its saved real-space field covariance has a maximum magnetic relative residual of 2.12e-11. That check excludes complex PAW augmentation matrices. Some wavefunction subspaces still fail closure; saved-density covariance alone is insufficient to certify all individual band labels. Distinguishing finite numerical precision, near-band mixing, and PAW/Hamiltonian symmetry residuals requires further targeted checks; the present data do not establish a unique cause. No failed label is repaired by changing characters or relaxing thresholds.

The attempted new EDIFF=1e-10 full-SSG SCF oscillated near 1e-5 eV, was stopped, and is excluded from all final band/label claims. The accepted data use the independently converged historical density. LMAXMIX=2 and LASPH=F were retained from it; quantitative material predictions would require a separate physical convergence study. [VASP's ICHARG documentation](https://vasp.at/wiki/ICHARG) describes density reuse and recommends LMAXMIX=4 for d states.

## Reproduction

Use the [repository example guide](../README.md) for dependency setup, portable
WAVECAR analysis commands, and offline replay. The adapter scripts now import
the repository's reusable `ssg_wavecar` module. A copied nested source release
is no longer required. The original machine-specific job launchers are excluded.

Licensed POTCAR files, VASP binaries, charge densities, and large WAVECAR files
are not distributed. Input and program hashes remain in `provenance.json` and
`execution_validation.json`; the latter records the source release used for the
original calculation. Offline replay recomputes matrix/character decisions but
uses the saved coefficient-space leakage. Re-extracting coefficients requires
the matching original WAVECAR files.

Historical location fields use portable provenance identifiers instead of
absolute server paths. `Mn3Sn/...` identifies a location relative to the original
material project, `VaspBandUnfolding/vaspwfc.py` identifies the external reader,
and `TimesNewRoman/Roman.ttf` identifies the plotting font. These identifiers
describe the original calculation; they are not paths to files bundled here.
Numerical values, acceptance decisions, and input/program hashes are preserved.

## Band-count convergence control

[Measured comparison](band_count_convergence.json): increasing NBANDS from 64 to 96 improves some boundary blocks but does not eliminate all closure failures. There are 530/582 accepted blocks at 14 distinct k points, plus a separate 60/60 repeat at the generic point.
