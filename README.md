# SSGReps

**Little-group representations and corepresentations of spin-space groups.**

Release: [v0.2.0](https://github.com/zine-phy/SSGReps/releases/tag/v0.2.0).

SSGReps calculates irreducible little-group representations at numerical
wavevectors and along parameterized paths from a spin-space-group (SSG) database.
It provides representation matrices, unitary
character tables, symbolic constructions, and tools for assigning symmetry
labels to spinor wavefunction subspaces.

[Installation](#installation) · [Usage](#usage) · [Examples](#examples) ·
[Mathematical methods](#mathematical-methods) · [Documentation](docs/usage.md) ·
[References](#references)

## Features

| Feature | Description |
| --- | --- |
| Little-group representations | Single- and double-valued representations, including corepresentations of groups with antiunitary operations |
| Character tables | Dimensions and unitary characters of irreducible corepresentations, with optional grouping of identical columns |
| Symbolic paths | One-parameter path expressions with symbolic Bloch phases and validated numerical constant matrices |
| Exact constructions | Fully algebraic matrices for the supplied exact families |
| Basis optimization | Optional unitary basis changes that reduce matrix density |
| Wavefunction analysis | Spinor coefficient transformations and decomposition of complete band-block characters into irreducible components and multiplicities |
| Reports and exports | Aligned text, offline HTML, and full-precision JSON/NPY output |
| Consistency checks | Matrix products, unitarity, character consistency, irreducibility, inequivalence, and completeness |

The general solver accepts numerical k points or symbolic path expressions.
General paths retain exact symbolic phases with numerical constant matrices;
fully algebraic constructions are provided separately for selected families. WAVECAR
analysis uses a separate material adapter with explicit cell, origin, and spin
conventions; it is not an automatic interface for arbitrary materials.

## Installation

Run from the repository root:

```bash
git clone https://github.com/zine-phy/SSGReps.git
cd SSGReps
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.
A source archive can be extracted and used with the same commands after `cd`.

The core dependencies are NumPy, SciPy, SymPy, and spglib, pinned in
[requirements.txt](requirements.txt). The pinned set targets Python 3.9–3.12;
installation and regression tests were run on Python 3.9.6/macOS arm64.
A GitHub Actions template is included for Linux with Python 3.9–3.12; it is not
enabled in this release, and those CI environments have not been validated.
The bundled database is read directly from `ssg_data/identify.pkl.tar.gz`.

GAP is required only to rebuild the exact datasets. The cached exact examples
run with the core Python dependencies. The WAVECAR example uses the optional
MOM2SSG and VaspBandUnfolding tools; plotting additionally requires Matplotlib.
See the [wavefunction example guide](examples/wavecar_mn3sn/README.md) for setup.

## Usage

### Character tables

Specify an SSG identifier and a k point:

```bash
python ssgreps.py --ssgNum 63.1.4.67 --kp 0 0 0.5 --out character
```

Rows label irreducible representations or corepresentations. Columns contain
**unitary operations only**. Antiunitary operations enter the theoretical
construction and validation; traces of their matrix parts are not characters.

### Representation matrices

Select a representation and operations by their one-based local indices:

```bash
python ssgreps.py --ssgNum 221.8.6.3 --kp 0.5 0.5 0.5 \
  --out rep_matrix --rep 3 --ops 2 193 --optimize --seed 0 \
  --text report.txt --html report.html
```

This example selects a 16-dimensional corepresentation. Open `report.html` in a
browser for the formatted matrices. `--optimize` searches for a sparser basis;
it does not guarantee maximal sparsity or simultaneous diagonalization of
noncommuting matrices.

### Symbolic path input

Use `--kpath` instead of `--kp` to retain a real path parameter:

```bash
python ssgreps.py --ssgNum 194.1.6.1.P \
  --kpath "1/3" "1/3" "t" --interval 0 1/2 \
  --out both --rep 1 --ops 4 --seed 19 \
  --text path.txt --html path.html --fileType json --export path.json
```

The path need not be a predefined named line. Expressions can contain rational
arithmetic, powers, `sqrt`, `sin`, `cos`, and `exp`; `--parameter` selects a name
other than `t`. A seeded sample supplies the numerical basis, while symbolic
integer conditions certify the generic little group and identify where extra
symmetries can occur. Endpoints and enhanced-symmetry points require separate
point calculations.

The result has the form `D_g(t)=exp(-2*pi*i*q(t).tau_g)*M_g`, with symbolic phases
and numerical constant matrices. The default open interval is `(0,1)`.
See the [general symbolic-path guide](docs/symbolic_paths.md) for nonlinear
paths, exceptional-point conditions, Python evaluation, and export formats.

### Numerical exports

```bash
python ssgreps.py --ssgNum 184.12.2.9 --kp 0.137 0.219 0.317 \
  --out character --fileType json --export result.json
```

Use `--fileType npy` for an NPY export. JSON complex numbers are encoded as
`{"real": ..., "imag": ...}`. Exported values retain full numerical precision;
rounding and special-number formatting apply only to the display.

| Option | Meaning |
| --- | --- |
| `--ssgNum ID` | SSG database identifier |
| `--kp kx ky kz` | Numerical k coordinates in the pure-translation reciprocal basis |
| `--kpath EXPR EXPR EXPR` | Symbolic path coordinates; alternative to `--kp` |
| `--parameter NAME`, `--interval A B` | Real path parameter and open interval; defaults `t`, `(0,1)` |
| `--groupType 1/2` | Single-valued / double-valued; default `2` |
| `--out TYPE` | `character`, `rep_matrix`, `rep_degree`, or `both` |
| `--rep N`, `--ops ...` | Matrix selections using one-based local indices |
| `--convention bloch/projective` | Display convention; default `bloch` |
| `--optimize [true/false]` | Optional basis optimization; default `false` |
| `--seed N` | Reproducible solver seed; default `0` |
| `--compact-characters` | Group exactly equal character columns and list their members |
| `--text PATH`, `--html PATH` | Readable report destinations |
| `--fileType json/npy`, `--export PATH` | Data export format and destination |

Run `python ssgreps.py --help` for all options. See the
[usage guide](docs/usage.md#command-line-usage) and
[Python interfaces](docs/usage.md#python-interfaces-and-exports) for more examples.

## Conventions

- **Coordinates:** k points use the reciprocal basis of the pure-translation
  lattice, which can differ from the parent conventional cell. Reciprocal
  vectors include the factor `2*pi`.
- **Projective matrices:** JSON/NPY `repMatrix` stores `M(g)`. Its `character`
  field contains unitary characters in the same convention. `linearCharacter`
  stores the Bloch-phase-weighted unitary characters in numerical-point exports;
  path exports use `symbolicBlochCharacters` and retain symbolic phases separately.
- **Bloch matrices:** the default display is
  $D(g)=e^{-i\mathbf{k}_{\mathrm{cart}}\cdot\boldsymbol\tau_g}M(g)$.
  Translation defects and spin-lift signs can leave a nontrivial factor system
  on the finite quotient even after restoring this phase.
- **Antiunitary action:** an antiunitary operation acts as `D(g)K`, where `K` is
  complex conjugation. Products and basis changes include the corresponding
  conjugation.
- **Labels:** representation labels and operation indices are local to each
  calculation and convention. They do not specify band connectivity between
  different k points.

Exact arithmetic and projectivity are independent: an exact matrix can belong
to a projective representation. Full definitions are in the
[conventions guide](docs/usage.md#coordinates-phases-and-antiunitary-operations).

## Examples

### Exact 16-dimensional corepresentation

For SSG 221.8.6.3 at `k=(1/2,1/2,1/2)`, generate an exact report with:

```bash
python examples/readable_output/demo.py --rep r3 --ops 2 193 \
  --convention bloch --outdir exact_report
```

The finite little group has 384 operations, of which 192 are unitary. Its compact
Bloch character table is:

```text
  rep | dim | g001 | g002 | g005 | g008 | g011 | g016
------+-----+------+------+------+------+------+-----
count |   - |    1 |  147 |    5 |    7 |   12 |   20
   r1 |   8 |    8 |    0 |  -4i |   4i |    2 |   -2
   r2 |   8 |    8 |    0 |   4i |  -4i |    2 |   -2
   r3 |  16 |   16 |    0 |    0 |    0 |   -2 |    2
```

`count` gives the number of operations with identical columns, not conjugacy-class
sizes. For the 16D branch, one antiunitary operation has matrix part

$$
D(g_{193})=i\begin{pmatrix}0&I_8\\I_8&0\end{pmatrix},
\qquad \widehat D(g_{193})=D(g_{193})K.
$$

The full corepresentation is irreducible; antiunitary operations exchange the
two 8D unitary blocks. The [complete report](examples/readable_output/representation_report.txt)
contains the aligned matrices. These k coordinates refer to the pure-translation
lattice `2I`, and correspond to `(1/4,1/4,1/4)` in the parent conventional
reciprocal coordinates.

### Fully algebraic high-symmetry paths

For general user-defined paths, use `--kpath` as described above. The separate
cached symbolic-path module provides exact matrices on six open paths of SSG
221.8.6.3: GX, GM, GR, XM, MR, and XR. For example, on MR with
`k(t)=(1/2,1/2,t)` and `0<t<1/2`, one generator is

$$
D_t(g_4)=\frac{e^{-i\pi t}}{\sqrt{2}}
\begin{pmatrix}
1&-1&0&0\\-1&-1&0&0\\0&0&1&-1\\0&0&-1&-1
\end{pmatrix}.
$$

The parameter dependence is explicit. Endpoints have separate little groups
and datasets. See the [Python example](docs/usage.md#obtain-an-exact-symbolic-path-expression)
and [analytic generators](examples/symbolic_paths/analytic_generators.txt).

### WAVECAR band labels

The Mn3Sn example analyzes noncollinear wavefunctions without SOC for SSG
194.1.6.1.P. It aligns the spatial and spin conventions, extracts unitary
representation matrices, and assigns complete band blocks by their characters.
A reducible block is reported as a direct sum with multiplicities, such as
`r1 + r3` or `2*r1 + r2`. Explicit block diagonalization of the measured
wavefunction matrices is outside the current interface.

At A=`(0,0,1/2)`, all 19 tested blocks pass the acceptance checks. Bands 53–56 form
a 4D `r3` at mean `E-EF=+0.21084044 eV`, with character residual `6.43e-9` and
relative leakage `6.62e-4`. The
[A-point report](examples/wavecar_mn3sn/results/A_point.md) contains its character
table and measured matrices.

Across 14 distinct tested k points, **530/582 blocks pass; 52 are `UNRESOLVED`**.
The general-point repeat gives 590/642 over 15 evaluations. Reports retain failed
checks. The [example guide](examples/wavecar_mn3sn/README.md) documents the
material-specific scope, PAW assumptions, input files, thresholds, band plots,
and real-WAVECAR commands. IRSSG [3] provides related SSG identification and
band-symmetry workflows.

Replay the saved matrix checks without VASP or a WAVECAR reader:

```bash
python examples/wavecar_mn3sn/replay_validation.py
```

Replay recomputes characters, products, unitarity, and label decisions. Subspace
leakage is read from the recorded coefficient analysis; recomputing it requires
the original WAVECAR files.

## Mathematical methods

The numerical solver decomposes the twisted regular representation using a
Hamiltonian/group-averaging method [1]. Wigner's corepresentation construction
accounts for antiunitary pairing and doubling. Basis optimization uses unitary
transformations compatible with the semilinear action.

The exact modules use finite central extensions, cyclotomic arithmetic in GAP
[6], induction from abelian subgroups, character projectors, and exact basis
construction. Relevant projective and corepresentation theory is discussed in
[4,5]. The supplied exact constructions support sign-valued factor systems and
Wigner cases `+1` and `0`. Along supported paths with constant little group and
factor system, fixed exact matrices combine with symbolic Bloch phases.
SymPy [7] handles the symbolic expressions; recognition of special numbers from
floating-point data is reported as a numerical candidate, not an exact proof.

The general path solver reconstructs rational spatial geometry and checks
symbolic reciprocal-lattice conditions for the generic little group. These
conditions establish that the projective factor system is constant along the
path in the chosen convention. A numerical solve at a seeded generic point
then supplies constant matrices, multiplied by symbolic Bloch phases. Additional
samples validate the transferred matrices, and exact integer conditions record
where extra little-group operations can appear.

Wavefunction analysis uses the forward spinor symmetry action and a common
Gram-orthonormalized pseudo-state basis. Matching uses only unitary characters,
with the actual restriction norms of full irreducible corepresentations.
Nonnegative integer multiplicities, full-character reconstruction, subspace
leakage, unitarity, and unitary products determine acceptance. The construction
of the theoretical references includes antiunitary operations.

See the [mathematical guide](docs/usage.md#mathematical-methods) and
[wavefunction derivation](docs/usage.md#comparison-with-vasp-wavefunctions).

## Validation

```bash
python -m unittest discover -s tests -p 'test_*.py'
python verify_release.py
```

The regression suite contains 28 tests covering numerical representations,
exports, coefficient conventions, Wigner norms, reducible character matching,
and cached material results. The extended numerical survey covers 129
configurations across 16 SSGs, including translation indices up to 12, and checks
20,398,105 products in each of the projective and Bloch conventions. Symbolic-path
validation covers 42 sampled points on six exact families. General-path checks
include centered and nonorthogonal lattices, high translation indices, nonlinear
expressions, and enhanced-symmetry parameters; see the
[general-path validation report](validation/SYMBOLIC_PATHS.md).

These checks define the tested scope; they are not exhaustive over the database.
The [CI template guide](docs/ci/README.md) explains how to enable the included
workflow, which installs pinned dependencies, checks the release manifest, and
runs all software regressions.
See the [numerical validation report](validation/RELEASE_VALIDATION.md) and
[WAVECAR validation report](validation/WAVECAR_INTEGRATION.md). Release archives
include a SHA-256 file manifest. Verify it before regenerating bundled outputs.

## Documentation

- [Full usage guide, conventions, mathematics, and Python API](docs/usage.md)
- [General symbolic paths and parameter-dependent output](docs/symbolic_paths.md)
- [Exact construction and GAP reconstruction](examples/exact_16d/README.txt)
- [WAVECAR analysis and material example](examples/wavecar_mn3sn/README.md)
- [Release changes](CHANGELOG.md)
- [BibTeX references](references.bib)

## References

Cite [1] for the representation method and [2] for the SSG database. Reference [3]
describes IRSSG and related electronic-band symmetry analysis. For exact and
symbolic calculations, cite the relevant theoretical background [4,5] and
software [6,7]; cite [8] when using spglib-based utilities. Record the code revision
used in your calculation. BibTeX entries are provided in [references.bib](references.bib).

1. Ziyin Song, A. Z. Yang, Yi Jiang, Zhong Fang, Jian Yang, Chen Fang,
   Hongming Weng, and Zheng-Xin Liu,
   **Constructions and applications of irreducible representations
   of spin-space groups**, *Physical Review B* **111**, 134407 (2025).
   [DOI](https://doi.org/10.1103/PhysRevB.111.134407) · [arXiv](https://arxiv.org/abs/2409.13601)
2. Yi Jiang, Ziyin Song, Tiannian Zhu, Zhong Fang, Hongming Weng,
   Zheng-Xin Liu, Jian Yang, and Chen Fang,
   **Enumeration of Spin-Space Groups: Toward a Complete Description
   of Symmetries of Magnetic Orders**, *Physical Review X* **14**, 031039 (2024).
   [DOI](https://doi.org/10.1103/PhysRevX.14.031039) · [arXiv](https://arxiv.org/abs/2307.10371)
3. Sheng Zhang, Ziyin Song, Zhong Fang, Hongming Weng, and Zhijun Wang,
   **IRSSG: An Open-Source Software Package for Spin Space Groups**,
   *Computer Physics Communications* **326**, 110190 (2026).
   [DOI](https://doi.org/10.1016/j.cpc.2026.110190) · [arXiv](https://arxiv.org/abs/2511.21821)
4. Zheng Zhang, Peiyuan Wang, and Y. X. Zhao, **Projective representation theory of
   spin space groups**, arXiv:2609.21289 (2026), v1.
   [arXiv](https://arxiv.org/abs/2609.21289v1)
5. Zeying Zhang, Gui-Bin Liu, Mu Tian, Run-Wu Zhang, Zhi-Ming Yu, and Yugui Yao,
   **Spin layer groups and their corepresentations**,
   arXiv:2605.25484 (2026), v2.
   [arXiv](https://arxiv.org/abs/2605.25484v2)
6. The GAP Group, **GAP — Groups, Algorithms, and Programming**, version 4.16.1.
   [Reference manual](https://gap-system.github.io/gap/doc/ref/chap71_mj.html)
7. Aaron Meurer, Christopher P. Smith, Mateusz Paprocki, Ondřej Čertík,
   Sergey B. Kirpichev, Matthew Rocklin, Amit Kumar, Sergiu Ivanov, Jason K. Moore,
   Sartaj Singh, Thilina Rathnayake, Sean Vig, Brian E. Granger, Richard P. Muller,
   Francesco Bonazzi, Harsh Gupta, Shivam Vats, Fredrik Johansson, Fabian Pedregosa,
   Matthew J. Curry, Andy R. Terrel, Štěpán Roučka, Ashutosh Saboo, Isuru Fernando,
   Sumith Kulal, Robert Cimrman, and Anthony Scopatz,
   **SymPy: symbolic computing in Python**,
   *PeerJ Computer Science* **3**, e103 (2017).
   [DOI](https://doi.org/10.7717/peerj-cs.103)
8. Atsushi Togo, Kohei Shinohara, and Isao Tanaka, **Spglib: a software library for
   crystal symmetry search**, *Science and Technology of Advanced Materials:
   Methods* **4**, 2384822 (2024).
   [DOI](https://doi.org/10.1080/27660400.2024.2384822)

## License

[MIT](LICENSE).
