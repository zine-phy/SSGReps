# Usage and mathematical methods

[Back to the repository overview](../README.md)

**Validated numerical representations, exact example constructions, and readable output for spin-space groups.**

SSGReps computes little-group representations and corepresentations at a specified
momentum. This release retains the original SSG database and adds corrected numerical
postprocessing, automatic mathematical validation, exact example backends, symbolic
high-symmetry paths, and aligned English text/HTML reports.

| Workflow | Input | Result | Current scope |
|---|---|---|---|
| Numerical CLI | SSG identifier and numerical k point | All computed irreducible little-group corepresentations | Bundled database; final matrices validated on every run |
| Exact 16D example | Cached data or GAP reconstruction | Exact cyclotomic matrices | SSG 221.8.6.3 at the documented boundary point |
| General symbolic paths | SSG identifier, coordinate expressions, parameter interval | Symbolic Bloch phases with numerical constant matrices | Database SSGs with certified generic geometry and explicit exceptional conditions |
| Exact path families | Path name and symbolic parameter t | Fully algebraic matrices with explicit Bloch phases | Six paths and four endpoints of SSG 221.8.6.3 |
| Symbolic formatting | Exact expressions or floating-point samples | Radicals/exponentials, or provisional numerical candidates | Exact rewriting and numerical inference are separate interfaces |

**The generic CLI supports symbolic parameter dependence with numerical matrix
coefficients.** Choosing a different SSG does not automatically invoke a fully
algebraic solver. See the [general symbolic-path guide](symbolic_paths.md). The exact examples demonstrate an additional construction
that has not yet been generalized to the complete database.

## Contents

- [Installation and quick start](#installation-and-quick-start)
- [Command-line usage](#command-line-usage)
- [Output examples](#output-examples)
- [Coordinates, phases, and antiunitary operations](#coordinates-phases-and-antiunitary-operations)
- [Comparison with VASP wavefunctions](#comparison-with-vasp-wavefunctions)
- [Mathematical methods](#mathematical-methods)
- [Python interfaces and exports](#python-interfaces-and-exports)
- [Exact reconstruction and validation](#exact-reconstruction-and-validation)
- [Changes from the original implementation](#changes-from-the-original-implementation)
- [Repository layout and limitations](#repository-layout-and-limitations)
- [References and citation guidance](#references-and-citation-guidance)

## Installation and quick start

Run all commands below from the repository root, including when reading this guide from `docs/`.
The pinned dependencies target Python 3.9–3.12; the release was actually tested with
Python 3.9.6 on macOS arm64. Other Python versions and platforms were not part of
that installation test.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_release.py
python ssgreps.py --ssgNum 1.2.1.3 --kp 0 0 0 --out both
```

For Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
Run checksum verification before regenerating bundled reports: regeneration can
intentionally change the files listed in the release manifest.

Dependencies: NumPy 1.26.4, SciPy 1.13.1, SymPy 1.14.0, and spglib 2.7.0.
The numerical CLI does not require Pymatgen, Tk/Turtle, a browser, or GAP.
GAP is needed only to reconstruct the exact examples; reading their cached results
requires only the Python dependencies. GAP 4.16.1 was used for the exact construction.

The bundled database is read directly from `ssg_data/identify.pkl.tar.gz`, without
manual extraction. If an unpacked `ssg_data/identify.pkl` exists, it takes precedence.
Use the bundled, trusted database: pickle files are not an untrusted input format.

## Command-line usage

### Compute a character table

```bash
python ssgreps.py --ssgNum 63.1.4.67 --kp 0 0 0.5 --out character --groupType 2
```

### Select a high-dimensional numerical representation

```bash
python ssgreps.py --ssgNum 221.8.6.3 --kp 0.5 0.5 0.5 --out both --rep 3 --ops 2 193 --optimize --seed 0 --html report.html --text report.txt
```

For this case, the numerical dimensions are 8, 8, and 16. The numerical basis need
not match the sparse exact basis shown below. Representation labels are local to
each calculation; they are not standard crystallographic labels.

### Compute a general point in a group with translation index 12

```bash
python ssgreps.py --ssgNum 184.12.2.9 --kp 0.137 0.219 0.317 --out character --fileType json --export result.json
```

### Evaluate a point on a high-symmetry line

```bash
python ssgreps.py --ssgNum 221.8.6.3 --kp 0.5 0.5 0.25 --groupType 2 --out character
```

This evaluates a single point on the MR path. For an expression containing a
symbolic parameter `t`, use `PathFamily` below; `--kp` accepts numerical coordinates.

### Options

| Option | Meaning / default |
|---|---|
| `--ssgNum ID` | Required database identifier |
| `--kp kx ky kz` | Numerical coordinates; mutually exclusive with `--kpath` |
| `--kpath EXPR EXPR EXPR` | Symbolic path coordinates in the pure-translation reciprocal basis |
| `--parameter NAME` | Real path parameter; default `t` |
| `--interval A B` | Open path interval; default `(0,1)` |
| `--groupType 1/2` | Single-valued / double-valued; default `2` |
| `--out TYPE` | `character`, `rep_matrix`, `rep_degree`, or `both`; default `character` |
| `--rep N` | One-based representation index; filters matrix output |
| `--ops ...` | One-based operation indices within the current little group; filters matrices |
| `--optimize [true/false]` | Safe unitary basis optimization; default `false`; bare flag means `true` |
| `--seed N` | Local random seed for reproducible reduction; default `0` |
| `--convention bloch/projective` | Display convention for both matrices and characters; default `bloch` |
| `--compact-characters` | Merge exactly equal character vectors, with member indices retained |
| `--html PATH` | Write an offline HTML report |
| `--text PATH` | Also save the aligned stdout report |
| `--fileType json/npy` | Export full-precision projective matrices and metadata |
| `--export PATH` | Export destination; default `output.json` or `output.npy` |

Without `--rep`, HTML selects the largest-dimensional representation. Without
`--ops`, it shows up to four initial operations and the first antiunitary operation.
Text matrix output includes every representation/operation unless filtered.
Character tables retain all representations; matrix-selection flags do not filter them.

Reports use six significant digits for numerical data; real or imaginary components
below `1e-10` are displayed as zero. These display rules do not modify the matrices.
Wide text matrices use column panels with global row/column numbers. HTML tables
scroll horizontally when necessary. Exact inputs retain exact symbolic expressions.

`--compact-characters` groups identical complete character vectors, **not conjugacy
classes**. It compares underlying values rather than rounded strings, so numerical
roundoff can leave visually identical columns separate. Solver diagnostics go to
stderr; stdout contains the report. Invalid input or failed validation produces a
nonzero exit status. Both `--optimize False` and `--optimize false` disable optimization.

## Output examples

### Exact character table and a 16D matrix

Generate these outputs from the bundled exact data:

```bash
python examples/readable_output/demo.py --rep r3 --ops 2 193 --convention bloch --outdir exact_report
```

This writes `exact_report/representation_report.txt` and
`exact_report/representation_report.html`. The following excerpts come from the
actual [bundled report](../examples/readable_output/representation_report.txt).

SSG 221.8.6.3 at `k=(1/2,1/2,1/2)` has 384 operations in the finite little group,
192 of them unitary. Its irreducible corepresentation dimensions are 8, 8, and 16.
The compact Bloch character table is:

```text
  rep | dim | g001 | g002 | g005 | g008 | g011 | g016
------+-----+------+------+------+------+------+-----
count |   - |    1 |  147 |    5 |    7 |   12 |   20
   r1 |   8 |    8 |    0 |  -4i |   4i |    2 |   -2
   r2 |   8 |    8 |    0 |   4i |  -4i |    2 |   -2
   r3 |  16 |   16 |    0 |    0 |    0 |   -2 |    2
```

`count` is the number of operations sharing that character vector. The report
lists the complete membership of every compressed column.
For the 16D branch, `r=sqrt(3)` and the common factor is `i/4`:

```text
D(g002) = i/4 * matrix
      1  2  3  4  5  6  7  8 |  9 10 11 12 13 14 15 16
 1 | -1  r  r  1  1  1  r  r |  0  0  0  0  0  0  0  0 |
 2 |  r  1  1 -r  r  r -1 -1 |  0  0  0  0  0  0  0  0 |
 3 |  r  1  1 -r -r -r  1  1 |  0  0  0  0  0  0  0  0 |
 4 |  1 -r -r -1  1  1  r  r |  0  0  0  0  0  0  0  0 |
 5 |  1  r -r  1  1 -1  r -r |  0  0  0  0  0  0  0  0 |
 6 |  1  r -r  1 -1  1 -r  r |  0  0  0  0  0  0  0  0 |
 7 |  r -1  1  r  r -r -1  1 |  0  0  0  0  0  0  0  0 |
 8 |  r -1  1  r -r  r  1 -1 |  0  0  0  0  0  0  0  0 |
   | ------------------------------------------------- |
 9 |  0  0  0  0  0  0  0  0 | -1  r  r  1  1  1  r  r |
10 |  0  0  0  0  0  0  0  0 |  r  1  1 -r  r  r -1 -1 |
11 |  0  0  0  0  0  0  0  0 |  r  1  1 -r -r -r  1  1 |
12 |  0  0  0  0  0  0  0  0 |  1 -r -r -1  1  1  r  r |
13 |  0  0  0  0  0  0  0  0 |  1  r -r  1  1 -1  r -r |
14 |  0  0  0  0  0  0  0  0 |  1  r -r  1 -1  1 -r  r |
15 |  0  0  0  0  0  0  0  0 |  r -1  1  r  r -r -1  1 |
16 |  0  0  0  0  0  0  0  0 |  r -1  1  r -r  r  1 -1 |
  r = sqrt(3)
```

Equivalently, `D(g002)=(i/4) diag(B,B)`, with `B` the displayed 8-by-8 block.
An antiunitary operation exchanges the blocks:

$$
D(g_{193})=i\begin{pmatrix}0&I_8\\I_8&0\end{pmatrix},
\qquad \widehat D(g_{193})=D(g_{193})K.
$$

Thus a block-diagonal unitary matrix does not imply that the full corepresentation
is reducible. The factor `i` is the Bloch phase in this example; in the projective
convention the corresponding matrix part is the block-exchange matrix without it.

### A parameter-dependent 4D matrix

On MR, `k(t)=(1/2,1/2,t)` with `0<t<1/2`, branch `r1` contains the database
operation `g4=[C2y_spin || I_space | (1,0,1)]`:

$$
D_t(g_4)=\frac{e^{-i\pi t}}{\sqrt{2}}
\begin{pmatrix}
1&-1&0&0\\-1&-1&0&0\\0&0&1&-1\\0&0&-1&-1
\end{pmatrix}.
$$

Its square is `exp(-2*i*pi*t)*I4`; its eigenvalues are `+exp(-i*pi*t)` and
`-exp(-i*pi*t)`, each twice. This is one generator of the complete irreducible
family. All generators are in [analytic_generators.txt](../examples/symbolic_paths/analytic_generators.txt).

The available paths and full irreducible corepresentation dimensions are:

| Path | k(t), 0 < t < 1/2 | Little-group order | Dimensions |
|---|---|---:|---|
| GX | (t,0,0) | 64 | 2, 2, 2, 2, 4 |
| GM | (t,t,0) | 32 | 2, 2, 2, 2 |
| GR | (t,t,t) | 48 | 2, 2, 4 |
| XM | (1/2,t,0) | 32 | 4 |
| MR | (1/2,1/2,t) | 64 | 4, 4 |
| XR | (1/2,t,t) | 32 | 4 |

Endpoint dimensions: Gamma `2,2,2,2,4,4,6,6,6,6`; X `4,4,8`; M `8`; R `8,8,16`.
Endpoint substitution in a path matrix gives a representation of the **path
subgroup**. Additional endpoint operations require the separate endpoint matrices.

## Coordinates, phases, and antiunitary operations

The input `k` is expressed in the reciprocal basis of the **pure-translation
lattice**, including the factor `2*pi` in its reciprocal vectors. It is generally
different from the parent space group's conventional reciprocal coordinates.
For SSG 221.8.6.3, the pure-translation basis is `2I`, so input
`(1/2,1/2,1/2)` corresponds to parent conventional coordinates `(1/4,1/4,1/4)`.
The labels Gamma/X/M/R in the examples refer to the pure-translation Brillouin zone.

The translation index is `abs(det(superCell))` relative to the parent primitive
translation lattice. It is not generally the conventional-cell volume.

Let `s_g=+1` for unitary elements and `s_g=-1` for antiunitary elements, and define
`X^[s_g]=X` or `conjugate(X)`, respectively. This notation never means a matrix inverse.
The multiplication index `gh` below denotes the chosen representative of the
quotient-group product. The projective co-group matrices obey

$$
M(g)M(h)^{[s_g]}=\omega(g,h)M(gh).
$$

For a chosen coset section, the Bloch small-representation matrix is

$$
D(g)=e^{-i\mathbf{k}_{\rm cart}\cdot\boldsymbol\tau_g}M(g).
$$

An antiunitary symmetry acts as `D(g)K`, where `K` denotes complex conjugation.
With `L(g,h)=tau_g+R_g*tau_h-tau_gh` a pure translation, its product law is

$$
D(g)D(h)^{[s_g]}=
\sigma_{\rm spin}(g,h)e^{-i\mathbf{k}_{\rm cart}\cdot\mathbf L(g,h)}D(gh).
$$

The spin-lift factor is omitted (`sigma_spin=1`) for single-valued representations.
The code checks both laws. Restoring the Bloch phase does not generally make
the finite operation table an ordinary linear representation: translation defects
and spin-lift signs can remain. Including the full translation group and the
spin double cover instead gives a full-group corepresentation, with linear
unitary operations and antilinear antiunitary operations. Exact arithmetic does
not change this distinction.

Characters are traces of unitary operations only;
traces of antiunitary matrix parts are not treated as ordinary characters.

## Comparison with VASP wavefunctions

The main CLI supplies theoretical representations and corepresentations. The
[Mn3Sn adapter](../examples/wavecar_mn3sn/README.md) additionally reads actual
spinor `WAVECAR` files and identifies complete band blocks for SSG 194.1.6.1.P
without SOC, using an audited identity cell/origin mapping. Its reusable
[NumPy kernel](../SSGReps/ssg_wavecar.py) is independent of the binary reader.
This material example does not implement automatic mappings for arbitrary
structures, SOC calculations, or band connectivity.

For an orthonormal Bloch-state subspace closed under the little group, its physical
symmetry matrices are

$$
B_{mn}(g)=\langle\psi_{m\mathbf k}|\widehat g\,\psi_{n\mathbf k}\rangle.
$$

For a unitary operation, the action on a spinor wavefunction is

$$
(\widehat g\psi)(\mathbf r)=
V_g\,\psi\!\left(R_g^{-1}(\mathbf r-\boldsymbol\tau_g)\right),
$$

where `V_g` is its spin matrix in the wavefunction's spin basis. Little-group
operations preserve k modulo a reciprocal vector; when using cell-periodic
functions instead of full Bloch functions, the reciprocal-vector gauge must also
be included.

**Label assignment uses only unitary characters.** Let `H` be the unitary
subgroup of the specified little group and let `chi_alpha` be the character of
the restriction of a full irreducible corepresentation to `H`. After aligning
the operation representatives, factor system, and phase convention, decompose
the character of a complete band block using

$$
m_\alpha=
\frac{\sum_{h\in H}\chi_\alpha(h)^*\,\operatorname{tr}B(h)}
     {\sum_{h\in H}|\chi_\alpha(h)|^2}.
$$

The result must be a nonnegative integer and must reconstruct the entire
unitary-character vector. The denominator cannot always be replaced by `|H|`:

| Wigner indicator | Restriction to `H` | Squared character norm on `H` |
| --- | --- | --- |
| `+1` | `Delta` | `1` |
| `-1` | `Delta + Delta` | `4` |
| `0` | `Delta + Delta_partner`, with inequivalent partners | `2` |

Here the norm is `sum_h abs(chi(h))**2 / |H|`. These restrictions are mutually
orthogonal for inequivalent irreducible corepresentations of the same full
group and factor system. Thus they suffice for identification; extracting
antiunitary matrices from the wavefunctions is not required. Full, unrounded
character arrays should be used; compact display columns require their operation
counts as weights. The theoretical construction still needs antiunitary
operations to determine partners, Wigner indicators, and doubling. See the
[Wigner construction](#4-antiunitary-assembly-using-the-wigner-indicator) and reference [4],
Appendix G.

Character-table rows therefore label full corepresentations, while columns
contain only unitary operations. The 16D example above must be compared with
its complete `8D + 8D` restriction, not with either constituent alone. A 16D
corepresentation acts on 16 Bloch states collectively; it does not mean that a
single wavefunction has 16 spin components. The abstract table alone does not
identify its energy or band indices. The selected subspace may also contain
several irreducible corepresentations.

**Optional matrix comparison.** If `Phi = Psi U` changes the VASP basis to a
theoretical basis, then

$$
D(g)=U^\dagger B(g)U\quad\text{(unitary)},\qquad
D(g)=U^\dagger B(g)U^*\quad\text{(antiunitary)}.
$$

For this optional extension, an antiunitary action conjugates the wavefunction
and includes the time-reversal spin matrix. Its trace is not a character:
even `U = exp(i*phi) I` multiplies its matrix part by `exp(-2*i*phi)` without
changing the corepresentation. Antiunitary square relations and a common basis
change can validate a reconstructed full action, but are not extra requirements
for unitary-character label assignment.

A practical interface needs to account for:

- **Actual symmetry:** the converged Hamiltonian must preserve the specified SSG.
  With SOC, use only the surviving symmetries; independent spin rotations of the
  SOC-free SSG generally cease to be symmetries.
- **Geometry:** match the simulation cell, pure-translation lattice, origin,
  operation representatives, and k coordinates; handle band folding if needed.
- **Spin conventions:** retain spin-1/2 double-valued representations for the full
  electron spinor problem even without SOC. Align the `SAXIS` spin basis and the
  chosen SU(2) lifts, including their signs.
- **Subspace closure:** include all states mixed by the relevant operations and
  assess numerical splittings and leakage outside the selected subspace.
- **PAW overlaps:** the physical inner product involves the PAW transformation.
  Working instead with pseudo-wavefunctions requires consistent handling of
  their Gram matrix and normalization; a bare coefficient overlap is not
  automatically the physical matrix element above.

See the official VASP documentation for
[LSORBIT](https://vasp.at/wiki/LSORBIT),
[SAXIS](https://vasp.at/wiki/SAXIS), and
[the PAW formalism](https://vasp.at/wiki/Projector-augmented-wave_formalism).

### Implemented coefficient action and validation

Write the pseudo-wavefunction as
`psi_n(x)=sum_G C[G,:,n] exp(2*pi*i*(G+k) dot x)`, with fractional real-space
columns and reciprocal rows. For a unitary operation `x -> R x + tau`,

$$
q'= (G+k)R^{-1},\qquad G'=q'-k,\qquad
C'[G',:,n]=e^{-2\pi i q'\cdot\tau}\,V_g C[G,:,n].
$$

`transform_coefficients` enforces an integer, bijective mapping into the same
G set. It uses the forward operator rather than its adjoint. In particular,
fractional translations involve the transformed full wavevector `q'`, not
just the original G or the Bloch k. The kernel can also apply an optional
antiunitary action with `q'=-(G+k)R^-1` and conjugated coefficients, for diagnostic
checks. The material matching path never requests that action.

For the selected band block, flatten spin and G into the rows of `C` and use

$$
S=C^\dagger C,\qquad Q=C S^{-1/2},\qquad B_g=Q^\dagger\widehat g Q.
$$

`orthonormal_basis` rejects rank-deficient or ill-conditioned inputs. One common
basis is used for every operation. Normalizing each column separately would
leave off-diagonal Gram overlaps, so even the identity matrix could be wrong.
This Euclidean change of pseudo-state basis gives a representation equivalent
to the physical PAW action only if the PAW transformation is injective and
intertwines the symmetry. It is not an all-electron overlap or spin calculation.

`validate_unitary_block` calls `match_unitary_characters` and checks

$$
\ell=\max_g\frac{\|\widehat gQ-QB_g\|_F}{\|\widehat gQ\|_F},\qquad
u=\max_g\|B_g^\dagger B_g-I\|_F,\qquad
p=\max_{g,h}\|B_gB_h-\omega_B(g,h)B_{gh}\|_F.
$$

Products run over the unitary subgroup. `omega_B` is obtained from the same
theoretical operation representatives, spin lifts, and Bloch phases; restoring
the Bloch phase does not mean this finite quotient has a trivial factor system.
Each accepted label must have nonnegative integer multiplicities and reproduce
the entire character vector. A character match alone is insufficient when the
measured subspace fails leakage or matrix-product checks. Numerical thresholds
express approximate acceptance, not exact symbolic proofs.

The public kernel interfaces are:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "SSGReps"))
from ssg_wavecar import (
    orthonormal_basis, transform_coefficients, representation_matrix,
    match_unitary_characters, validate_unitary_block,
)
```

- Coefficients have shape `(nG, 2, nBands)`; G vectors have shape `(nG, 3)`.
- Matching takes `observed[nH]` and `references[nCoreps, nH]`, identity first,
  with every unitary operation in the same order and convention.
- Block validation takes `matrices[nH,d,d]`, zero-based `products[nH,nH]`,
  complex Bloch `factors[nH,nH]`, and leakage computed from the coefficients.
- A failed numerical match returns `passed=False`, `label="UNRESOLVED"`, and
  residuals; malformed input raises an error. The character-only matcher raises
  `ValueError` when decomposition fails. Neither function accepts antiunitary
  traces as characters.

The [real Mn3Sn report](../examples/wavecar_mn3sn/results/README.md) records the
2 meV candidate gap, tolerance values, SCF provenance, unresolved blocks, and
NBANDS=64/96 comparison. The 96-band run passes 530/582 blocks at 14 distinct k
points. All 19 tested A-point blocks pass, including a measured four-dimensional
branch. Its input and WAVECAR basis are specific to that run; theoretical labels
are local, and dense measured matrices need not equal sparse theoretical
matrices entry by entry. A repeated general point brings the totals to 590/642.
The [offline replay](../examples/wavecar_mn3sn/replay_validation.py) recomputes
matrix checks and preserves all 52 unresolved decisions. It reads saved leakage;
it does not claim to reread absent WAVECAR coefficients.

## Mathematical methods

### 1. Numerical reduction: retained method, corrected implementation

The numerical route follows the Hamiltonian/group-averaging approach underlying
the original SSGReps work [1]. Construct the twisted regular action

$$
\mathcal R(g)|h\rangle=\omega(g,h)|gh\rangle.
$$

For a generic Hermitian matrix `X`, the semilinear average

$$
\overline X=\frac{1}{|G|}\sum_{g\in G}
\mathcal R(g)X^{[s_g]}\mathcal R(g)^\dagger
$$

has invariant eigenspaces. Restricting the regular action to these spaces gives
candidate irreducible corepresentations. Equivalent copies are identified through
their unitary characters. Irreducibility, inequivalence, and completeness are
checked before accepting the result; failed candidates trigger bounded retries.

The release evaluates this average through the multiplication table and monomial
phases, avoiding storage of all dense regular matrices. It uses a local random
seed and refreshes exported characters from the final matrices.

### 2. Safe basis optimization and sparse output

For a unitary basis matrix `U`, the valid transformations are

$$
M'(g)=U^\dagger M(g)U \quad (s_g=+1),\qquad
M'(g)=U^\dagger M(g)U^* \quad (s_g=-1).
$$

The corrected implementation preserves these laws. `--optimize` tests eigenbases
of Hermitian combinations of early unitary operations and accepts a basis only
when it increases the count of entries below the numerical zero threshold.
It is a bounded sparsity heuristic, not a guarantee of maximal sparsity or
simultaneous diagonalization of noncommuting matrices. Formatting common factors
or abbreviating radicals is separate from changing the basis.

### 3. Exact example backend: central extensions and cyclotomic arithmetic

The additional exact route replaces floating-point matrix construction for the
specified examples with algebra in GAP [6]. The unitary subgroup `H` has a
root-of-unity factor system. Writing
`omega(h,l)=exp(2*pi*i*c(h,l)/q)`, form the finite central extension

$$
(h,u)(l,v)=(hl,u+v+c(h,l)\bmod q).
$$

Select ordinary irreducible representations of this extension for which its
central generator acts as `exp(2*pi*i/q)*I`; these recover the required projective
sector. The implemented examples use `q=2`, so the central generator acts as `-I`.
The extension is constructed for the **unitary subgroup**; antiunitary operations
are assembled separately.

To construct compact exact matrices, the code induces a linear character from an
abelian subgroup, isolates the target character with the central projector

$$
P_\chi=\frac{d_\chi}{|\widetilde H|}
\sum_{x\in\widetilde H}\overline{\chi(x)}\,\rho_{\rm ind}(x),
$$

then uses subgroup-weight projectors and exact orthonormalization to choose a basis.
GAP's character/representation routines and exact roots of unity are used directly.
No floating-point eigendecomposition or decimal-to-radical fitting generates these
matrices. In the 16D example, entries lie in `Q(i,sqrt(3))`.

### 4. Antiunitary assembly using the Wigner indicator

For an irreducible projective character `chi` of `H`, the indicator used by the
example backend is

$$
W_\chi=\frac{1}{|H|}\sum_{a\in G\setminus H}
\omega(a,a)\chi(a^2).
$$

For `W=+1`, exact group averaging constructs an intertwiner, followed by exact
normalization and a check of the antiunitary square. For `W=0`, inequivalent
unitary partners are paired into a doubled corepresentation. The exact 16D
example pairs two inequivalent 8D unitary representations.

The supplied exact solver handles the `+1` and `0` cases needed by its examples
and explicitly rejects unsupported `-1` cases. The generic numerical reducer
is separate and also tests quaternionic/Kramers-type examples.

### 5. Analytic dependence along fixed-little-group paths

For `k(t)=k0+t*v` on a connected open segment with constant little group, the
reciprocal-lattice difference in the little-group condition is constant. Hence

$$
\mathbf v_{\rm cart}^{T}(R_g-s_gI)=0.
$$

In the repository's convention,

$$
\omega_t(g,h)=\sigma_{\rm spin}(g,h)
\exp\{-i\mathbf k_{\rm cart}(t)\cdot(R_g\boldsymbol\tau_h-s_g\boldsymbol\tau_h)\},
$$

so its path-parameter coefficient vanishes. Constant projective matrices `M(g)`
can be constructed once, and the path dependence remains explicit:

$$
D_t(g)=e^{-i\mathbf k_{\rm cart}(t)\cdot\boldsymbol\tau_g}M(g).
$$

For the implemented lattice this is stored as
`exp[-2*pi*i*(alpha_g+beta_g*t)]*M(g)` with exact rational coefficients.
The implementation checks the phase identities symbolically, in addition to
numerical sampling. It does not infer an unknown function from sampled matrices.

### 6. Relation to the literature

[4] develops a broader Mackey/group-extension treatment of SSG projective
representations, including separate translation, mixed, and point-group factors.
It provides theoretical context for handling factor systems and antiunitary
symmetries. This release does not implement its full momentum-orbit induction,
projective translation algebras, or reconstructed Brillouin-space classification.

The finite-central-extension route implemented here corresponds directly to the
construction described in Sec. IV.3 of [5]: obtain ordinary representations of the
extension, select the required central character, and restore the Bloch phase.
The method is not restricted to layer groups. This code does not implement that
paper's dimensional reduction, complete layer-group classification, or
Smith-normal-form search for high-symmetry momenta. Its concrete GAP construction
also differs from the abstract-group lookup/polycyclic strategy discussed in
Appendix C of [5].

The projective antiunitary partners, assembly formulas, and Wigner criterion have
an explicit counterpart in Appendix G, Eqs. (192)-(197), of [4]; Sec. IV.2 of [5]
also discusses the corepresentation cases. These connections identify relevant
references for standard representation-theory tools, rather than attributing
all of those tools exclusively to either paper.

The abelian-subgroup basis selection, explicit path-phase separation in this
repository's convention, output formatting, and regression checks are concrete
implementation steps documented here; citing these papers does not imply that
their complete algorithms have been reproduced.

## Python interfaces and exports

### Use the numerical solver

Run this from the repository root, or replace the path with an absolute repository path:

```python
from pathlib import Path
import sys
import numpy as np

repo = Path.cwd()
sys.path.insert(0, str(repo / "SSGReps"))
from SSGReps import load_little_group
from ssg_database import load_database
from ssg_output import from_little_group, report_text

ssg = "1.2.1.3"
lg = load_little_group(ssg, np.zeros(3), True, "double",
                       load_database()[ssg], seed=0)
print(lg.rep_degree)
print(lg.validation)
print(report_text(from_little_group(lg), rep_label="r1",
                  operation_ids=[1, 2], output="rep_matrix"))
```

The main solve path keeps the regular action in tables. Call
`lg.get_regular_rep_list()` only if dense `lg.regularRep` matrices are needed.
All CLI operation numbers are one-based; Python array indices are zero-based.

### Read JSON or NPY exports

`repMatrix` and `character` use the projective convention even if the displayed
report uses `--convention bloch`. `linearCharacter` includes the Bloch phase; its legacy name does not imply
that the quotient-group factor system is trivial.
Exports also contain `timeReversal`, `tauC`, `rotC`, `spin`, `su2`, `repDegree`,
`torsion`, `pureTranslations`, `reciprocalBasis`, explicit convention names,
and the numerical validation summary.

JSON complex numbers are encoded as `{"real": ..., "imag": ...}`. Decode them with:

```python
import json
import numpy as np

def decode_complex(obj):
    if set(obj) == {"real", "imag"}:
        return complex(obj["real"], obj["imag"])
    return obj

with open("result.json", encoding="utf-8") as stream:
    result = json.load(stream, object_hook=decode_complex)
matrices = [np.asarray(rep, dtype=complex) for rep in result["repMatrix"]]
# Keep a list: different irreducible representations can have different dimensions.
```

To generate and read a local NPY export:

```bash
python ssgreps.py --ssgNum 1.2.1.3 --kp 0 0 0 --fileType npy --export output.npy
```

```python
import numpy as np
result = np.load("output.npy", allow_pickle=True).item()  # Load trusted files only.
```

Keep the antiunitary flags when multiplying matrices. `pureTranslations` and
`reciprocalBasis` store basis vectors as rows. The legacy `torsion` field is zero
for unitary groups; for groups with antiunitary elements it records the squared
character norm on the unitary subgroup, not the Wigner indicator itself.

### Obtain a general symbolic path expression

The CLI accepts `--kpath "1/3" "1/3" "t" --interval 0 1/2` in place of `--kp`.
The Python interface is `ssg_symbolic.SymbolicPath`. It keeps symbolic phases
with numerical constant matrices for user-defined paths and records exact
conditions for additional little-group operations. See
[the path guide](symbolic_paths.md) for runnable examples, scope, and the
`ssgreps.symbolic_path.v1` export format.

### Obtain an exact symbolic path expression

```python
from pathlib import Path
import sys
import sympy as sp

sys.path.insert(0, str(Path.cwd() / "examples" / "symbolic_paths"))
from symbolic_paths import PathFamily

family = PathFamily("MR")
index = family.geometry["global_indices_0based"].index(3)  # Database g4.
D, anti = family.operation("r1", index)
print(D)
print(D.subs(family.t, sp.sqrt(2) / 5))
# If anti is True, the complete operator is D*K.
```

### Preserve exact special numbers, or inspect numerical candidates

```python
from pathlib import Path
import sys
import sympy as sp

sys.path.insert(0, str(Path.cwd() / "examples" / "symbolic_output"))
from symbolic_numbers import format_exact, recognize_numeric

z = (1 + sp.sqrt(3) * sp.I) / 2
print(format_exact(z, "exponential"))  # exp(I*pi/3)
print(format_exact(z, "radical"))      # 1/2 + sqrt(3)*I/2
candidate = recognize_numeric(0.5 + 0.8660254037844386j)
print(candidate["exponential"], candidate["residual"])
print(candidate["certified_exact"])   # False: numerical evidence is not a proof.
```

Exact formatting rejects floating-point input. Numerical recognition searches a
bounded family `sqrt(a/b)*exp(2*pi*i*p/q)` and returns a candidate with a residual
or `unrecognized`; it is not automatically applied to arbitrary numerical bases.
A candidate with zero floating-point residual is still not a certified algebraic
number. GAP's `E(n)` means `exp(2*pi*i/n)`, not Euler's number raised to `n`.

## Exact reconstruction and validation

Reconstruct the fixed 16D example with GAP, from its own directory:

```bash
cd examples/exact_16d
gap -q -A -b --quitonbreak construct.g
cd ../..
```

Reconstruct all six symbolic paths and four endpoints, then regenerate formulas:

```bash
python examples/symbolic_paths/rebuild.py --gap /absolute/path/to/gap
python examples/symbolic_paths/compatibility.py
python examples/symbolic_paths/export_formulas.py
```

GAP is not bundled. Supply its actual executable path or install it on `PATH`.
Cached matrices are sufficient for the normal Python examples.

For release verification and regression testing:

```bash
python verify_release.py
python -m unittest discover -s tests -p 'test_*.py'
python tests/validate_suite.py
python examples/readable_output/check_output.py
python examples/symbolic_paths/verify_paths.py --repo .
python examples/symbolic_output/check_examples.py
```

The recorded release validation contains:

| Check | Recorded result |
|---|---:|
| SSGs / calculation configurations | 16 / 129, all passed |
| Translation indices tested | 1, 2, 4, 6, 8, 9, 10, 11, 12 |
| Projective matrix-product checks | 20,398,105 |
| Bloch matrix-product checks | 20,398,105 |
| Maximum projective product residual | 1.224e-14 |
| Maximum Bloch product residual | 8.565e-15 |
| Maximum unitarity residual | 1.491e-14 |
| Separate path sampling | 42 points on six paths |
| Exact-character comparisons in the extended suite | 22 |
| Fresh-environment/package regression tests | 7 passed |

Every run validates final matrix products, unitarity, character consistency,
irreducibility, inequivalence, and completeness. Mathematical residuals are checked
against `1e-7`; matrix-product and unitarity residuals use the absolute Frobenius
norm. The extended suite additionally checks associativity, the cocycle condition,
geometric membership, and values parsed from HTML. The display tolerance is `1e-5`,
reflecting six-significant-digit printing. JSON/NPY retain full numerical precision.

The exact path/endpoint backend separately records 2,024,192 exact matrix-product
checks for 33 corepresentations, plus symbolic phase identities. These algebraic
checks and numerical residual checks have different meanings.

See the [validation report](../validation/RELEASE_VALIDATION.md) and
[per-case records](../validation/extended_validation.json) for the actual tested
settings. These finite regressions are not an exhaustive proof for all 183498
records, every k point, or every platform.

## Changes from the original implementation

| Area | Release change |
|---|---|
| Numerical mathematics | Retains semilinear group averaging; uses monomial actions and validated final subspaces |
| Legacy phase postprocessing | Replaces independent entry rescaling with valid unitary basis changes; `solve_factor` is a compatibility no-op |
| Exact construction | Adds finite central extensions, induced representations, exact projectors, and Wigner assembly for the documented examples |
| Path dependence | Separates exact Bloch phases from constant projective matrices on fixed-little-group segments |
| Output | English aligned text/HTML, factor extraction, root notation, selected matrices, explicit conventions |
| Special numbers | Separate certified exact rewriting and provisional numerical recognition |
| Reliability | Final-matrix validation, finite-input checks, correct `--optimize False`, nonzero error exits |
| Packaging | Compressed database loading, pinned dependencies, standalone examples, tests, and checksums |

The old postprocessing failed multiplication/unitarity tests for SSGs 1.2.1.3 and
1.6.1.3, with residuals about 4.24 and 49.50. The release fixes these failures;
[baseline witnesses](../validation/baseline_failures.json) are retained. The little-group
phase tolerance was also tightened from `1e-3` to `1e-8`, avoiding spurious symmetry
at nearby generic points. Numerical matrices can differ from old output by a basis
choice and local ordering; compare characters or equivalence rather than unrelated
matrix entries. See [CHANGELOG.md](../CHANGELOG.md).

## Repository layout and limitations

```text
ssgreps.py                    Repository-root CLI
SSGReps/SSGReps.py            Geometry, little-group interface, exports, CLI logic
SSGReps/ssg_numeric.py        Numerical reduction, unitary basis changes, validation
SSGReps/ssg_output.py         Shared text/HTML formatter
SSGReps/ssg_database.py       Trusted bundled database loader
SSGReps/ssg_symbolic.py       General symbolic paths and validity conditions
SSGReps/ssg_wavecar.py        Spinor coefficient and unitary-character tools
ssg_data/                    Original compressed database and supporting data
examples/exact_16d/          One exact 16D corepresentation
examples/symbolic_paths/     Six parametric paths and four endpoint datasets
examples/symbolic_output/    Exact formatting and numerical candidate recognition
examples/readable_output/    Exact report demonstration using the shared formatter
tests/                      Regression and extended validation scripts
validation/                 Measurements, baseline failures, and output examples
references.bib              Bibliography for the references below
SHA256SUMS.txt               Release file manifest
```

The exact backend is currently limited to the documented SSG, factors `+/-1`, and
its supported Wigner cases. General symbolic paths use the numerical solver and are not restricted to these
exact examples. The symbolic-path separation applies in this gauge on segments
with a fixed little group. General-path exports record exceptional conditions;
they do not construct representations of the enhanced groups automatically.
The interface does not handle
branch monodromy around closed loops, Berry phases, or arbitrary projective translation
algebras. Collinear handling retains the original finite-subgroup convention.
Legacy k-dot-p helpers are outside the present validation scope.

## References and citation guidance

Bibliographic metadata was checked against publisher/arXiv and official software
sources on 2026-11-09. IRSSG is cited in its published journal form; the
projective-SSG and spin-layer works are cited as arXiv preprints.
Machine-readable entries are in
[references.bib](../references.bib).

1. **Original representation method.** Ziyin Song, A. Z. Yang, Yi Jiang, Zhong Fang,
   Jian Yang, Chen Fang, Hongming Weng, and Zheng-Xin Liu,
   *Constructions and applications of irreducible representations of spin-space groups*,
   **Physical Review B 111**, 134407 (2025).
   [Publisher / DOI](https://doi.org/10.1103/PhysRevB.111.134407);
   [arXiv:2409.13601](https://arxiv.org/abs/2409.13601).

2. **SSG enumeration and database.** Yi Jiang, Ziyin Song, Tiannian Zhu, Zhong Fang,
   Hongming Weng, Zheng-Xin Liu, Jian Yang, and Chen Fang,
   *Enumeration of Spin-Space Groups: Toward a Complete Description of Symmetries of Magnetic Orders*,
   **Physical Review X 14**, 031039 (2024).
   [Publisher / DOI](https://doi.org/10.1103/PhysRevX.14.031039);
   [arXiv:2307.10371](https://arxiv.org/abs/2307.10371).

3. **SSG software and electronic-band symmetry analysis.** Sheng Zhang, Ziyin Song,
   Zhong Fang, Hongming Weng, and Zhijun Wang,
   *IRSSG: An Open-Source Software Package for Spin Space Groups*,
   **Computer Physics Communications 326**, 110190 (2026).
   [Publisher / DOI](https://doi.org/10.1016/j.cpc.2026.110190);
   [arXiv:2511.21821](https://arxiv.org/abs/2511.21821).

4. **Projective SSG framework.** Zheng Zhang,
   Peiyuan Wang, and Y. X. Zhao,
   *Projective representation theory of spin space groups*,
   arXiv:2609.21289 (2026), version 1.
   [arXiv](https://arxiv.org/abs/2609.21289v1);
   [DOI](https://doi.org/10.48550/arXiv.2609.21289).

5. **Analytic spin-layer corepresentations.**
   Zeying Zhang, Gui-Bin Liu, Mu Tian, Run-Wu Zhang, Zhi-Ming Yu, and Yugui Yao,
   *Spin layer groups and their corepresentations*,
   arXiv:2605.25484 (2026), version 2.
   [arXiv](https://arxiv.org/abs/2605.25484v2);
   [DOI](https://doi.org/10.48550/arXiv.2605.25484).

6. **Exact finite-group computations.** The GAP Group,
   *GAP — Groups, Algorithms, and Programming*, version 4.16.1 used here.
   [Official reference manual: character tables and representation construction](https://gap-system.github.io/gap/doc/ref/chap71_mj.html).

7. **Symbolic expressions.** Aaron Meurer, Christopher P. Smith, Mateusz Paprocki,
   Ondřej Čertík, Sergey B. Kirpichev, Matthew Rocklin, Amit Kumar, Sergiu Ivanov,
   Jason K. Moore, Sartaj Singh, Thilina Rathnayake, Sean Vig, Brian E. Granger,
   Richard P. Muller, Francesco Bonazzi, Harsh Gupta, Shivam Vats, Fredrik Johansson,
   Fabian Pedregosa, Matthew J. Curry, Andy R. Terrel, Štěpán Roučka, Ashutosh Saboo,
   Isuru Fernando, Sumith Kulal, Robert Cimrman, and Anthony Scopatz,
   *SymPy: symbolic computing in Python*, **PeerJ Computer Science 3**, e103 (2017).
   [Publisher / DOI](https://doi.org/10.7717/peerj-cs.103);
   [official paper repository](https://github.com/sympy/sympy-paper).

8. **Crystallographic utilities.** Atsushi Togo, Kohei Shinohara, and Isao Tanaka,
   *Spglib: a software library for crystal symmetry search*,
   **Science and Technology of Advanced Materials: Methods 4**, 2384822 (2024).
   [Publisher / DOI](https://doi.org/10.1080/27660400.2024.2384822);
   [official repository](https://github.com/spglib/spglib).

Cite [1] for the representation approach and [2] for the database. Reference [3]
describes IRSSG and related electronic-band symmetry analysis.
For exact or symbolic workflows, also cite the relevant
background [4,5] and software [6,7]. Cite [8] when using the spglib-based utilities.
Record the actual code revision and validation scope in reproducible research;
this release is not itself assigned a publication DOI.

The upstream repository is [zine-phy/SSGReps](https://github.com/zine-phy/SSGReps).
This work is based on commit `97e839509ec1a7f08af68b047f75aad0c71468b7`.
The original [MIT License](../LICENSE) is retained.
