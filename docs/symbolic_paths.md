# General symbolic paths

[Repository overview](../README.md) · [Numerical and exact interfaces](usage.md)

The `--kpath` input constructs a parameter-dependent representation family for
an SSG in the bundled database. Coordinates are expressions in one real
parameter. The solver obtains a numerical basis at a reproducibly sampled
interior point and retains the parameter in the Bloch phases:

$$
D_g(t)=e^{-2\pi i\,q(t)\cdot\tau_g}M_g.
$$

**The parameter dependence is symbolic; the constant matrices `M_g` are
numerical.** This interface does not certify every constant matrix entry as an
exact radical or root of unity. The separate cached exact examples provide
fully algebraic matrices for their documented families.

## Command-line input

Run from the repository root with the core requirements installed. GAP is not
needed. Choose either `--kp` for a numerical point or `--kpath` for a path.

```bash
python ssgreps.py --ssgNum 221.8.6.3 \
  --kpath "1/2" "1/2" "t" --interval 0 1/2 \
  --out both --rep 1 --ops 3 --optimize --seed 19 \
  --text path.txt --html path.html \
  --fileType json --export path.json
```

This specifies the open MR segment `k(t)=(1/2,1/2,t)`, `0<t<1/2`. The generic
little group has order 64 and two four-dimensional corepresentations. Operation
indices are **local to that little group**; local operation 3 corresponds to
global database operation 4 in this example. The export records the complete
mapping in `globalOperationIds`.

Other groups and paths use the same entry point:

```bash
# Hexagonal path from K toward H.
python ssgreps.py --ssgNum 194.1.6.1.P \
  --kpath "1/3" "1/3" "t" --interval 0 1/2 --out character

# A group with translation index 12.
python ssgreps.py --ssgNum 184.12.2.9 \
  --kpath "t" "0" "0" --interval 0 1/2 --out character

# A nonlinear path with an interior symmetry-enhancement point.
python ssgreps.py --ssgNum 221.8.6.3 \
  --kpath "s**2" "0" "0" --parameter s --interval 0 1 \
  --fileType json --export quadratic_path.json

# A trigonometric parametrization.
python ssgreps.py --ssgNum 63.1.4.67 \
  --kpath "0" "0" "sin(t)/5" --interval 0 "2*pi" \
  --fileType json --export trigonometric_path.json
```

`--kp-symbolic` is an alias of `--kpath`. `--parameter` defaults to `t` and the
open interval defaults to `(0,1)`. Expressions support `+`, `-`, `*`, `/`, rational
powers using `**`, and `sqrt`, `sin`, `cos`, `exp`, and `pi`. Decimal literals
are interpreted as exact decimal rationals; use `1/3` for an exact third.
Quote expressions for the shell. Parenthesize a coordinate beginning with a
minus sign, such as `"(-t)"`, so it is not interpreted as a command-line option.
The parser does not execute Python expressions or accept undeclared symbols.

Coordinates use the reciprocal basis of the **pure-translation lattice**, just
as in `--kp`. A different SSG may have a different lattice basis. `--groupType 1`
and `2` select single- and double-valued representations. Matrix filters,
`--convention`, `--seed`, text/HTML reports, and JSON/NPY exports work in both
input modes. `--seed` controls both path sampling and the numerical reduction.

## Scope and exceptional points

One family describes the full **generic** little group of the path. At special
parameter values, the little group may contain additional operations. The
reported family still restricts to its original subgroup there; it is not the
complete representation table of the enhanced group.

For example, `k(t)=(t,0,0)` on `(0,1)` has an interior enhancement at `t=1/2`.
For `k(s)=(s**2,0,0)`, the same k point is reached at `s=sqrt(2)/2`. These values
are listed in the report/export, and the checked Python evaluation interface
rejects them. Compute the full point with `--kp 0.5 0 0`, or construct a separate
constant path. Interval endpoints are recorded separately and excluded from
checked path evaluation even if their little group happens to be unchanged.

The export gives an exact integer condition for each potentially additional
operation. An operation becomes a symmetry when **all three listed expressions
are integers simultaneously**. The generic family is valid only when no
additional operation satisfies its condition.

A finite exceptional-point list is provided when the conditions can be fully
enumerated by the polynomial constraint algorithm. Rational affine paths are
covered, as are many polynomial paths. For more general expressions, such as
trigonometric paths, the export may retain integer conditions instead of a
complete finite list. In this case `finiteExceptionListComplete` is `false`;
absence from the point list must not be interpreted as proof of generic symmetry.
`check_parameter`, `numeric`, and `validate_at` still check the requested value.

The supported input is a real continuous one-parameter expression on a finite
open interval. Split paths at singularities and piecewise branches. If symbolic
membership cannot be reconciled with a sampled generic group, the solver stops
instead of assuming that the sampled group is valid everywhere. This interface
does not construct multi-parameter surfaces, automatic endpoint compatibility
relations, a periodic gauge around closed loops, or Berry phases.

## Why one numerical solve is sufficient

Let `P` contain the pure-translation basis vectors as rows, so
`q(t)=k(t)^T P^(-T)` is a row in conventional reciprocal coordinates with the
factor `2*pi` removed. Let `s_g=+1` for unitary and `-1` for antiunitary elements.
An operation belongs to the little group if

$$
q(t)(s_gR_g-I)P^T\in\mathbb Z^3.
$$

On a connected continuous segment, an integer-valued continuous expression is
constant. The implementation checks this constancy symbolically for every
operation in the proposed generic group. Consequently
`q(t)(R_g-s_g I)` is constant, and so is the projective factor in the existing
SSGReps convention:

$$
\omega(g,h)=\sigma_{\rm spin}(g,h)
\exp\!\left[-2\pi i\,q(t)\cdot(R_g\tau_h-s_g\tau_h)\right].
$$

The numerical matrices at one generic sample can therefore be kept fixed.
They obey the semilinear law

$$
M_g M_h^{[s_g]}=\omega(g,h)M_{gh},
$$

where `[s_g]` means identity for `+1` and complex conjugation for `-1`.
Restoring the exact symbolic Bloch phase gives the path family. Its physical
quotient multiplication law includes the translation defect
`L(g,h)=tau_g+R_g tau_h-tau_gh`:

$$
D_g(t)D_h(t)^{[s_g]}=
\sigma_{\rm spin}(g,h)e^{-2\pi i q(t)\cdot L(g,h)}D_{gh}(t).
$$

The database spatial geometry is reconstructed as rationals with denominator
at most 4096 and absolute error at most `1e-10`; unsupported reconstruction is
rejected. Exact lattice preservation, spatial products, and lattice translation
defects are checked. Numerical matrix products, unitarity, irreducibility,
inequivalence, completeness, associativity, and the cocycle are checked at the
reference point. Three additional seeded generic samples check transferred
matrices against factors recomputed by the numerical geometry code.

The generic group and parameter dependence are certified for this reconstructed
geometry. Constant matrix accuracy remains numerical and is reported through
residuals. No polynomial interpolation of matrix entries is used.

## Python interface

```python
from pathlib import Path
import sys
import sympy as sp

sys.path.insert(0, str(Path.cwd() / "SSGReps"))
from ssg_symbolic import SymbolicPath

family = SymbolicPath(
    "194.1.6.1.P", ["1/3", "1/3", "t"],
    interval=("0", "1/2"), seed=19,
)
print(family.k)
print(family.labels)

# Local zero-based index; an antiunitary operation has full action D*K.
D, anti = family.operation("r1", 3)
print(D)

# Only unitary characters are returned, with local one-based operation IDs.
operation_ids, characters = family.characters("r1")
print(list(zip(operation_ids, characters)))

# Checked evaluation rejects endpoints and enhanced-symmetry parameters.
matrices = family.numeric("r1", "sqrt(2)/5")
checks = family.validate_at("1/7")
print(checks["passed"])

# Integer coefficients in the pure-translation lattice basis.
print(family.translation([0, 0, 1]))
```

`operation(..., bloch=False)` and `characters(..., bloch=False)` return the
constant projective matrices/characters. `operation` returns a SymPy expression;
manual `.subs(...)` does not check little-group changes. Use the checked methods
above when the full little group at the substituted point matters.

Character tables contain only unitary operations. Antiunitary operations are
included in the theoretical construction and the semilinear checks, never as
ordinary character columns. A representation can have parameter-independent
characters or matrices for some operations even on a nonconstant path; pure
translations retain their Bloch phase through `translation(...)`.

## Export format

Symbolic-path JSON/NPY exports have schema `ssgreps.symbolic_path.v1`, distinct
from a numerical point export. Important fields are:

| Field | Content |
| --- | --- |
| `kPath`, `parameter`, `interval` | Symbolic coordinate strings and open parameter interval |
| `referenceParameter`, `referenceK`, `seed` | Reproducible numerical reference |
| `repMatrix` | Full-precision constant numerical projective matrices |
| `blochPhases` | Symbolic strings multiplying each operation's constant matrix |
| `character` | Numerical projective characters, unitary operations only |
| `symbolicBlochCharacters` | Unitary characters retaining the symbolic phase |
| `globalOperationIds`, `unitaryLocalOperationIds` | Operation-index conventions |
| `factorPhaseExponents`, `spinFactors` | Constant factor-system data |
| `validity` | Excluded integer conditions, known enhanced points, completeness flag, endpoints |
| `exactness` | Symbolic parameter dependence and explicitly numerical constant matrices |
| `validation` | Geometry checks, reference residuals, and sample checks |

Symbolic expressions are stored separately from numerical coefficient arrays;
no display rounding is used in exported matrices. JSON complex values use
`{"real": ..., "imag": ...}`. `to_dict()` returns the Python export data.
The numerical-point export format is unchanged.

## Validation

```bash
python -m unittest discover -s tests -p 'test_symbolic.py'
python tests/validate_symbolic_paths.py
```

The dedicated regressions cover input parsing, nonorthogonal/centered cells,
high translation indices, both representation types, antiunitary products,
polynomial and trigonometric paths, enhanced-point rejection, mixed symbolic
formatting, and numerical agreement with independently constructed exact data.
The cross-check script also compares families with independent numerical solves
at rational and irrational parameters. Results are recorded in
[the general-path validation data](../validation/symbolic_paths_general.json).
