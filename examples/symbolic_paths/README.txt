Exact SSG small corepresentations with symbolic path parameters
==============================================================

Implemented scope
-----------------
SSG 221.8.6.3, noncoplanar, double-valued representations. Six high-symmetry
paths and four endpoints: Gamma, X, M, and R.
Coordinates use the reciprocal basis of the pure-translation lattice T0=2 Z^3:
Gamma=(0,0,0), X=(1/2,0,0), M=(1/2,1/2,0), R=(1/2,1/2,1/2).
Parent conventional reciprocal coordinates are half these values; these labels
must not be transferred directly to the parent's conventional cell.
All open paths use 0<t<1/2. Labels r1,r2,... are local branch identifiers.

Path      k(t)                  Little-group order    Irreducible dimensions
GX        (t,0,0)               64                    2,2,2,2,4
GM        (t,t,0)               32                    2,2,2,2
GR        (t,t,t)               48                    2,2,4
XM        (1/2,t,0)             32                    4
MR        (1/2,1/2,t)           64                    4,4
XR        (1/2,t,t)             32                    4

Each order includes unitary and antiunitary elements; the unitary subgroup has
half that order. Full endpoint little-group corepresentation dimensions are:
Gamma: 2,2,2,2,4,4,6,6,6,6; X: 4,4,8; M: 8; R: 8,8,16.
Exact rational checks exhaust the extra little-group conditions in each open
interval, so no interior symmetry-enhancement points are omitted.

Main mathematical simplification
--------------------------------
On a straight segment k(t)=k0+t v with a fixed little group, let s_g=+1 for
unitary elements and -1 for antiunitary elements. The reciprocal-vector difference
in the little-group condition is discrete, so it is constant on a connected open
interval. Consequently,
    v_conv^T (R_g-s_g I)=0.
In the original SSGReps convention, the factor system is
    omega_t(g,h)=sigma_spin(g,h)
                 exp[-i k_conv(t).(R_g tau_h-s_g tau_h)].
Its t coefficient is exactly zero. The projective co-group matrices M(g) therefore
need to be constructed only once per segment. The parameter-dependent matrices are
    D_t(g)=exp[-i k_conv(t).tau_g] M(g),
with antiunitary operators D_t(g) K, where K denotes complex conjugation.

Here k_conv=pi*k and tau_g is integral, so the stored form is
    D_t(g)=exp[-2*pi*i*(alpha_g+beta_g*t)] M(g),
    alpha_g=(k0.tau_g)/2, beta_g=(v.tau_g)/2.
Both coefficients are exact rationals, and irrational parameters can be substituted
directly. This avoids repeated eigendecompositions and finite central extensions
at arbitrary irrational values of t.

The code verifies two distinct, compatible multiplication laws:
    M(g) M(h)^{s_g} = omega(g,h) M(gh),
    D_t(g) D_t(h)^{s_g}
      = sigma_spin(g,h) exp[-i k_conv(t).L(g,h)] D_t(gh),
where L(g,h)=tau_g+R_g tau_h-tau_gh belongs to the pure-translation lattice.
The exponent s_g=-1 means entrywise conjugation, not matrix inversion.
These laws use a coset section; omitting the Bloch phase of L gives an incorrect
multiplication check.

A readable 4D parameter-dependent example
----------------------------------------
For branch r1 on MR, k(t)=(1/2,1/2,t), database operation g4 is
    [C2y_spin || I_space | (1,0,1)].
Its analytic matrix is
    D_t(g4)=exp(-i*pi*t)/sqrt(2) *
     [[1,-1,0,0],[-1,-1,0,0],[0,0,1,-1],[0,0,-1,-1]].
Its eigenvalues are +/-exp(-i*pi*t), each with multiplicity two, and its square
is exp(-2*i*pi*t) I4. This is one generator of the full irreducible 4D family.
The other generators and antiunitary operators are also exported.
See analytic_generators.txt / .json for all branches on all six paths.

Exact construction of constant matrices
--------------------------------------
All path factors in this example are +/-1. Construct a twofold central extension
of the unitary subgroup and retain irreducible characters with central element
-I. Induce linear characters of abelian subgroups, then apply exact central
and subgroup-weight projectors to extract subspaces and select orthonormal bases.
All matrix arithmetic takes place in cyclotomic fields.

All six paths have Wigner indicator +1. Exact group averaging constructs an
antiunitary intertwiner U, followed by exact normalization and the check
    U conjugate(U)=omega(a,a)D(a^2).
Endpoints X and R also use inequivalent partner pairing with Wigner indicator 0.
The current finite-group backend explicitly supports +1 and 0 and raises an
error for -1; the latter case has not been implemented.

Exact compatibility relations
-----------------------------
All 12 path/endpoint combinations have been decomposed by character inner products.
Every multiplicity is checked to be a nonnegative integer, and dimensions are
preserved. See compatibility.txt / .json.
The earlier R-point 16D corepresentation restricts as
    R(16D)|MR = 2 MR.r1(4D) + 2 MR.r2(4D),
    R(16D)|XR = 4 XR.r1(4D),
    R(16D)|GR = 2 GR.r1(2D) + 2 GR.r2(2D) + 2 GR.r3(4D).
These are representation restrictions, without assuming a band dispersion.
Substituting an endpoint value of t gives a representation of the path subgroup
only. Additional endpoint operations require separately computed endpoint matrices;
path matrices alone do not form the complete endpoint little-group representation.

Validation
----------
1. All 16 inequivalent irreducible corepresentation families on the six paths
   were constructed exactly. Their constant matrices satisfy 41728 checked
   semilinear multiplication laws.
2. For every pair of elements on each path, the constant phase term is checked
   modulo integers and the t coefficient is checked identically: 13568 symbolic
   phase identities in total. The result holds over the interval, not just at
   sampled points.
3. Including four endpoints gives 33 irreducible corepresentations and 2024192
   exact multiplication checks. Every representation also passes unitarity and
   exact irreducibility-indicator checks with indicator 1.
4. Each path was additionally evaluated at seven parameters:
   1/17,1/8,1/4,3/8,8/17,sqrt(2)/5,sqrt(3)/7, for 42 path points in total.
   Comparisons use the original k_op, get_table, get_trans_factor,
   get_regular_rep_list, and get_irreducible_rep, excluding the previously
   identified problematic solve_factor postprocessing. Maximum character
   discrepancy was about 2.18e-15; maximum Frobenius residual for the Bloch
   multiplication law was about 7.54e-15. These numerical checks do not construct
   the exact matrices.
5. Original-code little-group identification at all 12 path endpoints agrees
   with the separately constructed exact endpoint data.

Usage
-----
Python dependencies are NumPy and SymPy. Reading existing results does not need GAP:
    from symbolic_paths import PathFamily
    import sympy as sp
    f = PathFamily('MR')
    t = f.t
    i = f.geometry['global_indices_0based'].index(3)  # Database operation g4.
    D, anti = f.operation('r1', i)  # Exact SymPy matrix containing t.
    print(D)
    print(D.subs(t, sp.sqrt(2)/5))
    print(f.translation([1,0,0]))  # Bloch phase of pure translation (2,0,0).

Recompute everything (tested with GAP 4.16.1):
    python rebuild.py --gap /path/to/gap
    python compatibility.py
    python export_formulas.py
    python verify_paths.py --repo /path/to/SSGReps

rebuild.py runs each GAP case in a clean temporary directory and copies results
only after success. Without --repo, verify_paths.py checks Bloch multiplication
and unitarity only and does not claim comparison with the original repository.
That comparison requires the original project's additional Python dependencies.
The included validation.json was produced by an actual run with --repo.

Files
-----
parent_group.json        Certified parent operations and R-point factors
prepare_paths.py         Exact path little groups, exceptional parameters,
                         constant factors, and symbolic phase proofs
solve_exact.g            Exact constant corepresentations, intertwiners, and checks
rebuild.py               Recompute six paths and four endpoints
symbolic_paths.py        SymPy interface with symbolic parameter t
compatibility.py         Exact endpoint compatibility relations
verify_paths.py          Pointwise comparison with the original code
export_formulas.py       Export generators and verify the compact MR example
analytic_generators.txt  Readable analytic generators
analytic_generators.json Machine-readable analytic generators
worked_example.txt       The 4-by-4 parameter-dependent example above
compatibility.txt/.json  All 12 compatibility relations
validation.json          Errors at all 42 sampled path points
GX/,...,R/               Group data, matrices, characters, and GAP logs per case

SSG data: https://github.com/zine-phy/SSGReps
Reference commit: 97e839509ec1a7f08af68b047f75aad0c71468b7.
The construction uses central extensions and antiunitary corepresentations.
Factor constancy along each path follows directly from the repository's factor
formula and is verified exactly for every path.
Background:
https://arxiv.org/abs/2609.21289
https://arxiv.org/abs/2605.25484

Limits of generalization
------------------------
The data entry point remains specific to SSG 221.8.6.3. The finite-group solver
supports +/-1 factors and Wigner indicators +1/0, not all SSGs. Parameter separation
applies to affine intervals with a constant little group in the original convention.
Other lattices need general primitive/reciprocal transformations, higher-order
roots of unity, and separate segments wherever the little group changes.
Branch permutations around closed loops, periodic gauges, and Berry phases are
outside this example's scope.
