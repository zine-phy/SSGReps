An exact 16-dimensional SSGReps corepresentation
==============================================

Example and coordinates
-----------------------
SSG = 221.8.6.3, noncoplanar, double-valued (spin-1/2 representations).
k = (1/2,1/2,1/2) in the reciprocal basis of pure translations used by SSGReps.
The pure-translation lattice is T0 = 2 Z^3. The same point has conventional
reciprocal coordinates (1/4,1/4,1/4) in the parent space group; it is not the
R point of the parent's conventional cell.
The parent space group is No. 221. The finite little group Gk/T0 has 384 elements:
192 unitary and 192 antiunitary. The 16D object is an irreducible corepresentation
of the little group at a single k point, without induction over a k star.

Source: https://github.com/zine-phy/SSGReps
Reference commit: 97e839509ec1a7f08af68b047f75aad0c71468b7.
Operation order and the phase convention for SU(2) lifts match that version.
This package is a standalone backend prototype with complete exact results.
It does not require the later output-formatting changes to the working repository.

Usage
-----
Read the precomputed analytic matrices with Python and SymPy:
    python exact_ssg.py

Python interface:
    from exact_ssg import ExactSSG
    rep = ExactSSG()
    M, anti = rep.operation(192)  # Zero-based index: matrix part of g193.
    # If anti is True, the operator is M K, with K denoting complex conjugation.
    D8 = rep.matrices[2][:8,:8]   # 8D block of the third unitary operation.
    numeric_fields = rep.as_ssgreps_dict()

Recompute the exact representation and algebraic checks using a standard GAP
installation (tested with GAP 4.16.1). Run from this directory:
    gap -q -A -b --quitonbreak construct.g

Independent numerical cross-check (NumPy and SymPy required):
    python verify_numeric.py

Re-export group data and certify integer geometry, exact SU(2) signs, and
all associativity relations:
    python prepare_case.py --repo /absolute/path/to/SSGReps
First extract ssg_data/identify.pkl.tar.gz and install the original repository's
dependencies. This exporter is explicitly limited to this example's integer
coordinates and signed-permutation spin matrices; it does not guess radicals
from arbitrary floats. Use the reference commit to retain operation ordering.

Mathematical construction
-------------------------
The original project's projective co-group matrices obey
    M(g) M(h)^{s(g)} = omega(g,h) M(gh),
where s(g)=+1 leaves the matrix unchanged and s(g)=-1 means entrywise complex
conjugation. Here omega(g,h) = (-1)^ce[g,h], with values +1 and -1 only.

1. Construct a central extension of order 384 for the unitary subgroup H of
   order 192:
       (g,u)(h,v) = (gh, u+v+ce[g,h] mod 2).
   Compute exact irreducible characters in GAP's cyclotomic field and retain
   branches where the central element z is represented by -I. Their dimensions
   are 4,4,4,4,8,8, whose squared dimensions sum to 192. Adding a separate spin
   double group would count the same factor system twice.

2. Choose an 8D character chi. Find a linear character theta of an abelian
   subgroup B of order 32 that occurs with multiplicity one. Induce it to the
   central extension, giving a 12D monomial representation. The exact projector
       P_chi = (8/384) sum_x conjugate(chi(x)) R_ind(x)
   selects the 8D irreducible subspace.

3. Choose a basis using character projectors of B, then orthonormalize exactly.
   This choice controls expression complexity. All final entries belong to
       0, +/-1, +/-i, +/-1/4, +/-i/4, +/-sqrt(3)/4, +/-i*sqrt(3)/4,
   a set of 13 numbers in Q(zeta_12) = Q(i,sqrt(3)). Random choices are confined
   to selecting finite subgroups, with a fixed seed. They do not approximate
   entries or infer analytic expressions. The construction uses induction,
   exact projection, and exact basis selection, without floating-point
   eigendecomposition.

4. The Wigner indicator
       (1/192) sum_{x in G-H} omega(x,x) chi(x^2)
   is exactly zero, pairing the 8D unitary representation with an inequivalent
   8D partner. Choose the antiunitary element
       a = g193 = [-I_spin || I_space | (1,1,1)].
   In this example's gauge, the 16D matrices are
       M(h) = diag(D(h), conjugate(D(h))), h in H,
       M(a) = [[0,I8],[I8,0]], with operator M(a) K,
       M(ah) = omega(a,h)^(-1) M(a) conjugate(M(h)).
   Here omega is +/-1, so its inverse equals itself; the inverse must be
   retained in generalizations. Unitary operations preserve the two blocks,
   while antiunitary operations exchange them, making the full corepresentation
   irreducible. In particular, D(g97) = -i I8, whereas its partner is +i I8,
   explicitly showing their inequivalence.

5. This package stores projective matrices in the original code's convention.
   The Bloch small-representation matrices are
       D_small(g) = exp(-i k_conv . tau_g) M(g).
   Here k_conv = (pi/2,pi/2,pi/2). Use rep.operation(index, bloch=True) in Python.
   Always retain antiunitary flags: antiunitary operators cannot be multiplied
   as ordinary complex matrices.

Validation results
------------------
prepare_case.py certifies signs for 147456 SU(2) products, all 56623104 group
associativity triples, and all 56623104 factor-system triples.
construct.g checks all 147456 semilinear 16-by-16 multiplication relations
and 384 unitarity conditions exactly.
The 8D character norm is 1, its inner product with the antiunitary partner is 0,
and the Wigner indicator is 0. The 16D restriction to the unitary subgroup has
character norm 2, while the full corepresentation irreducibility indicator is 1.
These are algebraic equalities, not tolerance-based decisions.

Independent numerical regular-representation reduction yields dimensions 8,8,16
and agrees with the exact 16D character. The maximum character discrepancy was
about 7.1e-15; the Frobenius multiplication residual was about 4.5e-16.
This check reproduces the original group-averaging reduction, using monomial
matrices for efficiency, and excludes the original solve_factor postprocessing.
Numerical results do not participate in constructing the exact matrices.
See construction.log and numeric_validation.json for the logs and errors.
The GAP construction plus full multiplication checks took about 24 seconds,
excluding data export and GAP installation. This was not a controlled performance
benchmark against the original code.

Files
-----
construct.g                  Exact construction and all algebraic checks
prepare_case.py               Case-specific data export and factor certification
group_data.json / .g          Geometry, multiplication table, and factors
                             (JSON indices are zero-based; GAP is one-based)
exact_representation.json     All 384 exact 16-by-16 matrices, characters, and checks
exact_matrices.g              Matrices readable directly in GAP
exact_ssg.py                  SymPy reader and numeric matrix-consumer adapter
generators.txt               Five unitary generators as 8-by-8 blocks, operations,
                             and antiunitary assembly formulas
generators_exact.json         Machine-readable generators
verify_numeric.py            Independent numerical cross-check
export_generators.py          Regenerate readable generator output

Scope and extensions
--------------------
This entry point selects this example's 8D unitary representation and produces
one 16D type-c corepresentation. It is not a general backend for all SSGs and
arbitrary k points. This case has only second-order root-of-unity factors.
Generalization requires parameterizing the root order q, target character,
exact coordinates, lift certification, and k point, and adding exact intertwiners
and normalization for Wigner indicators +1 and -1. Arbitrary irrational k points
also require symbolic Bloch phases. as_ssgreps_dict supplies matrix-consumer fields
for the selected 16D corepresentation only; it does not replace the original
interface that enumerates all corepresentations and does not include SU(2) metadata.

The method connects the referenced papers through root-of-unity factors, central
extensions of unitary subgroups, exact representations, and antiunitary
corepresentations. Abelian-subgroup projectors additionally select a compact basis.
References:
https://arxiv.org/abs/2609.21289
https://arxiv.org/abs/2605.25484
GAP documentation for exact induced representations and character algorithms:
https://gap-system.github.io/gap/doc/ref/chap71_mj.html
