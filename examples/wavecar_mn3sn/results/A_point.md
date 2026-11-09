# A-point characters and wavefunction representations

SSG: **194.1.6.1.P**. A = **(0, 0, 1/2)** in reciprocal fractional coordinates of the VASP cell.
The full little group has 48 operations, with 24 unitary operations. The full corepresentations have dimensions 2, 2, and 4.

## Unitary Bloch character table

Rows are full irreducible corepresentations restricted to the unitary subgroup. Values in this display are rounded to 10 decimal places before identical columns are combined; all matching uses the unrounded full arrays. These column groups are not asserted to be conjugacy classes.

| Corep | Dimension | g001 | g002 | g005 | g014 | g015 |
| --- | --- | --- | --- | --- | --- | --- |
| Count | - | 1 | 18 | 2 | 1 | 2 |
| r1 | 2 | 2 | 0 | -2 | -2i | 2i |
| r2 | 2 | 2 | 0 | -2 | 2i | -2i |
| r3 | 4 | 4 | 0 | 2 | 0 | 0 |

Column membership:

- g001: g001
- g002: g002, g003, g004, g007, g008, g009, g010, g011, g012, g013, g017, g018, g019, g020, g021, g022, g023, g024
- g005: g005, g006
- g014: g014
- g015: g015, g016

## Band assignments

Energies refer to the source SCF Fermi level, 6.192000 eV. Band indices are one-based.

| Bands | Dimension | Mean E-EF (eV) | Corep | Character error | Leakage | Product error |
| --- | --- | --- | --- | --- | --- | --- |
| 1-4 | 4 | -8.83464133 | r3 | 6.719e-10 | 7.180e-05 | 1.037e-08 |
| 5-8 | 4 | -4.73355958 | r3 | 6.495e-09 | 4.758e-04 | 4.903e-07 |
| 9-10 | 2 | -3.66404032 | r1 | 4.944e-09 | 7.032e-05 | 6.994e-09 |
| 11-14 | 4 | -3.41296141 | r3 | 4.467e-09 | 1.449e-04 | 4.216e-08 |
| 15-16 | 2 | -3.06701784 | r2 | 3.928e-10 | 1.986e-05 | 5.578e-10 |
| 17-18 | 2 | -2.87183680 | r1 | 2.737e-09 | 5.233e-05 | 3.873e-09 |
| 19-22 | 4 | -2.71868765 | r3 | 5.507e-09 | 9.936e-04 | 2.241e-06 |
| 23-26 | 4 | -2.59744670 | r3 | 1.089e-08 | 8.698e-04 | 1.677e-06 |
| 27-30 | 4 | -2.15787678 | r3 | 2.181e-07 | 7.216e-04 | 1.065e-06 |
| 31-32 | 2 | -2.14009458 | r2 | 2.258e-07 | 4.754e-04 | 3.207e-07 |
| 33-34 | 2 | -2.07268765 | r2 | 2.712e-09 | 5.208e-05 | 3.836e-09 |
| 35-36 | 2 | -1.78192727 | r1 | 7.201e-10 | 2.691e-05 | 1.025e-09 |
| 37-40 | 4 | -1.64477811 | r3 | 6.219e-09 | 4.605e-04 | 4.829e-07 |
| 41-42 | 2 | -1.43361843 | r2 | 1.693e-10 | 1.301e-05 | 2.400e-10 |
| 43-46 | 4 | -1.15680497 | r3 | 1.037e-08 | 7.805e-04 | 1.222e-06 |
| 47-48 | 2 | -0.99682559 | r2 | 2.314e-09 | 4.811e-05 | 3.274e-09 |
| 49-52 | 4 | -0.39306286 | r3 | 1.840e-08 | 9.143e-04 | 1.683e-06 |
| 53-56 | 4 | 0.21084044 | r3 | 6.432e-09 | 6.617e-04 | 8.927e-07 |
| 57-58 | 2 | 0.90143366 | r1 | 1.219e-08 | 1.104e-04 | 1.724e-08 |

## Four-dimensional example

Bands **53-56** form **r3**, at mean E-EF = **0.21084044 eV**.
The matrices below use one common Gram-orthonormalized pseudo-wavefunction basis. Their entries depend on the VASP basis; the unitary characters do not.

### Wavefunction matrix D(g001)

```text
    1 2 3 4
1 | 1 0 0 0 |
2 | 0 1 0 0 |
3 | 0 0 1 0 |
4 | 0 0 0 1 |
```

### Wavefunction matrix D(g002)

```text
                                   1                                2                                 3                                 4
1 |                     0.281566326i       0.020375543 - 0.959284411i -0.000544311249 - 0.000183979192i    -0.0082811068 + 0.00311969354i |
2 |      -0.020375543 - 0.959284411i                    -0.281555538i   -0.00876300936 - 0.00275602105i -0.000532129777 + 0.000213477502i |
3 | 0.000544311249 - 0.000183979192i   0.00876300936 - 0.00275602105i                      0.281684221i        0.607939977 + 0.742279091i |
4 |    0.0082811068 + 0.00311969354i 0.000532129777 + 0.000213477502i       -0.607939977 + 0.742279091i                     -0.281695009i |
```

### Wavefunction matrix D(g005)

```text
                                   1                                2                                 3                                4
1 |            0.5 + 3.05007858e-09i 3.73857909e-06 + 7.59763591e-08i   -1.85963749e-05 + 5.502205e-05i         0.3053069 + 0.810424122i |
2 |  -3.73857896e-06 + 7.597637e-08i            0.5 - 2.78330735e-09i       -0.259822771 + 0.826130646i -2.1618405e-05 - 5.38909759e-05i |
3 | 1.85963766e-05 + 5.50220551e-05i       0.259822771 + 0.826130646i             0.5 + 2.78314562e-09i 2.88583773e-06 - 2.35911569e-06i |
4 |        -0.3053069 + 0.810424122i 2.16184032e-05 - 5.38909714e-05i -2.88583792e-06 - 2.35911583e-06i            0.5 - 3.05059597e-09i |
```

### Wavefunction matrix D(g014)

```text
                                    1                                2                                 3                                4
1 |                       0.47191255i 3.27756189e-06 - 0.000154505714i        0.23902574 + 0.0807916519i       0.790534196 - 0.297813873i |
2 | -3.27756189e-06 - 0.000154505714i                     0.472920733i        -0.80531608 - 0.253276471i      0.234173151 - 0.0939444876i |
3 |       -0.23902574 + 0.0807916519i        0.80531608 - 0.253276471i                     -0.472922888i 9.33534577e-05 + 0.000113975548i |
4 |       -0.790534196 - 0.297813873i     -0.234173151 - 0.0939444876i -9.33534577e-05 + 0.000113975548i                    -0.471910395i |
```

### Sparse theoretical matrix for comparison

The r3 matrix D(g005) in the theoretical basis is diagonal:

```text
                     1                  2                  3                  4
1 | 0.5 + 0.866025404i                  0                  0                  0 |
2 |                  0 0.5 - 0.866025404i                  0                  0 |
3 |                  0                  0 0.5 - 0.866025404i                  0 |
4 |                  0                  0                  0 0.5 + 0.866025404i |
```

Its entries agree numerically with exp(+/- i*pi/3). This special-number identification does not turn the full numerical construction into a symbolic derivation.

## Operation and gauge metadata

The spatial action is x -> R x + tau. The spin lift used on WAVECAR is W @ SU2_database @ W.dagger, with W recorded in `symmetry/spin_basis_mapping.npz`. The theoretical label is tied to that lift convention.

- g002: R = `[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]`; tau = `[0.0, 0.0, 0.0]`.
- g005: R = `[[0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 1.0]]`; tau = `[0.0, 0.0, 0.0]`.
- g014: R = `[[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]`; tau = `[0.0, 0.0, 0.0]`.
