"""Validated numerical corepresentations in a unitary basis.

The reducer averages a Hermitian matrix over the semilinear regular action.
All subsequent basis changes are unitary, including conjugation on the right
for antiunitary elements. No matrix entries are rescaled independently.
"""
import numpy as np


class RepresentationError(RuntimeError):
    """Raised when a computed representation fails a mathematical check."""


def multiplication_table(group, tol=2e-7):
    """Match products modulo pure translations and certify their SU(2) signs."""
    rotations = np.asarray(group.rotC)
    translations = np.asarray(group.tauC)
    spins = np.asarray(group.spin)
    lifts = np.asarray(group.su2s)
    signs = np.asarray(group.time_reversal)
    inverse_lattice = np.linalg.inv(np.asarray(group.pure_T).T)
    n = len(signs)

    def key(r, t, s):
        frac = np.mod(np.round(inverse_lattice @ t, 7), 1)
        return (tuple(np.round(r, 7).flat), tuple(frac), tuple(np.round(s, 7).flat))

    lookup = {key(r, t, s): i for i, (r, t, s) in enumerate(zip(rotations, translations, spins))}
    if len(lookup) != n:
        raise RepresentationError('Duplicate group operations modulo pure translations.')
    table = np.empty((n, n), dtype=int)
    factors = np.empty((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            r = rotations[i] @ rotations[j]
            t = translations[i] + rotations[i] @ translations[j]
            s = spins[i] @ spins[j]
            k = lookup.get(key(r, t, s))
            # Rounded keys are only an accelerator; tolerance checks decide matches.
            if k is None:
                delta = (t - translations) @ inverse_lattice.T
                good = ((np.max(abs(rotations-r), axis=(1, 2)) < tol)
                        & (np.max(abs(spins-s), axis=(1, 2)) < tol)
                        & (np.max(abs(delta-np.rint(delta)), axis=1) < tol)
                        & (signs == signs[i]*signs[j]))
                matches = np.flatnonzero(good)
                if len(matches) != 1:
                    raise RepresentationError(f'Product ({i+1}, {j+1}) has no unique group element.')
                k = int(matches[0])
            delta = inverse_lattice @ (t-translations[k])
            if (np.max(abs(r-rotations[k])) > tol or np.max(abs(s-spins[k])) > tol
                    or np.max(abs(delta-np.rint(delta))) > tol or signs[k] != signs[i]*signs[j]):
                raise RepresentationError('A rounded operation key failed its geometric check.')
            product = lifts[i] @ (lifts[j] if signs[i] > 0 else lifts[j].conj())
            phase = np.vdot(lifts[k], product)/2
            sign = 1 if phase.real >= 0 else -1
            if np.linalg.norm(product-sign*lifts[k]) > tol:
                raise RepresentationError('SU(2) lifts do not close up to a sign.')
            table[i, j] = k+1
            factors[i, j] = sign if group.group_type == 'double' else 1
    if not (np.array_equal(table[0], np.arange(1, n+1))
            and np.array_equal(table[:, 0], np.arange(1, n+1))):
        raise RepresentationError('The first database operation is not the identity.')
    return table, factors


def basis_transform(matrices, signs, basis):
    """Apply one unitary basis change to the complete semilinear representation."""
    return np.asarray([basis.conj().T @ m @ (basis if s > 0 else basis.conj())
                       for m, s in zip(matrices, signs)])


def sparse_basis(matrices, signs, tol=1e-9):
    """Choose a generator eigenbasis only when it increases the number of zeros.

    This is a bounded heuristic, not simultaneous diagonalization of a
    noncommutative group. All changes preserve the complete corepresentation.
    """
    best = np.array(matrices, dtype=complex, copy=True)
    best_count = int(np.count_nonzero(abs(best) < tol))
    # Distinct early unitary operations provide inexpensive candidate bases.
    for m, sign in list(zip(matrices, signs))[:32]:
        if sign < 0:
            continue
        hermitian = (m+m.conj().T)/2 + np.sqrt(2)*(m-m.conj().T)/(2j)
        _, basis = np.linalg.eigh(hermitian)
        candidate = basis_transform(matrices, signs, basis)
        count = int(np.count_nonzero(abs(candidate) < tol))
        if count > best_count:
            best, best_count = candidate, count
    return best


def validate(lg, matrices=None, tol=1e-7, check_group=False):
    """Check every matrix product, unitarity, characters, and completeness.

    The Bloch law is computed independently from spatial translation defects.
    Frobenius residuals are absolute, not divided by matrix dimension.
    """
    reps = lg.correp_matrix if matrices is None else matrices
    mt = np.asarray(lg.mul_table, dtype=int)-1
    w = np.asarray(lg.factor)
    signs = np.asarray(lg.time_reversal)
    unitary = np.flatnonzero(signs > 0)
    anti = np.flatnonzero(signs < 0)
    n = len(signs)
    kcart = np.asarray(lg.kvec) @ np.asarray([lg.b1, lg.b2, lg.b3])
    translations = np.asarray(lg.tauC)
    rotations = np.asarray(lg.rotC)
    phase = np.exp(-1j*(translations @ kcart))
    spin = np.asarray(lg.ssg.factor_su2)[np.ix_(lg.oplist, lg.oplist)]
    defects = translations[:, None, :] + np.einsum('iab,jb->ija', rotations, translations) - translations[mt]
    physical_factor = spin*np.exp(-1j*np.einsum('ijc,c->ij', defects, kcart))
    stats = dict(projective_product_max=0., bloch_product_max=0., unitarity_max=0.,
                 character_max=0., irreducibility_max=0., product_checks=0,
                 representations=len(reps), little_group_order=n)
    chars = []
    weights = []
    for ri, rep in enumerate(reps):
        rep = np.asarray(rep, dtype=complex)
        if rep.ndim != 3 or rep.shape[0] != n or rep.shape[1] != rep.shape[2] or not np.isfinite(rep).all():
            raise RepresentationError('Invalid or non-finite representation matrix array.')
        d = rep.shape[1]
        stats['unitarity_max'] = max(stats['unitarity_max'], float(np.max(np.linalg.norm(
            rep @ rep.conj().transpose(0, 2, 1)-np.eye(d), axis=(1, 2)))))
        bloch = phase[:, None, None]*rep
        for i in range(n):
            error = rep[i] @ (rep if signs[i] > 0 else rep.conj()) - w[i, :, None, None]*rep[mt[i]]
            stats['projective_product_max'] = max(stats['projective_product_max'], float(np.max(np.linalg.norm(error, axis=(1, 2)))))
            error = bloch[i] @ (bloch if signs[i] > 0 else bloch.conj()) - physical_factor[i, :, None, None]*bloch[mt[i]]
            stats['bloch_product_max'] = max(stats['bloch_product_max'], float(np.max(np.linalg.norm(error, axis=(1, 2)))))
        stats['product_checks'] += n*n
        ch = np.trace(rep, axis1=1, axis2=2)
        chi = ch[unitary]
        chars.append(chi)
        stats['character_max'] = max(stats['character_max'], float(np.max(abs(chi-lg.character[ri]))))
        weight = float(np.vdot(chi, chi).real/len(unitary))
        indicator = weight
        if len(anti):
            indicator = (np.vdot(chi, chi)+np.sum(w[anti, anti]*ch[mt[anti, anti]]))/n
        stats['irreducibility_max'] = max(stats['irreducibility_max'], float(abs(indicator-1)))
        weights.append(d*d/weight)
    gram = np.asarray(chars).conj() @ np.asarray(chars).T/len(unitary)
    offdiagonal = gram-np.diag(np.diag(gram))
    stats['inequivalence_max'] = float(np.max(abs(offdiagonal)))
    stats['completeness_error'] = float(abs(sum(weights)-len(unitary)))
    stats['cocycle_max'] = 0.
    if check_group:
        for i in range(n):
            if not np.array_equal(mt[mt[i], :], mt[i, mt]):
                raise RepresentationError('Group multiplication is not associative.')
            error = w[i, :, None]*w[mt[i], :] - w[i, mt]*(w if signs[i] > 0 else w.conj())
            stats['cocycle_max'] = max(stats['cocycle_max'], float(np.max(abs(error))))
    errors = [v for key, v in stats.items() if key.endswith('_max') or key.endswith('_error')]
    if not all(np.isfinite(e) and e <= tol for e in errors):
        raise RepresentationError('Representation validation failed: '+str(stats))
    stats['passed'] = True
    return stats


def reduce_regular(lg, seed=0, attempts=5):
    """Reduce the twisted regular action and validate final output before return."""
    mt = np.asarray(lg.mul_table, dtype=int)-1
    w = np.asarray(lg.factor)
    signs = np.asarray(lg.time_reversal)
    unitary = np.flatnonzero(signs > 0)
    anti = np.flatnonzero(signs < 0)
    n = len(signs)
    rng = np.random.default_rng(seed)
    failure = None
    for attempt in range(attempts):
        x = rng.normal(size=(n, n))+1j*rng.normal(size=(n, n))
        average = np.zeros((n, n), dtype=complex)
        for i in range(n):
            average[np.ix_(mt[i], mt[i])] += w[i, :, None]*(x if signs[i] > 0 else x.conj())*w[i, None, :].conj()
        average = (average+average.conj().T)/(2*n)
        values, vectors = np.linalg.eigh(average)
        gap_tol = 1e-8*max(1., float(np.max(abs(values))))
        bounds = np.r_[0, np.flatnonzero(np.diff(values) > gap_tol)+1, n]
        chars, reps = [], []
        for start, end in zip(bounds[:-1], bounds[1:]):
            basis = vectors[:, start:end]
            chi = np.asarray([np.einsum('ja,ja->', basis[mt[i]].conj(), w[i, :, None]*basis)
                              for i in unitary])
            if any(np.max(abs(chi-old)) < 1e-7 for old in chars):
                continue
            matrices = np.asarray([basis[mt[i]].conj().T @ (w[i, :, None]*(basis if signs[i] > 0 else basis.conj()))
                                   for i in range(n)])
            chars.append(chi)
            reps.append(matrices)
        # Stable character ordering avoids overwriting slots when keys tie.
        order = sorted(range(len(chars)), key=lambda i: (round(chars[i][0].real),
            tuple(np.round(chars[i].real, 7)), tuple(np.round(chars[i].imag, 7))))
        lg.correp_matrix = [reps[i] for i in order]
        lg.character = [chars[i] for i in order]
        lg.rep_degree = [r.shape[1] for r in lg.correp_matrix]
        lg.rep_num = len(reps)
        lg.torsion = [int(round(float(np.vdot(c, c).real/len(unitary)))) if len(anti) else 0 for c in lg.character]
        try:
            validate(lg)
            if lg.need_beautify:
                lg.correp_matrix = [sparse_basis(r, signs) for r in lg.correp_matrix]
            # Recompute exported characters from the final matrices.
            lg.character = [np.trace(r[unitary], axis1=1, axis2=2) for r in lg.correp_matrix]
            kcart = np.asarray(lg.kvec) @ np.asarray([lg.b1, lg.b2, lg.b3])
            phase = np.exp(-1j*np.asarray(lg.tauC)[unitary] @ kcart)
            lg.linear_character = [phase*c for c in lg.character]
            lg.validation = validate(lg)
            lg.validation['seed'] = seed
            lg.validation['attempts'] = attempt+1
            return
        except RepresentationError as exc:
            failure = exc
    raise RepresentationError(f'Regular reduction failed after {attempts} attempts: {failure}')
