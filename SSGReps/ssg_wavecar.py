"""Spinor coefficient actions and unitary-character corepresentation matching.

The reader-independent kernel is exercised by the Mn3Sn material example.
It does not parse WAVECAR files or determine a material-to-database mapping.
Fractional real-space coordinates are columns: x -> R @ x + tau.
Reciprocal coefficients G and k are rows, so q -> s * q @ inv(R).
Coefficients have shape (number_of_G_vectors, 2, number_of_bands).
All operations in a comparison must use the same orthonormalized subspace.
"""
import numpy as np


def orthonormal_basis(coefficients, relative_rank_tol=1e-12):
    """Return a common Euclidean-orthonormal basis of a pseudo-state subspace.

    An equivariant, injective PAW map preserves representation equivalence.
    This changes the basis in the pseudo-state subspace; it does not reconstruct
    all-electron matrix elements or certify the PAW map for a given material.
    """
    if not np.isfinite(relative_rank_tol) or not 0 <= relative_rank_tol < 1:
        raise ValueError("relative_rank_tol must be finite and in [0, 1).")
    coefficients = np.asarray(coefficients, dtype=complex)
    if (coefficients.ndim != 3 or coefficients.shape[1] != 2
            or min(coefficients.shape) == 0 or not np.isfinite(coefficients).all()):
        raise ValueError("Expected finite coefficients with shape (nG, 2, nBands).")
    c = coefficients.reshape(-1, coefficients.shape[-1])
    gram = c.conj().T @ c
    values, vectors = np.linalg.eigh(gram)
    if values[-1] <= 0 or values[0] <= relative_rank_tol * values[-1]:
        raise ValueError("The selected coefficient subspace is rank deficient or ill-conditioned.")
    inverse_root = (vectors / np.sqrt(values)) @ vectors.conj().T
    q = c @ inverse_root
    return q.reshape(coefficients.shape), {
        "gram_condition": float(values[-1] / values[0]),
        "orthonormality_residual": float(np.linalg.norm(q.conj().T @ q - np.eye(q.shape[1]))),
    }


def transform_coefficients(coefficients, g_vectors, k, rotation, translation,
                           spin_matrix, antiunitary=False, tolerance=1e-8):
    """Apply the forward active operator V K^a psi(R^-1 (x-tau)).

    spin_matrix is the complete unitary matrix part, including the spin action
    of time reversal for an antiunitary operation. It is never conjugate-transposed
    implicitly. The selected G set must be closed under this operation.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative.")
    c = np.asarray(coefficients, complex)
    g = np.asarray(g_vectors, float)
    k = np.asarray(k, float)
    r = np.asarray(rotation, float)
    tau = np.asarray(translation, float)
    v = np.asarray(spin_matrix, complex)
    if (c.ndim != 3 or c.shape[1] != 2 or min(c.shape) == 0
            or g.shape != (c.shape[0], 3)):
        raise ValueError("Inconsistent coefficient or reciprocal-vector shapes.")
    if k.shape != (3,) or tau.shape != (3,) or r.shape != (3, 3) or v.shape != (2, 2):
        raise ValueError("Invalid operation shapes.")
    if not all(np.isfinite(x).all() for x in (c, g, k, r, tau, v)):
        raise ValueError("Inputs must be finite.")
    if not isinstance(antiunitary, (bool, np.bool_)):
        raise ValueError("antiunitary must be a boolean.")
    if np.max(abs(g - np.rint(g))) > tolerance:
        raise ValueError("G vectors must be integer reciprocal coefficients.")
    if np.linalg.norm(v.conj().T @ v - np.eye(2)) > tolerance:
        raise ValueError("The spin matrix must be unitary.")
    index = {tuple(row): i for i, row in enumerate(np.rint(g).astype(int))}
    if len(index) != len(g):
        raise ValueError("Duplicate G vectors.")
    q_out = (-1 if antiunitary else 1) * ((g + k) @ np.linalg.inv(r))
    g_out = q_out - k
    if np.max(abs(g_out - np.rint(g_out))) > tolerance:
        raise ValueError("Operation does not preserve this k point modulo the reciprocal lattice.")
    try:
        destination = np.array([index[tuple(row)] for row in np.rint(g_out).astype(int)])
    except KeyError as error:
        raise ValueError("The plane-wave set is not closed under the operation.") from error
    if len(np.unique(destination)) != len(g):
        raise ValueError("The reciprocal-vector action is not a permutation.")
    phase = np.exp(-2j * np.pi * (q_out @ tau))
    source = c.conj() if antiunitary else c
    out = np.empty_like(c)
    out[destination] = phase[:, None, None] * np.einsum("ab,gbn->gan", v, source)
    return out


def representation_matrix(basis, g_vectors, k, rotation, translation, spin_matrix,
                          antiunitary=False, tolerance=1e-8):
    """Extract one matrix and reject a nonclosed band subspace.

    For antiunitary elements, the resulting coefficient-space operator is B K.
    This local check must be supplemented by group-product and equivalence tests.
    """
    q = np.asarray(basis, complex)
    transformed = transform_coefficients(q, g_vectors, k, rotation, translation,
                                         spin_matrix, antiunitary=antiunitary,
                                         tolerance=tolerance)
    q_flat = q.reshape(-1, q.shape[-1])
    y = transformed.reshape(q_flat.shape)
    if np.linalg.norm(q_flat.conj().T @ q_flat - np.eye(q.shape[-1])) > tolerance:
        raise ValueError("Use one common orthonormal_basis for the selected band block first.")
    b = q_flat.conj().T @ y
    leakage = float(np.linalg.norm(y - q_flat @ b) / np.linalg.norm(y))
    unitarity = float(np.linalg.norm(b.conj().T @ b - np.eye(b.shape[0])))
    if max(leakage, unitarity) > tolerance:
        raise ValueError(f"Subspace check failed: leakage={leakage:.3e}, unitarity={unitarity:.3e}")
    return b, {"leakage": leakage, "unitarity_residual": unitarity}


def match_unitary_characters(observed, references, labels=None, atol=1e-6, rtol=1e-6):
    """Return integer corepresentation multiplicities or reject the match.

    Supply all unitary operations, identity first, in matching order. References
    are restrictions of full irreducible corepresentations in the same lift and
    factor-system convention. Rounded or compressed display columns are invalid
    inputs. No antiunitary wavefunction traces are used.

    The observed band block must carry a representation of the specified full
    group. Unitary-subgroup closure alone does not establish that assumption.
    Restriction character norms are included, so doubled and paired Wigner
    cases are not mistakenly counted as two or four copies.
    """
    observed = np.asarray(observed, dtype=complex)
    references = np.asarray(references, dtype=complex)
    if (observed.ndim != 1 or references.ndim != 2 or min(references.shape) == 0
            or references.shape[1] != len(observed)):
        raise ValueError("Expected observed (nH,) and references (nCoreps, nH).")
    if not np.isfinite(observed).all() or not np.isfinite(references).all():
        raise ValueError("Characters must be finite.")
    if not (np.isfinite(atol) and np.isfinite(rtol)) or min(atol, rtol) < 0:
        raise ValueError("Tolerances must be finite and nonnegative.")
    labels = list(labels) if labels is not None else [f"r{i+1}" for i in range(len(references))]
    if len(labels) != len(references) or len(set(labels)) != len(labels):
        raise ValueError("Provide one unique label per reference corepresentation.")
    dimensions = np.rint(references[:, 0].real).astype(int)
    if np.any(dimensions < 1) or not np.allclose(references[:, 0], dimensions, atol=atol, rtol=rtol):
        raise ValueError("The first operation must be the identity with positive integer characters.")
    gram = references.conj() @ references.T
    norms = np.diag(gram).real
    if np.any(norms <= 0):
        raise ValueError("Reference character norms must be positive.")
    normalized_gram = gram / np.sqrt(norms[:, None] * norms[None, :])
    if not np.allclose(normalized_gram, np.eye(len(references)), atol=atol, rtol=rtol):
        raise ValueError("Reference restrictions are not mutually orthogonal corepresentation characters.")
    coefficients = (references.conj() @ observed) / norms
    multiplicities = np.rint(coefficients.real).astype(int)
    if np.any(multiplicities < 0) or not np.allclose(coefficients, multiplicities, atol=atol, rtol=rtol):
        raise ValueError("No nonnegative integer corepresentation decomposition; check block completeness and conventions.")
    reconstructed = multiplicities @ references
    if not np.allclose(observed, reconstructed, atol=atol, rtol=rtol):
        raise ValueError("The full unitary character vector is not reproduced by the candidate decomposition.")
    return {
        "multiplicities": {label: int(m) for label, m in zip(labels, multiplicities)},
        "dimension": int(multiplicities @ dimensions),
        "unitary_restriction_norms": {label: float(n / len(observed)) for label, n in zip(labels, norms)},
        "maximum_character_residual": float(np.max(abs(observed - reconstructed))),
        "unitary_operations": len(observed),
        "uses_antiunitary_wavefunction_data": False,
    }


def validate_unitary_block(matrices, references, products, factors, leakage,
                           labels=None, character_tol=0.002, leakage_tol=0.005,
                           matrix_tol=0.002):
    """Check a measured unitary block, retaining diagnostics on failed matches.

    products[i,j] is the zero-based index of the representative of h_i h_j;
    factors[i,j] is its Bloch factor, including spin-lift and translation signs.
    leakage is the maximum relative norm outside this band subspace, measured
    from the original coefficients. It cannot be recovered from matrices alone.
    Malformed inputs raise ValueError; a physically unresolved block returns
    passed=False and label='UNRESOLVED'. A successful unitary check assumes the
    physical subspace belongs to the specified full symmetry group.
    """
    b = np.asarray(matrices, complex)
    ref = np.asarray(references, complex)
    product_input = np.asarray(products)
    factor = np.asarray(factors, complex)
    if (b.ndim != 3 or min(b.shape) == 0 or b.shape[1] != b.shape[2]
            or not np.isfinite(b).all()):
        raise ValueError("Expected finite matrices with shape (nH, dimension, dimension).")
    n = len(b)
    if (product_input.shape != (n, n)
            or not np.issubdtype(product_input.dtype, np.integer)
            or np.any(product_input < 0) or np.any(product_input >= n)):
        raise ValueError("products must be an integer (nH, nH) index array.")
    product = product_input.astype(int)
    if factor.shape != (n, n) or not np.isfinite(factor).all():
        raise ValueError("Expected finite factors with shape (nH, nH).")
    if not np.allclose(abs(factor), 1, atol=1e-8, rtol=0):
        raise ValueError("Bloch factors must have unit modulus.")
    if (ref.ndim != 2 or min(ref.shape) == 0 or ref.shape[1] != n
            or not np.isfinite(ref).all()):
        raise ValueError("Expected finite references with shape (nCoreps, nH).")
    tolerances = np.array([character_tol, leakage_tol, matrix_tol, leakage], float)
    if not np.isfinite(tolerances).all() or np.any(tolerances < 0):
        raise ValueError("Residuals and tolerances must be finite and nonnegative.")
    characters = np.trace(b, axis1=1, axis2=2)
    norms = np.sum(abs(ref)**2, axis=1)
    if np.any(norms == 0):
        raise ValueError("Reference character norms must be positive.")
    fit = ref.conj() @ characters / norms
    nearest = np.rint(fit.real).astype(int)
    character_error = float(np.max(abs(characters - nearest @ ref)))
    unitarity = float(np.max(np.linalg.norm(
        b.conj().transpose(0, 2, 1) @ b - np.eye(b.shape[1]), axis=(1, 2))))
    product_error = max(float(np.max(np.linalg.norm(
        b[i] @ b - factor[i, :, None, None] * b[product[i]], axis=(1, 2))))
        for i in range(n))
    result = dict(passed=False, label="UNRESOLVED", failure="", match=None,
                  characters=characters, fit_coefficients=fit,
                  character_residual=character_error, leakage=float(leakage),
                  unitarity_residual=unitarity, product_residual=product_error,
                  uses_antiunitary_wavefunction_data=False)
    try:
        match = match_unitary_characters(characters, ref, labels, atol=character_tol, rtol=0)
        if leakage > leakage_tol:
            raise ValueError("Band subspace leakage exceeds tolerance.")
        if max(unitarity, product_error) > matrix_tol:
            raise ValueError("Unitary matrix validation exceeds tolerance.")
        result.update(passed=True, match=match, label=" + ".join(
            (str(m) + "*" if m > 1 else "") + name
            for name, m in match["multiplicities"].items() if m))
    except ValueError as error:
        result["failure"] = str(error)
    return result
