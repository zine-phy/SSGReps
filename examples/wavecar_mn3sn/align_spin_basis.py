"""Audit the Mn3Sn database-to-material spatial and spin basis mapping."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "SSGReps"))
    from MOM2SSG import findAllOp, search4ssg, load_ssg_list
    from MOM2SSG.poscar_io import read_poscar_no_elements
    import SSGReps as native
    from ssg_database import load_database
    from rep_utils import SU2

    cell, elements = read_poscar_no_elements(str(run / "probe/POSCAR"))
    ops = findAllOp(cell, tol=1e-4, tolm=1e-4)
    ssg = search4ssg(cell, load_ssg_list(), tol=1e-4, tolm=1e-4)
    require(ssg == "194.1.6.1.P", "This adapter supports Mn3Sn SSG 194.1.6.1.P only.")
    group = native.loadSsgGroup(ssg, [0, 0, 0], "double", load_database()[ssg])
    unit = np.flatnonzero(np.asarray(group.time_reversal) > 0)
    require(len(unit) == 24 and len(group.time_reversal) == 48, "Unexpected group orders.")
    rot, tau = np.array(ops["RotC"]), np.array(ops["TauC"])
    reflection = np.diag([1, 1, -1])
    spin = np.array([s if np.linalg.det(s) > 0 else reflection @ s for s in ops["spin"]])
    require(np.allclose(ops["transformation_matrix"], np.eye(3), atol=1e-8, rtol=0), "Nonidentity cell mapping is unsupported.")
    require(np.allclose(ops["original_shift"], 0, atol=1e-8, rtol=0), "Nonzero origin shift is unsupported.")
    indices = []
    for r, t in zip(np.asarray(group.rotC)[unit], np.asarray(group.tauC)[unit]):
        matches = [i for i in range(len(rot)) if np.max(abs(r-rot[i])) < 1e-7
                   and np.max(abs(t-tau[i]-np.rint(t-tau[i]))) < 1e-7]
        require(len(matches) == 1, "Spatial operation matching is not unique.")
        indices.append(matches[0])
    target = spin[indices]
    source = np.asarray(group.spin)[unit]

    def residual(vector):
        O = Rotation.from_rotvec(vector).as_matrix()
        return (np.einsum("ij,gjk,lk->gil", O, source, O) - target).ravel()

    fits = [least_squares(residual, v, gtol=1e-12, xtol=1e-12, ftol=1e-12)
            for v in [[0, 0, 0], [np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi]]]
    fit = min(fits, key=lambda f: np.linalg.norm(f.fun))
    O = Rotation.from_rotvec(fit.x).as_matrix()
    W = SU2(O)
    require(np.max(abs(fit.fun)) < 1e-8, "Spin-operation alignment failed.")
    require(np.linalg.det(O) > 0, "Expected a proper common spin rotation.")
    # Check the actual positions and magnetic moments, not only the operation finder.
    positions, species, moments = np.asarray(cell[1]), np.asarray(cell[2]), np.asarray(cell[3])
    position_error = 0.0
    moment_error = 0.0
    for g in unit:
        transformed = positions @ np.asarray(group.rotC[g]).T + group.tauC[g]
        physical_spin = O @ group.spin[g] @ O.T
        for i, point in enumerate(transformed):
            delta = point - positions
            errors = np.linalg.norm(delta - np.rint(delta), axis=1)
            errors[species != species[i]] = np.inf
            j = int(np.argmin(errors))
            position_error = max(position_error, float(errors[j]))
            moment_error = max(moment_error, float(np.linalg.norm(physical_spin @ moments[i] - moments[j])))
    require(position_error < 1e-6, "Position covariance failed.")
    require(moment_error < 1e-4, "Moment covariance failed.")
    output = run / "symmetry"
    output.mkdir(parents=True, exist_ok=True)
    def encode(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    (output / "identified_operations.json").write_text(json.dumps({"ssg": ssg, "operations": ops}, default=encode, indent=2) + "\n")
    np.savez(output / "spin_basis_mapping.npz", O=O, W=W, spatial_match_indices=indices,
             max_spin_residual=np.max(abs(fit.fun)))
    audit = dict(ssg=ssg, full_order=48, unitary_order=24, spin_rotation=O,
                 spin_operation_residual=float(np.max(abs(fit.fun))),
                 position_covariance_residual=position_error, moment_covariance_residual_muB=moment_error,
                 input_poscar_sha256=hashlib.sha256((run / "probe/POSCAR").read_bytes()).hexdigest())
    (output / "mapping_validation.json").write_text(json.dumps(audit, default=encode, indent=2) + "\n")
    print(json.dumps(audit, default=encode, indent=2))


if __name__ == "__main__":
    main()
