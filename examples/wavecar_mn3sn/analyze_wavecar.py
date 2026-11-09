"""Validate Mn3Sn WAVECAR blocks against SSGReps using unitary characters.

This material adapter requires audited fractional coordinates, identity cell
mapping, and a precomputed common spin-basis transformation. It is not an
automatic structure-to-SSG interface for arbitrary materials.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def energy_blocks(energies, gap):
    """Build adjacent-gap candidate blocks; symmetry checks decide acceptance."""
    energies = np.asarray(energies, float)
    if (energies.ndim != 1 or len(energies) == 0 or not np.isfinite(energies).all()
            or not np.isfinite(gap) or gap <= 0 or np.any(np.diff(energies) < -1e-8)):
        raise ValueError("Expected sorted finite energies and a positive finite gap.")
    starts = [0] + (np.flatnonzero(np.diff(energies) > gap) + 1).tolist()
    return [np.arange(a, b) for a, b in zip(starts, starts[1:] + [len(energies)])]


def select_blocks(energies, gap, last_band):
    """Omit entire blocks crossing the requested or final WAVECAR boundary."""
    if last_band < 1:
        raise ValueError("last_band must be positive.")
    selected = [b for b in energy_blocks(energies, gap)
                if b[-1] < min(last_band, len(energies) - 1)]
    if not selected:
        raise ValueError("No complete candidate block remains below the band boundary.")
    return selected


def check_material_inputs(run, stage, wave):
    """Reject input files outside this audited NSOC, identity-cell example."""
    values = {}
    for line in (run / stage / "INCAR").read_text().splitlines():
        line = line.split("!")[0].split("#")[0]
        for item in line.split(";"):
            if "=" in item:
                key, value = item.split("=", 1)
                values[key.strip().upper()] = value.strip().upper()
    if values.get("LNONCOLLINEAR") not in {".TRUE.", "T", "TRUE"}:
        raise ValueError("This example requires LNONCOLLINEAR=T.")
    if values.get("LSORBIT") not in {".FALSE.", "F", "FALSE"}:
        raise ValueError("This audited example requires explicit LSORBIT=F.")
    if values.get("SAXIS", "0 0 1").split() != ["0", "0", "1"]:
        if not np.allclose(np.fromstring(values["SAXIS"], sep=" "), [0, 0, 1]):
            raise ValueError("This example supports SAXIS=(0,0,1) only.")
    audit = json.loads((run / "symmetry/mapping_validation.json").read_text())
    expected_hash = audit.get("input_poscar_sha256")
    if expected_hash is None:
        execution = json.loads((run / "execution_validation.json").read_text())
        expected_hash = execution["stages"]["probe"]["input_hashes"]["POSCAR"]
    poscar = run / stage / "POSCAR"
    if hashlib.sha256(poscar.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("POSCAR differs from the audited spin/position mapping; rerun align_spin_basis.py.")
    lines = poscar.read_text().splitlines()
    scale = float(lines[1])
    if scale <= 0:
        raise ValueError("This example requires a positive POSCAR lattice scale.")
    lattice = scale * np.array([[float(v) for v in line.split()] for line in lines[2:5]])
    if not np.allclose(wave._Acell, lattice, atol=1e-8, rtol=0):
        raise ValueError("WAVECAR lattice differs from the audited POSCAR.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--stage", choices=["probe", "bands"], default="probe")
    parser.add_argument("--kpoints", type=int, nargs="+")
    parser.add_argument("--last-band", type=int, default=60)
    parser.add_argument("--gap", type=float, default=0.002)
    parser.add_argument("--character-tol", type=float, default=0.002)
    parser.add_argument("--leakage-tol", type=float, default=0.005)
    parser.add_argument("--matrix-tol", type=float, default=0.002)
    parser.add_argument("--output", type=Path, help="Report directory; defaults to RUN/analysis_STAGE.")
    parser.add_argument("--require-all", action="store_true",
                        help="Return exit code 2 after saving reports if any block is unresolved.")
    args = parser.parse_args()
    for name in ("gap", "character_tol", "leakage_tol", "matrix_tol"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name + " must be finite and positive.")
    if args.last_band < 1:
        parser.error("--last-band must be positive.")
    run = args.run.resolve()
    sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "SSGReps"),
                   str(args.reader.resolve())]
    import SSGReps as native
    from ssg_database import load_database
    from ssg_output import from_little_group, character_text, matrix_text, matrix_view, aligned_table
    from ssg_wavecar import orthonormal_basis, transform_coefficients, validate_unitary_block
    from vaspwfc import vaspwfc

    provenance = json.loads((run / "provenance.json").read_text())
    identification = json.loads((run / "symmetry/identified_operations.json").read_text())
    ssg = identification["ssg"]
    if ssg != "194.1.6.1.P":
        raise ValueError("This material adapter supports SSG 194.1.6.1.P only.")
    database = load_database()
    entry = database[ssg]
    mapping = np.load(run / "symmetry/spin_basis_mapping.npz")
    W = mapping["W"]
    if W.shape != (2, 2) or np.linalg.norm(W.conj().T @ W - np.eye(2)) >= 1e-10:
        raise ValueError("Invalid common spin-basis transformation.")
    if not (np.allclose(entry["superCell"], np.eye(3), atol=1e-8, rtol=0)
            and np.allclose(identification["operations"]["transformation_matrix"], np.eye(3), atol=1e-8, rtol=0)
            and np.allclose(identification["operations"]["original_shift"], 0, atol=1e-8, rtol=0)):
        raise ValueError("This adapter requires the audited identity cell/origin mapping.")
    wave = vaspwfc(str(run / args.stage / "WAVECAR"), lsorbit=True)
    if wave._nspin != 1:
        raise ValueError("Expected noncollinear spinor WAVECAR with one spin channel.")
    check_material_inputs(run, args.stage, wave)
    indices = args.kpoints or list(range(1, wave._nkpts + 1))
    if len(set(indices)) != len(indices) or any(i < 1 or i > wave._nkpts for i in indices):
        raise ValueError("Provide unique one-based k indices within the WAVECAR.")
    out = args.output.resolve() if args.output else run / ("analysis_" + args.stage)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    point_summaries = []
    report = [f"SSG {ssg}: unitary-character WAVECAR validation",
              "Rows label full corepresentations; character columns contain unitary operations only.",
              f"Stage: {args.stage}; source SCF Fermi energy: {provenance['source_fermi_eV']:.6f} eV",
              f"Candidate block gap: {args.gap} eV; final band boundary: {args.last_band}",
              f"Tolerances: character={args.character_tol}, leakage={args.leakage_tol}, matrix={args.matrix_tol}",
              "Antiunitary wavefunction matrices and traces are not extracted."]
    for ik in indices:
        k = wave._kvecs[ik - 1]
        energies = wave._bands[0, ik - 1]
        lg = native.load_little_group(ssg, k, True, "double", entry, seed=19)
        signs = np.asarray(lg.time_reversal)
        unit = np.flatnonzero(signs > 0)
        phase = np.exp(-2j * np.pi * (np.asarray(lg.tauC) @ k))
        references = np.array([np.trace(np.asarray(m)[unit], axis1=1, axis2=2)
                               * phase[unit] for m in lg.correp_matrix])
        labels = [f"r{i+1}" for i in range(len(references))]
        permutations = np.asarray(lg.mul_table, dtype=int) - 1
        local = {int(g): j for j, g in enumerate(unit)}
        products = np.array([[local[int(permutations[g, h])] for h in unit] for g in unit])
        factors = np.array([[lg.factor[g, h] * phase[g] * phase[h] / phase[permutations[g, h]]
                             for h in unit] for g in unit])
        gvec = wave.gvectors(ikpt=ik)
        selected = select_blocks(energies, args.gap, args.last_band)
        count = max(b[-1] for b in selected) + 1
        raw = np.array([wave.readBandCoeff(ispin=1, ikpt=ik, iband=i+1, norm=False)
                        for i in range(count)])
        if raw.shape != (count, 2 * len(gvec)):
            raise ValueError("Coefficient count does not match the reader's reciprocal-vector ordering.")
        coefficients = raw.reshape(count, 2, len(gvec)).transpose(2, 1, 0)
        blocks = []
        for band in selected:
            c = coefficients[:, :, band]
            q, gram_checks = orthonormal_basis(c)
            cflat, qflat = c.reshape(-1, len(band)), q.reshape(-1, len(band))
            root = np.linalg.solve(cflat.conj().T @ cflat, cflat.conj().T @ qflat)
            blocks.append(dict(band=band, q=qflat, root=root, matrices=[],
                               leakage=0.0, unitarity=0.0, **gram_checks))

        for g in unit:
            spin = W @ np.asarray(lg.su2s[g]) @ W.conj().T
            transformed = transform_coefficients(coefficients, gvec, k,
                                                 lg.rotC[g], lg.tauC[g], spin)
            for block in blocks:
                band = block["band"]
                y = transformed[:, :, band].reshape(-1, len(band)) @ block["root"]
                q = block["q"]
                b = q.conj().T @ y
                leakage = np.linalg.norm(y - q @ b) / np.linalg.norm(y)
                unitarity = np.linalg.norm(b.conj().T @ b - np.eye(len(band)))
                block["leakage"] = max(block["leakage"], float(leakage))
                block["unitarity"] = max(block["unitarity"], float(unitarity))
                block["matrices"].append(b)

        report += ["", f"k index {ik}: ({', '.join(f'{v:.10g}' for v in k)})",
                   f"Little group: {len(signs)} operations; unitary subgroup: {len(unit)} operations.",
                   "Theoretical Bloch character table:",
                   character_text(from_little_group(lg), compress=True, width=160)]
        summary_rows = []
        point_records = []
        arrays = {"k": k, "unitary_local_ids": unit + 1,
                  "unitary_global_ids": np.asarray(lg.oplist)[unit] + 1,
                  "unitary_references": references, "unitary_product_indices": products,
                  "unitary_factors": factors, "energies": energies}
        for bi, block in enumerate(blocks):
            band = block["band"]
            b = np.asarray(block["matrices"])
            assessment = validate_unitary_block(
                b, references, products, factors, block["leakage"], labels,
                character_tol=args.character_tol, leakage_tol=args.leakage_tol,
                matrix_tol=args.matrix_tol)
            label = assessment["label"]
            char_error = assessment["character_residual"]
            product_error = assessment["product_residual"]
            record = dict(k_index=ik, k=k.tolist(), band_start=int(band[0]+1),
                          band_end=int(band[-1]+1), dimension=len(band),
                          energy_eV=float(np.mean(energies[band])),
                          energy_relative_eV=float(np.mean(energies[band])-provenance["source_fermi_eV"]),
                          energy_spread_eV=float(np.ptp(energies[band])), **assessment)
            point_records.append(record)
            records.append(record)
            arrays[f"block_{bi}_bands"] = band + 1
            arrays[f"block_{bi}_unitary_matrices"] = b
            summary_rows.append([f"{band[0]+1}-{band[-1]+1}", len(band),
                                 f"{record['energy_relative_eV']:.6f}", label,
                                 f"{char_error:.2e}", f"{block['leakage']:.2e}",
                                 f"{product_error:.2e}"])
        report += ["", "Wavefunction blocks:", aligned_table(
            ["bands", "dim", "E-EF (eV)", "corep", "char error", "leakage", "product error"], summary_rows)]
        good = [(i, r) for i, r in enumerate(point_records) if r["passed"] and r["dimension"] > 1]
        if good:
            index, chosen = min(good, key=lambda pair: (-pair[1]["dimension"], abs(pair[1]["energy_relative_eV"])))
            report += ["", f"Example band-block matrices: bands {chosen['band_start']}-{chosen['band_end']}, {chosen['label']}.",
                       "Common Gram-orthonormalized wavefunction basis; unitary operations only."]
            from ssg_output import scalar
            report += [aligned_table(["operation", "wavefunction character"],
                       [[f"g{g+1:03d}", scalar(chosen["characters"][j], precision=9)]
                        for j, g in enumerate(unit)])]
            selected_ops = [j for j in [0, 1, 4, 13] if j < len(unit)]
            for j in selected_ops:
                report += [f"D_wave(g{unit[j]+1:03d}):",
                           matrix_text(matrix_view(blocks[index]["matrices"][j]), width=160)]
        np.savez_compressed(out / f"k{ik:03d}_matrices.npz", **arrays)
        theoretical = lg.to_dic_json()
        (out / f"k{ik:03d}_theory.json").write_text(json.dumps(theoretical, default=encode, indent=2) + "\n")
        point_summaries.append(dict(k_index=ik, k=k.tolist(), little_group_order=len(signs),
                                    unitary_order=len(unit), corep_dimensions=lg.rep_degree,
                                    theory_validation=lg.validation,
                                    blocks=len(point_records), passed=sum(r["passed"] for r in point_records)))
        print(f"k {ik} {k}: {sum(r['passed'] for r in point_records)}/{len(point_records)} blocks passed", flush=True)
        (out / "report.txt").write_text("\n".join(report) + "\n")
    result = dict(ssg=ssg, stage=args.stage, wavecar=str(run / args.stage / "WAVECAR"),
                  uses_antiunitary_wavefunction_data=False,
                  thresholds={k: v for k, v in vars(args).items() if k.endswith('tol') or k == 'gap'},
                  points=point_summaries, blocks=records,
                  blocks_passed=sum(r["passed"] for r in records), blocks_total=len(records),
                  all_blocks_passed=all(r["passed"] for r in records))
    (out / "validation.json").write_text(json.dumps(result, default=encode, indent=2) + "\n")
    fields = ["k_index", "band_start", "band_end", "dimension", "energy_relative_eV", "energy_spread_eV",
              "label", "passed", "leakage", "unitarity_residual", "product_residual", "character_residual", "failure"]
    with (out / "band_labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"Total: {result['blocks_passed']}/{result['blocks_total']} blocks passed", flush=True)
    if not result["all_blocks_passed"]:
        print("PARTIAL: unresolved blocks remain; see validation.json for diagnostics.")
        if args.require_all:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
