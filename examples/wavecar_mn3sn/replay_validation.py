"""Recheck cached Mn3Sn matrices and labels without VASP or a WAVECAR reader.

This recomputes unitary traces, products, unitarity, and label decisions. Leakage
is taken from the original coefficient calculation; the omitted WAVECAR files
are needed to recompute it. A passing replay includes correct rejection of the
recorded unresolved blocks; it does not imply all physical band blocks pass.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "SSGReps"))
from ssg_wavecar import validate_unitary_block, match_unitary_characters
from ssg_output import aligned_table


def decode(value):
    if set(value) == {"real", "imag"}:
        return complex(value["real"], value["imag"])
    return value


def read_json(path):
    return json.loads(path.read_text(), object_hook=decode)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def replay(results):
    """Return recomputed counts, or fail on a changed record or decision."""
    results = Path(results)
    summary = read_json(results / "summary.json")
    rows, unique, points = [], {}, set()
    count = passed = 0
    example = None
    for stage in ("probe", "bands"):
        directory = results / ("analysis_" + stage)
        report = read_json(directory / "validation.json")
        require(report["uses_antiunitary_wavefunction_data"] is False, "Unexpected antiunitary matching data.")
        thresholds = {key: report["thresholds"][key]
                      for key in ("character_tol", "leakage_tol", "matrix_tol")}
        for point in report["points"]:
            ik = point["k_index"]
            blocks = [r for r in report["blocks"] if r["k_index"] == ik]
            successes = 0
            with np.load(directory / f"k{ik:03d}_matrices.npz") as cache:
                refs = cache["unitary_references"]
                np.testing.assert_allclose(cache["k"], point["k"], atol=1e-12, rtol=0)
                require(refs.shape[1] == point["unitary_order"], "Unitary subgroup size changed.")
                for i, record in enumerate(blocks):
                    matrices = cache[f"block_{i}_unitary_matrices"]
                    bands = cache[f"block_{i}_bands"]
                    require(bands[0] == record["band_start"] and bands[-1] == record["band_end"], "Band indices changed.")
                    assessment = validate_unitary_block(
                        matrices, refs, cache["unitary_product_indices"], cache["unitary_factors"],
                        record["leakage"], **thresholds)
                    for key in ("characters", "fit_coefficients", "character_residual",
                                "unitarity_residual", "product_residual"):
                        np.testing.assert_allclose(assessment[key], record[key], atol=1e-11, rtol=1e-8,
                                                   err_msg=f"{stage} k{ik} bands {bands[0]}-{bands[-1]}: {key}")
                    require(assessment["passed"] == record["passed"], "Acceptance decision changed.")
                    require(assessment["label"] == record["label"], "Corepresentation label changed.")
                    key_k = tuple(np.round(point["k"], 12))
                    key = (key_k, int(bands[0]), int(bands[-1]))
                    if key in unique:
                        require(unique[key] == assessment["passed"], "Repeated point has different acceptance.")
                    unique[key] = assessment["passed"]
                    points.add(key_k)
                    successes += assessment["passed"]
                    if stage == "probe" and ik == 4 and int(bands[0]) == summary["A_point_example"]["band_start"]:
                        example = (matrices.copy(), refs.copy(), thresholds)
                require(successes == point["passed"] and len(blocks) == point["blocks"], "Per-point counts changed.")
            rows.append([stage, ik, f"{successes}/{len(blocks)}"])
            passed += successes
            count += len(blocks)
    require(example is not None, "Missing measured 4D A-point example.")
    matrices, refs, thresholds = example
    partial = matrices[:, :2, :2]
    rejected = False
    try:
        match_unitary_characters(np.trace(partial, axis1=1, axis2=2), refs,
                                 atol=thresholds["character_tol"], rtol=0)
    except ValueError:
        rejected = True
    partial_error = float(np.max(np.linalg.norm(
        partial.conj().transpose(0, 2, 1) @ partial - np.eye(2), axis=(1, 2))))
    negative = read_json(results / "incomplete_block_test.json")
    require(rejected and negative["rejected_by_characters"], "Incomplete multiplet was accepted.")
    np.testing.assert_allclose(partial_error, negative["unitarity_residual"], atol=1e-11, rtol=0)
    counts = dict(blocks_passed=passed, blocks_total=count, blocks_unresolved=count-passed,
                  distinct_kpoints=len(points), distinct_blocks_total=len(unique),
                  distinct_blocks_passed=sum(unique.values()), point_evaluations=len(rows))
    for key, value in counts.items():
        require(summary[key] == value, f"Summary mismatch: {key}.")
    return dict(replay_passed=True, **counts, all_physical_blocks_passed=passed == count,
                incomplete_4D_block_rejected=rejected, uses_antiunitary_wavefunction_data=False,
                leakage_source="Recorded coefficient calculation; not recomputed by this replay.",
                per_point=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--json", type=Path, help="Optional destination for the replay audit.")
    args = parser.parse_args()
    audit = replay(args.results)
    print(aligned_table(["stage", "k index", "accepted/blocks"], audit["per_point"]))
    print(f"PASS: reproduced {audit['blocks_passed']}/{audit['blocks_total']} accepted blocks; "
          f"{audit['blocks_unresolved']} remain UNRESOLVED.")
    print(f"Distinct points: {audit['distinct_kpoints']}; "
          f"distinct accepted blocks: {audit['distinct_blocks_passed']}/{audit['distinct_blocks_total']}.")
    print("PASS: incomplete measured 4D block rejected. Leakage uses the recorded coefficient calculation.")
    if args.json:
        args.json.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
