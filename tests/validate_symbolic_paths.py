"""Compare general symbolic paths with independent numerical point calculations."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'SSGReps'))
import SSGReps as native
from ssg_database import load_database
from ssg_symbolic import SymbolicPath, parse_expression

CASES = [
    ('cubic_MR', '221.8.6.3', ['1/2', '1/2', 't'], ('0', '1/2')),
    ('interior_X', '221.8.6.3', ['t', '0', '0'], ('0', '1')),
    ('quadratic', '221.8.6.3', ['t**2', '0', '0'], ('0', '1')),
    ('hexagonal_axis', '194.1.6.1.P', ['0', '0', 't'], ('0', '1/2')),
    ('hexagonal_edge', '194.1.6.1.P', ['1/3', '1/3', 't'], ('0', '1/2')),
    ('general_curve', '194.1.6.1.P', ['137/1000+t/10', '219/1000+t**2/10', '317/1000+t/11'], ('0', '1/2')),
    ('centered', '63.1.4.67', ['0', '0', 't'], ('0', '1/2')),
    ('rhombohedral', '146.1.3.1', ['t', 't', 't'], ('0', '1/2')),
    ('index12', '184.12.2.9', ['t', '0', '0'], ('0', '1/2')),
    ('index11', '173.11.1.1', ['1/7', '2/11', 't'], ('0', '1/2')),
    ('index10', '194.10.4.17', ['t', 't', '0'], ('0', '1/2')),
    ('index9', '194.9.4.19', ['t', '0', '0'], ('0', '1/2')),
    ('face_centered', '225.1.12.1', ['t', 't', 't'], ('0', '1/2')),
    ('body_centered', '229.1.12.1', ['t', '0', '0'], ('0', '1/2')),
    ('trigonometric', '63.1.4.67', ['0', '0', 'sin(t)/5'], ('0', '2*pi')),
    ('constant_R_16D', '221.8.6.3', ['1/2', '1/2', '1/2'], ('0', '1')),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'validation/symbolic_paths_general.json')
    args = parser.parse_args()
    database = load_database()
    results = []
    for name, ssg, coordinates, interval in CASES:
        started = time.monotonic()
        family = SymbolicPath(ssg, coordinates, interval=interval, seed=19, database=database)
        comparisons = []
        for expression in ['1/7', 'sqrt(2)/5']:
            value = parse_expression(expression)
            group = native.load_little_group(ssg, family._k_numeric(value), False,
                                             'double', database[ssg], seed=31)
            if group.oplist != family.global_indices:
                raise RuntimeError('Native little group mismatch: ' + name)
            unit = np.flatnonzero(np.array(group.time_reversal) > 0)
            characters = [np.trace(family.numeric(label, expression)[unit], axis1=1, axis2=2) for label in family.labels]
            cost = np.array([[np.max(abs(a-b)) for b in group.linear_character] for a in characters])
            rows, cols = linear_sum_assignment(cost)
            error = float(np.max(cost[rows, cols]))
            if cost.shape[0] != cost.shape[1] or error > 1e-8:
                raise RuntimeError('Native character comparison failed: ' + name)
            comparisons.append(dict(parameter=expression, character_set_residual=error))
        item = dict(case=name, ssg=ssg, k=coordinates, interval=interval,
                    order=len(family.global_indices), dimensions=family.reference_group.rep_degree,
                    reference_parameter=str(family.reference_parameter),
                    exceptional_points=family.exceptional_points,
                    finite_exception_list_complete=family.exceptions_fully_enumerated,
                    validation=family.validation, independent_point_comparisons=comparisons,
                    elapsed_seconds=round(time.monotonic()-started, 3), passed=True)
        results.append(item)
        print(f"PASS {name}: order={item['order']}, dimensions={item['dimensions']}, "
              f"time={item['elapsed_seconds']}s", flush=True)
    report = dict(passed=True, paths=len(results), ssgs=len({r['ssg'] for r in results}),
                  independent_point_comparisons=sum(len(r['independent_point_comparisons']) for r in results),
                  parameter_dependence='symbolic', constant_matrices='floating-point',
                  cases=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
