"""Symbolic one-parameter paths with validated numerical corepresentation bases.

A seeded generic sample supplies the constant numerical projective matrices.
Exact rational geometry and symbolic little-group conditions certify their
parameter-independent factor system. Bloch phases retain the input parameter.
This is not a general exact-algebraic matrix solver.
"""
import ast
import copy
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
import re
import numpy as np
import sympy as sp
from sympy.calculus.util import continuous_domain


class SymbolicPathError(ValueError):
    """An input path or its certified domain is unsupported or inconsistent."""


def parse_expression(text, parameter=None):
    """Parse arithmetic and a small real-function grammar without eval/sympify.

    Decimal literals are exact decimal rationals. Attribute access, indexing,
    arbitrary function calls, and undeclared symbols are rejected.
    """
    text = str(text).strip()
    if len(text) > 512:
        raise SymbolicPathError('An expression must contain at most 512 characters.')
    try:
        tree = ast.parse(text, mode='eval')
    except SyntaxError as error:
        raise SymbolicPathError('Invalid expression: ' + text) from error
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise SymbolicPathError('Expression is too complex.')
    functions = {'sqrt': sp.sqrt, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp}

    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(text, node).replace('_', '')
            if len(literal) > 40:
                raise SymbolicPathError('Numeric literal is too long.')
            if 'e' in literal.lower() and abs(int(literal.lower().split('e')[1])) > 1000:
                raise SymbolicPathError('Scientific-notation exponent is too large.')
            return sp.Rational(literal)
        if isinstance(node, ast.Name):
            if parameter is not None and node.id == str(parameter):
                return parameter
            if node.id == 'pi':
                return sp.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return a+b
            if isinstance(node.op, ast.Sub):
                return a-b
            if isinstance(node.op, ast.Mult):
                return a*b
            if isinstance(node.op, ast.Div):
                if b == 0:
                    raise SymbolicPathError('Division by zero in path expression.')
                return a/b
            if isinstance(node.op, ast.Pow):
                if not b.is_Rational or abs(b) > 16:
                    raise SymbolicPathError('Powers require a rational exponent with absolute value at most 16.')
                return a**b
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in functions and len(node.args) == 1 and not node.keywords):
            return functions[node.func.id](walk(node.args[0]))
        raise SymbolicPathError('Use arithmetic, the declared parameter, pi, sqrt, sin, cos, or exp only.')

    result = sp.simplify(walk(tree.body))
    if result.has(sp.nan, sp.zoo, sp.oo, -sp.oo) or result.is_real is False:
        raise SymbolicPathError('Path expressions must be finite and real on their domain.')
    if result.has(sp.Abs, sp.Piecewise, sp.sign):
        raise SymbolicPathError('Split branch-dependent or piecewise paths into separate analytic intervals.')
    return result


def _rational_matrix(values, audit):
    values = np.asarray(values, float)
    result = []
    for value in values.flat:
        rational = Fraction(float(value)).limit_denominator(4096)
        error = abs(float(rational)-value)
        if not np.isfinite(value) or error > 1e-10:
            raise SymbolicPathError('Database spatial geometry cannot be reconstructed within 1e-10 with denominator <=4096.')
        audit['maximum_reconstruction_error'] = max(audit['maximum_reconstruction_error'], float(error))
        result.append(sp.Rational(rational.numerator, rational.denominator))
    return sp.ImmutableMatrix(values.shape[0], values.shape[1], result)


def _member_status(values):
    status = [sp.simplify(value).is_integer for value in values]
    if any(value is False for value in status):
        return False
    if all(value is True for value in status):
        return True
    return None


def _check_spatial_products(lattice, rotations, translations, indices, table):
    """Certify spatial products modulo the exact reconstructed pure lattice."""
    inverse_columns = lattice.inv().T
    lattice_rotations = np.array([inverse_columns*rotations[g]*lattice.T for g in indices], dtype=int)
    lattice_translations = np.array([
        [Fraction(int(x.p), int(x.q)) for x in inverse_columns*translations[g]] for g in indices], dtype=object)
    for i in range(len(indices)):
        if not np.array_equal(lattice_rotations[i] @ lattice_rotations, lattice_rotations[table[i]]):
            raise SymbolicPathError('Reconstructed spatial rotations fail the multiplication table.')
        defects = (lattice_translations[i] + (lattice_rotations[i] @ lattice_translations.T).T
                   - lattice_translations[table[i]])
        if any(x.denominator != 1 for x in defects.flat):
            raise SymbolicPathError('A reconstructed translation defect is not a pure-lattice vector.')
    return len(indices)**2


def _finite_exceptions(values, parameter, domain):
    """Enumerate roots using a nonconstant polynomial constraint of degree <=2.

    None means that the exact integer conditions remain authoritative; no claim
    of a complete finite exceptional-point list is made for that operation.
    """
    if any(not value.has(parameter) and value.is_integer is False for value in values):
        return []
    candidates = []
    for value in values:
        try:
            polynomial = sp.Poly(value, parameter)
        except sp.PolynomialError:
            continue
        if 1 <= polynomial.degree() <= 2:
            candidates.append((polynomial.degree(), value))
    if not candidates:
        return None
    _, constraint = min(candidates, key=lambda pair: pair[0])
    endpoints = [constraint.subs(parameter, domain.start), constraint.subs(parameter, domain.end)]
    derivative_roots = sp.solve(sp.diff(constraint, parameter), parameter)
    endpoints += [constraint.subs(parameter, r) for r in derivative_roots if domain.contains(r) is sp.S.true]
    try:
        lower = int(sp.ceiling(sp.Min(*endpoints)))
        upper = int(sp.floor(sp.Max(*endpoints)))
    except (TypeError, ValueError):
        return None
    if upper-lower > 10000:
        return None
    found = set()
    for integer in range(lower, upper+1):
        roots = sp.solveset(constraint-integer, parameter, domain=domain)
        if roots is sp.S.EmptySet:
            continue
        if not isinstance(roots, sp.FiniteSet):
            return None
        for root in roots:
            status = _member_status([v.subs(parameter, root) for v in values])
            if status is None:
                return None
            if status:
                found.add(root)
    return sorted(found, key=lambda x: float(x.evalf()))


class SymbolicPath:
    """Construct D_g(t)=exp(-2*pi*i*q(t).tau_g) M_g for a generic path group.

    k contains three expression strings in the pure-translation reciprocal
    basis. Each expression must be real and continuous on the open interval.
    Additional little-group operations are excluded by exact integer conditions.
    Numerical M_g coefficients are retained without claiming algebraic exactness.
    """
    def __init__(self, ssg, k, parameter='t', interval=('0', '1'), group_type='double',
                 optimize=False, seed=0, database=None, validation_samples=3):
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', parameter) or parameter in {'pi', 'sqrt', 'sin', 'cos', 'exp', 'I', 'E'}:
            raise SymbolicPathError('Choose a parameter name distinct from mathematical constants and functions.')
        if group_type not in {'single', 'double'} or len(k) != 3 or len(interval) != 2:
            raise SymbolicPathError('Expected three coordinates, two interval bounds, and a valid group type.')
        if validation_samples < 2:
            raise SymbolicPathError('At least two independent validation samples are required.')
        self.ssg = ssg
        self.t = sp.Symbol(parameter, real=True)
        self.k = sp.ImmutableMatrix([parse_expression(x, self.t) for x in k])
        bounds = [parse_expression(x) for x in interval]
        if any(x.free_symbols or x.is_real is not True or x.is_finite is not True for x in bounds):
            raise SymbolicPathError('Interval bounds must be finite real constants.')
        if not bool(bounds[0] < bounds[1]):
            raise SymbolicPathError('The parameter interval must have lower < upper.')
        self.domain = sp.Interval.open(*bounds)
        for expression in self.k:
            try:
                domain = continuous_domain(expression, self.t, self.domain)
            except (NotImplementedError, ValueError) as error:
                raise SymbolicPathError('Cannot certify a real continuous path; simplify the expression or split the interval.') from error
            if domain != self.domain:
                raise SymbolicPathError('A path coordinate is not real and continuous throughout the interval; split at singularities.')
        self.seed = seed
        self.group_type = group_type
        self.geometry_audit = dict(maximum_denominator=4096, reconstruction_tolerance=1e-10,
                                   maximum_reconstruction_error=0.0)
        from ssg_database import load_database
        import SSGReps as native
        database = load_database() if database is None else database
        if ssg not in database:
            raise SymbolicPathError('Unknown SSG number: ' + ssg)
        rng = np.random.default_rng(seed)
        denominator = 2**30

        def sample():
            return bounds[0] + (bounds[1]-bounds[0])*sp.Rational(int(rng.integers(1, denominator)), denominator)

        value = sample()
        lg = native.LittleGroup(ssg, self._k_numeric(value), group_type, database[ssg], optimize)
        self.lattice = _rational_matrix(lg.ssg.pure_T, self.geometry_audit)
        self.reciprocal_without_2pi = self.lattice.inv().T
        self.q = self.k.T*self.reciprocal_without_2pi
        parent = lg.ssg
        self._rotations = [_rational_matrix(r, self.geometry_audit) for r in parent.rotC]
        self._translations = [_rational_matrix(np.asarray(t).reshape(3, 1), self.geometry_audit) for t in parent.tauC]
        self._signs = list(map(int, parent.time_reversal))
        self.membership = []
        self.global_indices = []
        for i, (rotation, sign) in enumerate(zip(self._rotations, self._signs)):
            lattice_rotation = self.lattice.inv().T*rotation*self.lattice.T
            if any(x.is_integer is not True for x in lattice_rotation):
                raise SymbolicPathError('A reconstructed rotation does not preserve the pure-translation lattice.')
            condition = tuple(sp.simplify(x) for x in self.q*(sign*rotation-sp.eye(3))*self.lattice.T)
            self.membership.append(condition)
            if not any(x.has(self.t) for x in condition) and _member_status(condition) is True:
                self.global_indices.append(i)
        if not self.global_indices or self.global_indices[0] != 0:
            raise SymbolicPathError('The generic little group has no identity.')
        self.exclusions = []
        exceptional = {}
        cache = {}
        for i, condition in enumerate(self.membership):
            if i in self.global_indices:
                continue
            if any(not x.has(self.t) and x.is_integer is False for x in condition):
                continue
            if condition not in cache:
                cache[condition] = _finite_exceptions(condition, self.t, self.domain)
            roots = cache[condition]
            self.exclusions.append(dict(global_operation_id=i+1, integer_conditions=list(map(str, condition)),
                                        exceptional_values=None if roots is None else list(map(str, roots))))
            for root in roots or []:
                exceptional.setdefault(root, []).append(i+1)
        self.exceptional_points = []
        for root in sorted(exceptional, key=lambda value: float(value.evalf())):
            # Other, non-enumerated conditions may hold at the same known root.
            statuses = [_member_status([x.subs(self.t, root) for x in c]) for c in self.membership]
            known = [i+1 for i, status in enumerate(statuses) if status is True and i not in self.global_indices]
            determined = all(status is not None for status in statuses)
            self.exceptional_points.append(dict(parameter=str(root), additional_global_operation_ids=known,
                                                membership_certified=determined,
                                                full_little_group_order=sum(status is True for status in statuses) if determined else None))
        self.exceptions_fully_enumerated = all(x['exceptional_values'] is not None for x in self.exclusions)
        self.endpoints = [self._endpoint_info(x) for x in bounds]
        # Samples propose a little group; symbolic membership decides validity.
        for attempt in range(64):
            if attempt:
                value = sample()
            lg.kvec = self._k_numeric(value)
            lg.k_op()
            if lg.oplist == self.global_indices:
                break
        else:
            raise SymbolicPathError('No sample realizes the symbolic generic group; split the path into intervals with fixed symmetry.')
        self.reference_parameter = value
        lg.ssg.get_trans_factor(lg.kvec)
        lg.seed = seed
        lg.get_regular_rep_list(build_dense=False)
        spatial_checks = _check_spatial_products(
            self.lattice, self._rotations, self._translations, self.global_indices,
            np.asarray(lg.mul_table, dtype=int)-1)
        # Constancy follows from the exact reciprocal-lattice membership rows.
        rows = [tuple(sp.simplify(x) for x in self.q*(self._rotations[g]-self._signs[g]*sp.eye(3)))
                for g in self.global_indices]
        if any(x.has(self.t) for row in rows for x in row):
            raise SymbolicPathError('The projective factor system is not constant in this path convention.')
        translations = [self._translations[g] for g in self.global_indices]
        self.factor_exponents = [[sp.simplify(sum(x*y for x, y in zip(row, tau))) for tau in translations]
                                 for row in rows]
        spin = np.asarray(parent.factor_su2)[np.ix_(self.global_indices, self.global_indices)]
        self.spin_factors = np.rint(spin.real).astype(int)
        exact_factors = np.array([[complex(sp.exp(-2*sp.pi*sp.I*x).evalf(17)) for x in row]
                                 for row in self.factor_exponents])*self.spin_factors
        factor_error = float(np.max(abs(exact_factors-lg.factor)))
        if factor_error > 1e-8:
            raise SymbolicPathError('Symbolic factors disagree with the numerical reference group.')
        lg.get_irreducible_rep()
        from ssg_numeric import validate
        group_validation = validate(lg, check_group=True)
        self.reference_group = lg
        self.labels = ['r'+str(i+1) for i in range(lg.rep_num)]
        self.phase_exponents = [sp.simplify((self.q*tau)[0]) for tau in translations]
        self.phases = []
        for exponent in self.phase_exponents:
            offset, dependent = sp.expand(exponent).as_independent(self.t, as_Add=True)
            self.phases.append(sp.simplify(sp.exp(-2*sp.pi*sp.I*offset))*sp.exp(-2*sp.pi*sp.I*dependent))
        self.validation = dict(geometry=self.geometry_audit, generic_membership_certified=True,
                               exact_spatial_product_checks=spatial_checks,
                               projective_factor_parameter_independent=True,
                               factor_reference_residual=factor_error, reference=lg.validation,
                               group_and_cocycle=group_validation, samples=[])
        # Check transferred matrices against freshly evaluated native factors.
        for unused in range(validation_samples):
            for attempt in range(64):
                test_value = sample()
                if self._native_indices(test_value) == self.global_indices:
                    break
            else:
                raise SymbolicPathError('Could not find independent generic validation samples.')
            self.validation['samples'].append(self.validate_at(test_value))

    def _k_numeric(self, value):
        try:
            k = np.array([float(x.subs(self.t, value).evalf(17)) for x in self.k])
        except (TypeError, ValueError) as error:
            raise SymbolicPathError('The path is not finite and real at this parameter.') from error
        if not np.isfinite(k).all():
            raise SymbolicPathError('The path is not finite at this parameter.')
        return k

    def _endpoint_info(self, value):
        k = [sp.simplify(x.subs(self.t, value)) for x in self.k]
        if any(x.is_finite is not True or x.is_real is not True for x in k):
            return dict(parameter=str(value), status='undefined')
        statuses = [_member_status([x.subs(self.t, value) for x in c]) for c in self.membership]
        if any(x is None for x in statuses):
            return dict(parameter=str(value), k=list(map(str, k)), status='membership_undetermined')
        ids = [i+1 for i, status in enumerate(statuses) if status]
        return dict(parameter=str(value), k=list(map(str, k)), status='separate_point',
                    full_little_group_order=len(ids), global_operation_ids=ids)

    def _native_indices(self, value):
        candidate = copy.copy(self.reference_group)
        candidate.kvec = self._k_numeric(value)
        candidate.k_op()
        return candidate.oplist

    def check_parameter(self, value):
        value = parse_expression(value)
        if self.domain.contains(value) is not sp.S.true:
            raise SymbolicPathError('Parameter must lie strictly inside ' + str(self.domain) + '.')
        for i, condition in enumerate(self.membership):
            if i not in self.global_indices and _member_status([x.subs(self.t, value) for x in condition]) is True:
                raise SymbolicPathError('The little group is enhanced at this parameter; solve this point separately.')
        if self._native_indices(value) != self.global_indices:
            raise SymbolicPathError('This parameter has additional symmetry or is too close to an enhancement for numerical tolerance.')
        return value

    @lru_cache(None)
    def operation(self, label, index, bloch=True):
        """Return a SymPy matrix and antiunitary flag; index is local zero-based."""
        if label not in self.labels or not 0 <= index < len(self.global_indices):
            raise SymbolicPathError('Unknown representation label or local operation index.')
        matrix = sp.ImmutableMatrix(self.reference_group.correp_matrix[self.labels.index(label)][index])
        return (self.phases[index]*matrix if bloch else matrix), self.reference_group.time_reversal[index] < 0

    def characters(self, label, bloch=True):
        """Return unitary local operation IDs and parameter-dependent characters."""
        if label not in self.labels:
            raise SymbolicPathError('Unknown representation label.')
        unit = np.flatnonzero(np.array(self.reference_group.time_reversal) > 0)
        values = self.reference_group.character[self.labels.index(label)]
        return [int(i+1) for i in unit], [sp.sympify(v)*(self.phases[i] if bloch else 1) for i, v in zip(unit, values)]

    def translation(self, primitive_lattice_coefficients):
        """Return the symbolic phase of a pure translation in integer lattice units."""
        values = [parse_expression(x) for x in primitive_lattice_coefficients]
        if len(values) != 3 or any(x.is_integer is not True for x in values):
            raise SymbolicPathError('A pure translation requires three integer lattice coefficients.')
        return sp.exp(-2*sp.pi*sp.I*self.k.dot(sp.Matrix(values)))

    def numeric(self, label, value, bloch=True):
        """Evaluate inside the generic stratum; enhanced points are rejected."""
        value = self.check_parameter(value)
        if label not in self.labels:
            raise SymbolicPathError('Unknown representation label.')
        matrices = self.reference_group.correp_matrix[self.labels.index(label)].copy()
        if bloch:
            phase = np.array([complex(x.subs(self.t, value).evalf(17)) for x in self.phases])
            matrices *= phase[:, None, None]
        return matrices

    def validate_at(self, value):
        """Validate every semilinear product using the native geometry at value."""
        from ssg_numeric import validate
        value = self.check_parameter(value)
        lg = copy.copy(self.reference_group)
        lg.ssg = copy.copy(lg.ssg)
        lg.kvec = self._k_numeric(value)
        lg.ssg.get_trans_factor(lg.kvec)
        lg.get_regular_rep_list(build_dense=False)
        stats = validate(lg)
        return dict(parameter=str(value), k=lg.kvec.tolist(), **stats)

    def report_data(self):
        from ssg_output import from_little_group
        data = from_little_group(self.reference_group)
        data.update(k=list(map(str, self.k)), symbolic=True, exact=False)
        for i, op in enumerate(data['operations']):
            op['phase'] = self.phases[i]
            op['global_id'] = self.global_indices[i]+1
        for rep in data['representations']:
            rep['characters'] = [c if not op['anti'] else None for c, op in zip(rep['characters'], data['operations'])]
        notes = [f'Parameter: {self.t}; open interval: {self.domain}.',
                 'Symbolic Bloch phases with numerical constant matrices; not fully exact algebraic matrices.',
                 'The family describes the full generic little group only where no excluded operation becomes a symmetry.']
        if self.exceptional_points:
            notes.append('Interior enhanced-symmetry parameters: ' + ', '.join(x['parameter'] for x in self.exceptional_points) + '.')
        if not self.exceptions_fully_enumerated:
            notes.append('Some exceptional sets are specified by integer conditions, not a complete finite point list; see the export.')
        notes.append('Endpoints require separate point calculations. Operation indices in this report are local to the path group.')
        data['notes'] = notes
        return data

    def to_dict(self):
        """Export raw constant matrices and symbolic phases without rounding."""
        lg = self.reference_group
        unit = np.flatnonzero(np.array(lg.time_reversal) > 0)
        return dict(schema='ssgreps.symbolic_path.v1', ssgNum=self.ssg, groupType=self.group_type,
                    parameter=str(self.t), kPath=list(map(str, self.k)), interval=[str(self.domain.start), str(self.domain.end)],
                    exactness=dict(parameter_dependence='symbolic', constant_matrices='floating-point', fully_exact=False),
                    referenceParameter=str(self.reference_parameter), referenceK=lg.kvec.tolist(), seed=self.seed,
                    pureTranslations=[[str(x) for x in row] for row in self.lattice.tolist()],
                    reciprocalBasisWithout2Pi=[[str(x) for x in row] for row in self.reciprocal_without_2pi.tolist()],
                    globalOperationIds=[x+1 for x in self.global_indices], timeReversal=list(map(int, lg.time_reversal)),
                    rotC=[[[str(x) for x in row] for row in self._rotations[g].tolist()] for g in self.global_indices],
                    tauC=[[str(x) for x in self._translations[g]] for g in self.global_indices],
                    spin=np.asarray(lg.spin), su2=np.asarray(lg.su2s),
                    unitaryLocalOperationIds=(unit+1).tolist(), repDegree=lg.rep_degree,
                    matrixConvention='projective_constant', repMatrix=lg.correp_matrix,
                    blochPhaseExponents=list(map(str, self.phase_exponents)), blochPhases=list(map(str, self.phases)),
                    character=lg.character, characterConvention='projective_unitary',
                    symbolicBlochCharacters=[list(map(str, self.characters(label)[1])) for label in self.labels],
                    multiplicationTable=np.asarray(lg.mul_table, int),
                    spinFactors=self.spin_factors, factorPhaseExponents=[list(map(str, row)) for row in self.factor_exponents],
                    validity=dict(interval_open=True, excludedOperationConditions=self.exclusions,
                                  exceptionalPoints=self.exceptional_points, finiteExceptionListComplete=self.exceptions_fully_enumerated,
                                  endpoints=self.endpoints), validation=self.validation)


def load_one_ssg_path(ssg, coordinates, parameter='t', interval=('0', '1'), single=2,
                      out='character', file_type=None, optimize=False, convention='bloch',
                      rep=None, ops=None, compact_characters=False, seed=0,
                      html_path=None, text_path=None, export_path=None):
    """CLI-compatible symbolic path construction, reports, and JSON/NPY export."""
    from SSGReps import CustomEncoder
    from ssg_output import report_text, report_html
    if single not in (1, 2) or out not in {'character', 'rep_matrix', 'rep_degree', 'both'}:
        raise SymbolicPathError('Invalid group type or output selection.')
    if file_type not in (None, 'json', 'npy') or convention not in {'bloch', 'projective'}:
        raise SymbolicPathError('Invalid export type or matrix convention.')
    if export_path and file_type is None:
        raise SymbolicPathError('--export requires --fileType json or npy.')
    if rep is not None and rep < 1 or ops is not None and (not ops or min(ops) < 1):
        raise SymbolicPathError('Representation and operation indices start at 1.')
    family = SymbolicPath(ssg, coordinates, parameter, interval,
                          'single' if single == 1 else 'double', optimize, seed)
    if rep is not None and rep > len(family.labels):
        raise SymbolicPathError('Representation index exceeds the number of corepresentations.')
    if ops is not None and max(ops) > len(family.global_indices):
        raise SymbolicPathError('Operation index exceeds the generic little group order.')
    data = family.report_data()
    text = report_text(data, 'r'+str(rep) if rep is not None else None, ops,
                       convention, out, compact_characters)

    def write(path, content):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')

    if html_path:
        chosen_rep = rep if rep is not None else int(np.argmax(family.reference_group.rep_degree))+1
        selected = ops
        if selected is None:
            selected = list(range(1, min(4, len(family.global_indices))+1))
            anti = [i+1 for i, s in enumerate(family.reference_group.time_reversal) if s < 0]
            if anti and anti[0] not in selected:
                selected.append(anti[0])
        write(html_path, report_html(data, 'r'+str(chosen_rep), selected, convention, compact_characters))
    if text_path:
        write(text_path, text)
    if file_type:
        destination = Path(export_path or 'path_output.'+file_type)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if file_type == 'json':
            write(destination, json.dumps(family.to_dict(), cls=CustomEncoder, indent=2))
        else:
            with destination.open('wb') as stream:
                np.save(stream, family.to_dict())
    print(text, end='')
    return family
