"""Exact symbolic formatting plus bounded, explicitly provisional float recovery.

GAP E(n) means exp(2*pi*I/n), not Euler's number to the power n.
format_exact never infers algebraic numbers from floating point inputs.
recognize_numeric returns a candidate and its residual, never a proof.
"""
import ast
import cmath
import math
from fractions import Fraction
from functools import lru_cache
import sympy as sp


def parse_gap(text):
    """Parse the arithmetic-only cyclotomic expressions emitted by GAP safely."""
    def walk(node):
        if isinstance(node,ast.Constant) and type(node.value) is int:
            return sp.Integer(node.value)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            x=walk(node.operand);return x if isinstance(node.op,ast.UAdd) else -x
        if isinstance(node,ast.BinOp):
            a,b=walk(node.left),walk(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.Div):return a/b
            if isinstance(node.op,ast.Pow) and b.is_Integer:return a**b
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='E' and len(node.args)==1 and not node.keywords:
            n=walk(node.args[0])
            if n.is_Integer and n>0:return sp.exp(2*sp.pi*sp.I/n)
        raise ValueError('Expected an arithmetic GAP cyclotomic expression.')
    return walk(ast.parse(text.replace('^','**'),mode='eval').body)


@lru_cache(None)
def _polar_constant(value):
    """Return exact magnitude and rational multiple of pi when recognized exactly."""
    if value.free_symbols:return None
    value=sp.simplify(sp.expand_complex(value))
    if value==0:return (sp.S.Zero,sp.S.Zero)
    amplitude=sp.sqrt(sp.simplify(value*sp.conjugate(value)))
    angle=sp.simplify(sp.arg(value)/sp.pi)
    if angle.is_Rational and sp.simplify(sp.expand_complex(amplitude*sp.exp(sp.I*sp.pi*angle))-value)==0:
        return amplitude,angle
    return None


def format_exact(value,style='exponential'):
    """Return an exact SymPy scalar or matrix in radical or exponential style.

    Exponential style combines known constant phases with symbolic exp factors.
    Constants without a recognized rational phase retain their exact expression.
    """
    if isinstance(value,sp.MatrixBase):return value.applyfunc(lambda x:format_exact(x,style))
    expr=sp.sympify(value)
    if expr.has(sp.Float):raise ValueError('Floating input: use recognize_numeric first and inspect its residual.')
    if style not in ('radical','exponential'):raise ValueError('Choose radical or exponential.')
    if expr==0:return sp.S.Zero
    exponent=sp.S.Zero;coefficient=sp.S.One
    for part in sp.Mul.make_args(sp.factor_terms(expr)):
        if part.func==sp.exp:exponent+=part.args[0]
        else:coefficient*=part
    if style=='radical':
        c=sp.simplify(sp.expand_complex(coefficient)) if not coefficient.free_symbols else coefficient
        # Keep a variable Bloch exponent intact; constant exponent may be expanded.
        if exponent.free_symbols:return c*sp.exp(exponent)
        return sp.simplify(sp.expand_complex(c*sp.exp(exponent)))
    polar=_polar_constant(coefficient)
    if polar is not None:
        coefficient,angle=polar;exponent+=sp.I*sp.pi*angle
    exponent=sp.simplify(exponent)
    if exponent==0:return coefficient
    exponential=sp.exp(exponent,evaluate=False)
    return exponential if coefficient==1 else sp.Mul(coefficient,exponential,evaluate=False)


def recognize_numeric(value,*,atol=1e-10,rtol=1e-10,max_root_order=48,
                      max_denominator=64,max_numerator=4096):
    """Fit sqrt(rational)*root_of_unity with bounded complexity.

    Supports rationals, simple radicals and commensurate complex phases.
    An accepted result is only a numerical candidate, including for rounded 0/1.
    Unknown constants stay unknown; no random-basis matrix is forced into a field.
    """
    z=complex(value)
    if not math.isfinite(z.real) or not math.isfinite(z.imag):raise ValueError('Input must be finite.')
    if atol<0 or rtol<0 or min(max_root_order,max_denominator,max_numerator)<1:
        raise ValueError('Invalid tolerance or search bound.')
    threshold=atol+rtol*abs(z)
    result=dict(status='unrecognized',candidate=None,exponential=None,latex=None,
                residual=None,tolerance=threshold,certified_exact=False)
    if abs(z)<=threshold:
        candidate=sp.S.Zero
    else:
        square=Fraction(abs(z)**2).limit_denominator(max_denominator)
        if square<=0 or square.numerator>max_numerator:return result
        turn=Fraction(cmath.phase(z)/(2*math.pi)).limit_denominator(max_root_order)
        candidate=sp.sqrt(sp.Rational(square.numerator,square.denominator))*sp.exp(2*sp.pi*sp.I*sp.Rational(turn.numerator,turn.denominator))
    residual=abs(complex(candidate.evalf(40))-z)
    result['residual']=residual
    if residual>threshold:return result
    symbolic=format_exact(candidate,'exponential')
    result.update(status='numeric_candidate',candidate=sp.simplify(sp.expand_complex(candidate)),
                  exponential=symbolic,latex=sp.latex(symbolic))
    return result


def latex_exact(value,style='exponential'):
    return sp.latex(format_exact(value,style))
