SSG symbolic numbers and exponential output
==========================================

The exact backend already retains algebraic numbers and path parameters. This
utility adds display choices and bounded candidate recognition with residuals
for legacy floating-point data. It does not change the representation basis.

Dependencies: Python and SymPy (tested with SymPy 1.14.0).

Exact input
-----------
    import sympy as sp
    from symbolic_numbers import format_exact, latex_exact, parse_gap
    t = sp.Symbol('t', real=True)
    x = (1 + sp.sqrt(3)*sp.I)/2
    print(format_exact(x, 'exponential'))  # exp(I*pi/3)
    print(format_exact(x, 'radical'))      # 1/2 + sqrt(3)*I/2
    print(latex_exact(x))                 # LaTeX for e^{i pi/3}
    print(format_exact(-sp.I*sp.exp(-sp.I*sp.pi*t)/4))

GAP E(n) denotes the nth root of unity exp(2*pi*I/n), not Euler's number raised
to n. parse_gap('E(12)^7') reads this exact number. General cyclotomic numbers
are linear combinations of roots of unity and need not fit a single
exp(i*pi*p/q); their exact expressions are retained in that case.

format_exact accepts both SymPy scalars and matrices:
    formatted_matrix = format_exact(M, 'exponential')
    tex = latex_exact(M)

It combines constant phases with symbolic exponents in products. Floating-point
input raises an error to avoid silently treating approximations as exact.
Existing sqrt, Rational, and exp expressions require no numerical recognition.

Legacy floating-point input
---------------------------
    from symbolic_numbers import recognize_numeric
    result = recognize_numeric(0.5 + 0.8660254037844386j)
    print(result['exponential'])     # exp(I*pi/3)
    print(result['residual'])
    print(result['status'])          # numeric_candidate
    print(result['certified_exact']) # Always False

Default candidates have the form sqrt(a/b)*exp(2*pi*i*p/q), with
q <= 48, b <= 64, a <= 4096. The tolerance is atol + rtol*abs(z), with both
atol and rtol equal to 1e-10 by default. This includes rational numbers, simple
radicals, roots of unity, and roots of unity with algebraic amplitudes.
An unmatched input returns unrecognized rather than a forced replacement.

A floating-point sample cannot uniquely prove an exact expression. Even a zero
residual means only that the floating-point candidate agrees with the input.
Treating candidate matrices as an exact representation requires separate checks
of all multiplication laws, antiunitary conjugation, unitarity, and irreducibility.
Do not recover exact numbers from legacy output rounded to three decimal places:
0.5 + 0.866j is rejected at the default tolerance. Use the original complex arrays.
A numerical sample cannot recover an unknown function such as exp(-i*pi*t).
Path parameters and phase formulas must be retained from group operations and
k(t), as in the symbolic path implementation.

Export all verified path generators
-----------------------------------
    python export_all.py
    python check_examples.py

input_generators.json comes from the six verified paths of SSG 221.8.6.3 and
retains antiunitary flags. export_all.py exports exponential matrices, LaTeX
fragments, and JSON, verifying entrywise symbolic equality with the input.
The export does not use floating-point recognition.

generators_exponential.txt   Readable matrices in exponential form
generators_latex.tex         LaTeX matrix fragments; antiunitary terms include K
generators_exponential.json  Machine-readable output with antiunitary flags
export_validation.json      Exact rewrite verification counts
numeric_examples.json       Recognition candidates, residuals, and rejected cases
exact_rewrites.json         Exact radical/exponential rewrite examples

In symbolic_numbers.py, format_exact and parse_gap handle exact input;
recognize_numeric handles provisional numerical candidates. They are separate
interfaces with distinct guarantees.

SymPy symbolic and LaTeX printing documentation:
https://docs.sympy.org/latest/tutorials/intro-tutorial/printing.html
