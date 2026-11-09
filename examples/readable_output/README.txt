Exact 16D readable-output example
================================

From the repository root:
    python examples/readable_output/demo.py --rep r3 --ops 2 193
    python examples/readable_output/check_output.py

This uses the shared SSGReps/ssg_output.py formatter and exact data in
examples/symbolic_paths/R. No GAP installation or numerical recognition is needed.
Reports contain the unitary character table and selected 16D matrices for SSG
221.8.6.3 at k=(1/2,1/2,1/2), using pure-translation reciprocal coordinates.
The default is the Bloch convention; use --convention projective for M(g).
Use --characters full for all unitary columns or --outdir PATH to change the
report destination. Labels r1/r2/r3 are local to this cached exact calculation.

The generic numerical CLI and current release validation are documented in the
repository-root README.md. The HTML report in this directory shows the exact example.
