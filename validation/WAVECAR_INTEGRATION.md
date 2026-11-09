# WAVECAR integration audit

The repository now includes the reusable NumPy coefficient/matching module,
portable Mn3Sn adapters, and the original material records. All scientific
claims below distinguish code verification from physical block acceptance.

## Integration checks

- All 16 tests in `test_regressions.py` and `test_wavecar.py` passed with the
  pinned core requirements on Python 3.9.6/macOS arm64. Nine tests cover the new
  interface; seven retain the numerical-solver regression coverage.
- Synthetic tests compare the coefficient action with independent real-space
  evaluation, including fractional translation phases and a 16D spinful C4/T
  subspace. They check common Gram normalization and nonclosed-subspace rejection.
- Character tests cover Wigner restriction norms 1, 2, and 4, cached exact 8D/16D
  SSG branches, direct sums, incomplete multiplets, and cases where matching
  traces must still be rejected because products or leakage fail.
- A synthetic binary-reader substitute exercises the material CLI through native
  theory construction, coefficient transformation, report writing, successful
  complete blocks, rejected split blocks, and rejection of SOC inputs. This
  checks adapter wiring; it does not revalidate the external WAVECAR parser.
- Offline replay recomputes all unitary traces, unitarity, products, labels, and
  acceptance decisions from 642 measured block-matrix sets. It uses the original
  recorded leakage because large WAVECAR files are not distributed.
- A negative control truncating the measured A-point bands 53-56 to two basis
  vectors is rejected. Its unitarity residual is approximately 1.09859.
- The cached band-plot CLI runs without a WAVECAR reader; its PNG was visually
  checked for labels, legend placement, and the excluded general point.
- New Python syntax, English text, Markdown links/anchors/fences, and whitespace
  checks passed. The source manifest is verified after packaging.

## Material results retained without relabeling

The recorded NSOC SSG 194.1.6.1.P calculations cover 14 distinct k points:
530/582 blocks pass and 52 remain `UNRESOLVED`. A separately computed 60/60
general-point repeat gives 590/642 across 15 evaluations. At A=(0,0,1/2), all 19
tested blocks pass. Bands 53-56 form a 4D `r3` with mean E-EF=0.21084044 eV,
character residual 6.43e-9, product residual 8.93e-7, and leakage 6.62e-4.

The full theoretical corepresentations still use antiunitary operations.
Characters and wavefunction matching use only their unitary restrictions.
The final data use an independently converged historical density. The tighter
SCF trial was stopped without convergence and is excluded. Raising NBANDS from
64 to 96 does not eliminate the remaining failures; no unique cause is claimed.

See the [material report](../examples/wavecar_mn3sn/results/README.md),
[replay audit](mn3sn_replay.json), and
[example usage](../examples/wavecar_mn3sn/README.md). Original measured JSON/NPZ
records are preserved, including failed blocks. The numerical representation
solver was not changed during this integration; its earlier 129-configuration
survey was not rerun merely for these documentation and adapter changes.

## Reproduce

```bash
python -m unittest discover -s tests -p 'test_*.py'
python examples/wavecar_mn3sn/replay_validation.py
```

The full measured coefficient extraction was performed in the recorded remote
run. This integration replays cached matrices locally; it does not claim a new
VASP calculation or a new real-file parse after the portability refactor.
