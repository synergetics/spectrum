# spectrum C Core — HOSA-Complete Reimplementation

**Date:** 2026-10-18 **Status:** Approved design **Supersedes:** none

## 1. Context

An audit against the original HOSA MATLAB toolbox sources (Swami / United
Signals & Systems, recovered via the author's `synergetics/HOSA_Octave` mirror)
established that the current pure-Python `spectrum` package is a structurally
faithful but buggy port of a subset of HOSA:

- 7 confirmed crash/deviation bugs (default-`nsamp` ZeroDivisionError in
  `bicoherence`, `cumest` order-4 dispatching to `cum3est`, `cumest`
  `nsamp=None` TypeError, `cum4est` crash for `k2=0`, `cum4x` `k2 > 2`
  transcription bug, `bispectrumd` 254x254 double-crop + invalid-window
  AttributeError, 1-D input rejection everywhere).
- One HOSA-own bug already silently fixed by the Python port (`CUM4EST.M`
  returns after the first segment).
- Missing from the full toolbox: `wig2`, `wig2c`, `wig4`, `wig4c`, `tfrwv`,
  `trispec`, `doa`, `armaqs`, `armarts`, `cumtrue`.
- `trispectrum`/`tricoherence`/`armafit`/`armasel` are original Python
  extensions, not HOSA; `armafit` is currently broken (6 failing tests).
- Tests are mostly smoke/shape checks with self-regression pins; no MATLAB
  reference values.

This spec defines a two-step reimplementation: (1) move all numeric code to
native C with Python bindings, (2) implement the missing HOSA functions, also in
C.

## 2. Decisions (from brainstorming)

| Question        | Decision                                                                                                         |
| --------------- | ---------------------------------------------------------------------------------------------------------------- |
| Fidelity target | **Correct HOSA semantics** — fix all transcription bugs and HOSA's own bugs; match intended MATLAB behavior      |
| FFT engine      | **Vendored pocketfft** (BSD-3) by default; **optional FFTW3** when detected/enabled at build time                |
| Step-1 scope    | All numeric functions: 12 HOSA ports + `trispectrum`, `tricoherence`, `armafit`, `armasel`, `harmgen`, `nlgen`   |
| Step-2 scope    | All 10 missing HOSA functions                                                                                    |
| Python API      | **Drop-in**: same names/signatures/returns; dispatch to C extension, automatic fallback to pure-Python reference |
| Architecture    | **Layered**: pure C library (no Python types) + one thin CPython C-API binding module                            |
| Process         | **One commit per action** — each port/fix/new function/test-update is its own commit                             |

## 3. Semantics contract (what "correct HOSA" means concretely)

For each estimator, the C implementation is validated against the corrected
pure-Python reference (itself validated against the `.m` sources and analytical
properties). Behavior changes vs today's Python:

1. `bicoherence`: `nsamp<=0` with a single realization selects
   `fix(ly/(8-7*overlap/100))` (MATLAB `nrecs == 1` condition, not `> 1`);
   `overlap` clamped to [0,99]; 1-D numpy input accepted.
2. `cumest`: `nsamp=None` defaults to data length (no TypeError); `norder=4`
   dispatches to `cum4est` with `k1,k2`.
3. `cum4est`: works for all `k1,k2` including 0; averages over all segments with
   `/nrecord` (keeps the fix of HOSA's early-return bug); no debug output.
4. `cum4x`: `k2 >= 0` branch (not `k2 > 2`); all `k1,k2` combinations valid.
5. `bispectrumd`/`bispectrumdx`: output is exactly `nfft x nfft` (full
   convolution + center crop, i.e. scipy `mode='same'` equivalence, no second
   crop); invalid 2-D or 1-D windows (non-square, even-length,
   imaginary/negative components) warn, are ignored, and **skip smoothing
   entirely while keeping the `nfft x nfft` output** (the intended behavior;
   MATLAB's accidental crop-to-`(nfft-lby2)` quirk in this path is not
   reproduced); 1-D window path follows MATLAB's zero-padded
   `hankel(flipud(window), window)` construction with `winsize = 2*len(wind)-1`.
6. All estimators accept 1-D arrays (normalized to a column internally).
7. `armafit`: fixed (shape bug at armafit.py:143), still 2nd-order Yule-Walker
   based (documented as an extension, not HOSA).
8. `trispectrum`/`tricoherence`: keep current computational contract (reduced
   frequency grid, named windows) but fix mean-removal order (before windowing)
   for consistency with the rest of the family.

Known behavior-change risk: existing test pins encode buggy outputs (254x254
shape, `cumest(4) == cum3est`); each pin is updated in the same commit as its
semantic fix.

## 4. Architecture

```
spectrum/
  src/                          pure C11, no Python headers
    common.h                    error codes, arena allocator, threading macros
    estimators_bispec.c         bicoher, bicoherx, bispecd, bispecdx, bispeci
    estimators_cum.c            cum2est/x, cum3est/x, cum4est/x, cumest
    estimators_ext.c            trispectrum, tricoherence, armafit, armasel,
                                harmgen, nlgen
    wigner.c                    wig2, wig2c, wig4, wig4c, tfrwv
    trispec.c                   trispectrum (HOSA) direct 4th-order
    linalg.c                    small eigensolver, Toeplitz solve, GEPP for
                                doa, armaqs, armarts, cumtrue
    rng_util.c                  xoshiro256++ / box-muller for harmgen, nlgen
    fft/
      pocketfft/                vendored BSD-3 C pocketfft
      fft_iface.c               plan-cache LRU + engine dispatch
  _spectrum_core.c              CPython C-API binding (flat functions)
_spectrum_core.<abi>.so         built artifact
spectrum/_backend.py            import-time dispatch, env override
```

### Binding and dispatch

- `_spectrum_core` exposes one flat C function per estimator, taking contiguous
  buffers + parameter ints, returning ndarrays it allocates.
- `spectrum/_backend.py`: `get_backend() -> "c" | "python"`, honoring
  `SPECTRUM_BACKEND`; each Python module calls
  `_backend.call("bispecd", args...)` which tries the C path and falls back to
  the local pure-Python implementation on ImportError, missing symbol, or
  explicit env override. Warnings are logged once.
- GIL released around every C kernel call.
- Inputs coerced via `np.ascontiguousarray(..., dtype=complex128|float64)`;
  outputs are freshly-allocated ndarrays (no shared scratch state).

### FFT layer

- `fft_plan_get(n, direction, kind)` returns a cached plan; LRU cap ~64.
- Default engine: vendored pocketfft. Engine selection at build time:
  `SPECTRUM_FFTW=1` forces FFTW (error if not found); otherwise FFTW is
  auto-detected via pkg-config and used when present, else pocketfft. Both
  link-time paths compiled behind one macro.
- 2-D FFTs implemented as row/column passes through the 1-D plan cache.

### Threading

- OpenMP `parallel for` over segments (all FFT-domain estimators), over
  lag-slices (`bispectrumi` cumulant loops, `wig4`), and over `k1` blocks
  (trispectrum). Per-thread accumulators reduced at the end.
- Serial fallback (`#define` no-ops) when OpenMP is disabled at build.
- Thread count from `OMP_NUM_THREADS`, default hardware concurrency.

### Performance plan

- Fused per-segment loops (mean + window + accumulate in one pass).
- Exact symmetry mirroring: bispectrum plane is symmetric under (f1,f2) swap;
  compute the lower triangle and mirror — bit-exact by construction of the
  formula.
- Single arena allocation per estimator call; no malloc inside loops.
- `-O3 -funroll-loops` always; `-march=native` only when `SPECTRUM_NATIVE=1`
  (never in distributed wheels).
- Benchmark suite target: >= 10x vs pure Python on 4096-sample `bispectrumd`
  (nfft=256, 50% overlap); >= 50x on `trispectrum`.

## 5. Step 1 — ported functions (order)

Each item is: fix Python semantics (if needed) -> port to C -> parity test ->
commit. One function per commit.

1. `cum2est`, `cum2x` (foundation of everything else)
2. `cum3est`, `cum3x`, `cum4est`, `cum4x`, `cumest`
3. `bicoherence`, `bicoherencex`
4. `bispectrumd`, `bispectrumdx`
5. `bispectrumi`
6. `trispectrum`, `tricoherence`
7. `harmgen`, `harmgen_complex`
8. `nlgen`
9. `armafit` (fix included), `armasel`

## 6. Step 2 — missing HOSA functions

| Function        | Provenance of semantics                                                                        |
| --------------- | ---------------------------------------------------------------------------------------------- |
| `cumtrue`       | verbatim `CUMTRUE.M` from the Octave mirror                                                    |
| `wig2`, `wig2c` | HOSA manual + Wigner-Ville literature; re-attempt source recovery online during implementation |
| `wig4`, `wig4c` | HOSA manual + Wigner bispectrum definitions (Nikias)                                           |
| `tfrwv`         | HOSA manual: pseudo-Wigner-Ville time-frequency map                                            |
| `trispec`       | HOSA manual: direct-method trispectrum with full symmetry handling                             |
| `doa`           | cumulant-based DOA (cumulant MUSIC/ESPRIT family per HOSA docs)                                |
| `armaqs`        | q-slice method (Swami's papers, HOSA manual)                                                   |
| `armarts`       | residual time-series method (HOSA manual)                                                      |

Every Step-2 function ships as: C kernel + thin pure-Python twin (reference +
fallback) + tests (parity + property) + docstring stating provenance. `linalg.c`
provides what `doa`/`armaqs`/`armarts` need (symmetric eigensolver via Jacobi,
GEPP solve, Toeplitz Levinson-Durbin); no external LAPACK dependency.

## 7. Testing

- `tests/test_parity_*.py`: C vs corrected Python,
  `np.allclose(rtol=1e-10, atol=1e-12)` across parameter grids (flag
  biased/unbiased, k1/k2 sign combos, scalar/1-D/2-D windows, overlap 0..99,
  single/multi realization, real/complex input).
- `tests/test_property_*.py`: QPC peak at (f1,f2); Gaussian cumulants ~ 0;
  bicoherence in [0,1]; `bispectrumd` ~ `bispectrumi`; `cumtrue` vs empirical
  cumulants on simulated ARMA.
- Existing smoke/pin tests updated alongside semantic fixes.
- `tests/bench/`: timing comparisons C vs Python (informational, not CI gates).

## 8. Packaging and build

- `setup.py`: add `Extension("spectrum._spectrum_core", ...)`; compile the C
  library and binding together; `#define SPECTRUM_USE_FFTW` only when enabled.
  `extra_compile_args` per platform (msvc vs gcc/clang).
- `make build-c`, `make test` targets; sdist ships `spectrum/src/**`.
- Version 0.3.0 -> 0.4.0 (semantics change); README gains a "Backends" section
  documenting `SPECTRUM_BACKEND` and `SPECTRUM_FFTW`.
- The package must remain importable with the extension absent (pure Python
  fallback), so a failed C build never bricks the install.

## 9. Process constraints

- **One commit per action.** A port, a semantic fix, a new function, a pin
  update, a benchmark, a doc section — each is a single focused commit. Fix +
  its pin update are one commit (they are one action).
- TDD per the implementation plan: parity/property tests land with (or before)
  each kernel.
- No commits of build artifacts (`.so`, `.coverage`, `build/`).

## 10. Risks and mitigations

| Risk                                               | Mitigation                                                                                                      |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Missing `.m` sources for 8 of the Step-2 functions | Implement from manual + literature; document provenance; re-attempt recovery; property tests anchor correctness |
| FFTW GPL contamination                             | FFTW strictly optional, opt-in, not in wheels                                                                   |
| `-march=native` wheel breakage                     | Off by default, env-gated                                                                                       |
| Behavior changes break downstream users            | Version bump, CHANGELOG entry listing every semantic fix, Python fallback preserved                             |
| Numerical drift C vs Python                        | Parity tolerance 1e-10; pocketfft vs numpy FFT both use pocketfft lineage                                       |
| Python 3.14 C-API churn                            | Stable subset of C-API only (buffer protocol, PyModule, ndarray C-API)                                          |

## 11. Out of scope

- GPU backends (CUDA/SYCL).
- Plotting in C (all `plot_*` stay matplotlib/Python).
- `torch.Tensor` support (the commented-out `tools/types.py` idea).
- Reimplementing MATLAB demo scripts beyond what tests need.
