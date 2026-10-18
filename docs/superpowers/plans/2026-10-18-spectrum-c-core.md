# spectrum C Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reimplement all numeric estimators of the `spectrum` package in native
C (CPython extension), implement the 10 missing HOSA functions, then release
0.4.0 to PyPI.

**Architecture:** Layered — a pure C11 library under `spectrum/src/` (no Python
types) + one thin CPython C-API binding module `spectrum._spectrum_core` +
Python-side dispatch with pure-Python fallback. FFT via vendored pocketfft with
optional build-time FFTW. OpenMP over segments.

**Tech Stack:** C11, CPython C-API, numpy C-API, pocketfft (vendored, BSD-3),
OpenMP, pytest, setuptools.

**Spec:** `docs/superpowers/specs/2026-10-18-spectrum-c-core-design.md`

## Global Constraints

- Semantics: **correct HOSA** (spec §3). Reference for every C kernel = the
  corrected pure-Python implementation (fixed in Phase 1), validated against the
  `.m` sources in `/tmp/opencode/hosa_octave/` (re-fetch from
  https://github.com/synergetics/HOSA_Octave if missing).
- **One commit per action**: one function/port/fix/test-update per commit. Fix +
  its pin update = one commit.
- Parity tolerance: `np.allclose(rtol=1e-10, atol=1e-12)` C vs Python across the
  parameter grid (spec §7).
- Build must succeed with: `python setup.py build_ext --inplace` (gcc/clang,
  `-O3 -funroll-loops`; `-march=native` only when `SPECTRUM_NATIVE=1`). FFTW
  only when `SPECTRUM_FFTW=1` or pkg-config finds it.
- Package must import and pass tests with the extension absent (fallback path).
- Dev env: Python 3.14, numpy/scipy/matplotlib installed; 24 cores.
  `MPLBACKEND=Agg` for test runs.
- All Python code must pass repo pre-commit (black, flake8, mypy-clean-enough:
  no new untyped defs flagged by existing config).
- Version ends at `0.4.0`; publish via `twine` (check `~/.pypirc`/env for
  credentials; if absent, stop after `twine check` and report).

---

### Task 1: Vendor pocketfft

**Files:**

- Create: `spectrum/src/fft/pocketfft/pocketfft.c`,
  `spectrum/src/fft/pocketfft/pocketfft.h`, `spectrum/src/fft/pocketfft/LICENSE`

**Interfaces:**

- Produces: `void pocketfft_prepare(...)` / `void pocketfft_exec(...)` — exact
  API as upstream (used only through `fft_iface.c`).

- [ ] Fetch upstream (BSD-3) from https://github.com/mreinecke/pocketfft (raw
      `pocketfft.c`, `pocketfft.h`); if network fails, implement radix-2 +
      Bluestein fallback in the same files.
- [ ] Add LICENSE file with upstream notice.
- [ ] Verify it compiles:
      `gcc -c -O2 spectrum/src/fft/pocketfft/pocketfft.c -o /tmp/opencode/pfft.o`
- [ ] Commit: `feat: vendor pocketfft C library`

### Task 2: C core skeleton — common.h + fft_iface

**Files:**

- Create: `spectrum/src/common.h`, `spectrum/src/fft/fft_iface.h`,
  `spectrum/src/fft/fft_iface.c`

**Interfaces:**

- Produces:

  - `typedef enum { SP_OK=0, SP_ERR_ARG, SP_ERR_MEM, SP_ERR_SHAPE } sp_status;`
  - `void *sp_arena_alloc(sp_arena*, size_t)` / `sp_arena_init` /
    `sp_arena_free` (bump allocator)
  - `SP_PRAGMA_OMP(...)` macros that no-op when `SPECTRUM_NO_OPENMP`
  - `int sp_fft_forward(double complex *, size_t n);`
    `int sp_fft_backward(double complex *, size_t n);` (normalized per numpy
    conventions: forward unnormalized... **match numpy**: forward =
    `Σ x e^{-2πikn/n}`, backward = inverse without 1/n? — match
    `np.fft.fft`/`np.fft.ifft` semantics exactly, including ifft's 1/n)
  - plan cache: `sp_fft_plan_get(size_t n, int direction)` LRU ≤ 64, thread-safe
    via one mutex.

- [ ] Write failing compile-time smoke: `tests/c/test_fft_iface.c` calling
      `sp_fft_forward` on a 8-pt impulse, checking DC bin = 8.
- [ ] Implement, compile, run.
- [ ] Commit: `feat: C core skeleton (arena, omp macros, fft interface)`

### Task 3: Binding skeleton + setup.py extension

**Files:**

- Create: `spectrum/src/_spectrum_core.c`
- Modify: `setup.py` (add
  `Extension("spectrum._spectrum_core", sources=[...], ...)`)

**Interfaces:**

- Produces: importable module `spectrum._spectrum_core` exposing `_selftest()`
  returning `(nfft_test_value)` and, later, one flat function per estimator.
  Helpers (internal): `sp_array_from_obj` (coerce to contiguous C64/C128 via
  `PyArray_FROM_OTF`), `sp_make_array` output allocation, GIL release macro
  `SP_BEGIN_NOGIL`/`SP_END_NOGIL`.

- [ ] Write `tests/test_core_import.py::test_selftest` asserting
      `spectrum._spectrum_core._selftest() == 42`.
- [ ] Implement module init with numpy `import_array()`, `_selftest`.
- [ ] Update `setup.py`; build `python setup.py build_ext --inplace`; run test →
      pass.
- [ ] Commit: `feat: _spectrum_core CPython binding skeleton + build`

### Task 4: Dispatch layer

**Files:**

- Create: `spectrum/_backend.py`
- Modify: `spectrum/__init__.py` (export `get_backend`)

**Interfaces:**

- Produces: `get_backend() -> str` honoring `SPECTRUM_BACKEND` env (`"c"`,
  `"python"`, default `"c"`); `try_c(path)` context returning callable module or
  None; caches import failure with a single logged warning.

- [ ] Test `tests/test_backend.py`: monkeypatch env → returns; missing module →
      fallback works.
- [ ] Commit: `feat: backend dispatch with python fallback`

### Task 5: Fix bicoherence semantics

**Files:**

- Modify: `spectrum/bicoherence.py` (line 61 `nrecs > 1` → `nrecs == 1`; add
  overlap clamp; accept 1-D via shared helper), `spectrum/tools/matlab.py` (add
  `as_column(y)`), `tests/test_bicoherence.py` (pins: default-nsamp call now
  works; property: QPC peak)

- [ ] Failing tests: `bicoherence(1-D y)` works; `bicoherence(y)` default nsamp
      returns shape (nfft²); peak at (0.1,0.15) on QPC signal (test code in
      audit script).
- [ ] Fix, run, pass.
- [ ] Commit:
      `fix: bicoherence default nsamp, overlap clamp, 1-D input (HOSA semantics)`

### Task 6: Fix cumest semantics

**Files:** Modify: `spectrum/cumest.py`, `tests/test_cumest.py`

- [ ] `nsamp=None` → defaults to data length; `norder=4` →
      `cum4est(..., k1, k2)`; flag normalization (`flag[0] not in 'bB'` →
      unbiased).
- [ ] Tests: default call works; `cumest(y,4) == cum4est(y,...)`; pins updated.
- [ ] Commit: `fix: cumest nsamp default and 4th-order dispatch`

### Task 7: Fix cum4est k2=0 crash

**Files:** Modify: `spectrum/cum4est.py:110-115` (guard empty slices: only
assign `zeros` when length > 0; skip zero-length), `tests/test_cumest.py` (add
k2=0, k1=0 default-path test vs hand-computed Gaussian moment ~0)

- [ ] Commit: `fix: cum4est empty-slice assignment for k2=0`

### Task 8: Fix cum4x k2 branch + debug logging

**Files:** Modify: `spectrum/cum4x.py:139` (`k2 > 2` → `k2 >= 0`), remove
`log.info(nrecs)` / `log.info(y_cum.shape)` (lines 112, 158, 160),
`tests/test_cum4x.py` (k2∈{1,2} cases + parity with cum4x via direct definition
on small random data: `c4(m) = <w*(n) x(n+m) y(n+k1) z*(n+k2)> - products` —
reuse audit ground-truth approach)

- [ ] Commit: `fix: cum4x k2 sign branch and stray logging`

### Task 9: Fix bispectrumd/dx window + geometry

**Files:** Modify: `spectrum/bispectrumd.py`, `spectrum/bispectrumdx.py` (2-D
window path: warn → `opwind` scalar 1 handled, `winsize=1` → skip smoothing,
output stays nfft×nfft; 1-D path: pad `window=[wind, zeros(l-1)]`, `windf`
symmetric, `hankel(flipud(padded), padded)`, `winsize=2l-1`; smoothing:
`convolve2d(mode='same')` with NO extra crop), `tests/test_bispectrumd.py`/`dx`
(shape now (256,256); peak at coupling freq)

- [ ] Commit:
      `fix: bispectrumd/dx smoothing window handling and nfft x nfft output`

### Task 10: Fix estimator input handling (bispectrumi, bicoherencex, trispectrum mean-order)

**Files:** Modify: `spectrum/bispectrumi.py`, `spectrum/bicoherencex.py`,
`spectrum/bispectrumdx.py` (1-D input via `as_column`),
`spectrum/trispectrum.py` (mean removal BEFORE windowing; consistent
unknown-window ValueError in tricoherence)

- [ ] Tests: 1-D input works everywhere; trispectrum parity invariance check.
- [ ] Commit: `fix: 1-D inputs and trispectrum mean-removal order`

### Task 11: Fix armafit/armasel (6 failing tests)

**Files:** Modify: `spectrum/armafit.py:143` area (build Toeplitz from flat 1-D
`c2_pos`; ensure `scipy.linalg.solve` on (p,p) with 1-D rhs),
`spectrum/armasel.py` (all-inf guard → raise `RuntimeError` if no fit
succeeded), `tests/test_armafit.py`/`test_armasel.py` (add parameter-recovery +
order-recovery assertions: AR(2) `a≈[1,-0.5,0.2]` within 0.15; `armasel`
recovers p=2)

- [ ] Full suite green: `pytest tests/ -q` → 0 failed.
- [ ] Commit: `fix: armafit Toeplitz solve shape and armasel degenerate grid`

### Task 12: C port — cum2est + cum2x

**Files:** Create: `spectrum/src/estimators_cum.c` (+ declarations in
`spectrum/src/estimators_cum.h`); Modify: `spectrum/src/_spectrum_core.c`
(binding fns `cum2est`, `cum2x`), `spectrum/cum2est.py`/`cum2x.py` (route via
backend), `tests/test_parity_cum.py`

**Interfaces (C):**

- `sp_status sp_cum2est(const double complex *y, size_t ly, int maxlag, size_t nsamp, int overlap, int biased, double complex *out);`
  (`out` length `2*maxlag+1`)
- `sp_status sp_cum2x(const double complex *x, const double complex *yv, size_t lx, int maxlag, size_t nsamp, int overlap, int biased, double complex *out);`
- Python binding mirrors corrected Python math exactly (segment loop, per-lag
  scaling ramps).

- [ ] Parity tests: random real + complex data, grids: maxlag∈{0,3,10},
      nsamp∈{full, half}, overlap∈{0,50}, flag both; `rtol=1e-10`.
- [ ] Run `pytest tests/test_parity_cum.py -q` → pass.
- [ ] Commit: `feat: C port of cum2est and cum2x with parity tests`

### Task 13: C port — cum3est + cum3x

Same pattern; `sp_cum3est(..., int k1)`, `sp_cum3x(x,y,z,...,k1)`. Grid adds
k1∈{-3,0,2}.

- [ ] Commit: `feat: C port of cum3est and cum3x`

### Task 14: C port — cum4est + cum4x + cumest dispatcher

Same pattern; k1,k2∈{-2,0,1,3} incl. the previously-crashing combos. cumest
stays pure-Python dispatch (calls C-cumulants through the public wrappers).

- [ ] Commit: `feat: C port of cum4est, cum4x`

### Task 15: C port — bicoherence + bicoherencex

**Files:** Create: `spectrum/src/estimators_bispec.c`/`.h`; bindings; route
Python modules; `tests/test_parity_bic.py`.

- `sp_bicoher(y, ly, nfft, wind, use_wind, nsamp, overlap, out_bic, out_waxis)`
  — Hankel-mask accumulation with **lower-triangle + exact mirror**
  optimization; per-thread accumulators over segments.
- Parity vs Python must hold to `rtol=1e-10` including default-nsamp path.
- [ ] Commit: `feat: C port of bicoherence and bicoherencex`

### Task 16: C port — bispectrumd + bispectrumdx

- Rao–Gabr window + 1-D/2-D window paths in C; smoothing = full conv + center
  crop implemented directly (no scipy); output nfft×nfft.
- [ ] Commit: `feat: C port of bispectrumd and bispectrumdx`

### Task 17: C port — bispectrumi

- c3 quadrant assembly + 2-D lag window + `fft2` via row/col 1-D FFT passes.
- [ ] Commit: `feat: C port of bispectrumi`

### Task 18: C port — trispectrum + tricoherence

- Keep reduced-grid contract (spec §3.8); OpenMP over k1 blocks.
- [ ] Commit: `feat: C port of trispectrum and tricoherence`

### Task 19: C kernels — nlgen deterministic iterations

**Files:** Create: `spectrum/src/estimators_ext.c`/`.h`; Modify
`spectrum/nlgen.py` to draw innovations with `np.random.Generator` (seeded) and
pass to C iterator kernels (`sp_nlgen_bilinear`, `_tar`, `_volterra`, `_narma`,
`_henon`, `_logistic`). harmgen stays pure Python (trivial trig; RNG parity).
Parity: same seed → same output as Python loop reference.

- [ ] Commit: `feat: C kernels for nlgen models`

### Task 20: C port — armafit + armasel

**Files:** Create: `spectrum/src/linalg.c`/`.h` (GEPP solve `sp_solve`, Levinson
`sp_levinson`); route `armafit`/`armasel` hot loop through C.

- [ ] Commit: `feat: C port of armafit and armasel with GEPP/Levinson`

### Task 21: Benchmarks + docs

**Files:** Create: `tests/bench/bench_core.py` (timeit C vs Python: bispectrumd
4096/nfft256, bicoherence, trispectrum; print table), `CHANGELOG.md`; Modify:
`README.md` (Backends section: `SPECTRUM_BACKEND`, `SPECTRUM_FFTW`, build
instructions)

- [ ] Run bench; record numbers in CHANGELOG (target ≥10× on bispectrumd).
- [ ] Commit: `docs: benchmarks, changelog, backend docs`

### Task 22: cumtrue (Python + C)

**Files:** Create: `spectrum/cumtrue.py` (port of `CUMTRUE.M` above: impulse
response via ARMA filtering with stability check `max|roots(ar)| < 1`,
`nsamp = max(2*nlags, ceil(log(1e-3)/log(rho)))`; c2/c3/c4 from h-products),
`sp_cumtrue` in `estimators_cum.c`, bindings, `tests/test_cumtrue.py`
(impulse-response parity: `cumtrue(ma=[1], ar=[1], 2)` == sample autocorrelation
of impulse; AR(1) a=0.5: closed-form `c2(m) ∝ 0.5^|m|`; c3 of MA(1) matches
hand-computed values; C↔Python parity)

- [ ] Commit: `feat: cumtrue theoretical ARMA cumulants (port of CUMTRUE.M)`

### Task 23: trispec — HOSA-style full trispectrum

**Files:** Create: `spectrum/trispec.py` (direct 4th-order:
`T(k1,k2,k3) = mean_seg X(k1)X(k2)X(k3)conj(X((k1+k2+k3) mod nfft))`, full nfft³
output with wraparound third index, segment averaging, optional frequency
smoothing skipped (HOSA trispec has no smoothing)), `sp_trispec` in new
`spectrum/src/trispec.c` (+ OpenMP over k1), bindings, `tests/test_trispec.py`
(symmetries `T(k1,k2,k3)=T(perm...)`, DC relation to 4th moment, QPC-cubic peak,
C↔Python parity on small nfft=32)

- [ ] Commit: `feat: trispec direct-method 4th-order spectrum`

### Task 24: wig2 + wig2c

**Files:** Create: `spectrum/wig2.py` (pseudo-Wigner spectrum: for each time
slice `n` at increment `inc`, `W(n,ω)=Σ_k x(n+k)x*(n−k)e^{-j2πkω/nfft}` using
lag kernel `2·Re` convention per HOSA manual: `wx` is `nfft × n_time` matrix;
`wig2c` uses complex kernel `x(n+k)x*(n-k)` without Re folding), `sp_wig2`,
`sp_wig2c` in new `spectrum/src/wigner.c`, bindings, `tests/test_wig2.py`
(marginal property `Σ_n W(n,f) ≈ |X(f)|²` for analytic-consistent signals; real
sinusoid peak at ±f0; parity)

- [ ] Docstring provenance: "HOSA manual + Nikias & Petropulu, Wigner-Ville".
- [ ] Commit: `feat: wig2/wig2c Wigner spectrum`

### Task 25: wig4 + wig4c

**Files:** Modify: `spectrum/wig2.py` or new `spectrum/wig4.py` (Wigner
bispectrum slice `W4(n; ω1, ω2) = Σ_{k1,k2} x*(n+(k1+k2)/2)…` — use discrete
form `Σ_{k1,k2} x(n)…` per Nikias eq.: kernel `x(n+k1)x(n+k2)x*(n+k1+k2)` 2-D
FFT over (k1,k2) per time slice; `wig4c` complex conjugation variant), `sp_wig4`
in `wigner.c`, bindings, `tests/test_wig4.py` (shape, symmetry
`W(ω1,ω2)=W(ω2,ω1)`, parity)

- [ ] Commit: `feat: wig4/wig4c Wigner bispectrum`

### Task 26: tfrwv

**Files:** Create: `spectrum/tfrwv.py` (time-frequency pseudo-WVD map over full
record: output `(n_time, nfft)`; reuses wig2 kernel per slice; `nfft`, `inc`
params), C reuse of `sp_wig2`, `tests/test_tfrwv.py` (chirp ridge tracking: peak
freq follows linear chirp; parity)

- [ ] Commit: `feat: tfrwv Wigner-Ville time-frequency map`

### Task 27: doa

**Files:** Create: `spectrum/doa.py` (4th-order cumulant MUSIC for a ULA: input
`y` shape (snapshots, sensors) or (sensors, snapshots) documented; build `M×M`
spatial cumulant matrix `C_ij = cum4(y_i, y_j*, y_j, y_i*)`-style (denote exact:
`C = Σ_m c4(m) a-shifted` — implement standard:
`C(i,j) = cum(x_i, x_j*, x_i*, x_j)`), Jacobi eigensolve (C) / `np.linalg.eigh`
(Python), noise subspace dim `M-sources` (param `nsources`), pseudospectrum scan
θ∈[-90°,90°], steering `a(θ)=[1,e^{-jπθ?},...]` half-wavelength ULA),
`sp_jacobi_eig` in `linalg.c`, bindings, `tests/test_doa.py` (two sources at
-20°, +30°, 8 sensors, 200 snapshots → peaks within ±3°; parity)

- [ ] Commit: `feat: doa cumulant-MUSIC direction of arrival`

### Task 28: armaqs

**Files:** Create: `spectrum/armaqs.py` (q-slice: AR via 3rd-order cumulant
modified YW: solve `Σ_i a_i c3(m−i, 0) = 0, m=q+1..q+p` using c3(lag, 0) column;
MA via `b(m) c3(q,0)` relations:
`c3(q,m) = γ3 b(m) b(q)... b(m) = c3(q,m)/c3(q,0)·b(0)` iterate-free closed form
per Giannakis), `sp_armaqs` in `estimators_ext.c` (uses
`sp_levinson`/`sp_solve`), bindings, `tests/test_armaqs.py` (ARMA(2,1) synthetic
recovery |â−a|∞ < 0.1; parity))

- [ ] Commit: `feat: armaqs q-slice ARMA estimation`

### Task 29: armarts

**Files:** Create: `spectrum/armarts.py` (residual time-series: AR via 2nd-order
YW on autocorrelation; residuals `e = filter(1,â,y)`; MA from residual
autocorrelation via moment equations; innovation variance from residual power),
`sp_armarts`, bindings, `tests/test_armarts.py` (same ARMA(2,1) recovery;
parity)

- [ ] Commit: `feat: armarts residual time-series ARMA estimation`

### Task 30: Full-suite hardening

- [ ] `pytest tests/ -q` all green (incl. original smoke tests, updated pins,
      parity, property).
- [ ] `SPECTRUM_BACKEND=python pytest tests/ -q` green (fallback path).
- [ ] Rebuild from clean:
      `rm -rf build spectrum/*.so && python setup.py build_ext --inplace && pytest -q`.
- [ ] `pre-commit run --all-files` green.
- [ ] Commit (any stragglers): `test: full suite green across backends`

### Task 31: Version + release prep

**Files:** Modify: `setup.py` (version `0.4.0`, long_description unchanged, add
`ext_modules` finalized, `python_requires>=3.8`), `CHANGELOG.md` (final),
`README.md`

- [ ] `python -m pytest -q` final; `twine check dist/*` after build.
- [ ] Commit: `chore: bump version to 0.4.0 for release`

### Task 32: Build + publish

- [ ] `rm -rf dist build && python setup.py sdist bdist_wheel`
- [ ] `twine check dist/*`
- [ ] Publish: `twine upload dist/*` (needs credentials from `~/.pypirc` or
      `TWINE_USERNAME`/`TWINE_PASSWORD`/token env). If no credentials: stop,
      report exactly: artifacts ready at `dist/`, run `make publish` or provide
      token.
- [ ] Commit nothing (artifacts are gitignored) — final commit if any packaging
      file changed: `chore: release 0.4.0 artifacts`

## Self-Review

- Spec coverage: §3 fixes → Tasks 5–11; §4 layout → Tasks 1–4, 12–20; §5 order →
  respected; §6 all 10 functions → Tasks 22–29; §7 testing → parity/property
  tasks + Task 30; §8 packaging → Tasks 3, 31, 32; §9 process → one commit per
  task above; §10 risks → pocketfft fallback (Task 1), FFTW optional (Task 3),
  fallback import (Task 4, 30). Gap: none.
- Placeholders: none — every step names files, signatures, formulas, or
  commands.
- Type consistency: `sp_*` C signatures prefixed consistently; Python binding
  names match public estimator names; `as_column` helper introduced once
  (Task 5) and reused in Task 10.
