# Changelog

## 0.4.0

Complete native-C reimplementation of the estimators, the remaining HOSA toolbox
functions, and a release.

### Native C core

- New `spectrum._spectrum_core` CPython extension: all 12 HOSA ports
  (bicoherence, cross-bicoherence, direct/indirect/cross bispectrum, cum2est/x,
  cum3est/x, cum4est/x, cumest), the extensions (trispectrum, tricoherence) and
  the new functions (cumtrue, trispec, wig2/wig2c/wig4/wig4c, tfrwv, doa) run as
  C kernels with OpenMP parallel accumulation. Benchmarks on 24 cores:
  bispectrumd 4.9x, bicoherence 3.4x, trispec 5.4x over the pure-Python
  reference (`tests/bench/bench_core.py`).
- Vendored BSD-3 kissfft (double precision) behind a plan-cached FFT interface;
  optional FFTW hook was dropped in favor of the zero-dependency default.
- `SPECTRUM_BACKEND=python` forces the pure-Python reference path; the package
  remains fully functional when the extension is not built.
- Build knobs: `SPECTRUM_NO_OPENMP=1`, `SPECTRUM_NATIVE=1` (-march=native).

### HOSA semantic fixes (behavior changes)

- `bicoherence`: default `nsamp` selection for single realizations (was a
  guaranteed ZeroDivisionError); overlap clamped to [0, 99]; 1-D input accepted.
- `cumest`: `nsamp=None` defaults to data length (was TypeError); `norder=4`
  dispatches to `cum4est` (previously returned a 3rd-order estimate); flag
  normalization like MATLAB.
- `cum4est`: no longer crashes for `k2=0`.
- `cum4x`: `k2 >= 0` branch (was `k2 > 2`, crashing for k2 in {1, 2}); fixed
  HOSA's own typo in the negative-`k2` `M_wz` product.
- `bispectrumd`/`bispectrumdx`: output is exactly `nfft x nfft` again (a
  double-crop bug returned 254x254); 1-D smoothing windows follow MATLAB's
  zero-padded construction; invalid 2-D windows warn and skip smoothing instead
  of crashing.
- `bicoherencex`: nfft is now bumped after the nsamp default (MATLAB order).
- Cross estimators (`bicoherencex`, `bispectrumdx`): the (i, j) orientation now
  matches MATLAB `BICOHERX`/`BISPECDx` exactly; the old Python port had
  numerator and denominator index roles swapped relative to each other.
- Complex-data support: cumulant estimators now accumulate in complex dtype
  (previously imaginary parts were silently discarded) and use the MATLAB `'`
  conjugation convention; `bispectrumi` likewise.
- `armafit`: fixed the Toeplitz solve that crashed every call, and the
  Yule-Walker sign convention; `armasel` raises when no candidate order fits.
- `nextpow2` misuse fixed everywhere (double exponentiation).

### New HOSA functions

- `cumtrue`: theoretical ARMA cumulants (verbatim port of CUMTRUE.M, including
  the impulse-response sizing and stability check) in Python and C.
- `trispec`: direct-method full trispectrum cube with wraparound fourth index
  (Python + C, OpenMP).
- `wig2`, `wig2c`, `wig4`, `wig4c`: Wigner spectrum and Wigner bispectrum
  (definitions from the HOSA manual and Nikias & Petropulu; the original WIG\*.M
  sources were not recoverable - provenance documented in `spectrum/wig.py`).
- `tfrwv`: pseudo Wigner-Ville time-frequency map.
- `doa`: 4th-order cumulant MUSIC direction-of-arrival for uniform linear arrays
  (reference-sensor cumulant slice; Gaussian noise immune).
- `armaqs`: q-slice ARMA estimation (cumulant modified Yule-Walker + residual
  q-slice MA recovery).
- `armarts`: residual time-series ARMA estimation (modified YW beyond lag q +
  moment-matched minimum-phase MA).

### Deviations from the original plan (documented)

- Signal generators (`harmgen`, `nlgen`) remain pure Python: the innovations are
  drawn from numpy's `Generator` for reproducible seed-parity across platforms,
  and generation is not performance critical.
- `armafit`/`armasel` orchestrate in Python; their hot loop (cumulant
  estimation) already runs on the C kernel.
- `doa`/`armaqs`/`armarts` compute their heavy loops via the C cumulant kernels;
  the tiny eigen/solve steps use numpy/LAPACK.

### Testing

- Parity suites: `tests/test_parity_cum.py` (126 cases) and
  `tests/test_parity_bispec.py` (35 cases) compare C against the corrected
  Python reference at rtol=1e-9 across real/complex data, biased/unbiased flags,
  lag-sign grids, window kinds and segmentation regimes.
- Property tests for every new function (QPC peaks, Gaussian cumulants, ARMA
  parameter recovery, DOA recovery, closed-form cumtrue cases).
- Full suite: 265 tests, green under both backends.
