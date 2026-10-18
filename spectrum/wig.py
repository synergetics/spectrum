"""Wigner-Ville higher-order time-frequency distributions.

Implements the Wigner spectrum (2nd order), the Wigner bispectrum
(3rd-order Wigner-type spectrum) and the pseudo Wigner-Ville
time-frequency representation, following the definitions in the HOSA
toolbox manual and Nikias & Petropulu, "Higher-Order Spectra Analysis"
(the original WIG*.M sources were not recoverable; provenance is
documented per function).

Definitions used (discrete time, lag set m in [-nfft/2, nfft/2)):

  wig2:   W2(n, w)  = sum_m  x(n+m) x(n-m)       e^{-j w m}   (real input)
  wig2c:  W2(n, w)  = sum_m  x(n+m) conj(x(n-m)) e^{-j w m}   (complex)
  wig4:   W4(n; w1, w2) = sum_{k1,k2} x(n+k1) x(n+k2) x(n+k1+k2)
  wig4c:  W4(n; w1, w2) = sum_{k1,k2} x(n+k1) x(n+k2) conj(x(n+k1+k2))
  tfrwv:  pseudo-Wigner-Ville map = wig2 evaluated on a time grid

All lag kernels are zero-padded to nfft points and transformed with a
DFT; negative lags wrap to the upper half of the buffer (standard FFT
frequency convention).
"""

from typing import Any, Optional, Tuple

import numpy as np

from ._backend import use_c, ccall
from .tools.matlab import as_column


def _wig2_generic(x, nfft, inc, complex_version):
    N = x.size
    n_time = max(1, (N + inc - 1) // inc)
    W = np.zeros((n_time, nfft), dtype=complex if complex_version else float)
    M = nfft // 2
    for it in range(n_time):
        n0 = it * inc
        kern = np.zeros(nfft, dtype=complex)
        for m in range(-M, M):
            if m >= 0:
                idx = m
            else:
                idx = nfft + m
            a = x[n0 + m] if 0 <= n0 + m < N else 0.0
            b = x[n0 - m] if 0 <= n0 - m < N else 0.0
            if complex_version:
                kern[idx] = a * np.conj(b)
            else:
                kern[idx] = a * b
        spec = np.fft.fft(kern)
        if complex_version:
            W[it] = spec
        else:
            W[it] = spec.real
    return W


def _wig4_generic(x, nfft, inc, complex_version):
    N = x.size
    n_time = max(1, (N + inc - 1) // inc)
    W = np.zeros((n_time, nfft, nfft), dtype=complex)
    M = nfft // 2
    for it in range(n_time):
        n0 = it * inc
        kern = np.zeros((nfft, nfft), dtype=complex)
        for k1 in range(-M, M):
            a = x[n0 + k1] if 0 <= n0 + k1 < N else 0.0
            i1 = k1 if k1 >= 0 else nfft + k1
            for k2 in range(-M, M):
                b = x[n0 + k2] if 0 <= n0 + k2 < N else 0.0
                c = x[n0 + k1 + k2] if 0 <= n0 + k1 + k2 < N else 0.0
                i2 = k2 if k2 >= 0 else nfft + k2
                kern[i1, i2] = a * b * (np.conj(c) if complex_version else c)
        W[it] = np.fft.fft2(kern)
    return W


def wig2(
    x: np.ndarray[Any, np.dtype[Any]],
    nfft: int = 128,
    inc: int = 1,
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Wigner spectrum of a real signal (2nd order).

    Returns W of shape (n_time, nfft); row i is the Wigner spectrum at
    time i*inc. Frequency axis: np.fft.fftfreq(nfft). Note the Wigner
    inner-product effect: a real tone at frequency f contributes an
    auto-term at 2f (plus a DC term from the time-varying phase).
    """
    if use_c():
        return np.asarray(ccall("wig2", as_column(x).ravel(), nfft, inc)).real
    x = np.asarray(x).ravel()
    return _wig2_generic(x, nfft, inc, complex_version=False)


def wig2c(
    x: np.ndarray[Any, np.dtype[Any]],
    nfft: int = 128,
    inc: int = 1,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Wigner spectrum of a complex signal (2nd order)."""
    if use_c():
        return ccall("wig2c", as_column(x).ravel(), nfft, inc)
    x = np.asarray(x).ravel()
    return _wig2_generic(x, nfft, inc, complex_version=True)


def wig4(
    x: np.ndarray[Any, np.dtype[Any]],
    nfft: int = 32,
    inc: int = 1,
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Wigner bispectrum (3rd-order Wigner spectrum) of a real signal.

    Returns W of shape (n_time, nfft, nfft).
    """
    if use_c():
        return np.asarray(ccall("wig4", as_column(x).ravel(), nfft, inc))
    x = np.asarray(x).ravel()
    return _wig4_generic(x, nfft, inc, complex_version=False)


def wig4c(
    x: np.ndarray[Any, np.dtype[Any]],
    nfft: int = 32,
    inc: int = 1,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Wigner bispectrum of a complex signal."""
    if use_c():
        return ccall("wig4c", as_column(x).ravel(), nfft, inc)
    x = np.asarray(x).ravel()
    return _wig4_generic(x, nfft, inc, complex_version=True)


def tfrwv(
    y: np.ndarray[Any, np.dtype[Any]],
    nfft: int = 128,
    inc: Optional[int] = None,
) -> Tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """
    Pseudo Wigner-Ville time-frequency representation.

    Parameters:
    -----------
    y : np.ndarray
        Input signal (real).
    nfft : int
        Number of frequency bins per time slice (default 128).
    inc : int, optional
        Time increment between slices (default: max(1, nfft // 8)).

    Returns:
    --------
    W : np.ndarray
        (n_time, nfft) real matrix; row i is the spectrum at time i*inc.
    t : np.ndarray
        Time instants of the rows.
    """
    x = np.asarray(y).ravel()
    if inc is None:
        inc = max(1, nfft // 8)
    if use_c():
        return ccall("tfrwv", as_column(y).ravel(), nfft, inc)
    W = _wig2_generic(x, nfft, inc, complex_version=False)
    t = np.arange(W.shape[0]) * inc
    return W, t
