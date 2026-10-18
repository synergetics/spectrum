"""Theoretical cumulants of an ARMA model.

Port of the HOSA MATLAB toolbox function CUMTRUE.M (A. Swami,
United Signals & Systems, Rev 1.3): computes second-, third- and
fourth-order cumulants of an ARMA process from its impulse response
h(n), via

    c2(i)     = sum_n h(n) h(n+i)
    c3(i,j)   = sum_n h(n) h(n+i) h(n+j)
    c4(i,j,k) = sum_n h(n) h(n+i) h(n+j) h(n+k)
"""

import logging
from typing import Any, Optional

import numpy as np

from ._backend import use_c, ccall

log = logging.getLogger(__file__)


def _impulse_response(ma: np.ndarray, ar: np.ndarray, nlags: int) -> np.ndarray:
    """h = filter(ma, ar, [1; zeros(nsamp,1)]) with MATLAB sizing."""
    p = len(ar) - 1
    if p == 0:
        return np.asarray(ma, dtype=float).copy()
    rpoles = np.abs(np.roots(ar))
    if np.any(rpoles >= 1.0):
        raise ValueError("unstable AR polynomial passed")
    rho = float(np.max(rpoles))
    nsamp = max(2 * nlags, int(round(np.log(0.001) / np.log(rho))))
    h = np.zeros(nsamp + 1)
    ma = np.asarray(ma, dtype=float).ravel()
    ar = np.asarray(ar, dtype=float).ravel()
    for n in range(nsamp + 1):
        acc = ma[n] if n < len(ma) else 0.0
        for i in range(1, p + 1):
            if n - i >= 0:
                acc -= ar[i] * h[n - i]
        h[n] = acc / ar[0]
    return h


def cumtrue(
    ma: np.ndarray[Any, np.dtype[Any]],
    ar: Optional[np.ndarray[Any, np.dtype[Any]]] = None,
    norder: int = 3,
    nlags: Optional[int] = None,
    k: int = 0,
):
    """
    Theoretical cumulants of an ARMA model.

    Parameters:
    -----------
    ma : array-like
        MA parameter vector.
    ar : array-like, optional
        AR parameter vector (default [1], pure MA model).
    norder : int
        Cumulant order: 2, 3 or 4 (default 3).
    nlags : int, optional
        Maximum cumulant lag (default p + q).
    k : int
        For norder=4, the third cumulant lag (default 0).

    Returns:
    --------
    cmat : np.ndarray
        norder=2: vector of length 2*nlags+1, C2(m), -nlags <= m <= nlags.
        norder=3: (2*nlags+1) x (2*nlags+1) matrix, C3(i,j) at
                  cmat(i+nlags+1, j+nlags+1); origin at center.
        norder=4: slice C4(i,j,k) in the same layout as norder=3.
    """
    if use_c():
        ma_c = np.asarray(ma, dtype=float).ravel()
        ar_c = np.asarray([1.0]) if ar is None else np.asarray(ar, dtype=float).ravel()
        return ccall("cumtrue", ma_c, ar_c, norder, -1 if nlags is None else nlags, k)

    ma = np.asarray(ma, dtype=float).ravel()
    ar = np.array([1.0]) if ar is None else np.asarray(ar, dtype=float).ravel()
    if ma.ndim != 1 or ar.ndim != 1:
        raise ValueError("ma and ar must be vectors")
    q = len(ma) - 1
    p = len(ar) - 1
    if norder < 2 or norder > 4:
        raise ValueError("norder should be 2, 3 or 4")
    if nlags is None:
        nlags = p + q

    h = _impulse_response(ma, ar, nlags)
    nsamp = len(h)
    maxlag = nlags
    nlag1 = maxlag + 1

    if norder == 2:
        cmat = np.zeros(2 * nlags + 1)
        cmat[maxlag] = float(np.dot(h, h))
        for n in range(1, maxlag + 1):
            cmat[maxlag + n] = float(np.dot(h[: nsamp - n], h[n:])) if nsamp - n > 0 else 0.0
            cmat[maxlag - n] = cmat[maxlag + n]
        return cmat.reshape(-1, 1)

    cmat = np.zeros((2 * nlags + 1, 2 * nlags + 1))
    klag3 = k
    # 4th order: z2(n) = h(n) h(n+k); 3rd order: the chain starts from h
    z2 = np.zeros(nsamp)
    if norder == 3:
        z2[:] = h
    elif klag3 >= 0:
        z2[: nsamp - klag3] = h[: nsamp - klag3] * h[klag3:nsamp]
    else:
        z2[-klag3:nsamp] = h[-klag3:nsamp] * h[: nsamp + klag3]

    for n in range(-maxlag, maxlag + 1):
        z = np.zeros(nsamp)
        if n >= 0:
            z[: nsamp - n] = z2[: nsamp - n] * h[n:nsamp]
        else:
            z[-n:nsamp] = z2[-n:nsamp] * h[: nsamp + n]
        cmat[n + maxlag, maxlag] = float(np.dot(z, h))
        for kk in range(1, maxlag + 1):
            cmat[n + maxlag, maxlag + kk] = float(np.dot(z[: nsamp - kk], h[kk:nsamp])) if nsamp - kk > 0 else 0.0
            cmat[n + maxlag, maxlag - kk] = float(np.dot(h[: nsamp - kk], z[kk:nsamp])) if nsamp - kk > 0 else 0.0
    return cmat
