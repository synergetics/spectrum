"""ARMA parameter estimation via the q-slice method.

Implements the q-slice estimator (Giannakis; HOSA toolbox manual):

1. The output third-order cumulants C3(m, 0) obey the AR recursion
   beyond the MA order: C3(m, 0) + sum_i a_i C3(m-i, 0) = 0 for
   m = q+1 .. q+p. This Toeplitz system gives the AR parameters.
2. Filtering y by 1/A(z) gives residuals e(n) that form an MA(q).
   For an MA(q) the cumulant q-slice is diagonal:
   C3_e(q, m) = gamma3 * b(0) * b(q) * b(m), so
   b(m) = C3_e(q, m) / C3_e(q, 0).

Consistent for non-Gaussian driving noise; the heavy loops run on the
C cumulant kernels.
"""

from typing import Any, Tuple

import numpy as np
from scipy.linalg import solve as lin_solve

from ._backend import use_c  # noqa: F401  (hot loops dispatch inside cumest)
from .cumest import cumest
from .tools.matlab import as_column


def armaqs(
    y: np.ndarray[Any, np.dtype[Any]],
    p: int = 2,
    q: int = 1,
    maxlag: Any = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate ARMA(p, q) parameters via the cumulant q-slice method.

    Parameters:
    -----------
    y : np.ndarray
        Input data (single realization).
    p, q : int
        AR and MA orders.
    maxlag : int, optional
        Maximum cumulant lag (default p + q + 10).

    Returns:
    --------
    a : np.ndarray
        AR parameters [1, a(1), ..., a(p)].
    b : np.ndarray
        MA parameters [1, b(1), ..., b(q)].
    """
    y = as_column(y)
    if maxlag is None:
        maxlag = p + q + 10
    if p < 0 or q < 0:
        raise ValueError("p and q must be non-negative")

    # 3rd-order cumulant slice C3(m, 0), m = -maxlag..maxlag
    c3 = np.asarray(cumest(y, norder=3, maxlag=maxlag, nsamp=y.size, flag="biased")).ravel()
    z = maxlag  # index of lag 0

    if p > 0:
        # modified Yule-Walker on the cumulant slice: rows m = q+1..q+p
        A = np.empty((p, p))
        r = np.empty(p)
        for k in range(p):
            m = q + 1 + k
            for i in range(p):
                A[k, i] = c3[z + m - (i + 1)]
            r[k] = -c3[z + m]
        try:
            ar = lin_solve(A, r)
        except Exception:
            ar = np.linalg.lstsq(A, r, rcond=None)[0]
        a = np.concatenate([[1.0], ar])
    else:
        a = np.array([1.0])

    if q > 0:
        # residual time series e(n) = A(z) y(n)
        yr = np.asarray(y).ravel()
        e = yr.copy()
        for i in range(1, p + 1):
            e[i:] += a[i] * yr[:-i]
        e = e - e.mean()
        ec = e.reshape(-1, 1)
        # q-slice of the residual cumulants: C3_e(q, m), m = 0..q
        cslice = np.empty(q + 1)
        for m in range(q + 1):
            c_m = np.asarray(cumest(ec, norder=3, maxlag=q, nsamp=ec.size, flag="biased", k1=m)).ravel()
            cslice[m] = c_m[q + q]  # lag +q entry of the C3(m1, m) slice
        b = np.ones(q + 1)
        denom = cslice[0]
        for m in range(1, q + 1):
            b[m] = cslice[m] / denom if denom != 0 else 0.0
    else:
        b = np.array([1.0])

    return a, b
