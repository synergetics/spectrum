"""ARMA parameter estimation via the residual time-series method.

Implements the HOSA toolbox's ARMARTS approach (manual description; the
original ARMARTS.M source was not recoverable):

1. AR parameters from the modified Yule-Walker equations on the
   estimated autocorrelation (2nd-order statistics).
2. Residual time series e(n) = A(z) y(n), which is (asymptotically)
   the MA(q) process B(z) u(n).
3. MA parameters by matching the residual autocorrelation to the
   MA moment equations

       r_e(m) = sigma^2 * sum_j b(j) b(j+m),  m = 0..q

   solved as a small nonlinear least-squares problem (minimum-phase
   solution).
"""

from typing import Any, Optional, Tuple

import numpy as np
from scipy.linalg import solve as lin_solve
from scipy.optimize import least_squares

from ._backend import use_c  # noqa: F401  (hot loops dispatch inside cumest)
from .cumest import cumest
from .tools.matlab import as_column


def armarts(
    y: np.ndarray[Any, np.dtype[Any]],
    p: int = 2,
    q: int = 1,
    maxlag: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate ARMA(p, q) parameters via the residual time-series method.

    Parameters:
    -----------
    y : np.ndarray
        Input data (single realization).
    p, q : int
        AR and MA orders.
    maxlag : int, optional
        Maximum autocorrelation lag (default p + q + 10).

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

    r = np.asarray(cumest(y, norder=2, maxlag=maxlag, nsamp=y.size, flag="biased")).ravel()
    z = maxlag

    if p > 0:
        A = np.empty((p, p))
        rhs = np.empty(p)
        # the autocorrelation obeys the AR recursion only beyond lag q
        for k in range(p):
            m = q + 1 + k
            for i in range(1, p + 1):
                A[k, i - 1] = r[z + (i - m)] if i >= m else r[z + (m - i)]
            rhs[k] = -r[z + m]
        try:
            ar = lin_solve(A, rhs)
        except Exception:
            ar = np.linalg.lstsq(A, rhs, rcond=None)[0]
        a = np.concatenate([[1.0], ar])
    else:
        a = np.array([1.0])

    if q > 0:
        yr = np.asarray(y).ravel()
        e = yr.copy()
        for i in range(1, p + 1):
            e[i:] += a[i] * yr[:-i]
        e = e - e.mean()
        re = np.asarray(cumest(e.reshape(-1, 1), norder=2, maxlag=q, nsamp=e.size, flag="biased")).ravel()[q:]
        r0 = re[0]

        def moments_err(bv):
            bfull = np.concatenate([[1.0], bv])
            ssum = 1.0 + float(np.sum(bfull[1:] ** 2))
            errs = [r0 - r0]  # m = 0 is used to eliminate sigma^2
            for m in range(1, q + 1):
                conv = float(np.sum(bfull[: q + 1 - m] * bfull[m:]))
                errs.append(r0 * conv / ssum - re[m])
            return np.asarray(errs)[1:]

        if q == 1:
            # closed form: rho(1) = b / (1 + b^2) -> minimum-phase root
            rho = re[1] / r0
            disc = 1.0 - 4.0 * rho**2
            if disc < 0:
                bv = np.array([np.sign(rho)])
            else:
                bv = np.array([(1.0 - np.sqrt(disc)) / (2.0 * rho)])
        else:
            res = least_squares(moments_err, np.full(q, 0.1), method="lm")
            bv = res.x
        b = np.concatenate([[1.0], bv])
    else:
        b = np.array([1.0])

    return a, b
