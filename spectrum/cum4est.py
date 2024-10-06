#!/usr/bin/env python


import numpy as np
import logging
from typing import Any

from .tools.matlab import shape
from .cum2est import cum2est
from .cum2x import cum2x


log = logging.getLogger(__file__)


def cum4est(
    y: np.ndarray[Any, np.dtype[Any]],
    maxlag: int = 0,
    nsamp: int = 0,
    overlap: int = 0,
    flag: str = "biased",
    k1: int = 0,
    k2: int = 0,
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Estimate the fourth-order cumulants of a signal.

    Parameters:
    -----------
    y : np.ndarray[Any, np.dtype[Any]]
        Input data vector or time-series (column vector).
    maxlag : int, optional
        Maximum lag to be computed (default is 0).
    nsamp : int, optional
        Samples per segment (default is 0, which uses the entire data length).
    overlap : int, optional
        Percentage overlap of segments (default is 0).
        Overlap is clipped to the allowed range of [0,99].
    flag : str, optional
        'biased': biased estimates are computed (default)
        'unbiased': unbiased estimates are computed.
    k1 : int, optional
        The fixed lag in C4(m,k1,k2) (default is 0).
    k2 : int, optional
        The fixed lag in C4(m,k1,k2) (default is 0).

    Returns:
    --------
    y_cum : np.ndarray[Any, np.dtype[Any]]
        Estimated fourth-order cumulant slice C4(m,k1,k2), -maxlag <= m <= maxlag

    Notes:
    ------
    The fourth-order cumulant is a higher-order statistic that can reveal
    non-linear dependencies in a signal that are not captured by lower-order
    cumulants. It's particularly useful for analyzing non-Gaussian processes.
    """

    (n1, n2) = shape(y, 2)
    N = n1 * n2
    overlap0 = overlap
    overlap = int(np.fix(overlap / 100 * nsamp))
    nrecord = int(np.fix((N - overlap) / (nsamp - overlap)))
    nadvance = nsamp - overlap

    # scale factors for unbiased estimates
    nlags = 2 * maxlag + 1
    zlag = maxlag
    tmp = np.zeros([nlags, 1])
    if flag == "biased":
        scale = np.ones([nlags, 1]) / nsamp
    else:
        ind = np.arange(-maxlag, maxlag + 1).T
        kmin = min(0, min(k1, k2))
        kmax = max(0, max(k1, k2))
        scale = nsamp - np.maximum(ind, kmax) + np.minimum(ind, kmin)
        scale = np.ones(nlags) / scale
        scale = scale.reshape(-1, 1)

    mlag = maxlag + max(abs(k1), abs(k2))
    mlag = max(mlag, abs(k1 - k2))
    mlag1 = mlag + 1
    nlag = maxlag
    np.zeros([2 * maxlag + 1, 1])

    if np.any(np.any(np.imag(y) != 0)):
        complex_flag = 1
    else:
        complex_flag = 0

    # estimate second- and fourth-order moments combine
    y_cum = np.zeros([2 * maxlag + 1, 1])
    R_yy = np.zeros([2 * mlag + 1, 1])

    ind = np.arange(nsamp)
    for i in range(nrecord):
        tmp = np.zeros([2 * maxlag + 1, 1])
        x = y[ind]
        x = x.ravel(order="F") - np.mean(x)
        z = x * 0
        cx = np.conj(x)

        # create the "IV" matrix: offset for second lag
        if k1 >= 0:
            z[0 : nsamp - k1] = x[0 : nsamp - k1] * cx[k1:nsamp]
        else:
            z[-k1:nsamp] = x[-k1:nsamp] * cx[0 : nsamp + k1]

        # create the "IV" matrix: offset for third lag
        if k2 >= 0:
            z[0 : nsamp - k2] = z[0 : nsamp - k2] * x[k2:nsamp]
            z[nsamp - k2 : nsamp] = np.zeros([k2, 1])
        else:
            z[-k2:nsamp] = z[-k2:nsamp] * x[0 : nsamp + k2]
            z[0:-k2] = np.zeros([-k2, 1])

        tmp[zlag] = tmp[zlag] + np.dot(z.T, x)

        for k in range(1, maxlag + 1):
            tmp[zlag - k] = tmp[zlag - k] + np.dot(z[k:nsamp].T, x[0 : nsamp - k])
            tmp[zlag + k] = tmp[zlag + k] + np.dot(z[0 : nsamp - k].T, x[k:nsamp])

        y_cum = y_cum + tmp * scale

        R_yy = cum2est(x, mlag, nsamp, overlap0, flag)
        #  We need E x(t)x(t+tau) stuff also:
        if complex_flag:
            M_yy = cum2x(np.conj(x), x, mlag, nsamp, overlap0, flag)
        else:
            M_yy = R_yy

        y_cum = (
            y_cum
            - R_yy[mlag1 + k1 - 1] * R_yy[mlag1 - k2 - nlag - 1 : mlag1 - k2 + nlag]
            - R_yy[k1 - k2 + mlag1 - 1] * R_yy[mlag1 - nlag - 1 : mlag1 + nlag]
            - M_yy[mlag1 + k2 - 1].T * M_yy[mlag1 - k1 - nlag - 1 : mlag1 - k1 + nlag]
        )

        ind = ind + int(nadvance)

    y_cum = y_cum / nrecord

    return y_cum
