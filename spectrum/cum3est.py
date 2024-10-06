#!/usr/bin/env python


import numpy as np
import logging
from typing import Any

from .tools.matlab import make_arr


log = logging.getLogger(__file__)


def cum3est(
    y: np.ndarray[Any, np.dtype[Any]],
    maxlag: int,
    nsamp: int,
    overlap: int,
    flag: str,
    k1: int,
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Estimate the third-order cumulants of a signal.

    This function should be invoked via "CUMEST" for proper parameter checks.

    Parameters:
    -----------
    y : np.ndarray[Any, np.dtype[Any]]
        Input data vector (column).
    maxlag : int
        Maximum lag to be computed.
    nsamp : int
        Samples per segment.
    overlap : int
        Percentage overlap of segments.
    flag : str
        'biased': biased estimates are computed
        'unbiased': unbiased estimates are computed.
    k1 : int
        The fixed lag in C3(m,k1).

    Returns:
    --------
    y_cum : np.ndarray[Any, np.dtype[Any]]
        Estimated third-order cumulant, C3(m,k1) for -maxlag <= m <= maxlag.

    Notes:
    ------
    The third-order cumulant is a higher-order statistic that can reveal
    non-linear dependencies in a signal that are not captured by the
    autocorrelation function.
    """

    (n1, n2) = np.shape(y)
    N = n1 * n2
    minlag = -maxlag
    overlap = int(np.fix(overlap / 100 * nsamp))
    nrecord = int(np.fix((N - overlap) / (nsamp - overlap)))
    nadvance = nsamp - overlap

    y_cum = np.zeros([maxlag - minlag + 1, 1])

    ind = np.arange(nsamp).T
    nlags = 2 * maxlag + 1
    zlag = maxlag
    if flag == "biased":
        scale = np.ones([nlags, 1]) / nsamp
    else:
        lsamp = nsamp - abs(k1)
        scale = make_arr((range(lsamp - maxlag, lsamp + 1), range(lsamp - 1, lsamp - maxlag - 1, -1)), axis=1).T
        (m2, n2) = scale.shape
        scale = np.ones([m2, n2]) / scale

    y = y.ravel(order="F")
    for i in range(nrecord):
        x = y[ind]
        x = x - np.mean(x)
        cx = np.conj(x)
        z = x * 0

        # create the "IV" matrix: offset for second lag
        if k1 > 0:
            z[0 : nsamp - k1] = x[0 : nsamp - k1] * cx[k1:nsamp]
        else:
            z[-k1:nsamp] = x[-k1:nsamp] * cx[0 : nsamp + k1]

        # compute third-order cumulants
        y_cum[zlag] = y_cum[zlag] + np.dot(z.T, x)

        for k in range(1, maxlag + 1):
            y_cum[zlag - k] = y_cum[zlag - k] + np.dot(z[k:nsamp].T, x[0 : nsamp - k])
            y_cum[zlag + k] = y_cum[zlag + k] + np.dot(z[0 : nsamp - k].T, x[k:nsamp])

        ind = ind + int(nadvance)

    y_cum = y_cum * scale / nrecord

    return y_cum
