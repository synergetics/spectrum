#!/usr/bin/env python


import numpy as np
import logging
from typing import Any

from tools import make_arr


log = logging.getLogger(__file__)


def cum2x(
    x: np.ndarray[Any, np.dtype[Any]],
    y: np.ndarray[Any, np.dtype[Any]],
    maxlag: int = 0,
    nsamp: int = 0,
    overlap: int = 0,
    flag: str = "biased",
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Estimate the cross-covariance (2nd order cross-cumulant) function.

    Parameters:
    -----------
    x : np.ndarray[Any, np.dtype[Any]]
        First input data vector or matrix.
    y : np.ndarray[Any, np.dtype[Any]]
        Second input data vector or matrix.
        x and y must have identical dimensions.
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

    Returns:
    --------
    y_cum : np.ndarray[Any, np.dtype[Any]]
        Estimated cross-covariance,
        E[x^*(n)y(n+m)], -maxlag <= m <= maxlag

    Notes:
    ------
    The cross-covariance function is a measure of the linear dependence between
    two signals as a function of the time lag between them.
    If x and y are matrices, columns are assumed to correspond to independent realizations,
    overlap is set to 0, and samp_seg to the row dimension.
    """

    (lx, nrecs) = x.shape
    if (lx, nrecs) != y.shape:
        raise ValueError("x,y should have identical dimensions")

    if lx == 1:
        lx = nrecs
        nrecs = 1

    if maxlag < 0:
        raise ValueError("maxlag must be non-negative")
    if nrecs > 1:
        nsamp = lx
    if nsamp <= 0 or nsamp > lx:
        nsamp = lx
    if nrecs > 1:
        overlap = 0
    overlap = max(0, min(overlap, 99))

    overlap = int(np.fix(overlap / 100 * nsamp))
    nadvance = nsamp - overlap
    if nrecs == 1:
        nrecs = int(np.fix((lx - overlap) / nadvance))

    nlags = 2 * maxlag + 1
    zlag = maxlag
    y_cum = np.zeros([nlags, 1])

    if flag == "biased":
        scale = np.ones([nlags, 1]) / nsamp
    else:
        scale = make_arr((range(lx - maxlag, lx + 1), range(lx - 1, lx - maxlag - 1, -1)), axis=1).T
        scale = np.ones([2 * maxlag + 1, 1]) / scale

    ind = np.arange(nsamp).T
    for k in range(nrecs):
        xs = x[ind].ravel(order="F")
        xs = xs - np.mean(xs)
        ys = y[ind].ravel(order="F")
        ys = ys - np.mean(ys)

        y_cum[zlag] = y_cum[zlag] + np.dot(xs, ys)

        for m in range(1, maxlag + 1):
            y_cum[zlag - m] = y_cum[zlag - m] + np.dot(xs[m:nsamp].T, ys[0 : nsamp - m])
            y_cum[zlag + m] = y_cum[zlag + m] + np.dot(xs[0 : nsamp - m].T, ys[m:nsamp])

        ind = ind + int(nadvance)

    y_cum = y_cum * scale / nrecs

    return y_cum
