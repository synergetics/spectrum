#!/usr/bin/env python


import numpy as np
import logging
from typing import Any

from tools import make_arr


log = logging.getLogger(__file__)


def cum3x(
    x: np.ndarray[Any, np.dtype[Any]],
    y: np.ndarray[Any, np.dtype[Any]],
    z: np.ndarray[Any, np.dtype[Any]],
    maxlag: int = 0,
    nsamp: int = 0,
    overlap: int = 0,
    flag: str = "biased",
    k1: int = 0,
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Estimate the third-order cross-cumulants of three signals.

    Parameters:
    -----------
    x : np.ndarray[Any, np.dtype[Any]]
        First input data vector or matrix.
    y : np.ndarray[Any, np.dtype[Any]]
        Second input data vector or matrix.
    z : np.ndarray[Any, np.dtype[Any]]
        Third input data vector or matrix.
        x, y, and z must have identical dimensions.
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
        The fixed lag in C3(m,k1) (default is 0).

    Returns:
    --------
    y_cum : np.ndarray[Any, np.dtype[Any]]
        Estimated third-order cross-cumulant,
        E[x^*(n)y(n+m)z(n+k1)], -maxlag <= m <= maxlag

    Notes:
    ------
    The third-order cross-cumulant is a higher-order statistic that can reveal
    non-linear dependencies between three signals that are not captured by
    cross-correlation functions.
    If x, y, and z are matrices, columns are assumed to correspond to independent realizations,
    overlap is set to 0, and samp_seg to the row dimension.
    """

    (lx, nrecs) = x.shape
    if (lx, nrecs) != y.shape or (lx, nrecs) != z.shape:
        raise ValueError("x,y,z should have identical dimensions")

    if lx == 1:
        lx = nrecs
        nrecs = 1

    if maxlag < 0:
        raise ValueError('"maxlag" must be non-negative')
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
        lsamp = lx - abs(k1)
        scale = make_arr((range(lsamp - maxlag, lsamp + 1), range(lsamp - 1, lsamp - maxlag - 1, -1)), axis=1).T
        scale = np.ones([2 * maxlag + 1, 1]) / scale

    if k1 >= 0:
        indx = np.arange(nsamp - k1).T
        indz = np.arange(k1, nsamp).T
    else:
        indx = np.arange(-k1, nsamp).T
        indz = np.arange(nsamp + k1)

    ind = np.arange(nsamp)

    for k in range(nrecs):
        xs = x[ind]
        xs = xs - np.mean(xs)
        ys = y[ind]
        ys = ys - np.mean(ys)
        zs = z[ind]
        zs = zs - np.mean(zs)
        zs = np.conj(zs)

        u = np.zeros([nsamp, 1])
        u[indx] = xs[indx] * zs[indz]

        y_cum[zlag] = y_cum[zlag] + np.dot(u.T, ys)

        for m in range(1, maxlag + 1):
            y_cum[zlag - m] = y_cum[zlag - m] + np.dot(u[m:nsamp].T, ys[0 : nsamp - m])
            y_cum[zlag + m] = y_cum[zlag + m] + np.dot(u[0 : nsamp - m].T, ys[m:nsamp])

        ind = ind + int(nadvance)

    y_cum = y_cum * scale / nrecs

    return y_cum
