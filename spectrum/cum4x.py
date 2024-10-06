#!/usr/bin/env python


import numpy as np
import logging
from scipy.linalg import hankel
import scipy.io as sio
import matplotlib.pyplot as plt
from typing import Any

from tools import nextpow2, flat_eq, make_arr, shape
from cum2x import cum2x


log = logging.getLogger(__file__)


def cum4x(
    w: np.ndarray[Any, np.dtype[Any]],
    x: np.ndarray[Any, np.dtype[Any]],
    y: np.ndarray[Any, np.dtype[Any]],
    z: np.ndarray[Any, np.dtype[Any]],
    maxlag: int = 0,
    nsamp: int = 0,
    overlap: int = 0,
    flag: str = "biased",
    k1: int = 0,
    k2: int = 0,
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Estimate the fourth-order cross-cumulants of four signals.

    Parameters:
    -----------
    w, x, y, z : np.ndarray[Any, np.dtype[Any]]
        Input data vectors or matrices with identical dimensions.
        If matrices, columns are assumed to correspond to independent realizations.
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
    k1, k2 : int, optional
        The fixed lags in C4(m,k1,k2) (default is 0 for both).

    Returns:
    --------
    y_cum : np.ndarray[Any, np.dtype[Any]]
        Estimated fourth-order cross-cumulant,
        c4(t1,t2,t3) := cum( w^*(t), x(t+t1), y(t+t2), z^*(t+t3) )

    Notes:
    ------
    The fourth-order cross-cumulant is a higher-order statistic that can reveal
    non-linear dependencies between four signals that are not captured by
    lower-order statistics.
    """

    (lx, nrecs) = w.shape
    if (lx, nrecs) != x.shape or (lx, nrecs) != y.shape or (lx, nrecs) != z.shape:
        raise ValueError("w,x,y,z should have identical dimensions")

    if lx == 1:
        lx = nrecs
        nrecs = 1

    if maxlag < 0:
        raise ValueError('"maxlag" must be non-negative ')
    if nrecs > 1:
        nsamp = lx
    if nsamp <= 0 or nsamp > lx:
        nsamp = lx

    if nrecs > 1:
        overlap = 0
    overlap = max(0, min(overlap, 99))

    overlap0 = overlap
    overlap = int(np.fix(overlap / 100 * nsamp))
    nadvance = nsamp - overlap

    if nrecs == 1:
        nrecs = int(np.fix((lx - overlap) / nadvance))

    # scale factors for unbiased estimates
    nlags = 2 * maxlag + 1
    zlag = maxlag

    tmp = np.zeros([nlags, 1])
    if flag == "biased":
        scale = np.ones([nlags, 1]) / nsamp
        sc1 = 1 / nsamp
        sc2 = sc1
        sc12 = sc1
    else:
        ind = np.arange(-maxlag, maxlag + 1).T
        kmin = min(0, min(k1, k2))
        kmax = max(0, max(k1, k2))
        scale = nsamp - np.maximum(ind, kmax) + np.minimum(ind, kmin)
        scale = np.ones(nlags) / scale
        sc1 = 1 / (nsamp - abs(k1))
        sc2 = 1 / (nsamp - abs(k2))
        sc12 = 1 / (nsamp - abs(k1 - k2))
        scale = scale.reshape(-1, 1)

    # estimate second- and fourth-order moments combine
    y_cum = np.zeros([2 * maxlag + 1, 1])
    rind = np.arange(-maxlag, maxlag + 1)
    ind = np.arange(nsamp)

    log.info(nrecs)
    for i in range(nrecs):
        tmp = y_cum * 0
        R_zy = 0
        R_wy = 0
        M_wz = 0
        ws = w[ind]
        ws = ws - np.mean(ws)
        xs = x[ind]
        xs = xs - np.mean(xs)
        ys = y[ind]
        ys = ys - np.mean(ys)
        cys = np.conj(ys)
        zs = z[ind]
        zs = zs - np.mean(zs)

        ziv = xs * 0

        # create the "IV" matrix: offset for second lag
        if k1 >= 0:
            ziv[0 : nsamp - k1] = ws[0 : nsamp - k1] * cys[k1:nsamp]
            R_wy = R_wy + np.dot(ws[0 : nsamp - k1].T, ys[k1:nsamp])
        else:
            ziv[-k1:nsamp] = ws[-k1:nsamp] * cys[0 : nsamp + k1]
            R_wy = R_wy + np.dot(ws[-k1:nsamp].T, ys[0 : nsamp + k1])

        # create the "IV" matrix: offset for third lag
        if k2 > 2:
            ziv[0 : nsamp - k2] = ziv[0 : nsamp - k2] * zs[k2:nsamp]
            ziv[nsamp - k2 : nsamp] = np.zeros([k2, 1])
            M_wz = M_wz + np.dot(ws[0 : nsamp - k2].T, zs[k2:nsamp])
        else:
            ziv[-k2:nsamp] = ziv[-k2:nsamp] * zs[0 : nsamp + k2]
            ziv[0:-k2] = np.zeros([-k2, 1])
            M_wz = M_wz + np.dot(ws[-k2:nsamp].T, zs[0 : nsamp - k2])

        if k1 - k2 >= 0:
            R_zy = R_zy + np.dot(zs[0 : nsamp - k1 + k2].T, ys[k1 - k2 : nsamp])
        else:
            R_zy = R_zy + np.dot(zs[-k1 + k2 : nsamp].T, ys[0 : nsamp - k2 + k1])

        tmp[zlag] = tmp[zlag] + np.dot(ziv.T, xs)
        for k in range(1, maxlag + 1):
            tmp[zlag - k] = tmp[zlag - k] + np.dot(ziv[k:nsamp].T, xs[0 : nsamp - k])
            tmp[zlag + k] = tmp[zlag + k] + np.dot(ziv[0 : nsamp - k].T, xs[k:nsamp])

        log.info(y_cum.shape)
        y_cum = y_cum + tmp * scale  # fourth-order moment estimates done
        log.info(y_cum.shape)

        R_wx = cum2x(ws, xs, maxlag, nsamp, overlap0, flag)
        R_zx = cum2x(zs, xs, maxlag + abs(k2), nsamp, overlap0, flag)
        M_yx = cum2x(cys, xs, maxlag + abs(k1), nsamp, overlap0, flag)

        y_cum = (
            y_cum
            - R_zy * R_wx * sc12
            - R_wy * R_zx[rind - k2 + maxlag + abs(k2)] * sc1
            - M_wz.T * M_yx[rind - k1 + maxlag + abs(k1)] * sc2  # type: ignore
        )

        ind = ind + int(nadvance)

    y_cum = y_cum / nrecs

    return y_cum
