#!/usr/bin/env python


import numpy as np
import logging
from scipy.linalg import hankel
import scipy.io as sio
import matplotlib.pyplot as plt
from typing import Any

from tools import nextpow2, flat_eq, make_arr, shape


log = logging.getLogger(__file__)


def cum2est(
    y: np.ndarray[Any, np.dtype[Any]],
    maxlag: int,
    nsamp: int,
    overlap: int,
    flag: str,
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Estimate the covariance (2nd order cumulant) function.

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

    Returns:
    --------
    y_cum : np.ndarray[Any, np.dtype[Any]]
        Estimated covariance, C2(m) for -maxlag <= m <= maxlag.

    Notes:
    ------
    The covariance function is a measure of the linear dependence between
    two values of a signal that are separated by a time lag.
    """

    (n1, n2) = shape(y, 2)
    N = n1 * n2
    overlap = np.fix(overlap / 100 * nsamp)  # type: ignore
    nrecord = np.fix((N - overlap) / (nsamp - overlap))
    nadvance = nsamp - overlap

    y_cum = np.zeros([maxlag + 1, 1])
    ind = np.arange(nsamp)
    y = y.ravel(order="F")

    for i in range(nrecord):
        x = y[ind]
        x = x - np.mean(x)

        for k in range(maxlag + 1):
            y_cum[k] = y_cum[k] + np.dot(x[0 : nsamp - k].T, x[k:nsamp])

        ind = ind + int(nadvance)

    if flag == "biased":
        y_cum = y_cum / (nsamp * nrecord)
    else:
        y_cum = y_cum / (nrecord * (nsamp - np.matrix(range(maxlag + 1)).T))
        y_cum = np.asarray(y_cum)

    if maxlag > 0:
        y_cum = make_arr([np.conj(y_cum[maxlag:0:-1]), y_cum], axis=0)

    return y_cum
