#!/usr/bin/env python


import numpy as np
import logging
from scipy.linalg import hankel
import scipy.io as sio
import matplotlib.pyplot as plt
from typing import Any, Optional, Union

from tools import nextpow2, flat_eq, make_arr, shape
from cum2est import cum2est
from cum3est import cum3est
from cum4est import cum4est

log = logging.getLogger(__file__)


def cumest(
    y: np.ndarray[Any, np.dtype[Any]],
    norder: int = 2,
    maxlag: int = 0,
    nsamp: Optional[int] = None,
    overlap: int = 0,
    flag: str = "biased",
    k1: int = 0,
    k2: int = 0,
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Estimate the 2nd, 3rd, or 4th order cumulants of a time series.

    This function serves as a unified interface for estimating various orders of cumulants.

    Parameters:
    -----------
    y : np.ndarray[Any, np.dtype[Any]]
        Input time series data. Should be a vector (column or row).
    norder : int, optional
        Cumulant order to compute. Must be 2, 3, or 4 (default is 2).
    maxlag : int, optional
        Maximum cumulant lag to compute (default is 0).
    nsamp : Optional[int], optional
        Samples per segment (default is None, which uses the entire data length).
    overlap : int, optional
        Percentage overlap of segments. Range: [0, 99] (default is 0).
    flag : str, optional
        'biased' or 'unbiased' (default is 'biased').
    k1 : int, optional
        Specify the slice of 3rd or 4th order cumulants (default is 0).
    k2 : int, optional
        Specify the slice of 4th order cumulants (default is 0).

    Returns:
    --------
    y_cum : np.ndarray[Any, np.dtype[Any]]
        Estimated cumulant. The interpretation depends on the order:
        - For norder=2: C2(m), -maxlag <= m <= maxlag
        - For norder=3: C3(m,k1), -maxlag <= m <= maxlag
        - For norder=4: C4(m,k1,k2), -maxlag <= m <= maxlag

    Raises:
    -------
    ValueError
        If norder is not 2, 3, or 4, or if maxlag is negative.

    Notes:
    ------
    - For norder=2, this function computes the autocorrelation function.
    - For norder=3, it computes a slice of the third-order cumulant.
    - For norder=4, it computes a slice of the fourth-order cumulant.
    - The function uses the overlapped segment method for estimation.

    See Also:
    ---------
    cum2est, cum3est, cum4est : Individual functions for specific cumulant orders.
    """

    (ksamp, nrecs) = y.shape
    if ksamp == 1:
        ksamp = nrecs
        nrecs = 1

    if norder < 2 or norder > 4:
        raise ValueError("cumulant order must be 2, 3 or 4")

    if maxlag < 0:
        raise ValueError('"maxlag" must be non-negative')

    if nrecs > 1:
        nsamp = ksamp
    if nsamp <= 0 or nsamp > ksamp:  # type: ignore
        nsamp = ksamp

    if nrecs > 1:
        overlap = 0
    overlap = max(0, min(overlap, 99))

    # estimate the cumulants
    if norder == 2:
        y_cum = cum2est(y, maxlag, nsamp, overlap, flag)
    elif norder == 3:
        y_cum = cum3est(y, maxlag, nsamp, overlap, flag, k1)
    elif norder == 4:
        y_cum = cum3est(y, maxlag, nsamp, overlap, flag, k1, k2)

    return y_cum
