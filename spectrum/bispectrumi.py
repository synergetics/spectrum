#!/usr/bin/env python

import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, Any

from tools import nextpow2, make_arr

np.set_printoptions(linewidth=120)

log = logging.getLogger(__file__)


def bispectrumi(
    y: np.ndarray[Any, np.dtype[Any]],
    nlag: int = 0,
    nsamp: int = 0,
    overlap: int = 0,
    flag: str = "biased",
    nfft: int = 128,
    wind: Optional[Union[int, np.ndarray[Any, np.dtype[Any]]]] = None,
) -> Tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """
    Estimate the bispectrum using the indirect (time-domain) method.

    Parameters:
    -----------
    y : np.ndarray[Any, np.dtype[Any]]
        Input data vector or time-series.
    nlag : int, optional
        Number of lags to compute (default is 0, must be specified).
    nsamp : int, optional
        Samples per segment (default is 0, which uses the row dimension of y).
    overlap : int, optional
        Percentage overlap of segments, range [0, 99] (default is 0).
    flag : str, optional
        'biased' or 'unbiased' (default is 'biased').
    nfft : int, optional
        FFT length to use (default is 128).
    wind : Optional[Union[int, np.ndarray[Any, np.dtype[Any]]]], optional
        Window function to apply:
        If wind=0, the Parzen window is applied (default).
        Otherwise, the hexagonal window with unity values is applied.

    Returns:
    --------
    Bspec : np.ndarray[Any, np.dtype[Any]]
        Estimated bispectrum: an nfft x nfft array, with origin at the center,
        and axes pointing down and to the right.
    waxis : np.ndarray[Any, np.dtype[Any]]
        Vector of frequencies associated with the rows and columns of Bspec.
        Sampling frequency is assumed to be 1.

    Notes:
    ------
    The indirect method estimates the bispectrum by first computing third-order cumulants
    and then applying a 2D Fourier transform.
    """

    (ly, nrecs) = y.shape
    if ly == 1:
        y = y.reshape(1, -1)
        ly, nrecs = nrecs, 1

    overlap = min(99, max(overlap, 0))
    if nrecs > 1:
        overlap = 0
    if not nsamp:
        nsamp = ly
    if nsamp > ly or nsamp <= 0:
        nsamp = ly
    if not nfft:
        nfft = 128
    if wind is None:
        wind = 0

    if nlag == 0:
        nlag = min(nlag, nsamp - 1)
    if nfft < 2 * nlag + 1:
        nfft = 2 ** nextpow2(nsamp)

    # Create the lag window
    Bspec = np.zeros((nfft, nfft))
    if wind == 0:
        indx = np.array([range(1, nlag + 1)]).T
        window = make_arr((1, np.sin(np.pi * indx / nlag) / (np.pi * indx / nlag)), axis=0)
    else:
        window = np.ones((nlag + 1, 1))
    window = make_arr((window, np.zeros((nlag, 1))), axis=0)

    # Cumulants in non-redundant region
    overlap = int(np.fix(nsamp * overlap / 100))
    nadvance = nsamp - overlap
    nrecord = int(np.fix((ly * nrecs - overlap) / nadvance))

    c3 = np.zeros((nlag + 1, nlag + 1))
    ind = np.arange(nsamp)
    y = y.ravel(order="F")

    for k in range(nrecord):
        x = y[ind].ravel(order="F")
        x = x - np.mean(x)
        ind = ind + int(nadvance)

        for j in range(nlag + 1):
            z = x[: nsamp - j] * x[j:nsamp]
            for i in range(j, nlag + 1):
                Sum = np.dot(z[: nsamp - i].T, x[i:nsamp])
                if flag == "biased":
                    Sum = Sum / nsamp
                else:
                    Sum = Sum / (nsamp - i)
                c3[i, j] = c3[i, j] + Sum

    c3 = c3 / nrecord

    # Cumulants elsewhere by symmetry
    c3 = c3 + np.tril(c3, -1).T  # complete I quadrant
    c31 = c3[1 : nlag + 1, 1 : nlag + 1]
    c32 = np.zeros((nlag, nlag))
    c33 = np.zeros((nlag, nlag))
    c34 = np.zeros((nlag, nlag))
    for i in range(nlag):
        x = c31[i:nlag, i]
        c32[nlag - 1 - i, : nlag - i] = x.T
        c34[: nlag - i, nlag - 1 - i] = x
        if i + 1 < nlag:
            x = np.flipud(x[1:])
            c33 = c33 + np.diag(x, i + 1) + np.diag(x, -(i + 1))

    c33 = c33 + np.diag(c3[0, nlag:0:-1])

    cmat = make_arr(
        (
            make_arr((c33, c32, np.zeros((nlag, 1))), axis=1),
            make_arr((make_arr((c34, np.zeros((1, nlag))), axis=0), c3), axis=1),
        ),
        axis=0,
    )

    # Apply lag-domain window
    wcmat = cmat
    if wind != -1:
        indx = np.arange(-nlag, nlag + 1).T
        window = window.reshape(-1, 1)
        for k in range(-nlag, nlag + 1):
            wcmat[:, k + nlag] = (
                cmat[:, k + nlag].reshape(-1, 1) * window[abs(indx - k)] * window[abs(indx)] * window[abs(k)]
            ).reshape(
                -1,
            )

    # Compute 2d-fft, and shift and rotate for proper orientation
    Bspec = np.fft.fft2(wcmat, (nfft, nfft))
    Bspec = np.fft.fftshift(Bspec)  # axes d and r; orig at ctr

    if nfft % 2 == 0:
        waxis = np.transpose(np.arange(-nfft // 2, nfft // 2)) / nfft
    else:
        waxis = np.transpose(np.arange(-(nfft - 1) // 2, (nfft - 1) // 2 + 1)) / nfft

    return Bspec, waxis


def plot_bispectrumi(Bspec: np.ndarray[Any, np.dtype[Any]], waxis: np.ndarray[Any, np.dtype[Any]]) -> None:
    """
    Plot the bispectrum estimate.

    Parameters:
    -----------
    Bspec : np.ndarray[Any, np.dtype[Any]]
        Estimated bispectrum array.
    waxis : np.ndarray[Any, np.dtype[Any]]
        Frequency axis values.

    Returns:
    --------
    None
    """
    cont = plt.contourf(waxis, waxis, abs(Bspec), 100, cmap="viridis")
    plt.colorbar(cont)
    plt.title("Bispectrum estimated via the indirect method")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.show()


if __name__ == "__main__":
    # Example usage
    # Generate some sample data
    t = np.linspace(0, 10, 1000)
    y = (
        np.sin(2 * np.pi * 10 * t)
        + 0.5 * np.sin(2 * np.pi * 20 * t)
        + 0.3 * np.sin(2 * np.pi * 30 * t)
        + np.random.normal(0, 0.1, t.shape)
    )

    # Estimate bispectrum
    Bspec, waxis = bispectrumi(y.reshape(-1, 1), nlag=50)

    # Plot the results
    plot_bispectrumi(Bspec, waxis)
