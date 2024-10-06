#!/usr/bin/env python

import numpy as np
import logging
from scipy.linalg import hankel
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from typing import Tuple, Any, Union

from tools import nextpow2, flat_eq

log = logging.getLogger(__file__)


def bispectrumd(
    y: np.ndarray[Any, np.dtype[Any]],
    nfft: int = 128,
    wind: Union[int, np.ndarray[Any, np.dtype[Any]]] = 5,
    nsamp: int = 0,
    overlap: int = 50,
) -> Tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """
    Estimate the bispectrum using the direct (FFT) method.

    Parameters:
    -----------
    y : np.ndarray[Any, np.dtype[Any]]
        Input data vector or time-series.
    nfft : int, optional
        FFT length (default is 128). The actual size used is the next power of two greater than 'nsamp'.
    wind : Union[int, np.ndarray[Any, np.dtype[Any]]], optional
        Window specification for frequency-domain smoothing (default is 5).
        If 'wind' is a scalar, it specifies the length of the side of the square for the Rao-Gabr optimal window.
        If 'wind' is a vector, a 2D window will be calculated via w2(i,j) = wind(i) * wind(j) * wind(i+j).
        If 'wind' is a matrix, it specifies the 2-D filter directly.
    nsamp : int, optional
        Samples per segment (default is 0, which sets it to have 8 segments).
    overlap : int, optional
        Percentage overlap of segments, range [0, 99] (default is 50).

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
    The bispectrum is a higher-order spectral analysis technique that provides information
    about the interaction between different frequency components in a signal.
    """

    (ly, nrecs) = y.shape

    overlap = min(99, max(overlap, 0))
    if nrecs > 1:
        overlap = 0
        nsamp = ly
    if nrecs == 1 and nsamp <= 0:
        nsamp = int(np.fix(ly / (8 - 7 * overlap / 100)))
    if nfft < nsamp:
        nfft = 2 ** nextpow2(nsamp)
    overlap = int(np.fix(nsamp * overlap / 100))
    nadvance = nsamp - overlap
    nrecs = int(np.fix((ly * nrecs - overlap) / nadvance))

    # Create the 2-D window
    if isinstance(wind, (int, np.integer)):
        m = n = 1
        winsize = wind
        if winsize < 0:
            winsize = 5
        winsize = winsize - (winsize % 2) + 1
        if winsize > 1:
            mwind = np.fix(nfft / winsize)
            lby2 = (winsize - 1) / 2
            theta = np.array([np.arange(-lby2, lby2 + 1)])
            opwind = np.ones((winsize, 1)) * (theta**2)
            opwind = opwind + opwind.T + (theta.T * theta)
            opwind = 1 - ((2 * mwind / nfft) ** 2) * opwind
            Hex = np.ones((winsize, 1)) * theta
            Hex = abs(Hex) + abs(Hex.T) + abs(Hex + Hex.T)
            Hex = Hex < winsize
            opwind = opwind * Hex
            opwind = opwind * (4 * mwind**2) / (7 * np.pi**2)
        else:
            opwind = 1
    elif isinstance(wind, np.ndarray) and wind.ndim == 1:
        windf = np.concatenate((wind[:0:-1], wind))
        opwind = (windf[:, np.newaxis] * windf) * hankel(np.flipud(wind), wind)
        winsize = len(wind)
    elif isinstance(wind, np.ndarray):
        winsize = wind.shape[0]
        if wind.shape[0] != wind.shape[1]:
            log.info("2-D window is not square: window ignored")
            wind = 1
            winsize = wind.shape[0]
        if winsize % 2 == 0:
            log.info("2-D window does not have odd length: window ignored")
            wind = 1
            winsize = wind.shape[0]
        opwind = wind

    # Accumulate triple products
    Bspec = np.zeros((nfft, nfft))
    mask = hankel(np.arange(nfft), np.array([nfft - 1] + list(range(nfft - 1))))
    locseg = np.arange(nsamp).T
    y = y.ravel(order="F")

    for krec in range(nrecs):
        xseg = y[locseg].reshape(1, -1)
        Xf = np.fft.fft(xseg - np.mean(xseg), nfft) / nsamp
        CXf = np.conjugate(Xf).ravel(order="F")
        Bspec = Bspec + flat_eq(Bspec, (Xf * Xf.T) * CXf[mask].reshape(nfft, nfft))
        locseg = locseg + int(nadvance)

    Bspec = np.fft.fftshift(Bspec) / nrecs

    # Frequency-domain smoothing
    if winsize > 1:
        lby2 = int((winsize - 1) / 2)
        Bspec = convolve2d(Bspec, opwind, mode="same")
        Bspec = Bspec[lby2 : lby2 + nfft, lby2 : lby2 + nfft]

    if nfft % 2 == 0:
        waxis = np.transpose(np.arange(-nfft // 2, nfft // 2)) / nfft
    else:
        waxis = np.transpose(np.arange(-(nfft - 1) // 2, (nfft - 1) // 2 + 1)) / nfft

    return Bspec, waxis


def plot_bispectrumd(
    Bspec: np.ndarray[Any, np.dtype[Any]],
    waxis: np.ndarray[Any, np.dtype[Any]],
) -> None:
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
    plt.title("Bispectrum estimated via the direct (FFT) method")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.show()
