#!/usr/bin/env python

import numpy as np
import logging
from scipy.linalg import hankel
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from typing import Tuple, Union, Any

from .tools.matlab import nextpow2, flat_eq

log = logging.getLogger(__file__)


def bispectrumdx(
    x: np.ndarray[Any, np.dtype[Any]],
    y: np.ndarray[Any, np.dtype[Any]],
    z: np.ndarray[Any, np.dtype[Any]],
    nfft: int = 128,
    wind: Union[int, np.ndarray[Any, np.dtype[Any]]] = 5,
    nsamp: int = 0,
    overlap: int = 50,
) -> Tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """
    Estimate the cross-bispectrum using the direct (FFT) method.

    Parameters:
    -----------
    x : np.ndarray[Any, np.dtype[Any]]
        First input data vector or time-series.
    y : np.ndarray[Any, np.dtype[Any]]
        Second input data vector or time-series.
    z : np.ndarray[Any, np.dtype[Any]]
        Third input data vector or time-series.
        x, y, and z should have identical dimensions.
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
        Estimated cross-bispectrum: an nfft x nfft array, with origin at the center,
        and axes pointing down and to the right.
    waxis : np.ndarray[Any, np.dtype[Any]]
        Vector of frequencies associated with the rows and columns of Bspec.
        Sampling frequency is assumed to be 1.

    Notes:
    ------
    The cross-bispectrum is a higher-order spectral analysis technique that provides information
    about the interaction between different frequency components across multiple signals.
    """

    (lx, lrecs) = x.shape
    (ly, nrecs) = y.shape
    (lz, krecs) = z.shape

    if lx != ly or lrecs != nrecs or ly != lz or nrecs != krecs:
        raise ValueError("x, y and z should have identical dimensions")

    if ly == 1:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        z = z.reshape(1, -1)
        ly, nrecs = nrecs, 1

    overlap = max(0, min(overlap, 99))
    if nrecs > 1:
        overlap, nsamp = 0, ly
    if nrecs == 1 and nsamp <= 0:
        nsamp = int(np.fix(ly / (8 - 7 * overlap / 100)))
    if nfft < nsamp:
        nfft = 2 ** nextpow2(nsamp)

    overlap = int(np.fix(overlap / 100 * nsamp))
    nadvance = nsamp - overlap
    nrecs = int(np.fix((ly * nrecs - overlap) / nadvance))

    # Create the 2-D window
    if isinstance(wind, (int, np.integer)):
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
    else:
        winsize = wind.shape[0]
        if wind.shape[0] != wind.shape[1]:
            log.warning("2-D window is not square: window ignored")
            opwind = 1
            winsize = wind.shape[0]
        elif wind.shape[0] % 2 == 0:
            log.warning("2-D window does not have odd length: window ignored")
            opwind = 1
            winsize = wind.shape[0]
        else:
            opwind = wind

    # Accumulate triple products
    Bspec = np.zeros((nfft, nfft))
    mask = hankel(np.arange(nfft), np.array([nfft - 1] + list(range(nfft - 1))))
    locseg = np.arange(nsamp).T
    x = x.ravel(order="F")
    y = y.ravel(order="F")
    z = z.ravel(order="F")

    for krec in range(nrecs):
        xseg = x[locseg].reshape(1, -1)
        yseg = y[locseg].reshape(1, -1)
        zseg = z[locseg].reshape(1, -1)

        Xf = np.fft.fft(xseg - np.mean(xseg), nfft) / nsamp
        Yf = np.fft.fft(yseg - np.mean(yseg), nfft) / nsamp
        CZf = np.fft.fft(zseg - np.mean(zseg), nfft) / nsamp
        CZf = np.conjugate(CZf).ravel(order="F")

        Bspec = Bspec + flat_eq(Bspec, (Xf * Yf.T) * CZf[mask].reshape(nfft, nfft))
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


def plot_bispectrumdx(Bspec: np.ndarray[Any, np.dtype[Any]], waxis: np.ndarray[Any, np.dtype[Any]]) -> None:
    """
    Plot the cross-bispectrum estimate.

    Parameters:
    -----------
    Bspec : np.ndarray[Any, np.dtype[Any]]
        Estimated cross-bispectrum array.
    waxis : np.ndarray[Any, np.dtype[Any]]
        Frequency axis values.

    Returns:
    --------
    None
    """
    cont = plt.contourf(waxis, waxis, abs(Bspec), 100, cmap="viridis")
    plt.colorbar(cont)
    plt.title("Cross-Bispectrum estimated via the direct (FFT) method")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.show()
