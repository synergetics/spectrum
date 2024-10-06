#!/usr/bin/env python

import numpy as np
import logging
from scipy.linalg import hankel
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional, Any

from tools import nextpow2, flat_eq, make_arr, shape

log = logging.getLogger(__file__)


def bicoherence(
    y: np.ndarray[Any, np.dtype[Any]],
    nfft: int = 128,
    wind: Optional[np.ndarray[Any, np.dtype[Any]]] = None,
    nsamp: int = 0,
    overlap: int = 50,
) -> Tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """
    Estimate bicoherence using the direct (FFT) method.

    Parameters:
    -----------
    y : np.ndarray
        Input data vector or time-series. If y is a matrix, each column is treated as a separate realization.
    nfft : int, optional
        FFT length (default is 128). The actual size used is the next power of two greater than 'nsamp'.
    wind : np.ndarray, optional
        Time-domain window to be applied to each data segment (default is None, which uses a Hanning window).
    nsamp : int, optional
        Samples per segment (default is 0, which sets it to have 8 segments).
    overlap : int, optional
        Percentage overlap of segments, range [0, 99] (default is 50).

    Returns:
    --------
    bic : np.ndarray
        Estimated bicoherence: an nfft x nfft array, with origin at the center,
        and axes pointing down and to the right.
    waxis : np.ndarray
        Vector of frequencies associated with the rows and columns of bic.
        Sampling frequency is assumed to be 1.

    Notes:
    ------
    The bicoherence is a normalized bispectrum that measures the degree of coupling
    between triple combinations of frequency components.
    """

    # Parameter checks and adjustments
    (ly, nrecs) = y.shape
    if ly == 1:
        y = y.reshape(1, -1)
        ly, nrecs = nrecs, 1

    if nrecs > 1:
        overlap, nsamp = 0, ly

    if nrecs > 1 and nsamp <= 0:
        nsamp = int(np.fix(ly / (8 - 7 * overlap / 100)))
    if nfft < nsamp:
        nfft = 2 ** nextpow2(nsamp)

    overlap = int(np.fix(nsamp * overlap / 100))
    nadvance = nsamp - overlap
    nrecs = int(np.fix((ly * nrecs - overlap) / nadvance))

    if wind is None:
        wind = np.hanning(nsamp)

    if wind.size != nsamp:
        log.info(f"Segment size is {nsamp}")
        log.info(f"Wind array is {wind.shape}")
        log.info("Using default Hanning window")
        wind = np.hanning(nsamp)

    wind = wind.reshape(1, -1)

    # Accumulate triple products
    bic = np.zeros((nfft, nfft))
    Pyy = np.zeros((nfft, 1))

    mask = hankel(np.arange(nfft), np.array([nfft - 1] + list(range(nfft - 1))))
    Yf12 = np.zeros((nfft, nfft))
    ind = np.arange(nsamp)
    y = y.ravel(order="F")

    for k in range(nrecs):
        ys = y[ind]
        ys = (ys.reshape(1, -1) - np.mean(ys)) * wind

        Yf = np.fft.fft(ys, nfft) / nsamp
        CYf = np.conjugate(Yf)
        Pyy = Pyy + flat_eq(Pyy, (Yf * CYf))

        Yf12 = flat_eq(Yf12, CYf.ravel(order="F")[mask])

        bic = bic + ((Yf * np.transpose(Yf)) * Yf12)
        ind = ind + int(nadvance)

    bic = bic / nrecs
    Pyy = Pyy / nrecs
    mask = flat_eq(mask, Pyy.ravel(order="F")[mask])
    bic = abs(bic) ** 2 / ((Pyy * np.transpose(Pyy)) * mask)
    bic = np.fft.fftshift(bic)

    if nfft % 2 == 0:
        waxis = np.transpose(np.arange(-nfft // 2, nfft // 2)) / nfft
    else:
        waxis = np.transpose(np.arange(-(nfft - 1) // 2, (nfft - 1) // 2 + 1)) / nfft

    return bic, waxis


def plot_bicoherence(
    bic: np.ndarray[Any, np.dtype[Any]],
    waxis: np.ndarray[Any, np.dtype[Any]],
) -> None:
    """
    Plot the bicoherence estimate.

    Parameters:
    -----------
    bic : np.ndarray
        Estimated bicoherence array.
    waxis : np.ndarray
        Frequency axis values.

    Returns:
    --------
    None
    """
    cont = plt.contourf(waxis, waxis, bic, 100, cmap="viridis")
    plt.colorbar(cont)
    plt.title("Bicoherence estimated via the direct (FFT) method")
    plt.xlabel("f1")
    plt.ylabel("f2")

    colmax, row = bic.max(0), bic.argmax(0)
    maxval, col = colmax.max(), colmax.argmax()
    log.info(f"Max: bic({waxis[col]:.6f}, {waxis[row[col]]:.6f}) = {maxval:.6f}")
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

    # Estimate bicoherence
    bic, waxis = bicoherence(y.reshape(-1, 1))

    # Plot the results
    plot_bicoherence(bic, waxis)
