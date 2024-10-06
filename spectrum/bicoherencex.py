#!/usr/bin/env python

import numpy as np
from scipy.linalg import hankel
import logging
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Any

from tools import nextpow2, flat_eq, make_arr, shape

log = logging.getLogger(__file__)


def bicoherencex(
    w: np.ndarray[Any, np.dtype[Any]],
    x: np.ndarray[Any, np.dtype[Any]],
    y: np.ndarray[Any, np.dtype[Any]],
    nfft: int = 128,
    wind: Optional[np.ndarray[Any, np.dtype[Any]]] = None,
    nsamp: int = 0,
    overlap: int = 50,
) -> Tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """
    Estimate cross-bicoherence using the direct (FFT) method.

    Parameters:
    -----------
    w : np.ndarray
        First input data vector or time-series.
    x : np.ndarray
        Second input data vector or time-series.
    y : np.ndarray
        Third input data vector or time-series.
        w, x, and y should have identical dimensions.
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
        Estimated cross-bicoherence: an nfft x nfft array, with origin at the center,
        and axes pointing down and to the right.
    waxis : np.ndarray
        Vector of frequencies associated with the rows and columns of bic.
        Sampling frequency is assumed to be 1.

    Notes:
    ------
    The cross-bicoherence is a normalized cross-bispectrum that measures the degree of coupling
    between triple combinations of frequency components from three different signals.
    """

    if w.shape != x.shape or x.shape != y.shape:
        raise ValueError("w, x and y should have identical dimensions")

    (ly, nrecs) = y.shape
    if ly == 1:
        ly, nrecs = nrecs, 1
        w = w.reshape(1, -1)
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

    if nfft < nsamp:
        nfft = 2 ** nextpow2(nsamp)

    overlap = max(0, min(overlap, 99))
    if nrecs > 1:
        overlap, nsamp = 0, ly
    if nrecs == 1 and nsamp <= 0:
        nsamp = int(np.fix(ly / (8 - 7 * overlap / 100)))

    overlap = int(np.fix(overlap / 100 * nsamp))
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
    Pww = np.zeros((nfft, 1))
    Pxx = np.zeros((nfft, 1))

    mask = hankel(np.arange(nfft), np.array([nfft - 1] + list(range(nfft - 1))))
    Yf12 = np.zeros((nfft, nfft))
    ind = np.arange(nsamp)
    w = w.ravel(order="F")
    x = x.ravel(order="F")
    y = y.ravel(order="F")

    for k in range(nrecs):
        ws = w[ind]
        ws = (ws - np.mean(ws)) * wind
        Wf = np.fft.fft(ws, nfft) / nsamp
        CWf = np.conjugate(Wf)
        Pww = Pww + flat_eq(Pww, (Wf * CWf))

        xs = x[ind]
        xs = (xs - np.mean(xs)) * wind
        Xf = np.fft.fft(xs, nfft) / nsamp
        CXf = np.conjugate(Xf)
        Pxx = Pxx + flat_eq(Pxx, (Xf * CXf))

        ys = y[ind]
        ys = (ys - np.mean(ys)) * wind
        Yf = np.fft.fft(ys, nfft) / nsamp
        CYf = np.conjugate(Yf)
        Pyy = Pyy + flat_eq(Pyy, (Yf * CYf))

        Yf12 = flat_eq(Yf12, CYf.ravel(order="F")[mask])
        bic = bic + (Wf * np.transpose(Xf)) * Yf12

        ind = ind + int(nadvance)

    bic = bic / nrecs
    Pww = Pww / nrecs
    Pxx = Pxx / nrecs
    Pyy = Pyy / nrecs
    mask = flat_eq(mask, Pyy.ravel(order="F")[mask])

    bic = abs(bic) ** 2 / ((Pww * np.transpose(Pxx)) * mask)
    bic = np.fft.fftshift(bic)

    if nfft % 2 == 0:
        waxis = np.transpose(np.arange(-nfft // 2, nfft // 2)) / nfft
    else:
        waxis = np.transpose(np.arange(-(nfft - 1) // 2, (nfft - 1) // 2 + 1)) / nfft

    return bic, waxis


def plot_bicoherencex(
    bic: np.ndarray[Any, np.dtype[Any]],
    waxis: np.ndarray[Any, np.dtype[Any]],
) -> None:
    """
    Plot the cross-bicoherence estimate.

    Parameters:
    -----------
    bic : np.ndarray
        Estimated cross-bicoherence array.
    waxis : np.ndarray
        Frequency axis values.

    Returns:
    --------
    None
    """
    cont = plt.contourf(waxis, waxis, bic, 100, cmap="viridis")
    plt.colorbar(cont)
    plt.title("Cross-Bicoherence estimated via the direct (FFT) method")
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
    w = np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.1, t.shape)
    x = np.sin(2 * np.pi * 20 * t) + np.random.normal(0, 0.1, t.shape)
    y = np.sin(2 * np.pi * 30 * t) + 0.5 * np.sin(2 * np.pi * (10 + 20) * t) + np.random.normal(0, 0.1, t.shape)

    # Estimate cross-bicoherence
    bic, waxis = bicoherencex(w.reshape(-1, 1), x.reshape(-1, 1), y.reshape(-1, 1))

    # Plot the results
    plot_bicoherencex(bic, waxis)
