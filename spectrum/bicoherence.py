#!/usr/bin/env python


import numpy as np
import logging
from scipy.linalg import hankel
import scipy.io as sio
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
    Direct (FD) method for estimating bicoherence
    Parameters:
      y     - data vector or time-series
      nfft - fft length [default = power of two > segsamp]
             actual size used is power of two greater than 'nsamp'
      wind - specifies the time-domain window to be applied to each
             data segment; should be of length 'segsamp' (see below);
        otherwise, the default Hanning window is used.
      segsamp - samples per segment [default: such that we have 8 segments]
              - if x is a matrix, segsamp is set to the number of rows
      overlap - percentage overlap, allowed range [0,99]. [default = 50];
              - if x is a matrix, overlap is set to 0.

    Output:
      bic     - estimated bicoherence: an nfft x nfft array, with origin
                at the center, and axes pointing down and to the right.
      waxis   - vector of frequencies associated with the rows and columns
                of bic;  sampling frequency is assumed to be 1.
    """

    # Parameter checks

    (ly, nrecs) = y.shape
    if ly == 1:
        y = y.reshape(1, -1)
        ly = nrecs
        nrecs = 1

    if nrecs > 1:
        overlap = 0
    if nrecs > 1:
        nsamp = ly

    if nrecs > 1 and nsamp <= 0:
        nsamp = int(np.fix(ly / (8 - 7 * overlap / 100)))
    if nfft < nsamp:
        nfft = 2 ** nextpow2(nsamp)

    overlap = int(np.fix(nsamp * overlap / 100))
    nadvance = nsamp - overlap
    nrecs = int(np.fix((ly * nrecs - overlap) / nadvance))

    if not wind:
        wind = np.hanning(nsamp)

    try:
        (rw, cw) = wind.shape
    except ValueError:
        (rw,) = wind.shape
        cw = 1

    if min(rw, cw) == 1 or max(rw, cw) == nsamp:
        log.info("Segment size is " + str(nsamp))
        log.info("Wind array is " + str(rw) + " by " + str(cw))
        log.info("Using default Hanning window")
        wind = np.hanning(nsamp)

    wind = wind.reshape(1, -1)

    # Accumulate triple products

    bic = np.zeros([nfft, nfft])
    Pyy = np.zeros([nfft, 1])

    mask = hankel(np.arange(nfft), np.array([nfft - 1] + range(nfft - 1)))  # type: ignore
    Yf12 = np.zeros([nfft, nfft])
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
        waxis = np.transpose(np.arange(-1 * nfft / 2, nfft / 2)) / nfft
    else:
        waxis = np.transpose(np.arange(-1 * (nfft - 1) / 2, (nfft - 1) / 2 + 1)) / nfft

    cont = plt.contourf(waxis, waxis, bic, 100, cmap="viridis")
    plt.colorbar(cont)
    plt.title("Bicoherence estimated via the direct (FFT) method")
    plt.xlabel("f1")
    plt.ylabel("f2")

    colmax, row = bic.max(0), bic.argmax(0)
    maxval, col = colmax.max(0), colmax.argmax(0)
    log.info("Max: bic(" + str(waxis[col]) + "," + str(waxis[col]) + ") = " + str(maxval))
    plt.show()

    return (bic, waxis)
