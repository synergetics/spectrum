"""HOSA-style direct-method trispectrum.

Implements the 4th-order polyspectrum by segment-averaged triple
products with wraparound frequency indexing (the 4th-order analogue of
the direct bispectrum):

    T(k1, k2, k3) = E[ X(k1) X(k2) X(k3) conj(X((k1+k2+k3) mod nfft)) ]

Unlike :func:`spectrum.trispectrum` (which computes a reduced low-
frequency grid), this function returns the full nfft^3 cube. Axes are
frequency bins 0..nfft-1 (sampling frequency 1); the fourth frequency
is implied by wraparound.
"""

from typing import Any, Tuple

import numpy as np

from ._backend import use_c, ccall
from .tools.matlab import as_column


def trispec(
    y: np.ndarray[Any, np.dtype[Any]],
    nfft: int = 128,
    nsamp: int = 0,
    overlap: int = 50,
) -> Tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """
    Estimate the trispectrum using the direct (FFT) method.

    Parameters:
    -----------
    y : np.ndarray
        Input data vector or time-series (column = one realization).
    nfft : int
        FFT length (default 128); bumped up to the next power of two
        >= nsamp when too small. Limited to <= 512 (nfft^3 memory).
    nsamp : int
        Samples per segment (default 0: about 8 segments).
    overlap : int
        Percentage overlap of segments, [0, 99] (default 50).

    Returns:
    --------
    Tspec : np.ndarray
        Estimated trispectrum, shape (nfft, nfft, nfft).
    waxis : np.ndarray
        Frequency axis (bins 0..nfft-1, Fs = 1).
    """
    if use_c():
        return ccall("trispec", as_column(y), nfft, nsamp, overlap)

    y = as_column(y)
    (ly, nrecs) = y.shape
    if ly == 1:
        y = y.reshape(1, -1)
        ly, nrecs = nrecs, 1

    overlap = min(99, max(overlap, 0))
    if nrecs > 1:
        overlap, nsamp = 0, ly
    if nrecs == 1 and nsamp <= 0:
        nsamp = int(np.fix(ly / (8 - 7 * overlap / 100)))
    if nfft < nsamp:
        nfft = int(2 ** np.ceil(np.log2(nsamp)))
    if nfft > 512:
        raise ValueError("nfft too large for the full trispectrum cube (use spectrum.trispectrum)")

    overlap = int(np.fix(nsamp * overlap / 100))
    nadvance = nsamp - overlap
    nrecs = int(np.fix((ly * nrecs - overlap) / nadvance))

    y = y.ravel(order="F")
    Tspec = np.zeros((nfft, nfft, nfft), dtype=complex)
    idx = np.arange(nfft)
    wrap = (idx[:, None, None] + idx[None, :, None] + idx[None, None, :]) % nfft

    for krec in range(nrecs):
        xseg = y[krec * nadvance : krec * nadvance + nsamp]
        Xf = np.fft.fft(xseg - np.mean(xseg), nfft) / nsamp
        Xc = np.conj(Xf)
        Tspec += Xf[:, None, None] * Xf[None, :, None] * Xf[None, None, :] * Xc[wrap]

    Tspec /= nrecs
    waxis = np.arange(nfft) / nfft
    return Tspec, waxis
