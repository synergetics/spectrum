#!/usr/bin/env python

import numpy as np
import logging
from typing import Any, Optional, Tuple

from .tools.matlab import shape, nextpow2

log = logging.getLogger(__file__)


def trispectrum(
    y: np.ndarray[Any, np.dtype[Any]],
    nfft: Optional[int] = None,
    window: Optional[str] = None,
    nsamp: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """
    Estimate the trispectrum (4th order spectrum) using the direct (FFT) method.

    The trispectrum is the Fourier transform of the fourth-order cumulant function
    and provides information about phase coupling and nonlinear interactions in
    the signal at three frequencies.

    Parameters:
    -----------
    y : np.ndarray[Any, np.dtype[Any]]
        Input data vector (column).
    nfft : int, optional
        FFT length. Default is next power of 2 >= nsamp.
    window : str, optional
        Window function name:
        - None or 'rect': Rectangular window
        - 'hanning': Hanning window
        - 'hamming': Hamming window
        - 'blackman': Blackman window
        Default is None (rectangular).
    nsamp : int, optional
        Samples per segment. Default is length of y.
    overlap : int, optional
        Percentage overlap of segments. Default is 0.

    Returns:
    --------
    Tspec : np.ndarray[Any, np.dtype[Any]]
        Estimated trispectrum. 3D array with dimensions corresponding to
        the three frequency variables (f1, f2, f3).
    waxis : np.ndarray[Any, np.dtype[Any]]
        Frequency axis for the trispectrum.

    Notes:
    ------
    The trispectrum T(f1, f2, f3) is defined as:
    T(f1, f2, f3) = E[X(f1) * X(f2) * X(f3) * X*(f1+f2+f3)]

    where X(f) is the Fourier transform of the data segment, E[] denotes
    expectation, and * denotes complex conjugate.

    The trispectrum is useful for:
    - Detecting quadratic phase coupling
    - Analyzing nonlinear systems
    - Detecting non-Gaussian processes

    Due to computational complexity, the trispectrum is computed over a
    reduced frequency domain to make it tractable.
    """

    (n1, n2) = shape(y, 2)
    N = n1 * n2
    y = y.ravel(order="F")

    # Set default parameters
    if nsamp is None:
        nsamp = N
    if overlap is None:
        overlap = 0
    if nfft is None:
        nfft = 2 ** nextpow2(nsamp)

    # Convert overlap percentage to samples
    overlap = int(np.fix(overlap / 100 * nsamp))
    nadvance = nsamp - overlap
    nrecord = int(np.fix((N - overlap) / nadvance))

    # Window function
    if window is None or window == "rect":
        wind = np.ones(nsamp)
    elif window == "hanning":
        wind = np.hanning(nsamp)
    elif window == "hamming":
        wind = np.hamming(nsamp)
    elif window == "blackman":
        wind = np.blackman(nsamp)
    else:
        raise ValueError(f"Unknown window type: {window}")

    # Normalize window
    wind = wind / np.linalg.norm(wind) * np.sqrt(nsamp)

    # Frequency axis
    waxis = np.arange(nfft // 2 + 1) / nfft
    nfreq = len(waxis)

    # Reduce frequency domain for computational efficiency
    # Only compute trispectrum for lower frequencies
    max_freq_idx = min(nfreq, nfft // 8)  # Use only lower 1/8 of frequency range
    waxis_reduced = waxis[:max_freq_idx]

    # Initialize trispectrum accumulator
    Tspec = np.zeros((max_freq_idx, max_freq_idx, max_freq_idx), dtype=complex)

    # Process each segment
    ind = np.arange(nsamp)

    for i in range(nrecord):
        # Extract and window the segment
        x = y[ind] * wind
        x = x - np.mean(x)  # Remove DC

        # Compute FFT
        X = np.fft.fft(x, nfft)
        X = X[:max_freq_idx]  # Keep only lower frequencies

        # Compute trispectrum for this segment
        # T(f1, f2, f3) = X(f1) * X(f2) * X(f3) * X*(f1+f2+f3)
        for k1 in range(max_freq_idx):
            for k2 in range(max_freq_idx):
                for k3 in range(max_freq_idx):
                    # Check frequency constraint: f1 + f2 + f3 < Nyquist
                    k4 = k1 + k2 + k3
                    if k4 < max_freq_idx:
                        Tspec[k1, k2, k3] += X[k1] * X[k2] * X[k3] * np.conj(X[k4])

        ind = ind + nadvance

    # Normalize by number of records
    Tspec = Tspec / nrecord

    return Tspec, waxis_reduced


def tricoherence(
    y: np.ndarray[Any, np.dtype[Any]],
    nfft: Optional[int] = None,
    window: Optional[str] = None,
    nsamp: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """
    Estimate the tricoherence (normalized trispectrum).

    The tricoherence is the normalized version of the trispectrum and
    measures the degree of quadratic phase coupling.

    Parameters:
    -----------
    y : np.ndarray[Any, np.dtype[Any]]
        Input data vector (column).
    nfft : int, optional
        FFT length. Default is next power of 2 >= nsamp.
    window : str, optional
        Window function name. Default is None (rectangular).
    nsamp : int, optional
        Samples per segment. Default is length of y.
    overlap : int, optional
        Percentage overlap of segments. Default is 0.

    Returns:
    --------
    tricoh : np.ndarray[Any, np.dtype[Any]]
        Estimated tricoherence magnitude.
    waxis : np.ndarray[Any, np.dtype[Any]]
        Frequency axis for the tricoherence.

    Notes:
    ------
    The tricoherence is defined as:
    T_norm(f1, f2, f3) = |T(f1, f2, f3)| / sqrt(P(f1)*P(f2)*P(f3)*P(f1+f2+f3))

    where T(f1, f2, f3) is the trispectrum and P(f) is the power spectrum.

    Values range from 0 to 1, where 1 indicates perfect quadratic phase coupling.
    """

    # Compute trispectrum
    Tspec, waxis = trispectrum(y, nfft, window, nsamp, overlap)

    # Compute power spectrum for normalization
    (n1, n2) = shape(y, 2)
    N = n1 * n2
    y = y.ravel(order="F")

    if nsamp is None:
        nsamp = N
    if overlap is None:
        overlap = 0
    if nfft is None:
        nfft = 2 ** nextpow2(nsamp)

    overlap = int(np.fix(overlap / 100 * nsamp))
    nadvance = nsamp - overlap
    nrecord = int(np.fix((N - overlap) / nadvance))

    # Window function
    if window is None or window == "rect":
        wind = np.ones(nsamp)
    elif window == "hanning":
        wind = np.hanning(nsamp)
    elif window == "hamming":
        wind = np.hamming(nsamp)
    elif window == "blackman":
        wind = np.blackman(nsamp)
    else:
        wind = np.ones(nsamp)

    wind = wind / np.linalg.norm(wind) * np.sqrt(nsamp)

    # Compute power spectrum
    max_freq_idx = len(waxis)
    P = np.zeros(max_freq_idx)

    ind = np.arange(nsamp)
    for i in range(nrecord):
        x = y[ind] * wind
        x = x - np.mean(x)
        X = np.fft.fft(x, nfft)
        X = X[:max_freq_idx]
        P += np.abs(X) ** 2
        ind = ind + nadvance

    P = P / nrecord

    # Normalize trispectrum to get tricoherence
    tricoh = np.zeros_like(Tspec, dtype=float)

    for k1 in range(max_freq_idx):
        for k2 in range(max_freq_idx):
            for k3 in range(max_freq_idx):
                k4 = k1 + k2 + k3
                if k4 < max_freq_idx:
                    denom = np.sqrt(P[k1] * P[k2] * P[k3] * P[k4])
                    if denom > 0:
                        tricoh[k1, k2, k3] = min(1.0, np.abs(Tspec[k1, k2, k3]) / denom)

    return tricoh, waxis


def plot_trispectrum(
    Tspec: np.ndarray[Any, np.dtype[Any]],
    waxis: np.ndarray[Any, np.dtype[Any]],
    slice_type: str = "magnitude",
    f3_slice: Optional[int] = None,
    title: str = "Trispectrum",
) -> None:
    """
    Plot 2D slices of the trispectrum.

    Parameters:
    -----------
    Tspec : np.ndarray[Any, np.dtype[Any]]
        Trispectrum array.
    waxis : np.ndarray[Any, np.dtype[Any]]
        Frequency axis.
    slice_type : str, optional
        Type of plot:
        - 'magnitude': Plot magnitude
        - 'phase': Plot phase
        - 'real': Plot real part
        - 'imag': Plot imaginary part
        Default is 'magnitude'.
    f3_slice : int, optional
        Index for f3 slice. If None, uses middle frequency.
    title : str, optional
        Plot title.
    """
    try:
        import matplotlib.pyplot as plt

        if f3_slice is None:
            f3_slice = len(waxis) // 2

        # Extract 2D slice at fixed f3
        if slice_type == "magnitude":
            data = np.abs(Tspec[:, :, f3_slice])
            cmap = "viridis"
        elif slice_type == "phase":
            data = np.angle(Tspec[:, :, f3_slice])
            cmap = "hsv"
        elif slice_type == "real":
            data = np.real(Tspec[:, :, f3_slice])
            cmap = "RdBu_r"
        elif slice_type == "imag":
            data = np.imag(Tspec[:, :, f3_slice])
            cmap = "RdBu_r"
        else:
            data = np.abs(Tspec[:, :, f3_slice])
            cmap = "viridis"

        fig, ax = plt.subplots(figsize=(8, 6))

        F1, F2 = np.meshgrid(waxis, waxis, indexing="ij")
        im = ax.contourf(F1, F2, data, levels=20, cmap=cmap)

        ax.set_xlabel("Frequency f1")
        ax.set_ylabel("Frequency f2")
        ax.set_title(f"{title} ({slice_type}) - f3 = {waxis[f3_slice]:.3f}")

        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()

    except ImportError:
        log.warning("Matplotlib not available for plotting")


def plot_tricoherence_summary(
    tricoh: np.ndarray[Any, np.dtype[Any]], waxis: np.ndarray[Any, np.dtype[Any]], title: str = "Tricoherence Summary"
) -> None:
    """
    Plot summary views of tricoherence.

    Parameters:
    -----------
    tricoh : np.ndarray[Any, np.dtype[Any]]
        Tricoherence array.
    waxis : np.ndarray[Any, np.dtype[Any]]
        Frequency axis.
    title : str, optional
        Plot title.
    """
    try:
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Maximum projection along f3 axis
        max_proj = np.max(tricoh, axis=2)
        F1, F2 = np.meshgrid(waxis, waxis, indexing="ij")
        im1 = ax1.contourf(F1, F2, max_proj, levels=20, cmap="viridis")
        ax1.set_xlabel("Frequency f1")
        ax1.set_ylabel("Frequency f2")
        ax1.set_title("Max Projection (over f3)")
        plt.colorbar(im1, ax=ax1)

        # Mean projection along f3 axis
        mean_proj = np.mean(tricoh, axis=2)
        im2 = ax2.contourf(F1, F2, mean_proj, levels=20, cmap="viridis")
        ax2.set_xlabel("Frequency f1")
        ax2.set_ylabel("Frequency f2")
        ax2.set_title("Mean Projection (over f3)")
        plt.colorbar(im2, ax=ax2)

        # Diagonal slice (f1 = f2)
        diag_indices = np.arange(len(waxis))
        diag_slice = tricoh[diag_indices, diag_indices, :]
        im3 = ax3.contourf(waxis[diag_indices], waxis, diag_slice.T, levels=20, cmap="viridis")
        ax3.set_xlabel("Frequency f1 = f2")
        ax3.set_ylabel("Frequency f3")
        ax3.set_title("Diagonal Slice (f1 = f2)")
        plt.colorbar(im3, ax=ax3)

        # Global maximum over all frequencies
        max_values = np.max(np.max(tricoh, axis=1), axis=1)
        ax4.plot(waxis, max_values, "b-", linewidth=2)
        ax4.set_xlabel("Frequency")
        ax4.set_ylabel("Maximum Tricoherence")
        ax4.set_title("Global Maximum vs Frequency")
        ax4.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    except ImportError:
        log.warning("Matplotlib not available for plotting")


def detect_quadratic_coupling(
    tricoh: np.ndarray[Any, np.dtype[Any]], waxis: np.ndarray[Any, np.dtype[Any]], threshold: float = 0.7
) -> list:
    """
    Detect significant quadratic phase coupling.

    Parameters:
    -----------
    tricoh : np.ndarray[Any, np.dtype[Any]]
        Tricoherence array.
    waxis : np.ndarray[Any, np.dtype[Any]]
        Frequency axis.
    threshold : float, optional
        Threshold for significant coupling. Default is 0.7.

    Returns:
    --------
    couplings : list
        List of tuples (f1, f2, f3, tricoh_value) for significant couplings.
    """

    couplings = []

    for k1 in range(len(waxis)):
        for k2 in range(len(waxis)):
            for k3 in range(len(waxis)):
                if tricoh[k1, k2, k3] > threshold:
                    couplings.append((waxis[k1], waxis[k2], waxis[k3], tricoh[k1, k2, k3]))

    # Sort by tricoherence value (descending)
    couplings.sort(key=lambda x: x[3], reverse=True)

    return couplings
