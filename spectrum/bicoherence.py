import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional
import torch


def bicoherence(
    y: Union[np.ndarray, torch.Tensor],
    nfft: Optional[int] = None,
    window: Optional[Union[np.ndarray, torch.Tensor]] = None,
    nsamp: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Direct (FD) method for estimating bicoherence.

    Parameters:
    -----------
    y : array_like
        Input data vector or time-series.
    nfft : int, optional
        FFT length. If None, uses the next power of two > nsamp.
    window : array_like, optional
        Time-domain window to be applied to each data segment.
        If None, a Hanning window is used.
    nsamp : int, optional
        Samples per segment. If None, uses 8 segments.
    overlap : int, optional
        Percentage overlap of segments (0-99). If None, uses 50%.

    Returns:
    --------
    bic : ndarray
        Estimated bicoherence: an nfft x nfft array, with origin
        at the center, and axes pointing down and to the right.
    waxis : ndarray
        Vector of frequencies associated with the rows and columns of bic.
    """
    # Determine if we're using PyTorch
    use_torch = isinstance(y, torch.Tensor)

    # Use the appropriate library
    lib = torch if use_torch else np

    # Reshape input if necessary
    y = lib.asarray(y)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ly, nrecs = y.shape

    # Set default parameters
    if nfft is None:
        nfft = 128
    if overlap is None:
        overlap = 50 if nrecs == 1 else 0
    if nsamp is None:
        nsamp = ly if nrecs > 1 else 0

    # Adjust parameters
    if nrecs == 1 and nsamp <= 0:
        nsamp = int(ly / (8 - 7 * overlap / 100))
    if nfft < nsamp:
        nfft = 2 ** int(lib.ceil(lib.log2(lib.tensor(nsamp) if use_torch else nsamp)))

    overlap = int(nsamp * overlap / 100)
    nadvance = nsamp - overlap
    nrecs = int((ly * nrecs - overlap) / nadvance)

    # Create window
    if window is None:
        window = lib.hann(nsamp) if use_torch else np.hanning(nsamp)
    window = lib.asarray(window).ravel()

    if len(window) != nsamp:
        raise ValueError(f"Window length ({len(window)}) must match nsamp ({nsamp})")

    # Initialize arrays
    bic = lib.zeros((nfft, nfft), dtype=complex)
    Pyy = lib.zeros(nfft)
    mask = (
        lib.tensor(hankel(lib.arange(nfft), lib.array([nfft - 1] + list(range(nfft - 1)))))
        if use_torch
        else hankel(np.arange(nfft), np.array([nfft - 1] + list(range(nfft - 1))))
    )
    Yf12 = lib.zeros((nfft, nfft), dtype=complex)

    # Main loop for bispectrum estimation
    for k in range(nrecs):
        ind = slice(k * nadvance, k * nadvance + nsamp)
        ys = y.flatten()[ind]
        ys = (ys - lib.mean(ys)) * window

        Yf = lib.fft.fft(ys, nfft) / nsamp
        CYf = lib.conj(Yf)
        Pyy += lib.abs(Yf) ** 2

        Yf12 = CYf[mask].reshape(nfft, nfft)
        bic += lib.outer(Yf, Yf) * Yf12

    # Normalize and compute bicoherence
    bic /= nrecs
    Pyy /= nrecs
    bic = lib.abs(bic) ** 2 / (lib.outer(Pyy, Pyy) * Pyy[mask].reshape(nfft, nfft))
    bic = lib.fft.fftshift(bic)

    # Compute frequency axis
    waxis = lib.fft.fftshift(lib.fft.fftfreq(nfft))

    return bic, waxis


def plot_bicoherence(bic: Union[np.ndarray, torch.Tensor], waxis: Union[np.ndarray, torch.Tensor]) -> None:
    """
    Plot the bicoherence estimate.

    Parameters:
    -----------
    bic : ndarray
        Bicoherence estimate from the bicoherence function.
    waxis : ndarray
        Frequency axis from the bicoherence function.
    """
    # Convert to numpy if tensors
    if isinstance(bic, torch.Tensor):
        bic = bic.cpu().numpy()
    if isinstance(waxis, torch.Tensor):
        waxis = waxis.cpu().numpy()

    plt.figure(figsize=(10, 8))
    cont = plt.contourf(waxis, waxis, bic, 100, cmap=plt.cm.Spectral_r)
    plt.colorbar(cont)
    plt.title("Bicoherence estimated via the direct (FFT) method")
    plt.xlabel("f1")
    plt.ylabel("f2")

    # Find and annotate maximum
    max_val = np.max(bic)
    max_idx = np.unravel_index(np.argmax(bic), bic.shape)
    plt.plot(waxis[max_idx[1]], waxis[max_idx[0]], "r*", markersize=15)
    plt.annotate(
        f"Max: {max_val:.3f}",
        xy=(waxis[max_idx[1]], waxis[max_idx[0]]),
        xytext=(10, 10),
        textcoords="offset points",
        color="red",
    )

    plt.show()
