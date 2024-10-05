import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional


def bicoherencex(
    w: Union[np.ndarray, torch.Tensor],
    x: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    nfft: Optional[int] = None,
    window: Optional[Union[np.ndarray, torch.Tensor]] = None,
    nsamp: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Direct (FD) method for estimating cross-bicoherence.

    Parameters:
    -----------
    w, x, y : array_like
        Input data vectors or time-series with identical dimensions.
    nfft : int, optional
        FFT length. If None, uses the next power of two > nsamp.
    window : array_like, optional
        Time-domain window to be applied to each data segment.
        If None, a Hanning window is used.
    nsamp : int, optional
        Samples per segment. If None, uses the entire data length.
    overlap : int, optional
        Percentage overlap of segments (0-99). If None, uses 50%.

    Returns:
    --------
    bic : ndarray
        Estimated cross-bicoherence: an nfft x nfft array, with origin
        at the center, and axes pointing down and to the right.
    waxis : ndarray
        Vector of frequencies associated with the rows and columns of bic.
    """
    # Determine if we're using PyTorch
    use_torch = isinstance(w, torch.Tensor) or isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor)

    # Use the appropriate library
    lib = torch if use_torch else np

    # Ensure inputs are numpy arrays or PyTorch tensors and have the same shape
    w, x, y = map(lib.asarray, (w, x, y))
    if not (w.shape == x.shape == y.shape):
        raise ValueError("w, x, and y must have identical dimensions")

    # Reshape input if necessary
    if w.ndim == 1:
        w = w.reshape(1, -1)
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
    ly, nrecs = w.shape

    # Set default parameters
    if nfft is None:
        nfft = 128
    if overlap is None:
        overlap = 50 if nrecs == 1 else 0
    if nsamp is None:
        nsamp = ly

    # Adjust parameters
    overlap = lib.clip(overlap, 0, 99)
    overlap = int(nsamp * overlap / 100)
    nadvance = nsamp - overlap
    nrecs = int((ly - overlap) / nadvance)

    if nfft < nsamp:
        nfft = 2 ** int(lib.ceil(lib.log2(lib.tensor(nsamp) if use_torch else nsamp)))

    # Create window
    if window is None:
        window = lib.hann(nsamp) if use_torch else np.hanning(nsamp)
    window = lib.asarray(window).ravel()

    if len(window) != nsamp:
        raise ValueError(f"Window length ({len(window)}) must match nsamp ({nsamp})")

    # Initialize arrays
    bic = lib.zeros((nfft, nfft), dtype=complex)
    Pww = lib.zeros(nfft)
    Pxx = lib.zeros(nfft)
    Pyy = lib.zeros(nfft)

    mask = (
        lib.tensor(np.array([nfft - 1] + list(range(nfft - 1))))
        if use_torch
        else np.array([nfft - 1] + list(range(nfft - 1)))
    )
    Yf12 = lib.zeros((nfft, nfft), dtype=complex)

    # Main loop for cross-bispectrum estimation
    for k in range(nrecs):
        ind = slice(k * nadvance, k * nadvance + nsamp)
        ws = w.flatten()[ind]
        xs = x.flatten()[ind]
        ys = y.flatten()[ind]

        ws = (ws - lib.mean(ws)) * window
        xs = (xs - lib.mean(xs)) * window
        ys = (ys - lib.mean(ys)) * window

        Wf = lib.fft.fft(ws, nfft) / nsamp
        Xf = lib.fft.fft(xs, nfft) / nsamp
        Yf = lib.fft.fft(ys, nfft) / nsamp

        CYf = lib.conj(Yf)

        Pww += lib.abs(Wf) ** 2
        Pxx += lib.abs(Xf) ** 2
        Pyy += lib.abs(Yf) ** 2

        Yf12 = CYf[mask].reshape(nfft, nfft)
        bic += lib.outer(Wf, Xf) * Yf12

    # Normalize and compute cross-bicoherence
    bic /= nrecs
    Pww /= nrecs
    Pxx /= nrecs
    Pyy /= nrecs

    bic = lib.abs(bic) ** 2 / (lib.outer(Pww, Pxx) * Pyy[mask].reshape(nfft, nfft))
    bic = lib.fft.fftshift(bic)

    # Compute frequency axis
    waxis = lib.fft.fftshift(lib.fft.fftfreq(nfft))

    return bic, waxis


def plot_cross_bicoherence(
    bic: Union[np.ndarray, torch.Tensor], waxis: Union[np.ndarray, torch.Tensor], title: str = "Cross-Bicoherence"
) -> None:
    """
    Plot the cross-bicoherence estimate.

    Parameters:
    -----------
    bic : ndarray
        Cross-bicoherence estimate from the bicoherencex function.
    waxis : ndarray
        Frequency axis from the bicoherencex function.
    title : str, optional
        Title for the plot.
    """
    # Convert to numpy if tensors
    if isinstance(bic, torch.Tensor):
        bic = bic.cpu().numpy()
    if isinstance(waxis, torch.Tensor):
        waxis = waxis.cpu().numpy()

    plt.figure(figsize=(10, 8))
    cont = plt.contourf(waxis, waxis, bic, 100, cmap=plt.cm.Spectral_r)
    plt.colorbar(cont)
    plt.title(title)
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
