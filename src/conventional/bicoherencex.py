import numpy as np
from scipy.linalg import hankel
import matplotlib.pyplot as plt


def bicoherencex(w, x, y, nfft=None, window=None, nsamp=None, overlap=None):
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
    # Ensure inputs are numpy arrays and have the same shape
    w, x, y = map(np.asarray, (w, x, y))
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
    overlap = np.clip(overlap, 0, 99)
    overlap = int(nsamp * overlap / 100)
    nadvance = nsamp - overlap
    nrecs = int((ly - overlap) / nadvance)

    if nfft < nsamp:
        nfft = 2 ** int(np.ceil(np.log2(nsamp)))

    # Create window
    if window is None:
        window = np.hanning(nsamp)
    window = np.asarray(window).ravel()

    if len(window) != nsamp:
        raise ValueError(f"Window length ({len(window)}) must match nsamp ({nsamp})")

    # Initialize arrays
    bic = np.zeros((nfft, nfft), dtype=complex)
    Pww = np.zeros(nfft)
    Pxx = np.zeros(nfft)
    Pyy = np.zeros(nfft)

    mask = hankel(np.arange(nfft), np.array([nfft - 1] + list(range(nfft - 1))))
    Yf12 = np.zeros((nfft, nfft), dtype=complex)

    # Main loop for cross-bispectrum estimation
    for k in range(nrecs):
        ind = slice(k * nadvance, k * nadvance + nsamp)
        ws = w.flat[ind] - np.mean(w.flat[ind])
        xs = x.flat[ind] - np.mean(x.flat[ind])
        ys = y.flat[ind] - np.mean(y.flat[ind])

        ws *= window
        xs *= window
        ys *= window

        Wf = np.fft.fft(ws, nfft) / nsamp
        Xf = np.fft.fft(xs, nfft) / nsamp
        Yf = np.fft.fft(ys, nfft) / nsamp

        CWf = np.conj(Wf)
        CYf = np.conj(Yf)

        Pww += np.abs(Wf) ** 2
        Pxx += np.abs(Xf) ** 2
        Pyy += np.abs(Yf) ** 2

        Yf12 = CYf[mask].reshape(nfft, nfft)
        bic += np.outer(Wf, Xf) * Yf12

    # Normalize and compute cross-bicoherence
    bic /= nrecs
    Pww /= nrecs
    Pxx /= nrecs
    Pyy /= nrecs

    bic = np.abs(bic) ** 2 / (np.outer(Pww, Pxx) * Pyy[mask].reshape(nfft, nfft))
    bic = np.fft.fftshift(bic)

    # Compute frequency axis
    waxis = np.fft.fftshift(np.fft.fftfreq(nfft))

    return bic, waxis


def plot_cross_bicoherence(bic, waxis, title="Cross-Bicoherence"):
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


def test_bicoherencex():
    """
    Test function for cross-bicoherence estimation.
    """
    # Generate simple test signals
    t = np.linspace(0, 10, 1000)
    f1, f2 = 5, 10
    w = np.sin(2 * np.pi * f1 * t)
    x = np.sin(2 * np.pi * f2 * t)
    y = np.sin(2 * np.pi * (f1 + f2) * t) + 0.1 * np.random.randn(len(t))

    # Compute and plot cross-bicoherence
    bic, waxis = bicoherencex(w, x, y, nfft=256, nsamp=256)
    plot_bicoherence(bic, waxis, title="Cross-Bicoherence Estimate")


if __name__ == "__main__":
    test_bicoherencex()
