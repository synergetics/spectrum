import numpy as np
from scipy.linalg import hankel
import matplotlib.pyplot as plt


def bicoherence(y, nfft=None, window=None, nsamp=None, overlap=None):
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
    # Reshape input if necessary
    y = np.asarray(y)
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
        nfft = 2 ** int(np.ceil(np.log2(nsamp)))

    overlap = int(nsamp * overlap / 100)
    nadvance = nsamp - overlap
    nrecs = int((ly * nrecs - overlap) / nadvance)

    # Create window
    if window is None:
        window = np.hanning(nsamp)
    window = np.asarray(window).ravel()

    if len(window) != nsamp:
        raise ValueError(f"Window length ({len(window)}) must match nsamp ({nsamp})")

    # Initialize arrays
    bic = np.zeros((nfft, nfft), dtype=complex)
    Pyy = np.zeros(nfft)
    mask = hankel(np.arange(nfft), np.array([nfft - 1] + list(range(nfft - 1))))
    Yf12 = np.zeros((nfft, nfft), dtype=complex)

    # Main loop for bispectrum estimation
    for k in range(nrecs):
        ind = slice(k * nadvance, k * nadvance + nsamp)
        ys = y.flat[ind]
        ys = (ys - np.mean(ys)) * window

        Yf = np.fft.fft(ys, nfft) / nsamp
        CYf = np.conj(Yf)
        Pyy += np.abs(Yf) ** 2

        Yf12 = CYf[mask].reshape(nfft, nfft)
        bic += np.outer(Yf, Yf) * Yf12

    # Normalize and compute bicoherence
    bic /= nrecs
    Pyy /= nrecs
    bic = np.abs(bic) ** 2 / (np.outer(Pyy, Pyy) * Pyy[mask].reshape(nfft, nfft))
    bic = np.fft.fftshift(bic)

    # Compute frequency axis
    waxis = np.fft.fftshift(np.fft.fftfreq(nfft))

    return bic, waxis


def plot_bicoherence(bic, waxis):
    """
    Plot the bicoherence estimate.

    Parameters:
    -----------
    bic : ndarray
        Bicoherence estimate from the bicoherence function.
    waxis : ndarray
        Frequency axis from the bicoherence function.
    """
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


def test_bicoherence():
    """
    Test function for bicoherence estimation.
    """
    # Generate a simple test signal
    t = np.linspace(0, 10, 1000)
    f1, f2 = 5, 10
    y = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + 0.5 * np.sin(2 * np.pi * (f1 + f2) * t)
    y += 0.1 * np.random.randn(len(t))

    # Compute and plot bicoherence
    bic, waxis = bicoherence(y, nfft=256, nsamp=256)
    plot_bicoherence(bic, waxis)


if __name__ == "__main__":
    test_bicoherence()
