import numpy as np
from scipy.linalg import hankel
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


def bispectrumd(y, nfft=None, window=None, nsamp=None, overlap=None):
    """
    Estimate bispectrum using the direct (FFT) method.

    Parameters:
    -----------
    y : array_like
        Input data vector or time-series.
    nfft : int, optional
        FFT length. If None, uses the next power of two > nsamp.
    window : array_like or int, optional
        If array: 2D window for frequency-domain smoothing.
        If int: length of the side of the square for the Rao-Gabr optimal window.
        If None, no frequency-domain smoothing is applied.
    nsamp : int, optional
        Samples per segment. If None, uses 8 segments.
    overlap : int, optional
        Percentage overlap of segments (0-99). If None, uses 50%.

    Returns:
    --------
    Bspec : ndarray
        Estimated bispectrum: an nfft x nfft array, with origin
        at the center, and axes pointing down and to the right.
    waxis : ndarray
        Vector of frequencies associated with the rows and columns of Bspec.
    """
    # Ensure input is a numpy array
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

    # Create the 2-D window for frequency-domain smoothing
    if window is None:
        window = np.ones((1, 1))
    elif np.isscalar(window):
        winsize = int(window)
        if winsize < 0:
            winsize = 5
        winsize = winsize - (winsize % 2) + 1  # make it odd
        if winsize > 1:
            mwind = int(nfft / winsize)
            lby2 = (winsize - 1) // 2
            theta = np.arange(-lby2, lby2 + 1)
            opwind = (
                np.outer(theta**2, np.ones(winsize)) + np.outer(np.ones(winsize), theta**2) + np.outer(theta, theta)
            )
            opwind = 1 - ((2 * mwind / nfft) ** 2) * opwind
            Hex = (
                np.abs(np.outer(theta, np.ones(winsize)))
                + np.abs(np.outer(np.ones(winsize), theta))
                + np.abs(np.outer(theta, theta))
            )
            opwind = opwind * (Hex < winsize)
            opwind = opwind * (4 * mwind**2) / (7 * np.pi**2)
        else:
            opwind = np.ones((1, 1))
    else:
        opwind = np.asarray(window)
        if opwind.ndim != 2 or opwind.shape[0] != opwind.shape[1]:
            raise ValueError("2-D window must be square")

    # Initialize arrays
    Bspec = np.zeros((nfft, nfft), dtype=complex)
    mask = hankel(np.arange(nfft), np.array([nfft - 1] + list(range(nfft - 1))))

    # Main loop for bispectrum estimation
    for krec in range(nrecs):
        ind = slice(krec * nadvance, krec * nadvance + nsamp)
        xseg = y.flat[ind]
        xseg = xseg - np.mean(xseg)

        Xf = np.fft.fft(xseg, nfft) / nsamp
        CXf = np.conj(Xf)

        Bspec += np.outer(Xf, Xf) * CXf[mask].reshape(nfft, nfft)

    Bspec = np.fft.fftshift(Bspec) / nrecs

    # Frequency-domain smoothing
    if opwind.size > 1:
        Bspec = convolve2d(Bspec, opwind, mode="same")

    # Compute frequency axis
    waxis = np.fft.fftshift(np.fft.fftfreq(nfft))

    return Bspec, waxis


def plot_bispectrum(Bspec, waxis, title="Bispectrum"):
    """
    Plot the bispectrum estimate.

    Parameters:
    -----------
    Bspec : ndarray
        Bispectrum estimate from the bispectrumd function.
    waxis : ndarray
        Frequency axis from the bispectrumd function.
    title : str, optional
        Title for the plot.
    """
    plt.figure(figsize=(10, 8))
    cont = plt.contourf(waxis, waxis, np.abs(Bspec), 100, cmap=plt.cm.Spectral_r)
    plt.colorbar(cont)
    plt.title(title)
    plt.xlabel("f1")
    plt.ylabel("f2")

    # Find and annotate maximum
    max_val = np.max(np.abs(Bspec))
    max_idx = np.unravel_index(np.argmax(np.abs(Bspec)), Bspec.shape)
    plt.plot(waxis[max_idx[1]], waxis[max_idx[0]], "r*", markersize=15)
    plt.annotate(
        f"Max: {max_val:.3f}",
        xy=(waxis[max_idx[1]], waxis[max_idx[0]]),
        xytext=(10, 10),
        textcoords="offset points",
        color="red",
    )

    plt.show()


def test_bispectrumd():
    """
    Test function for bispectrum estimation using the direct method.
    """
    # Generate a simple test signal
    t = np.linspace(0, 10, 1000)
    f1, f2 = 5, 10
    y = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + 0.5 * np.sin(2 * np.pi * (f1 + f2) * t)
    y += 0.1 * np.random.randn(len(t))

    # Compute and plot bispectrum
    Bspec, waxis = bispectrumd(y, nfft=256, nsamp=256, window=5)
    plot_bispectrum(Bspec, waxis, title="Bispectrum Estimate (Direct Method)")


if __name__ == "__main__":
    test_bispectrumd()
