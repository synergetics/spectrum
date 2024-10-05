import numpy as np
from scipy.linalg import hankel
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


def bispectrumdx(x, y, z, nfft=None, window=None, nsamp=None, overlap=None):
    """
    Estimate cross-bispectrum using the direct (FFT) method.

    Parameters:
    -----------
    x, y, z : array_like
        Input data vectors or time-series with identical dimensions.
    nfft : int, optional
        FFT length. If None, uses the next power of two > nsamp.
    window : array_like or int, optional
        If array: 2D window for frequency-domain smoothing.
        If int: length of the side of the square for the Rao-Gabr optimal window.
        If None, no frequency-domain smoothing is applied.
    nsamp : int, optional
        Samples per segment. If None, uses the entire data length.
    overlap : int, optional
        Percentage overlap of segments (0-99). If None, uses 50%.

    Returns:
    --------
    Bspec : ndarray
        Estimated cross-bispectrum: an nfft x nfft array, with origin
        at the center, and axes pointing down and to the right.
    waxis : ndarray
        Vector of frequencies associated with the rows and columns of Bspec.
    """
    # Ensure inputs are numpy arrays and have the same shape
    x, y, z = map(np.asarray, (x, y, z))
    if not (x.shape == y.shape == z.shape):
        raise ValueError("x, y, and z must have identical dimensions")

    # Reshape input if necessary
    if x.ndim == 1:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        z = z.reshape(1, -1)
    ly, nrecs = x.shape

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

    # Main loop for cross-bispectrum estimation
    for krec in range(nrecs):
        ind = slice(krec * nadvance, krec * nadvance + nsamp)
        xseg = x.flat[ind] - np.mean(x.flat[ind])
        yseg = y.flat[ind] - np.mean(y.flat[ind])
        zseg = z.flat[ind] - np.mean(z.flat[ind])

        Xf = np.fft.fft(xseg, nfft) / nsamp
        Yf = np.fft.fft(yseg, nfft) / nsamp
        Zf = np.fft.fft(zseg, nfft) / nsamp
        CZf = np.conj(Zf)

        Bspec += np.outer(Xf, Yf) * CZf[mask].reshape(nfft, nfft)

    Bspec = np.fft.fftshift(Bspec) / nrecs

    # Frequency-domain smoothing
    if opwind.size > 1:
        Bspec = convolve2d(Bspec, opwind, mode="same")

    # Compute frequency axis
    waxis = np.fft.fftshift(np.fft.fftfreq(nfft))

    return Bspec, waxis


def plot_cross_bispectrum(Bspec, waxis, title="Cross-Bispectrum"):
    """
    Plot the cross-bispectrum estimate.

    Parameters:
    -----------
    Bspec : ndarray
        Cross-bispectrum estimate from the bispectrumdx function.
    waxis : ndarray
        Frequency axis from the bispectrumdx function.
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
