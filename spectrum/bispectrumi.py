import numpy as np
import matplotlib.pyplot as plt


def bispectrumi(y, nlag=None, nsamp=None, overlap=0, flag="biased", nfft=None, wind="parzen"):
    """
    Estimate bispectrum using the indirect method.

    Parameters:
    -----------
    y : array_like
        Input data vector or time-series.
    nlag : int, optional
        Number of lags to compute. Must be specified.
    nsamp : int, optional
        Samples per segment. If None, uses the entire data length.
    overlap : int, optional
        Percentage overlap of segments (0-99).
    flag : str, optional
        'biased' or 'unbiased'. Default is 'biased'.
    nfft : int, optional
        FFT length to use. If None, uses 128 or next power of 2 > 2*nlag+1.
    wind : str or int, optional
        Window function to apply:
        If 'parzen', the Parzen window is applied (default).
        If 0, a rectangular window is applied.
        If int > 0, a Parzen window of that length is applied.

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

    # Check and set parameters
    if nlag is None:
        raise ValueError("nlag must be specified")
    if nsamp is None or nsamp > ly:
        nsamp = ly
    if nfft is None:
        nfft = max(128, 2 ** int(np.ceil(np.log2(2 * nlag + 1))))
    overlap = np.clip(overlap, 0, 99)
    if nrecs > 1:
        overlap = 0

    # Adjust parameters
    nlag = min(nlag, nsamp - 1)
    overlap = int(nsamp * overlap / 100)
    nadvance = nsamp - overlap
    nrecord = int((ly * nrecs - overlap) / nadvance)

    # Create the lag window
    if isinstance(wind, str) and wind.lower() == "parzen":
        window = parzen_window(nlag)
    elif wind == 0:
        window = np.ones(nlag + 1)
    elif isinstance(wind, int) and wind > 0:
        window = parzen_window(wind)
    else:
        raise ValueError("Invalid window specification")

    window = np.concatenate((window, np.zeros(nlag)))

    # Initialize arrays
    c3 = np.zeros((nlag + 1, nlag + 1))
    y = y.ravel()

    # Estimate third-order cumulants
    for k in range(nrecord):
        index = slice(k * nadvance, k * nadvance + nsamp)
        x = y[index] - np.mean(y[index])

        for j in range(nlag + 1):
            z = x[: nsamp - j] * x[j:nsamp]
            for i in range(j, nlag + 1):
                sum_value = np.dot(z[: nsamp - i], x[i:nsamp])
                if flag == "biased":
                    c3[i, j] += sum_value / nsamp
                else:
                    c3[i, j] += sum_value / (nsamp - i)

    c3 /= nrecord

    # Complete the cumulant array by symmetry
    c3 += np.tril(c3, -1).T
    c31 = c3[1:, 1:]
    c32 = np.zeros((nlag, nlag))
    c33 = np.zeros((nlag, nlag))
    c34 = np.zeros((nlag, nlag))

    for i in range(nlag):
        c32[nlag - 1 - i, : nlag - i] = c31[i:nlag, i]
        c34[: nlag - i, nlag - 1 - i] = c31[i:nlag, i]
        if i < nlag - 1:
            x = np.flipud(c31[i + 1 : nlag, i])
            c33 += np.diag(x, i + 1) + np.diag(x, -(i + 1))

    c33 += np.diag(c3[0, nlag:0:-1])

    cmat = np.block([[c33, c32, np.zeros((nlag, 1))], [c34, c3]])

    # Apply lag window
    if wind != 0:
        ind = np.arange(-nlag, nlag + 1)
        window_2d = np.outer(window, window) * window[np.abs(ind)]
        cmat *= window_2d

    # Compute 2D FFT and shift
    Bspec = np.fft.fft2(cmat, (nfft, nfft))
    Bspec = np.fft.fftshift(Bspec)

    # Compute frequency axis
    waxis = np.fft.fftshift(np.fft.fftfreq(nfft))

    return Bspec, waxis


def parzen_window(n):
    """
    Create a Parzen window of length n+1.
    """
    n = int(n)
    if n % 2 == 0:
        n += 1  # Make sure n is odd

    window = np.zeros(n)
    m = (n - 1) / 2

    for k in range(int(m) + 1):
        if k <= m / 2:
            window[k] = 1 - 6 * (k / m) ** 2 * (1 - k / m)
        else:
            window[k] = 2 * (1 - k / m) ** 3

    window[int(m) + 1 :] = window[int(m) - 1 :: -1]
    return window


def plot_bispectrum(Bspec, waxis, title="Bispectrum (Indirect Method)"):
    """
    Plot the bispectrum estimate.

    Parameters:
    -----------
    Bspec : ndarray
        Bispectrum estimate from the bispectrumi function.
    waxis : ndarray
        Frequency axis from the bispectrumi function.
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
