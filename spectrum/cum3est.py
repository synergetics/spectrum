import numpy as np
import matplotlib.pyplot as plt


def cum3est(y, maxlag=0, nsamp=None, overlap=0, flag="biased", k1=0):
    """
    Estimate the third-order cumulants of a time series.

    Parameters:
    -----------
    y : array_like
        Input data vector or time-series.
    maxlag : int, optional
        Maximum lag to be computed. Default is 0.
    nsamp : int, optional
        Samples per segment. If None, nsamp is set to len(y).
    overlap : int, optional
        Percentage overlap of segments (0-99). Default is 0.
    flag : str, optional
        'biased' or 'unbiased'. Default is 'biased'.
    k1 : int, optional
        The fixed lag in C3(m,k1). Default is 0.

    Returns:
    --------
    y_cum : ndarray
        Estimated third-order cumulant,
        C3(m,k1) for -maxlag <= m <= maxlag
    """
    y = np.asarray(y).squeeze()

    # Check input dimensions
    if y.ndim != 1:
        raise ValueError("Input time series must be a 1-D array.")

    N = len(y)

    if nsamp is None or nsamp > N:
        nsamp = N

    overlap = np.clip(overlap, 0, 99)
    nadvance = nsamp - int(overlap / 100 * nsamp)

    nrecs = int((N - nsamp) / nadvance) + 1

    y_cum = np.zeros(2 * maxlag + 1)

    # Compute third-order cumulants
    for i in range(nrecs):
        ind = slice(i * nadvance, i * nadvance + nsamp)
        x = y[ind]
        x = x - np.mean(x)
        cx = np.conj(x)

        # Create the "IV" vector
        if k1 >= 0:
            iv = x[:-k1] * cx[k1:]
            ind1 = np.arange(nsamp - k1)
        else:
            iv = x[-k1:] * cx[:k1]
            ind1 = np.arange(-k1, nsamp)

        y_cum[maxlag] += np.dot(iv, x[ind1])

        for k in range(1, maxlag + 1):
            y_cum[maxlag + k] += np.dot(iv[k:], x[ind1[:-k]])
            y_cum[maxlag - k] += np.dot(iv[:-k], x[ind1[k:]])

    # Normalize
    if flag == "biased":
        y_cum = y_cum / (nsamp * nrecs)
    else:  # 'unbiased'
        lsamp = nsamp - abs(k1)
        y_cum = y_cum / (nrecs * (lsamp - np.abs(np.arange(-maxlag, maxlag + 1))))

    return y_cum


def plot_third_order_cumulant(lags, cumulant, k1, title="Third-Order Cumulant"):
    """
    Plot the third-order cumulant function.

    Parameters:
    -----------
    lags : array_like
        Lag values.
    cumulant : array_like
        Third-order cumulant values.
    k1 : int
        The fixed lag value.
    title : str, optional
        Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lags, cumulant, "b-")
    plt.title(f"{title} (k1 = {k1})")
    plt.xlabel("Lag m")
    plt.ylabel("C3(m,k1)")
    plt.grid(True)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.show()
