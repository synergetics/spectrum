import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel


def cum4est(y, maxlag=0, nsamp=None, overlap=0, flag="biased", k1=0, k2=0):
    """
    Estimate the fourth-order cumulants of a time series.

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
    k1, k2 : int, optional
        The fixed lags in C4(m,k1,k2). Default is 0.

    Returns:
    --------
    y_cum : ndarray
        Estimated fourth-order cumulant,
        C4(m,k1,k2) for -maxlag <= m <= maxlag
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

    # Compute fourth-order cumulants
    R_yy = np.zeros(2 * maxlag + 1)
    M_yy = np.zeros(2 * maxlag + 1)

    for i in range(nrecs):
        ind = slice(i * nadvance, i * nadvance + nsamp)
        x = y[ind]
        x = x - np.mean(x)
        cx = np.conj(x)

        # Create the "IV" vector
        z = np.zeros_like(x)
        if k1 >= 0:
            z[:-k1] = x[:-k1] * cx[k1:]
        else:
            z[-k1:] = x[-k1:] * cx[:k1]

        if k2 >= 0:
            z[:-k2] *= x[k2:]
            z[nsamp - k2 :] = 0
        else:
            z[-k2:] *= x[:k2]
            z[:k2] = 0

        y_cum[maxlag] += np.dot(z, x)
        R_yy[maxlag] += np.dot(x, cx)
        M_yy[maxlag] += np.dot(x, x)

        for k in range(1, maxlag + 1):
            y_cum[maxlag + k] += np.dot(z[k:], x[:-k])
            y_cum[maxlag - k] += np.dot(z[:-k], x[k:])
            R_yy[maxlag + k] += np.dot(x[k:], cx[:-k])
            R_yy[maxlag - k] += np.dot(x[:-k], cx[k:])
            M_yy[maxlag + k] += np.dot(x[k:], x[:-k])
            M_yy[maxlag - k] += np.dot(x[:-k], x[k:])

    # Normalize
    if flag == "biased":
        scale = np.ones(2 * maxlag + 1) / (nsamp * nrecs)
    else:  # 'unbiased'
        lsamp = nsamp - np.maximum(np.abs(np.arange(-maxlag, maxlag + 1)), np.maximum(np.abs(k1), np.abs(k2)))
        scale = 1 / (nrecs * lsamp)

    y_cum *= scale
    R_yy *= scale
    M_yy *= scale

    # Remove second-order statistics
    y_cum -= (
        R_yy[maxlag + k1] * R_yy[maxlag - k2 : maxlag - k2 + 2 * maxlag + 1]
        + R_yy[maxlag + k1 - k2] * R_yy[maxlag - maxlag : maxlag + maxlag + 1]
        + M_yy[maxlag + k2] * M_yy[maxlag - k1 : maxlag - k1 + 2 * maxlag + 1]
    )

    return y_cum


def plot_fourth_order_cumulant(lags, cumulant, k1, k2, title="Fourth-Order Cumulant"):
    """
    Plot the fourth-order cumulant function.

    Parameters:
    -----------
    lags : array_like
        Lag values.
    cumulant : array_like
        Fourth-order cumulant values.
    k1, k2 : int
        The fixed lag values.
    title : str, optional
        Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lags, cumulant, "b-")
    plt.title(f"{title} (k1 = {k1}, k2 = {k2})")
    plt.xlabel("Lag m")
    plt.ylabel("C4(m,k1,k2)")
    plt.grid(True)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.show()
