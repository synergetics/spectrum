import numpy as np
import matplotlib.pyplot as plt


def cum2x(x, y, maxlag=0, nsamp=None, overlap=0, flag="biased"):
    """
    Estimate the cross-covariance between two time series.

    Parameters:
    -----------
    x, y : array_like
        Input data vectors or time-series.
    maxlag : int, optional
        Maximum lag to be computed. Default is 0.
    nsamp : int, optional
        Samples per segment. If None, nsamp is set to min(len(x), len(y)).
    overlap : int, optional
        Percentage overlap of segments (0-99). Default is 0.
    flag : str, optional
        'biased' or 'unbiased'. Default is 'biased'.

    Returns:
    --------
    y_cum : ndarray
        Estimated cross-covariance,
        C_xy(m) for -maxlag <= m <= maxlag
    """
    x = np.asarray(x).squeeze()
    y = np.asarray(y).squeeze()

    # Check input dimensions
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input time series must be 1-D arrays.")

    if len(x) != len(y):
        raise ValueError("Input time series must have the same length.")

    N = len(x)

    if nsamp is None or nsamp > N:
        nsamp = N

    overlap = np.clip(overlap, 0, 99)
    nadvance = nsamp - int(overlap / 100 * nsamp)

    nrecs = int((N - nsamp) / nadvance) + 1

    y_cum = np.zeros(2 * maxlag + 1)

    # Compute cross-covariance
    for k in range(nrecs):
        ind = slice(k * nadvance, k * nadvance + nsamp)
        xs = x[ind]
        ys = y[ind]
        xs = xs - np.mean(xs)
        ys = ys - np.mean(ys)

        y_cum[maxlag] += np.dot(xs, ys)

        for m in range(1, maxlag + 1):
            y_cum[maxlag + m] += np.dot(xs[m:], ys[:-m])
            y_cum[maxlag - m] += np.dot(xs[:-m], ys[m:])

    # Normalize
    if flag == "biased":
        y_cum = y_cum / (nsamp * nrecs)
    else:  # 'unbiased'
        y_cum = y_cum / (nrecs * (nsamp - np.abs(np.arange(-maxlag, maxlag + 1))))

    return y_cum


def plot_cross_covariance(lags, ccov, title="Cross-Covariance"):
    """
    Plot the cross-covariance function.

    Parameters:
    -----------
    lags : array_like
        Lag values.
    ccov : array_like
        Cross-covariance values.
    title : str, optional
        Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lags, ccov, "b-")
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("Cross-Covariance")
    plt.grid(True)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.show()


def test_cum2x():
    """
    Test function for cross-covariance estimation.
    """
    # Generate test signals: two correlated AR(1) processes
    N = 1000
    phi = 0.5
    np.random.seed(0)
    x = np.zeros(N)
    y = np.zeros(N)
    for t in range(1, N):
        x[t] = phi * x[t - 1] + np.random.randn()
        y[t] = phi * y[t - 1] + 0.5 * x[t] + 0.5 * np.random.randn()

    # Estimate cross-covariance
    maxlag = 20
    ccov = cum2x(x, y, maxlag=maxlag, nsamp=N, flag="unbiased")

    # Plot results
    lags = np.arange(-maxlag, maxlag + 1)
    plot_cross_covariance(lags, ccov, title="Estimated Cross-Covariance of Correlated AR(1) Processes")

    # Compute and plot sample cross-correlation function (CCF)
    ccf = ccov / np.sqrt(np.var(x) * np.var(y))
    plot_cross_covariance(lags, ccf, title="Sample Cross-Correlation Function (CCF)")


if __name__ == "__main__":
    test_cum2x()
