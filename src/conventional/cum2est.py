import numpy as np
import matplotlib.pyplot as plt


def cum2est(y, maxlag=0, nsamp=None, overlap=0, flag="biased"):
    """
    Estimate the second-order cumulants (autocovariance) of a time series.

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

    Returns:
    --------
    y_cum : ndarray
        Estimated second-order cumulant (autocovariance),
        C2(m) for -maxlag <= m <= maxlag
    """
    y = np.asarray(y)
    y = y.squeeze()

    # Check input dimensions
    if y.ndim != 1:
        raise ValueError("Input time series must be a 1-D array.")

    N = len(y)

    if nsamp is None or nsamp > N:
        nsamp = N

    overlap = np.clip(overlap, 0, 99)
    nadvance = nsamp - int(overlap / 100 * nsamp)

    nrecs = int((N - overlap) / nadvance)

    y_cum = np.zeros(2 * maxlag + 1)

    ind = np.arange(nsamp)

    # Compute second-order cumulants
    for _ in range(nrecs):
        x = y[ind]
        x = x - np.mean(x)

        for k in range(maxlag + 1):
            y_cum[maxlag + k] += np.dot(x[0 : nsamp - k], x[k:nsamp])

        ind += nadvance

    # Normalize
    if flag == "biased":
        y_cum = y_cum / (nsamp * nrecs)
    else:  # 'unbiased'
        y_cum = y_cum / (nrecs * (nsamp - np.abs(np.arange(-maxlag, maxlag + 1))))

    # Fill in the negative lags
    y_cum[0:maxlag] = y_cum[-1:maxlag:-1].conj()

    return y_cum


def plot_autocovariance(lags, acov, title="Autocovariance"):
    """
    Plot the autocovariance function.

    Parameters:
    -----------
    lags : array_like
        Lag values.
    acov : array_like
        Autocovariance values.
    title : str, optional
        Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lags, acov, "b-")
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("Autocovariance")
    plt.grid(True)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.show()


def test_cum2est():
    """
    Test function for second-order cumulant estimation.
    """
    # Generate a test signal: AR(1) process
    N = 1000
    phi = 0.5
    np.random.seed(0)
    y = np.zeros(N)
    for t in range(1, N):
        y[t] = phi * y[t - 1] + np.random.randn()

    # Estimate second-order cumulants
    maxlag = 20
    y_cum = cum2est(y, maxlag=maxlag, nsamp=N, flag="unbiased")

    # Plot results
    lags = np.arange(-maxlag, maxlag + 1)
    plot_autocovariance(lags, y_cum, title="Estimated Autocovariance of AR(1) Process")

    # Compare with theoretical autocovariance
    theoretical_acov = np.array([phi ** abs(k) / (1 - phi**2) for k in lags])
    plot_autocovariance(lags, theoretical_acov, title="Theoretical Autocovariance of AR(1) Process")


if __name__ == "__main__":
    test_cum2est()
