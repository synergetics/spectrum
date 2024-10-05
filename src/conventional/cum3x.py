import numpy as np
import matplotlib.pyplot as plt


def cum3x(x, y, z, maxlag=0, nsamp=None, overlap=0, flag="biased", k1=0):
    """
    Estimate the third-order cross-cumulants of three time series.

    Parameters:
    -----------
    x, y, z : array_like
        Input data vectors or time-series.
    maxlag : int, optional
        Maximum lag to be computed. Default is 0.
    nsamp : int, optional
        Samples per segment. If None, nsamp is set to min(len(x), len(y), len(z)).
    overlap : int, optional
        Percentage overlap of segments (0-99). Default is 0.
    flag : str, optional
        'biased' or 'unbiased'. Default is 'biased'.
    k1 : int, optional
        The fixed lag in C_xyz(m,k1). Default is 0.

    Returns:
    --------
    y_cum : ndarray
        Estimated third-order cross-cumulant,
        C_xyz(m,k1) for -maxlag <= m <= maxlag
    """
    x = np.asarray(x).squeeze()
    y = np.asarray(y).squeeze()
    z = np.asarray(z).squeeze()

    # Check input dimensions
    if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
        raise ValueError("Input time series must be 1-D arrays.")

    if len(x) != len(y) or len(x) != len(z):
        raise ValueError("Input time series must have the same length.")

    N = len(x)

    if nsamp is None or nsamp > N:
        nsamp = N

    overlap = np.clip(overlap, 0, 99)
    nadvance = nsamp - int(overlap / 100 * nsamp)

    nrecs = int((N - nsamp) / nadvance) + 1

    y_cum = np.zeros(2 * maxlag + 1)

    # Compute third-order cross-cumulants
    for i in range(nrecs):
        ind = slice(i * nadvance, i * nadvance + nsamp)
        xs = x[ind] - np.mean(x[ind])
        ys = y[ind] - np.mean(y[ind])
        zs = z[ind] - np.mean(z[ind])

        cx = np.conj(xs)

        # Create the "IV" vector
        if k1 >= 0:
            iv = cx[:-k1] * ys[k1:]
            ind1 = np.arange(nsamp - k1)
        else:
            iv = cx[-k1:] * ys[:k1]
            ind1 = np.arange(-k1, nsamp)

        y_cum[maxlag] += np.dot(iv, zs[ind1])

        for k in range(1, maxlag + 1):
            y_cum[maxlag + k] += np.dot(iv[k:], zs[ind1[:-k]])
            y_cum[maxlag - k] += np.dot(iv[:-k], zs[ind1[k:]])

    # Normalize
    if flag == "biased":
        y_cum = y_cum / (nsamp * nrecs)
    else:  # 'unbiased'
        lsamp = nsamp - abs(k1)
        y_cum = y_cum / (nrecs * (lsamp - np.abs(np.arange(-maxlag, maxlag + 1))))

    return y_cum


def plot_third_order_cross_cumulant(lags, cumulant, k1, title="Third-Order Cross-Cumulant"):
    """
    Plot the third-order cross-cumulant function.

    Parameters:
    -----------
    lags : array_like
        Lag values.
    cumulant : array_like
        Third-order cross-cumulant values.
    k1 : int
        The fixed lag value.
    title : str, optional
        Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lags, cumulant, "b-")
    plt.title(f"{title} (k1 = {k1})")
    plt.xlabel("Lag m")
    plt.ylabel("C_xyz(m,k1)")
    plt.grid(True)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.show()


def test_cum3x():
    """
    Test function for third-order cross-cumulant estimation.
    """
    # Generate test signals: three related non-Gaussian processes
    N = 10000
    np.random.seed(0)
    e = np.random.randn(N) ** 3  # Non-Gaussian noise
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    for t in range(1, N):
        x[t] = 0.5 * x[t - 1] + e[t]
        y[t] = 0.3 * y[t - 1] + 0.4 * x[t - 1] + 0.5 * e[t]
        z[t] = 0.4 * z[t - 1] + 0.3 * x[t - 2] + 0.2 * y[t - 1] + 0.3 * e[t]

    # Estimate third-order cross-cumulants
    maxlag = 20
    k1_values = [0, 5, -5]

    for k1 in k1_values:
        y_cum = cum3x(x, y, z, maxlag=maxlag, nsamp=N, flag="unbiased", k1=k1)

        # Plot results
        lags = np.arange(-maxlag, maxlag + 1)
        plot_third_order_cross_cumulant(lags, y_cum, k1, title="Estimated Third-Order Cross-Cumulant")


if __name__ == "__main__":
    test_cum3x()
