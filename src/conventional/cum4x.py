import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel


def cum4x(w, x, y, z, maxlag=0, nsamp=None, overlap=0, flag="biased", k1=0, k2=0):
    """
    Estimate the fourth-order cross-cumulants of four time series.

    Parameters:
    -----------
    w, x, y, z : array_like
        Input data vectors or time-series.
    maxlag : int, optional
        Maximum lag to be computed. Default is 0.
    nsamp : int, optional
        Samples per segment. If None, nsamp is set to min(len(w), len(x), len(y), len(z)).
    overlap : int, optional
        Percentage overlap of segments (0-99). Default is 0.
    flag : str, optional
        'biased' or 'unbiased'. Default is 'biased'.
    k1, k2 : int, optional
        The fixed lags in C4(m,k1,k2). Default is 0.

    Returns:
    --------
    y_cum : ndarray
        Estimated fourth-order cross-cumulant,
        C4(m,k1,k2) for -maxlag <= m <= maxlag
    """
    w, x, y, z = map(np.asarray, (w, x, y, z))
    w, x, y, z = map(np.squeeze, (w, x, y, z))

    # Check input dimensions
    if not all(arr.ndim == 1 for arr in (w, x, y, z)):
        raise ValueError("All input time series must be 1-D arrays.")

    if len(set(map(len, (w, x, y, z)))) != 1:
        raise ValueError("All input time series must have the same length.")

    N = len(w)

    if nsamp is None or nsamp > N:
        nsamp = N

    overlap = np.clip(overlap, 0, 99)
    nadvance = nsamp - int(overlap / 100 * nsamp)

    nrecs = int((N - nsamp) / nadvance) + 1

    y_cum = np.zeros(2 * maxlag + 1)

    # Compute fourth-order cross-cumulants
    R_zy = np.zeros(2 * maxlag + 1)
    R_wy = np.zeros(2 * maxlag + 1)
    R_wx = np.zeros(2 * maxlag + 1)
    R_zx = np.zeros(2 * maxlag + 1)
    M_wz = np.zeros(2 * maxlag + 1)
    M_yx = np.zeros(2 * maxlag + 1)

    for i in range(nrecs):
        ind = slice(i * nadvance, i * nadvance + nsamp)
        ws, xs, ys, zs = w[ind], x[ind], y[ind], z[ind]
        ws, xs, ys, zs = map(lambda arr: arr - np.mean(arr), (ws, xs, ys, zs))

        cys = np.conj(ys)

        # Create the "IV" vector
        ziv = np.zeros_like(xs)
        if k1 >= 0:
            ziv[:-k1] = ws[:-k1] * cys[k1:]
            R_wy[maxlag] += np.dot(ws[:-k1], ys[k1:])
        else:
            ziv[-k1:] = ws[-k1:] * cys[:k1]
            R_wy[maxlag] += np.dot(ws[-k1:], ys[:k1])

        if k2 >= 0:
            ziv[:-k2] *= zs[k2:]
            ziv[nsamp - k2 :] = 0
            M_wz[maxlag] += np.dot(ws[:-k2], zs[k2:])
        else:
            ziv[-k2:] *= zs[:k2]
            ziv[:-k2] = 0
            M_wz[maxlag] += np.dot(ws[-k2:], zs[:k2])

        y_cum[maxlag] += np.dot(ziv, xs)
        R_zy[maxlag] += np.dot(zs, cys)
        R_wx[maxlag] += np.dot(ws, np.conj(xs))
        R_zx[maxlag] += np.dot(zs, np.conj(xs))
        M_yx[maxlag] += np.dot(ys, xs)

        for k in range(1, maxlag + 1):
            y_cum[maxlag + k] += np.dot(ziv[k:], xs[:-k])
            y_cum[maxlag - k] += np.dot(ziv[:-k], xs[k:])
            R_zy[maxlag + k] += np.dot(zs[k:], cys[:-k])
            R_zy[maxlag - k] += np.dot(zs[:-k], cys[k:])
            R_wx[maxlag + k] += np.dot(ws[k:], np.conj(xs[:-k]))
            R_wx[maxlag - k] += np.dot(ws[:-k], np.conj(xs[k:]))
            R_zx[maxlag + k] += np.dot(zs[k:], np.conj(xs[:-k]))
            R_zx[maxlag - k] += np.dot(zs[:-k], np.conj(xs[k:]))
            M_yx[maxlag + k] += np.dot(ys[k:], xs[:-k])
            M_yx[maxlag - k] += np.dot(ys[:-k], xs[k:])

    # Normalize
    if flag == "biased":
        scale = np.ones(2 * maxlag + 1) / (nsamp * nrecs)
    else:  # 'unbiased'
        lsamp = nsamp - np.maximum(np.abs(np.arange(-maxlag, maxlag + 1)), np.maximum(np.abs(k1), np.abs(k2)))
        scale = 1 / (nrecs * lsamp)

    y_cum *= scale
    R_zy *= scale
    R_wy *= scale
    R_wx *= scale
    R_zx *= scale
    M_wz *= scale
    M_yx *= scale

    # Remove second-order statistics
    y_cum -= (
        R_zy[maxlag] * R_wx
        + R_wy[maxlag] * R_zx[maxlag - k2 : maxlag - k2 + 2 * maxlag + 1]
        + M_wz[maxlag].conj() * M_yx[maxlag - k1 : maxlag - k1 + 2 * maxlag + 1]
    )

    return y_cum


def plot_fourth_order_cross_cumulant(lags, cumulant, k1, k2, title="Fourth-Order Cross-Cumulant"):
    """
    Plot the fourth-order cross-cumulant function.

    Parameters:
    -----------
    lags : array_like
        Lag values.
    cumulant : array_like
        Fourth-order cross-cumulant values.
    k1, k2 : int
        The fixed lag values.
    title : str, optional
        Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lags, cumulant.real, "b-", label="Real")
    plt.plot(lags, cumulant.imag, "r--", label="Imaginary")
    plt.title(f"{title} (k1 = {k1}, k2 = {k2})")
    plt.xlabel("Lag m")
    plt.ylabel("C4(m,k1,k2)")
    plt.grid(True)
    plt.axhline(y=0, color="k", linestyle=":")
    plt.legend()
    plt.show()


def test_cum4x():
    """
    Test function for fourth-order cross-cumulant estimation.
    """
    # Generate test signals: four related non-Gaussian processes
    N = 10000
    np.random.seed(0)
    e = np.random.randn(N) ** 3  # Non-Gaussian noise
    w = np.zeros(N)
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    for t in range(1, N):
        w[t] = 0.5 * w[t - 1] + e[t]
        x[t] = 0.3 * x[t - 1] + 0.4 * w[t - 1] + 0.5 * e[t]
        y[t] = 0.4 * y[t - 1] + 0.3 * w[t - 2] + 0.2 * x[t - 1] + 0.3 * e[t]
        z[t] = 0.2 * z[t - 1] + 0.1 * w[t - 3] + 0.2 * x[t - 2] + 0.3 * y[t - 1] + 0.2 * e[t]

    # Estimate fourth-order cross-cumulants
    maxlag = 20
    k1_k2_values = [(0, 0), (5, 0), (0, 5), (5, 5)]

    for k1, k2 in k1_k2_values:
        y_cum = cum4x(w, x, y, z, maxlag=maxlag, nsamp=N, flag="unbiased", k1=k1, k2=k2)

        # Plot results
        lags = np.arange(-maxlag, maxlag + 1)
        plot_fourth_order_cross_cumulant(lags, y_cum, k1, k2, title="Estimated Fourth-Order Cross-Cumulant")


if __name__ == "__main__":
    test_cum4x()
