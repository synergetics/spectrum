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
