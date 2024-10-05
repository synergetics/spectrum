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
