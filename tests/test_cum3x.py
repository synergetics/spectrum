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
