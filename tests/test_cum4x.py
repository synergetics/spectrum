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
