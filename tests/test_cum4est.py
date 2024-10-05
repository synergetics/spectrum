def test_cum4est():
    """
    Test function for fourth-order cumulant estimation.
    """
    # Generate a test signal: Non-Gaussian AR(1) process with quadratic nonlinearity
    N = 10000
    phi = 0.5
    np.random.seed(0)
    e = np.random.randn(N) ** 3  # Non-Gaussian noise
    y = np.zeros(N)
    for t in range(1, N):
        y[t] = phi * y[t - 1] + 0.1 * y[t - 1] ** 2 + e[t]

    # Estimate fourth-order cumulants
    maxlag = 20
    k1_k2_values = [(0, 0), (5, 0), (0, 5), (5, 5)]

    for k1, k2 in k1_k2_values:
        y_cum = cum4est(y, maxlag=maxlag, nsamp=N, flag="unbiased", k1=k1, k2=k2)

        # Plot results
        lags = np.arange(-maxlag, maxlag + 1)
        plot_fourth_order_cumulant(
            lags, y_cum, k1, k2, title="Estimated Fourth-Order Cumulant of Non-Gaussian AR(1) Process"
        )
