def test_cum3est():
    """
    Test function for third-order cumulant estimation.
    """
    # Generate a test signal: Non-Gaussian AR(1) process
    N = 10000
    phi = 0.5
    np.random.seed(0)
    e = np.random.randn(N) ** 3  # Non-Gaussian noise
    y = np.zeros(N)
    for t in range(1, N):
        y[t] = phi * y[t - 1] + e[t]

    # Estimate third-order cumulants
    maxlag = 20
    k1_values = [0, 5, -5]

    for k1 in k1_values:
        y_cum = cum3est(y, maxlag=maxlag, nsamp=N, flag="unbiased", k1=k1)

        # Plot results
        lags = np.arange(-maxlag, maxlag + 1)
        plot_third_order_cumulant(lags, y_cum, k1, title="Estimated Third-Order Cumulant of Non-Gaussian AR(1) Process")
