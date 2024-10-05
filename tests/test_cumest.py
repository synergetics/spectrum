def test_cumest():
    """
    Test function for the unified cumulant estimation.
    """
    # Generate a test signal: Non-Gaussian AR(1) process with quadratic nonlinearity
    N = 10000
    phi = 0.5
    np.random.seed(0)
    e = np.random.randn(N) ** 3  # Non-Gaussian noise
    y = np.zeros(N)
    for t in range(1, N):
        y[t] = phi * y[t - 1] + 0.1 * y[t - 1] ** 2 + e[t]

    # Estimate cumulants of different orders
    maxlag = 20
    nsamp = N
    flag = "unbiased"

    for norder in [2, 3, 4]:
        if norder == 2:
            y_cum = cumest(y, norder=norder, maxlag=maxlag, nsamp=nsamp, flag=flag)
            plot_cumulant(np.arange(-maxlag, maxlag + 1), y_cum, norder)
        elif norder == 3:
            for k1 in [0, 5]:
                y_cum = cumest(y, norder=norder, maxlag=maxlag, nsamp=nsamp, flag=flag, k1=k1)
                plot_cumulant(np.arange(-maxlag, maxlag + 1), y_cum, norder, k1=k1)
        else:  # norder == 4
            for k1, k2 in [(0, 0), (5, 0), (0, 5), (5, 5)]:
                y_cum = cumest(y, norder=norder, maxlag=maxlag, nsamp=nsamp, flag=flag, k1=k1, k2=k2)
                plot_cumulant(np.arange(-maxlag, maxlag + 1), y_cum, norder, k1=k1, k2=k2)
