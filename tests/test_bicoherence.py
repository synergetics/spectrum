def test_bicoherence():
    """
    Test function for bicoherence estimation.
    """
    # Generate a simple test signal
    t = np.linspace(0, 10, 1000)
    f1, f2 = 5, 10
    y = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + 0.5 * np.sin(2 * np.pi * (f1 + f2) * t)
    y += 0.1 * np.random.randn(len(t))

    # Compute and plot bicoherence
    bic, waxis = bicoherence(y, nfft=256, nsamp=256)
    plot_bicoherence(bic, waxis)
