def test_bicoherencex():
    """
    Test function for cross-bicoherence estimation.
    """
    # Generate simple test signals
    t = np.linspace(0, 10, 1000)
    f1, f2 = 5, 10
    w = np.sin(2 * np.pi * f1 * t)
    x = np.sin(2 * np.pi * f2 * t)
    y = np.sin(2 * np.pi * (f1 + f2) * t) + 0.1 * np.random.randn(len(t))

    # Compute and plot cross-bicoherence
    bic, waxis = bicoherencex(w, x, y, nfft=256, nsamp=256)
    plot_bicoherence(bic, waxis, title="Cross-Bicoherence Estimate")
