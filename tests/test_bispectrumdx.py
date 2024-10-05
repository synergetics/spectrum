def test_bispectrumdx():
    """
    Test function for cross-bispectrum estimation using the direct method.
    """
    # Generate simple test signals
    t = np.linspace(0, 10, 1000)
    f1, f2 = 5, 10
    x = np.sin(2 * np.pi * f1 * t)
    y = np.sin(2 * np.pi * f2 * t)
    z = np.sin(2 * np.pi * (f1 + f2) * t) + 0.1 * np.random.randn(len(t))

    # Compute and plot cross-bispectrum
    Bspec, waxis = bispectrumdx(x, y, z, nfft=256, nsamp=256, window=5)
    plot_cross_bispectrum(Bspec, waxis, title="Cross-Bispectrum Estimate (Direct Method)")
