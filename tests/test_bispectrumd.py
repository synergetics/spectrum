def test_bispectrumd():
    """
    Test function for bispectrum estimation using the direct method.
    """
    # Generate a simple test signal
    t = np.linspace(0, 10, 1000)
    f1, f2 = 5, 10
    y = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + 0.5 * np.sin(2 * np.pi * (f1 + f2) * t)
    y += 0.1 * np.random.randn(len(t))

    # Compute and plot bispectrum
    Bspec, waxis = bispectrumd(y, nfft=256, nsamp=256, window=5)
    plot_bispectrum(Bspec, waxis, title="Bispectrum Estimate (Direct Method)")
