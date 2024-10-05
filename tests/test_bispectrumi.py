def test_bispectrumi():
    """
    Test function for bispectrum estimation using the indirect method.
    """
    # Generate a simple test signal
    t = np.linspace(0, 10, 1000)
    f1, f2 = 5, 10
    y = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + 0.5 * np.sin(2 * np.pi * (f1 + f2) * t)
    y += 0.1 * np.random.randn(len(t))

    # Compute and plot bispectrum
    Bspec, waxis = bispectrumi(y, nlag=50, nsamp=1000, nfft=256)
    plot_bispectrum(Bspec, waxis)
