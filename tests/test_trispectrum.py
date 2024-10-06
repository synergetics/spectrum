import numpy as np
from spectrum import trispectrum, tricoherence, plot_trispectrum, plot_tricoherence_summary, detect_quadratic_coupling


def test_trispectrum():
    # Generate test signal with quadratic coupling
    np.random.seed(42)
    N = 512
    t = np.arange(N)
    
    # Create signal with quadratic phase coupling: f1 + f2 = f3
    f1, f2 = 0.1, 0.15
    f3 = f1 + f2
    
    y = (np.sin(2 * np.pi * f1 * t) + 
         np.sin(2 * np.pi * f2 * t) + 
         0.5 * np.sin(2 * np.pi * f3 * t + np.pi/4))
    y += 0.1 * np.random.randn(N)
    y = y.reshape(-1, 1)
    
    Tspec, waxis = trispectrum(y, nfft=128, nsamp=128)
    
    assert Tspec.ndim == 3
    assert len(waxis) > 0
    assert np.iscomplexobj(Tspec)


def test_tricoherence():
    # Test tricoherence computation
    np.random.seed(42)
    N = 256
    
    # Simple test signal
    y = np.random.randn(N).reshape(-1, 1)
    
    tricoh, waxis = tricoherence(y, nfft=64, nsamp=64)
    
    assert tricoh.ndim == 3
    assert len(waxis) > 0
    assert np.isreal(tricoh).all()
    assert np.all(tricoh >= 0) and np.all(tricoh <= 1)  # Tricoherence should be between 0 and 1


def test_trispectrum_windows():
    # Test different window functions
    np.random.seed(42)
    N = 200
    y = np.random.randn(N).reshape(-1, 1)
    
    windows = [None, 'hanning', 'hamming', 'blackman']
    
    for window in windows:
        Tspec, waxis = trispectrum(y, nfft=64, window=window, nsamp=64)
        assert Tspec.ndim == 3
        assert len(waxis) > 0


def test_plot_trispectrum():
    # Test plotting function
    np.random.seed(42)
    N = 128
    y = np.random.randn(N).reshape(-1, 1)
    
    Tspec, waxis = trispectrum(y, nfft=32, nsamp=32)
    
    # Test different plot types
    for slice_type in ['magnitude', 'phase', 'real', 'imag']:
        plot_trispectrum(Tspec, waxis, slice_type=slice_type, 
                        title=f"Test Trispectrum ({slice_type})")


def test_plot_tricoherence_summary():
    # Test tricoherence summary plotting
    np.random.seed(42)
    N = 128
    y = np.random.randn(N).reshape(-1, 1)
    
    tricoh, waxis = tricoherence(y, nfft=32, nsamp=32)
    
    plot_tricoherence_summary(tricoh, waxis, title="Test Tricoherence Summary")


def test_detect_quadratic_coupling():
    # Test coupling detection
    np.random.seed(42)
    N = 64
    
    # Create tricoherence with some high values
    tricoh = np.random.rand(8, 8, 8) * 0.5
    tricoh[2, 3, 1] = 0.8  # Add significant coupling
    tricoh[1, 1, 2] = 0.9  # Add another coupling
    
    waxis = np.linspace(0, 0.5, 8)
    
    couplings = detect_quadratic_coupling(tricoh, waxis, threshold=0.7)
    
    assert len(couplings) >= 1  # Should find at least the couplings we added
    for coupling in couplings:
        assert len(coupling) == 4  # (f1, f2, f3, tricoh_value)
        assert coupling[3] > 0.7  # Should exceed threshold


def test_trispectrum_small_signal():
    # Test with very small signal
    N = 64
    y = np.random.randn(N).reshape(-1, 1) * 0.01
    
    Tspec, waxis = trispectrum(y, nfft=32, nsamp=32)
    
    assert Tspec.ndim == 3
    assert len(waxis) > 0
    assert np.isfinite(Tspec).all()


def test_tricoherence_coherent_signal():
    # Test with coherent signal (should show high tricoherence)
    np.random.seed(42)
    N = 256
    t = np.arange(N)
    
    # Perfect quadratic coupling
    f1, f2 = 0.1, 0.15
    f3 = f1 + f2
    
    y = (np.sin(2 * np.pi * f1 * t) + 
         np.sin(2 * np.pi * f2 * t) + 
         np.sin(2 * np.pi * f3 * t))  # Perfect phase relationship
    y = y.reshape(-1, 1)
    
    tricoh, waxis = tricoherence(y, nfft=64, nsamp=64)
    
    # Should have some high tricoherence values
    max_tricoh = np.max(tricoh)
    assert max_tricoh > 0.1  # Should detect some coupling


if __name__ == "__main__":
    test_trispectrum()
    test_tricoherence()
    test_trispectrum_windows()
    test_plot_trispectrum()
    test_plot_tricoherence_summary()
    test_detect_quadratic_coupling()
    test_trispectrum_small_signal()
    test_tricoherence_coherent_signal()
    print("All trispectrum tests passed!")