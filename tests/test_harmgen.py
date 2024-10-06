import numpy as np
from spectrum import harmgen, harmgen_complex, plot_harmonic_signal, harmonic_snr


def test_harmgen_single():
    # Test single harmonic generation
    N = 1000
    A = 1.0
    f = 0.1
    phi = 0.0
    sigma_n = 0.1
    
    y = harmgen(N, A, f, phi, sigma_n, seed=42)
    
    assert len(y) == N
    assert np.isreal(y).all()


def test_harmgen_multiple():
    # Test multiple harmonics
    N = 1000
    A = [1.0, 0.5, 0.3]
    f = [0.1, 0.15, 0.25]
    phi = [0, np.pi/4, np.pi/2]
    sigma_n = 0.05
    sigma_m = 0.02
    
    y = harmgen(N, A, f, phi, sigma_n, sigma_m, seed=42)
    
    assert len(y) == N
    assert np.isreal(y).all()


def test_harmgen_complex():
    # Test complex harmonic generation
    N = 500
    A = [1.0, 0.8]
    f = [0.1, -0.2]  # Negative frequency allowed for complex
    phi = [0, np.pi/3]
    sigma_n = 0.1
    
    y = harmgen_complex(N, A, f, phi, sigma_n, seed=42)
    
    assert len(y) == N
    assert np.iscomplexobj(y)


def test_harmgen_noise_only():
    # Test noise-only generation
    N = 200
    A = 0.0
    f = 0.1
    sigma_n = 1.0
    
    y = harmgen(N, A, f, sigma_n=sigma_n, seed=42)
    
    assert len(y) == N
    # Should be mostly noise
    assert np.std(y) > 0.5


def test_plot_harmonic_signal():
    # Test plotting function
    N = 100
    A = 1.0
    f = 0.1
    
    y = harmgen(N, A, f, seed=42)
    
    # Should not raise an error
    plot_harmonic_signal(y, fs=1.0, title="Test Harmonic Signal")


def test_harmonic_snr():
    # Test SNR estimation
    N = 1000
    A = 1.0
    f = 0.1
    sigma_n = 0.1
    
    y = harmgen(N, A, f, sigma_n=sigma_n, seed=42)
    snr = harmonic_snr(y, signal_freqs=[f], fs=1.0)
    
    assert np.isfinite(snr)
    assert snr > 0  # Should have positive SNR for this case


def test_harmgen_frequency_validation():
    # Test frequency validation
    N = 100
    A = 1.0
    
    # Should raise error for out-of-range frequency
    try:
        y = harmgen(N, A, f=0.6)  # > 0.5
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        y = harmgen(N, A, f=-0.1)  # < 0
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_harmgen_complex_frequency_validation():
    # Test complex frequency validation
    N = 100
    A = 1.0
    
    # Should raise error for out-of-range frequency
    try:
        y = harmgen_complex(N, A, f=0.6)  # > 0.5
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        y = harmgen_complex(N, A, f=-0.6)  # < -0.5
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_harmgen_single()
    test_harmgen_multiple()
    test_harmgen_complex()
    test_harmgen_noise_only()
    test_plot_harmonic_signal()
    test_harmonic_snr()
    test_harmgen_frequency_validation()
    test_harmgen_complex_frequency_validation()
    print("All harmgen tests passed!")