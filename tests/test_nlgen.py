import numpy as np
from spectrum import nlgen, plot_nonlinear_series, nonlinear_measures


def test_nlgen_bilinear():
    # Test bilinear model
    N = 500
    a = [0.5]
    b = [1.0]
    c = [0.1]
    sigma = 0.5
    
    y = nlgen(N, model_type="bilinear", a=a, b=b, c=c, sigma=sigma, seed=42)
    
    assert len(y) == N
    assert np.isfinite(y).all()


def test_nlgen_tar():
    # Test threshold autoregressive model
    N = 400
    a = [0.6]
    c = [0.8]  # Different regime coefficients
    
    y = nlgen(N, model_type="tar", a=a, c=c, threshold=0.0, delay=1, a2=[0.3], seed=42)
    
    assert len(y) == N
    assert np.isfinite(y).all()


def test_nlgen_volterra():
    # Test Volterra series model
    N = 300
    a = [0.8, 0.3, 0.1]
    c = [0.1, 0.05, 0.02, 0.01]
    
    y = nlgen(N, model_type="volterra", a=a, c=c, order=2, memory=3, seed=42)
    
    assert len(y) == N
    assert np.isfinite(y).all()


def test_nlgen_narma():
    # Test nonlinear ARMA model
    N = 400
    a = [0.5]
    b = [1.0]
    c = [0.2]
    
    for nonlinear_func in ['tanh', 'sigmoid', 'cubic', 'quadratic']:
        y = nlgen(N, model_type="narma", a=a, b=b, c=c, 
                 nonlinear_func=nonlinear_func, seed=42)
        assert len(y) == N
        assert np.isfinite(y).all()


def test_nlgen_henon():
    # Test Hénon map
    N = 200
    a = [1.4]
    b = [0.3]
    sigma = 0.01
    
    y = nlgen(N, model_type="henon", a=a, b=b, sigma=sigma, x0=0.1, y0=0.1, seed=42)
    
    assert len(y) == N
    assert np.isfinite(y).all()


def test_nlgen_logistic():
    # Test logistic map
    N = 150
    a = [3.8]
    sigma = 0.01
    
    y = nlgen(N, model_type="logistic", a=a, sigma=sigma, x0=0.5, seed=42)
    
    assert len(y) == N
    assert np.isfinite(y).all()
    assert np.all(y >= 0) and np.all(y <= 1)  # Logistic map should stay in [0,1]


def test_plot_nonlinear_series():
    # Test plotting function
    N = 100
    y = nlgen(N, model_type="bilinear", seed=42)
    
    # Should not raise an error
    plot_nonlinear_series(y, model_type="Bilinear Test", fs=1.0)


def test_nonlinear_measures():
    # Test nonlinearity measures
    N = 300
    y = nlgen(N, model_type="henon", a=[1.4], b=[0.3], sigma=0.01, seed=42)
    
    measures = nonlinear_measures(y)
    
    required_keys = ['mean', 'std', 'skewness', 'kurtosis', 'bds_statistic', 'lyapunov_exponent']
    for key in required_keys:
        assert key in measures
        assert np.isfinite(measures[key])


def test_nlgen_invalid_model():
    # Test invalid model type
    N = 100
    
    try:
        y = nlgen(N, model_type="invalid_model")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_nlgen_parameter_arrays():
    # Test with array parameters
    N = 200
    a = np.array([0.5, 0.2])
    b = np.array([1.0, 0.3])
    c = np.array([0.1])
    
    y = nlgen(N, model_type="bilinear", a=a, b=b, c=c, seed=42)
    
    assert len(y) == N
    assert np.isfinite(y).all()


if __name__ == "__main__":
    test_nlgen_bilinear()
    test_nlgen_tar()
    test_nlgen_volterra()
    test_nlgen_narma()
    test_nlgen_henon()
    test_nlgen_logistic()
    test_plot_nonlinear_series()
    test_nonlinear_measures()
    test_nlgen_invalid_model()
    test_nlgen_parameter_arrays()
    print("All nlgen tests passed!")