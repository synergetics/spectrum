import numpy as np
from spectrum import armafit, plot_arma_poles_zeros


def test_armafit_ar():
    # Test pure AR model
    np.random.seed(42)
    N = 1000
    a_true = [1, -0.5, 0.2]  # AR(2) coefficients
    
    # Generate AR(2) process
    e = np.random.randn(N)
    y = np.zeros(N)
    for n in range(2, N):
        y[n] = -a_true[1] * y[n-1] - a_true[2] * y[n-2] + e[n]
    
    y = y.reshape(-1, 1)
    a, b, rho = armafit(y, p=2, q=0)
    
    assert len(a) == 3  # Including a[0] = 1
    assert len(b) == 1  # Only b[0] = 1 for pure AR
    assert a[0] == 1.0
    assert rho > 0


def test_armafit_ma():
    # Test pure MA model
    np.random.seed(42)
    N = 1000
    b_true = [1, 0.6, -0.3]  # MA(2) coefficients
    
    # Generate MA(2) process
    e = np.random.randn(N + 2)
    y = np.zeros(N)
    for n in range(N):
        y[n] = b_true[0] * e[n+2] + b_true[1] * e[n+1] + b_true[2] * e[n]
    
    y = y.reshape(-1, 1)
    a, b, rho = armafit(y, p=0, q=2)
    
    assert len(a) == 1  # Only a[0] = 1 for pure MA
    assert len(b) == 3  # Including b[0] = 1
    assert a[0] == 1.0
    assert b[0] == 1.0
    assert rho > 0


def test_armafit_arma():
    # Test ARMA model
    np.random.seed(42)
    N = 1000
    
    # Generate simple ARMA(1,1) process
    e = np.random.randn(N + 1)
    y = np.zeros(N)
    for n in range(1, N):
        y[n] = 0.5 * y[n-1] + e[n] + 0.3 * e[n-1]
    
    y = y.reshape(-1, 1)
    a, b, rho = armafit(y, p=1, q=1)
    
    assert len(a) == 2  # a[0] = 1, a[1]
    assert len(b) == 2  # b[0] = 1, b[1]
    assert a[0] == 1.0
    assert b[0] == 1.0
    assert rho > 0


def test_plot_arma_poles_zeros():
    # Test plotting function
    a = np.array([1, -0.5, 0.2])
    b = np.array([1, 0.3])
    
    # This should not raise an error
    plot_arma_poles_zeros(a, b, "Test ARMA Model")


if __name__ == "__main__":
    test_armafit_ar()
    test_armafit_ma()
    test_armafit_arma()
    test_plot_arma_poles_zeros()
    print("All armafit tests passed!")

def test_armafit_recovers_ar2():
    rng = np.random.default_rng(10)
    n = 20000
    e = rng.standard_normal(n)
    a_true = np.array([1.0, -0.5, 0.2])
    y = np.zeros(n)
    for i in range(2, n):
        y[i] = 0.5 * y[i - 1] - 0.2 * y[i - 2] + e[i]
    a, b, rho = armafit(y.reshape(-1, 1), p=2, q=0)
    assert np.allclose(a, a_true, atol=0.1)
    assert rho > 0


def test_armafit_1d_input():
    rng = np.random.default_rng(11)
    y = rng.standard_normal(4096)
    a, b, rho = armafit(y, p=1, q=0)
    assert a[0] == 1.0 and len(a) == 2
