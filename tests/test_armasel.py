import numpy as np
from spectrum import armasel, plot_ic_surface, plot_ic_comparison


def test_armasel():
    # Generate AR(2) test data
    np.random.seed(42)
    N = 500
    a_true = [1, -0.6, 0.2]  # AR(2) coefficients
    
    # Generate AR(2) process
    e = np.random.randn(N)
    y = np.zeros(N)
    for n in range(2, N):
        y[n] = -a_true[1] * y[n-1] - a_true[2] * y[n-2] + e[n]
    
    y = y.reshape(-1, 1)
    
    # Test AIC
    p_opt, q_opt, ic_min, ic_matrix = armasel(y, pmax=4, qmax=2, criterion="aic")
    
    assert isinstance(p_opt, int)
    assert isinstance(q_opt, int)
    assert 0 <= p_opt <= 4
    assert 0 <= q_opt <= 2
    assert ic_matrix.shape == (5, 3)  # (pmax+1, qmax+1)
    assert np.isfinite(ic_min)


def test_armasel_criteria():
    # Test different criteria
    np.random.seed(42)
    N = 300
    
    # Generate simple AR(1) process
    y = np.zeros(N)
    y[0] = np.random.randn()
    for n in range(1, N):
        y[n] = 0.7 * y[n-1] + np.random.randn()
    
    y = y.reshape(-1, 1)
    
    criteria = ["aic", "bic", "hq"]
    for criterion in criteria:
        p_opt, q_opt, ic_min, ic_matrix = armasel(y, pmax=3, qmax=1, criterion=criterion)
        assert isinstance(p_opt, int)
        assert isinstance(q_opt, int)
        assert np.isfinite(ic_min)


def test_plot_ic_surface():
    # Generate test data
    np.random.seed(42)
    N = 200
    y = np.random.randn(N).reshape(-1, 1)
    
    # Get IC matrix
    p_opt, q_opt, ic_min, ic_matrix = armasel(y, pmax=2, qmax=2, criterion="aic")
    
    # Test plotting function
    plot_ic_surface(ic_matrix, criterion="AIC", title="Test IC Surface")


def test_plot_ic_comparison():
    # Generate test data
    np.random.seed(42)
    N = 200
    y = np.random.randn(N).reshape(-1, 1)
    
    # Test comparison plotting
    plot_ic_comparison(y, pmax=2, qmax=2, criteria=["aic", "bic"])


if __name__ == "__main__":
    test_armasel()
    test_armasel_criteria()
    test_plot_ic_surface()
    test_plot_ic_comparison()
    print("All armasel tests passed!")