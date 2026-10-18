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
    
    assert isinstance(p_opt, (int, np.integer))
    assert isinstance(q_opt, (int, np.integer))
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
        assert isinstance(p_opt, (int, np.integer))
        assert isinstance(q_opt, (int, np.integer))
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

def test_armasel_recovers_order():
    rng = np.random.default_rng(12)
    n = 20000
    e = rng.standard_normal(n)
    y = np.zeros(n)
    for i in range(2, n):
        y[i] = 0.5 * y[i - 1] - 0.2 * y[i - 2] + e[i]
    p_opt, q_opt, ic_min, ic_matrix = armasel(y.reshape(-1, 1), pmax=3, qmax=2, criterion="bic")
    assert p_opt == 2
    assert q_opt == 0


def test_armasel_raises_when_all_fits_fail(monkeypatch):
    import pytest

    # simulate every candidate fit failing
    import importlib

    mod = importlib.import_module("spectrum.armasel")

    def boom(*args, **kwargs):
        raise ValueError("fit failed")

    monkeypatch.setattr(mod, "armafit", boom)
    y = np.random.default_rng(0).standard_normal(256).reshape(-1, 1)
    with pytest.raises(RuntimeError):
        armasel(y, pmax=2, qmax=1)
