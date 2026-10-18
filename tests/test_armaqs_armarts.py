import numpy as np
import pytest

from spectrum import armarts, armaqs


def _arma_process(a, b, n, rng, skewed=False):
    e = rng.exponential(1.0, n) - 1.0 if skewed else rng.standard_normal(n)
    p = len(a) - 1
    q = len(b) - 1
    y = np.zeros(n)
    for i in range(n):
        v = e[i]
        for j in range(1, q + 1):
            if i - j >= 0:
                v += b[j] * e[i - j]
        for j in range(1, p + 1):
            if i - j >= 0:
                v -= a[j] * y[i - j]
        y[i] = v
    return y


A_TRUE = np.array([1.0, -0.5, 0.2])
B_TRUE = np.array([1.0, 0.4])


def test_armaqs_recovers_arma21():
    rng = np.random.default_rng(11)
    y = _arma_process(A_TRUE, B_TRUE, 40000, rng, skewed=True)
    a, b = armaqs(y, p=2, q=1)
    assert np.allclose(a, A_TRUE, atol=0.1), a
    assert np.allclose(b, B_TRUE, atol=0.1), b


def test_armarts_recovers_arma21():
    rng = np.random.default_rng(11)
    y = _arma_process(A_TRUE, B_TRUE, 40000, rng)
    a, b = armarts(y, p=2, q=1)
    assert np.allclose(a, A_TRUE, atol=0.08), a
    assert np.allclose(b, B_TRUE, atol=0.1), b


def test_armaqs_pure_ar():
    rng = np.random.default_rng(3)
    y = _arma_process(np.array([1.0, 0.6, -0.3]), np.array([1.0]), 40000, rng, skewed=True)
    a, b = armaqs(y, p=2, q=0)
    assert np.allclose(a, [1.0, 0.6, -0.3], atol=0.08), a
    assert b == np.array([1.0])


def test_armarts_validation():
    rng = np.random.default_rng(0)
    y = _arma_process(A_TRUE, B_TRUE, 1000, rng)
    with pytest.raises(ValueError):
        armarts(y, p=-1, q=1)
