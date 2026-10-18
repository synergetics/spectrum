import scipy.io as sio
import numpy as np
from spectrum import bicoherence, plot_bicoherence


def test_specific():
    qpc = sio.loadmat("./tests/demo/qpc.mat")
    dbic = bicoherence(qpc["zmat"])


def test_bicoherence():
    t = np.linspace(0, 10, 1000)
    f1, f2 = 5, 10
    y = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + 0.5 * np.sin(2 * np.pi * (f1 + f2) * t)
    y += 0.1 * np.random.randn(len(t))
    y = y.reshape(-1, 1)  # Reshape to 2D array
    bic, waxis = bicoherence(y, nfft=256, nsamp=256)
    assert bic.shape == (256, 256)
    assert waxis.shape == (256,)
    plot_bicoherence(np.abs(bic), waxis)


def test_bicoherence_default_nsamp():
    # default nsamp must auto-select ~8 segments for a single realization
    rng = np.random.default_rng(0)
    y = rng.standard_normal(4096).reshape(-1, 1)
    bic, waxis = bicoherence(y)
    assert bic.shape[0] == bic.shape[1]
    assert bic.shape[0] >= 128
    assert np.all(np.isfinite(bic))


def test_bicoherence_1d_input():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(4096)
    bic, waxis = bicoherence(y, nfft=128, nsamp=256)
    # nfft is bumped up to the next power of two >= nsamp (MATLAB semantics)
    assert bic.shape == (256, 256)


def test_bicoherence_overlap_clamped():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(2048).reshape(-1, 1)
    bic, _ = bicoherence(y, nfft=128, nsamp=256, overlap=150)
    assert bic.shape == (256, 256)


def test_bicoherence_qpc_peak_and_bounds():
    rng = np.random.default_rng(3)
    n = 16384
    t = np.arange(n)
    f1, f2 = 0.10, 0.15
    y = (
        np.cos(2 * np.pi * f1 * t + 1.0)
        + np.cos(2 * np.pi * f2 * t + 2.0)
        + 0.5 * np.cos(2 * np.pi * (f1 + f2) * t + 3.0)
        + 0.1 * rng.standard_normal(n)
    )
    bic, waxis = bicoherence(y.reshape(-1, 1), nfft=256, nsamp=256, overlap=50)
    w = np.asarray(waxis).ravel()
    i = int(np.argmin(np.abs(w - f1)))
    j = int(np.argmin(np.abs(w - f2)))
    assert bic[i, j] > 0.8  # strong coupling at (f1, f2)
    assert bic.max() > 0.8
    assert bic.min() >= 0 and bic.max() <= 1.0000001
