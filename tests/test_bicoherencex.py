import scipy.io as sio
import numpy as np
from spectrum import bicoherencex, plot_bicoherencex


def test_specific():
    nl1 = sio.loadmat("./tests/demo/nl1.mat")
    dbic = bicoherencex(nl1["x"], nl1["x"], nl1["y"])


def test_bicoherencex():
    t = np.linspace(0, 10, 1000)
    w = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(len(t))
    x = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
    y = np.sin(2 * np.pi * 15 * t) + 0.1 * np.random.randn(len(t))
    w, x, y = w.reshape(-1, 1), x.reshape(-1, 1), y.reshape(-1, 1)  # Reshape to 2D arrays
    bic, waxis = bicoherencex(w, x, y, nfft=256, nsamp=256)
    assert bic.shape == (256, 256)
    assert waxis.shape == (256,)
    plot_bicoherencex(bic, waxis)
