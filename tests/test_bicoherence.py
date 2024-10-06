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
