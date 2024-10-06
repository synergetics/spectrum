import scipy.io as sio
import numpy as np
from spectrum import bispectrumi, plot_bispectrumi


def test_specific():
    qpc = sio.loadmat("./tests/demo/qpc.mat")
    dbic = bispectrumi(qpc["zmat"], 21, 64, 0, "unbiased", 128, 1)


def test_bispectrumi():
    t = np.linspace(0, 10, 1000)
    f1, f2 = 5, 10
    y = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + 0.5 * np.sin(2 * np.pi * (f1 + f2) * t)
    y += 0.1 * np.random.randn(len(t))
    y = y.reshape(-1, 1)  # Reshape to 2D array
    Bspec, waxis = bispectrumi(y, nlag=50, nsamp=256)
    assert Bspec.shape == (128, 128)
    assert waxis.shape == (128,)
    plot_bispectrumi(Bspec, waxis)
