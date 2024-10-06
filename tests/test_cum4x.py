import scipy.io as sio
import numpy as np
from spectrum import cum4x


def test_specific():
    y = sio.loadmat("./tests/demo/ma1.mat")["y"]

    biased = cum4x(y, y, y, y, 3, 100, 0, "biased").flatten().tolist()
    unbiased = cum4x(y, y, y, y, 3, 100, 0, "unbiased").flatten().tolist()

    print(biased)
    print(unbiased)

    assert biased == [
        -0.5234348330742729,
        -0.43056952713986557,
        1.1665058630740923,
        3.2158320429550487,
        1.980876165048158,
        -0.38022387563210686,
        -1.05836041124719,
    ]

    assert unbiased == [
        -0.5396235392518276,
        -0.43935666034680176,
        1.1782887505798905,
        3.2158320429550487,
        2.0008850152001614,
        -0.38798354656337414,
        -1.091093207471329,
    ]


def test_cum4x():
    t = np.linspace(0, 10, 1000)
    w = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(len(t))
    x = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
    y = np.sin(2 * np.pi * 15 * t) + 0.1 * np.random.randn(len(t))
    z = np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.randn(len(t))
    w, x, y, z = w.reshape(-1, 1), x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)  # Reshape to 2D arrays
    cum4 = cum4x(w, x, y, z, maxlag=50, nsamp=256, overlap=50, k1=5, k2=10)
    assert cum4.shape == (101, 1)
