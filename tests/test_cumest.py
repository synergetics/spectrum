import scipy.io as sio
import numpy as np
from spectrum import cum2est, cum3est, cum4est
from spectrum import cumest


def test_specific_2est():
    y = sio.loadmat("./tests/demo/ma1.mat")["y"]

    unbiased = cum2est(y, 2, 128, 0, "unbiased").flatten().tolist()
    biased = cum2est(y, 2, 128, 0, "biased").flatten().tolist()

    assert biased == [
        -0.12250512866728513,
        0.35963612544369206,
        1.0058694488562387,
        0.35963612544369206,
        -0.12250512866728513,
    ]

    assert unbiased == [
        -0.1244496545191468,
        0.3624679059589967,
        1.0058694488562387,
        0.3624679059589967,
        -0.1244496545191468,
    ]


def test_specific_3est():
    y = sio.loadmat("./tests/demo/ma1.mat")["y"]

    biased = cum3est(y, 2, 128, 0, "biased", 1).flatten().tolist()
    unbiased = cum3est(y, 2, 128, 0, "unbiased", 1).flatten().tolist()

    assert biased == [
        -0.18203038517731535,
        0.07751502877676983,
        0.671130353691816,
        0.7299529988645022,
        0.07751502877676983,
    ]

    assert unbiased == [
        -0.18639911442157092,
        0.07874542605894078,
        0.6764148446657673,
        0.7415395544020339,
        0.07937538946741231,
    ]


def test_specific_4est():
    y = sio.loadmat("./tests/demo/ma1.mat")["y"]

    biased = cum4est(y, 3, 128, 0, "biased", 1, 1).flatten().tolist()
    unbiased = cum4est(y, 3, 128, 0, "unbiased", 1, 1).flatten().tolist()

    assert biased == [
        -0.036420834253640295,
        0.47550259618188895,
        0.6352588000992427,
        1.3897523166421655,
        0.8379111729560189,
        0.41641134290356524,
        -0.9738632234863677,
    ]

    assert unbiased == [
        -0.0401138801490203,
        0.4873679301642249,
        0.6494892735626269,
        1.4073463302985982,
        0.8445088987273259,
        0.4230397918730159,
        -0.997249679819298,
    ]


def test_cumest_default_nsamp():
    # nsamp=None (default) falls back to data length instead of crashing
    rng = np.random.default_rng(0)
    y = rng.standard_normal(256).reshape(-1, 1)
    c = cumest(y, 2, 5)
    assert c.shape == (11, 1)


def test_cumest_order4_dispatch():
    # norder=4 must reach cum4est, not cum3est
    rng = np.random.default_rng(1)
    y = rng.standard_normal(512).reshape(-1, 1)
    via_cumest = cumest(y, 4, 3, nsamp=512, k1=1, k2=1)
    direct = cum4est(y, 3, 512, 0, "biased", 1, 1)
    assert np.allclose(via_cumest, direct)


def test_cumest_flag_normalization():
    # MATLAB: any flag not starting with b/B means unbiased
    rng = np.random.default_rng(2)
    y = rng.standard_normal(256).reshape(-1, 1)
    assert np.allclose(cumest(y, 2, 3, nsamp=256, flag="Unbiased"), cumest(y, 2, 3, nsamp=256, flag="unbiased"))
    assert np.allclose(cumest(y, 2, 3, nsamp=256, flag="b"), cumest(y, 2, 3, nsamp=256, flag="biased"))


def test_cumest_1d_input():
    rng = np.random.default_rng(3)
    y = rng.standard_normal(256)
    c = cumest(y, 3, 4, nsamp=256)
    assert c.shape == (9, 1)


def test_cum4est_k2_zero():
    # k2=0 (default) previously crashed on an empty-slice assignment
    rng = np.random.default_rng(4)
    y = rng.standard_normal(4096).reshape(-1, 1)
    c4 = cum4est(y, 5, 4096, 0, "biased", 0, 0)
    assert c4.shape == (11, 1)
    # 4th-order cumulants of Gaussian data are ~0 (estimator std ~ 0.08)
    assert np.abs(c4).max() < 0.25


def test_cum4est_gaussian_moment_products():
    # c4 = m4 - products of 2nd-order moments; for Gaussian the result is ~0
    rng = np.random.default_rng(5)
    y = rng.standard_normal(20000).reshape(-1, 1)
    for k1, k2 in [(0, 1), (1, -2), (-1, 1)]:
        c4 = cum4est(y, 3, 20000, 0, "biased", k1, k2)
        assert np.abs(c4).max() < 0.15
