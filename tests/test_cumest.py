import scipy.io as sio
import numpy as np
from spectrum import cum2est, cum3est, cum4est


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
