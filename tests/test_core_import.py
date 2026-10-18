import pytest

core = pytest.importorskip("spectrum._spectrum_core")


def test_selftest():
    assert core._selftest() == 42
