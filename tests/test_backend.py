import spectrum._backend as backend


def test_get_backend_prefers_c():
    # either "c" (extension built) or "python" (not built); must be one of
    assert backend.get_backend() in ("c", "python")


def test_env_forces_python(monkeypatch):
    monkeypatch.setenv("SPECTRUM_BACKEND", "python")
    assert backend.use_c() is False
    assert backend.get_backend() == "python"


def test_missing_extension_falls_back(monkeypatch):
    monkeypatch.delenv("SPECTRUM_BACKEND", raising=False)
    monkeypatch.setattr(backend, "_attempted", True)
    monkeypatch.setattr(backend, "_core_module", None)
    assert backend.use_c() is False
    assert backend.get_backend() == "python"
