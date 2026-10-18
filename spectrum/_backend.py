"""Backend dispatch for the spectrum package.

Estimators run on the native C extension (``spectrum._spectrum_core``)
when it is importable, and fall back to the pure-Python reference
implementations otherwise.

The selection can be forced with the ``SPECTRUM_BACKEND`` environment
variable: set it to ``python`` to always use the pure-Python path, or
``c`` (the default) to prefer the C core.
"""

import logging
import os
from typing import Any, Optional

log = logging.getLogger(__name__)

_core_module: Optional[Any] = None
_attempted = False


def core() -> Optional[Any]:
    """Return the C extension module, or None when unavailable."""
    global _core_module, _attempted
    if not _attempted:
        _attempted = True
        try:
            import spectrum._spectrum_core as m

            _core_module = m
        except Exception as exc:  # pragma: no cover - environment dependent
            _core_module = None
            log.warning(
                "spectrum._spectrum_core unavailable (%s); " "using pure-Python estimators",
                exc,
            )
    return _core_module


def use_c() -> bool:
    """True when estimators should dispatch to the C core."""
    if os.environ.get("SPECTRUM_BACKEND", "c").lower() == "python":
        return False
    return core() is not None


def get_backend() -> str:
    """Return the active backend name, "c" or "python"."""
    return "c" if use_c() else "python"
