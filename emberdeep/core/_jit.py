"""Optional Numba acceleration shim.

``njit`` is :func:`numba.njit` (with ``cache=True`` and ``nogil=True`` defaults) when Numba is
importable, otherwise an identity decorator.  Setting the environment variable
``EMBERDEEP_NO_NUMBA=1`` forces the pure-Python/NumPy fallback.  Either way every decorated
function exposes ``.py_func`` (the undecorated Python implementation), which the test-suite
uses to check that compiled and interpreted results agree.

All kernels in :mod:`emberdeep.core` are written in the Numba-compatible subset of Python, so
the same source runs in both modes.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

log = logging.getLogger(__name__)

_DISABLED = os.environ.get("EMBERDEEP_NO_NUMBA", "").strip().lower() in ("1", "true", "yes", "on")

HAVE_NUMBA = False
_numba: Any = None
if not _DISABLED:
    try:  # pragma: no cover - depends on environment
        import numba as _numba

        HAVE_NUMBA = True
    except Exception:  # noqa: BLE001 - any import problem means "no numba"
        log.debug("numba unavailable; using pure-Python fallback")


def _identity(fn: Callable) -> Callable:
    """Fallback decorator: mark ``fn`` as its own ``py_func`` and return it unchanged."""
    try:
        fn.py_func = fn  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover
        pass
    return fn


def njit(*args: Any, **kwargs: Any) -> Any:
    """Decorator usable as ``@njit`` or ``@njit(cache=True, ...)``.

    With Numba: compiles in nopython mode with on-disk caching.  Without: identity.
    """
    if HAVE_NUMBA:
        kwargs.setdefault("cache", True)
        kwargs.setdefault("nogil", True)
        if len(args) == 1 and callable(args[0]):
            return _numba.njit(**kwargs)(args[0])
        return _numba.njit(*args, **kwargs)
    if len(args) == 1 and callable(args[0]):
        return _identity(args[0])
    return _identity
