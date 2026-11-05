"""Emberdeep presentation layer.

Importing this package never imports pygame or moderngl (training mode stays
graphics-free). ``Renderer`` is resolved lazily on first attribute access.
"""
from __future__ import annotations

__all__ = ["Renderer"]


def __getattr__(name: str):
    if name == "Renderer":
        from emberdeep.render.renderer import Renderer
        return Renderer
    raise AttributeError(name)
