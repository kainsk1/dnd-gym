"""Shader loading (with ``#include``), full-screen quad helpers and texture utilities."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import moderngl
import numpy as np

log = logging.getLogger(__name__)
SHADER_DIR = Path(__file__).resolve().parent.parent / "shaders"
_INC = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.M)


def read_glsl(name: str, _depth: int = 0) -> str:
    src = (SHADER_DIR / name).read_text(encoding="utf-8")
    if _depth > 4:
        return src
    return _INC.sub(lambda m: read_glsl(m.group(1), _depth + 1), src)


class ShaderLib:
    """Compiles and caches programs from ``<name>.vert`` / ``<name>.frag`` pairs."""

    def __init__(self, ctx: moderngl.Context, defines: dict[str, object] | None = None):
        self.ctx = ctx
        self.defines = defines or {}
        self._cache: dict[str, moderngl.Program] = {}

    def _prep(self, src: str, stage: str = "FRAGMENT") -> str:
        lines = src.split("\n")
        head = "\n".join(f"#define {k} {v}" for k, v in self.defines.items()) + f"\n#define {stage}_SHADER"
        if lines and lines[0].startswith("#version"):
            return lines[0] + "\n" + head + "\n" + "\n".join(lines[1:])
        return "#version 330 core\n" + head + "\n" + src

    def get(self, vert: str, frag: str | None = None) -> moderngl.Program:
        frag = frag or vert
        key = f"{vert}|{frag}"
        p = self._cache.get(key)
        if p is None:
            vs = self._prep(read_glsl(vert + ".vert"), "VERTEX")
            fs = self._prep(read_glsl(frag + ".frag"))
            try:
                p = self.ctx.program(vertex_shader=vs, fragment_shader=fs)
            except Exception:
                log.error("shader compile failed: %s / %s", vert, frag)
                raise
            self._cache[key] = p
        return p


def set_uniform(prog: moderngl.Program, name: str, value) -> None:
    try:
        u = prog[name]
    except KeyError:
        return
    if isinstance(value, np.ndarray):
        u.write(np.ascontiguousarray(value, dtype="f4").tobytes())
    else:
        u.value = value


class Quad:
    """Unit quad (0..1) used for instanced sprites and as full-screen triangle strip."""

    def __init__(self, ctx: moderngl.Context):
        self.vbo = ctx.buffer(np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype="f4").tobytes())
        self.fs_vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())


def srgb_to_linear(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
