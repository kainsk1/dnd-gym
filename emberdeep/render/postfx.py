"""HDR target, bloom chain, tone-mapping + grading composite."""
from __future__ import annotations

import moderngl
import numpy as np

from emberdeep.render.gl.util import Quad, ShaderLib, set_uniform

# theme -> (exposure, lift rgb, gain rgb, saturation, contrast, vignette, bloom, grain)
GRADES = {
    "dungeon": (1.3, (0.9, 0.55, 0.3), (1.04, 0.98, 0.90), 1.05, 1.06, 0.55, 0.55, 0.035),
    "crypt":   (1.3, (0.2, 0.6, 0.7), (0.90, 1.0, 1.04), 0.92, 1.07, 0.6, 0.5, 0.04),
    "cave":    (1.3, (0.2, 0.55, 0.5), (0.92, 1.02, 1.0), 1.0, 1.06, 0.55, 0.55, 0.035),
    "castle":  (1.08, (0.6, 0.5, 0.4), (1.02, 1.0, 0.95), 1.05, 1.05, 0.45, 0.5, 0.03),
    "forest":  (1.15, (0.3, 0.55, 0.3), (1.0, 1.04, 0.94), 1.15, 1.04, 0.4, 0.45, 0.025),
    "sewer":   (1.3, (0.3, 0.5, 0.2), (0.95, 1.03, 0.92), 0.95, 1.06, 0.55, 0.5, 0.04),
    "temple":  (1.25, (0.8, 0.6, 0.3), (1.04, 1.0, 0.92), 1.05, 1.05, 0.45, 0.6, 0.03),
    "lair":    (1.3, (1.0, 0.25, 0.1), (1.06, 0.95, 0.88), 1.1, 1.08, 0.6, 0.65, 0.04),
    "arena":   (0.85, (0.8, 0.6, 0.4), (1.0, 0.98, 0.95), 0.92, 1.05, 0.45, 0.5, 0.03),
    "tavern":  (1.1, (1.0, 0.6, 0.3), (1.05, 0.98, 0.9), 1.05, 1.05, 0.5, 0.5, 0.03),
}


class PostFX:
    def __init__(self, ctx: moderngl.Context, shaders: ShaderLib, quad: Quad, levels: int = 4):
        self.ctx, self.sh, self.quad, self.levels = ctx, shaders, quad, levels
        self.size = (0, 0)
        self.scene_tex = self.scene_fbo = None
        self.chain: list[tuple[moderngl.Texture, moderngl.Framebuffer, tuple[int, int]]] = []
        self.out_tex = self.out_fbo = None
        self.p_down = shaders.get("post", "bloom_down")
        self.p_up = shaders.get("post", "bloom_up")
        self.p_comp = shaders.get("post", "composite")
        self.p_hud = shaders.get("post", "hud")
        self.p_blit = shaders.get("post", "blit")
        mk = lambda p: ctx.vertex_array(p, [(quad.fs_vbo, "2f", "in_pos")])
        self.vao_down, self.vao_up, self.vao_comp = mk(self.p_down), mk(self.p_up), mk(self.p_comp)
        self.vao_hud, self.vao_blit = mk(self.p_hud), mk(self.p_blit)
        self.ui_tex: moderngl.Texture | None = None

    def resize(self, rw: int, rh: int, fw: int, fh: int) -> None:
        ctx = self.ctx
        for o in (self.scene_tex, self.scene_fbo, self.out_tex, self.out_fbo):
            if o is not None:
                o.release()
        for t, f, _ in self.chain:
            t.release(); f.release()
        self.chain = []
        self.size = (rw, rh)
        self.full = (fw, fh)
        self.scene_tex = ctx.texture((rw, rh), 4, dtype="f2")
        self.scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_tex.repeat_x = self.scene_tex.repeat_y = False
        self.scene_fbo = ctx.framebuffer([self.scene_tex])
        w, h = rw, rh
        for _ in range(self.levels):
            w, h = max(2, w // 2), max(2, h // 2)
            t = ctx.texture((w, h), 4, dtype="f2")
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            self.chain.append((t, ctx.framebuffer([t]), (w, h)))
        self.out_tex = ctx.texture((fw, fh), 4, dtype="f1")
        self.out_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.out_fbo = ctx.framebuffer([self.out_tex])
        if self.ui_tex is not None:
            self.ui_tex.release()
        self.ui_tex = ctx.texture((fw, fh), 4)
        self.ui_tex.swizzle = "BGRA"
        self.ui_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def bloom(self, threshold: float = 0.9) -> None:
        ctx = self.ctx
        ctx.disable(moderngl.BLEND)
        src, (sw, sh) = self.scene_tex, self.size
        for i, (t, f, (w, h)) in enumerate(self.chain):
            f.use()
            ctx.viewport = (0, 0, w, h)
            src.use(0)
            self.p_down["u_src"].value = 0
            set_uniform(self.p_down, "u_texel", (1.0 / sw, 1.0 / sh))
            set_uniform(self.p_down, "u_first", 1.0 if i == 0 else 0.0)
            set_uniform(self.p_down, "u_threshold", threshold)
            self.vao_down.render(moderngl.TRIANGLE_STRIP)
            src, sw, sh = t, w, h
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        for i in range(len(self.chain) - 1, 0, -1):
            t, _, (w, h) = self.chain[i]
            _, f2, (w2, h2) = self.chain[i - 1]
            f2.use()
            ctx.viewport = (0, 0, w2, h2)
            t.use(0)
            self.p_up["u_src"].value = 0
            set_uniform(self.p_up, "u_texel", (1.0 / w, 1.0 / h))
            self.vao_up.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)

    def composite(self, params: dict) -> None:
        ctx = self.ctx
        self.out_fbo.use()
        ctx.viewport = (0, 0) + self.full
        ctx.disable(moderngl.BLEND)
        self.scene_tex.use(0)
        self.chain[0][0].use(1)
        p = self.p_comp
        p["u_scene"].value = 0
        p["u_bloom"].value = 1
        for k, v in params.items():
            set_uniform(p, k, v)
        self.vao_comp.render(moderngl.TRIANGLE_STRIP)

    def draw_ui(self, surf_bytes: bytes | None) -> None:
        if surf_bytes is None:
            return
        ctx = self.ctx
        self.ui_tex.write(surf_bytes)
        self.out_fbo.use()
        ctx.viewport = (0, 0) + self.full
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ui_tex.use(0)
        self.p_hud["u_ui"].value = 0
        self.vao_hud.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)

    def present(self, screen_fbo, size) -> None:
        ctx = self.ctx
        screen_fbo.use()
        ctx.viewport = (0, 0) + tuple(size)
        ctx.disable(moderngl.BLEND)
        self.out_tex.use(0)
        self.p_blit["u_src"].value = 0
        self.vao_blit.render(moderngl.TRIANGLE_STRIP)

    def read(self) -> np.ndarray:
        w, h = self.full
        raw = self.out_fbo.read(components=3, alignment=1)
        a = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
        return np.ascontiguousarray(a[::-1])
