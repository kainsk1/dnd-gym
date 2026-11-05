"""Emberdeep world renderer: OpenGL 3.3 (moderngl) + pygame-ce window, implements ``RendererProtocol``."""
from __future__ import annotations

import logging
import math
import os
import time

import numpy as np

from emberdeep.core.types import Material, Terrain
from emberdeep.render.api import ActionDescriptor, FrameState, StaticScene
from emberdeep.render.hud_api import UICommand, ViewInfo

log = logging.getLogger(__name__)
F4 = np.float32

QUALITY = {
    "high": dict(tex=192, sdf_s=4, steps=28, max_lights=64, particles=24000, bloom=5, scale=1.0),
    "medium": dict(tex=128, sdf_s=4, steps=20, max_lights=40, particles=12000, bloom=4, scale=1.0),
    "low": dict(tex=128, sdf_s=3, steps=12, max_lights=20, particles=5000, bloom=3, scale=0.75),
}

WALL_MATS = {Material.STONE_WALL, Material.CAVE_WALL, Material.BRICK_WALL, Material.WOOD_WALL, Material.HEDGE}
# material kinds: 0 floor, 1 wall, 2 water, 3 lava, 4 chasm, 5 deep water
# material table: struct, ragged, kind, tile
MAT_TABLE = {
    Material.NONE: (1, 0, 0, 1), Material.STONE_FLOOR: (0, 0, 0, 2.5), Material.COBBLE: (0, 0, 0, 2.0), Material.CAVE_FLOOR: (0, 1, 0, 3.0),
    Material.DIRT: (0, 1, 0, 2.5), Material.GRASS: (0, 1, 0, 2.5), Material.WOOD_PLANKS: (0, 0, 0, 2.5), Material.MARBLE: (0, 0, 0, 2.5),
    Material.SAND: (0, 1, 0, 2.5), Material.WATER: (0, 0, 2, 2.0), Material.DEEP_WATER: (0, 0, 5, 2.0), Material.LAVA: (0, 0, 3, 2.0),
    Material.STONE_WALL: (0, 0, 1, 2.0), Material.CAVE_WALL: (0, 0, 1, 2.5), Material.BRICK_WALL: (0, 0, 1, 2.0), Material.WOOD_WALL: (0, 0, 1, 2.0),
    Material.RUBBLE: (0, 1, 0, 2.0), Material.MOSS: (0, 1, 0, 2.5), Material.ICE: (0, 0, 0, 2.5), Material.CARPET_RED: (0, 0, 0, 1.5),
    Material.BONE_FLOOR: (0, 1, 0, 2.5), Material.METAL_GRATE: (1, 0, 0, 1), Material.CHASM: (0, 0, 4, 2.5), Material.SNOW: (0, 1, 0, 3.0),
    Material.HEDGE: (0, 0, 1, 1.5),
}

ZONE_KINDS = {"web": 0, "fog_cloud": 1, "stinking_cloud": 2, "darkness": 3, "spirit_guardians": 4, "moonbeam": 5, "spike_growth": 6,
              "entangle": 7, "grease": 8, "silence": 9, "sleet_storm": 10, "wall_of_fire": 11, "fire": 11, "flaming_sphere": 12,
              "daylight": 13, "faerie_fire": 14}
GROUND_ZONES = {0, 4, 5, 6, 7, 8, 9, 14, 15}

COND_OVERLAYS = {"poisoned": (3, (0.35, 0.9, 0.25)), "frightened": (4, (0.65, 0.4, 1.0)), "paralyzed": (5, (1.0, 0.95, 0.5)),
                 "stunned": (10, (1.0, 0.9, 0.4)), "petrified": (6, (0.75, 0.75, 0.72)), "restrained": (7, (0.95, 0.97, 1.0)),
                 "grappled": (7, (0.9, 0.85, 0.7)), "charmed": (12, (1.0, 0.4, 0.75))}
EFFECT_OVERLAYS = {"rage": (8, (1.0, 0.25, 0.08)), "raging": (8, (1.0, 0.25, 0.08)), "bless": (9, (1.0, 0.85, 0.4)),
                   "shield_of_faith": (11, (0.6, 0.8, 1.0)), "shield": (11, (0.5, 0.7, 1.0)), "mage_armor": (11, (0.55, 0.65, 1.0)),
                   "protection": (11, (0.9, 0.9, 1.0)), "sanctuary": (11, (1.0, 0.95, 0.7)), "barkskin": (11, (0.6, 0.8, 0.3)),
                   "heroism": (9, (1.0, 0.8, 0.3)), "haste": (13, (1.0, 0.9, 0.4)), "aid": (9, (1.0, 0.5, 0.5))}


class NullHUD:
    """No-op HUD used when the UI module is unavailable (or hud=False)."""

    def set_scene(self, scene): pass
    def on_frame(self, frame): pass
    def on_event(self, event, frame, view): pass
    def update(self, dt, view): pass
    def draw(self, surface, view): pass
    def handle_event(self, ev, view): return None
    def begin_human_choice(self, agent_id, actions): pass
    def end_human_choice(self): pass
    def world_click(self, cell, eid, button, view): return None


class NullAudio:
    def on_event(self, event): pass
    def update(self, dt): pass
    def close(self): pass


def _box_blur_np(a, r):
    for ax in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=ax, dtype=F4)
        n = a.shape[ax]
        a = (np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=ax) - np.take(c, np.arange(0, n), axis=ax)) / (2 * r + 1)
    return a


class ZoneLayer:
    def __init__(self, zid, kind, cells, center, radius, attached=-1, harmful=True):
        self.id, self.kind, self.attached, self.harmful = zid, kind, attached, harmful
        self.alpha = 0.0
        self.target = 1.0
        self.set_cells(cells, center, radius)
        self.base_center = tuple(center)
        self.tex = None
        self.dirty = True

    def set_cells(self, cells, center, radius):
        self.cells = np.array(cells, np.int32).reshape(-1, 2) if len(cells) else np.zeros((0, 2), np.int32)
        self.center = tuple(center)
        self.radius = float(radius)
        self.dirty = True


class Renderer:
    def __init__(self, mode: str = "human", width: int = 1600, height: int = 900, quality: str = "medium", fps: int = 60,
                 hud: bool = True, audio: bool = True, pov: str = "omniscient", seed: int = 0):
        assert mode in ("human", "rgb_array")
        self.mode, self.fps = mode, fps
        self.quality = QUALITY.get(quality, QUALITY["medium"])
        self.quality_name = quality if quality in QUALITY else "medium"
        self._closed = False
        self._init_gl(width, height)
        from emberdeep.render.camera import Camera
        from emberdeep.render.gl.util import Quad, ShaderLib
        from emberdeep.render.particles import ParticleSystem
        from emberdeep.render.postfx import PostFX
        from emberdeep.render.tokens import TokenAtlas
        from emberdeep.render.vfx.effects import FXContext
        self.shaders = ShaderLib(self.ctx, {"STEPS": self.quality["steps"]})
        self.quad = Quad(self.ctx)
        self.post = PostFX(self.ctx, self.shaders, self.quad, self.quality["bloom"])
        self.particles = ParticleSystem(self.ctx, self.shaders, self.quad, self.quality["particles"])
        self.tokens = TokenAtlas(self.ctx, 96)
        self.fx = FXContext(self)
        self.camera = Camera(40, 28, (self.fw, self.fh), self.ui_scale)
        self.hud = self._make_hud() if hud else NullHUD()
        self.audio = self._make_audio(audio)
        self.has_hud = not isinstance(self.hud, NullHUD)
        if self.has_hud:
            try:
                self.hud.audio = self.audio if not isinstance(self.audio, NullAudio) else getattr(self.hud, "audio", None)
            except Exception:  # noqa: BLE001
                pass
        self._set_pads()
        from emberdeep.render.director import Director
        self.director = Director(self)
        self.scene: StaticScene | None = None
        self.frame: FrameState | None = None
        self.vents: dict = {}
        self.zones: dict[int, ZoneLayer] = {}
        self.pov = pov
        self.pov_options = ["omniscient"]
        self.time = 0.0
        self.paused = False
        self.speed = 1.0
        self.options = dict(getattr(self.hud, "DEFAULT_OPTIONS", None) or {"grid": False, "labels": True, "log": True, "minimap": True})
        self.hover_cell = None
        self.hover_eid = -1
        self.selected_eid = -1
        self.human_pending = False
        self.human_result: int | None = None
        self.human_actions: list[ActionDescriptor] = []
        self._drag = None
        self._light_rows = np.zeros((0, 10), F4)
        self._rgb_dirty = True
        self.timings: dict[str, float] = {}
        self._ui_surface = None
        self._ambient_acc = 0.0
        self._step_request = False
        self._rng = np.random.default_rng(seed)
        self._frames_rendered = 0
        self.door_vis: dict = {}
        self.active_vis = None
        self._closed_sig = None

    # ------------------------------------------------------------------------------------------
    def _init_gl(self, width, height):
        import moderngl
        import pygame
        self._pg, self._mgl = pygame, moderngl
        if self.mode == "rgb_array" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self.clock = pygame.time.Clock()
        if self.mode == "human":
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
            try:
                pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, 1)
            except Exception:  # noqa: BLE001
                pass
            pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 0)
            pygame.display.set_caption("Emberdeep")
            self.window = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
            self.ctx = moderngl.create_context()
            self.screen_fbo = self.ctx.screen
            ww, wh = pygame.display.get_window_size()
            sw, sh = self.window.get_size()
            self.fw, self.fh = sw, sh
            self.win_size = (ww, wh)
        else:
            self.ctx = moderngl.create_standalone_context(require=330)
            self.screen_fbo = None
            self.fw, self.fh = width, height
            self.win_size = (width, height)
        self.ui_scale = float(min(max(self.fh / 900.0, 0.6), 2.5))
        self.ctx.gc_mode = "context_gc"

    def _set_pads(self):
        s = self.fh / 900.0
        self.camera.pad_l, self.camera.pad_r = (292.0 * s, 344.0 * s) if self.has_hud and getattr(self, "hud_visible", True) else (0.0, 0.0)

    def _make_hud(self):
        try:
            from emberdeep.render.ui.hud import HUD
        except Exception as e:  # noqa: BLE001
            log.info("HUD unavailable (%s); using NullHUD", e)
            return NullHUD()
        for args in ((), (self,)):
            try:
                return HUD(*args)
            except TypeError:
                continue
            except Exception as e:  # noqa: BLE001
                log.warning("HUD init failed: %s", e)
                return NullHUD()
        return NullHUD()

    def _make_audio(self, enabled):
        if not enabled or self.mode != "human":
            return NullAudio()
        try:
            from emberdeep.render.audio import AudioEngine
            return AudioEngine(True, 0.7)
        except Exception as e:  # noqa: BLE001
            log.info("audio unavailable (%s)", e)
            return NullAudio()

    # ------------------------------------------------------------------------------------------
    # scene compilation
    # ------------------------------------------------------------------------------------------
    def set_scene(self, scene: StaticScene) -> None:
        from emberdeep.render.fog import Fog
        from emberdeep.render.lighting import LightSet, build_info, build_occluder
        from emberdeep.render.props import build_atlas
        from emberdeep.render.textures import N_VARIANTS, build_texture_arrays
        ctx, mgl = self.ctx, self._mgl
        self._release_scene()
        self.scene = scene
        W, H = scene.width, scene.height
        self.W, self.H = W, H
        s = self.quality["sdf_s"]
        if W * s > 700:
            s = max(2, 700 // W)
        self.S = s
        self.camera = type(self.camera)(W, H, (self.fw, self.fh), self.ui_scale)
        self._set_pads()
        terrain = np.asarray(scene.terrain, np.uint8)
        material = np.asarray(scene.material, np.uint8).copy()
        # --- sanitise materials against terrain ---------------------------------------------------
        mat = material.copy()
        wallish = (terrain == Terrain.WALL)
        floorish = ~wallish
        for m in WALL_MATS:
            pass
        is_wall_mat = np.isin(mat, [int(m) for m in WALL_MATS])
        mat[wallish & ~is_wall_mat] = int(Material.STONE_WALL)
        mat[~wallish & is_wall_mat & (terrain != Terrain.PILLAR) & (terrain != Terrain.LOW_WALL)] = int(Material.STONE_FLOOR)
        mat[(mat == 0) & (terrain != Terrain.VOID) & ~wallish] = int(Material.STONE_FLOOR)
        mat[terrain == Terrain.PIT] = int(Material.CHASM)
        mat[(terrain == Terrain.LAVA)] = int(Material.LAVA)
        mat[(terrain == Terrain.WATER_DEEP)] = int(Material.DEEP_WATER)
        mat[(terrain == Terrain.WATER_SHALLOW) & (mat != int(Material.WATER)) & (mat != int(Material.DEEP_WATER))] = int(Material.WATER)
        mat[(terrain == Terrain.ICE)] = int(Material.ICE)
        mat[terrain == Terrain.PILLAR] = np.where(is_wall_mat[terrain == Terrain.PILLAR], mat[terrain == Terrain.PILLAR], int(Material.STONE_WALL))
        mat[terrain == Terrain.LOW_WALL] = np.where(is_wall_mat[terrain == Terrain.LOW_WALL], mat[terrain == Terrain.LOW_WALL], int(Material.BRICK_WALL))
        self.mat = mat
        shape = np.zeros((H, W), np.uint8)
        shape[terrain == Terrain.WALL] = 1
        shape[terrain == Terrain.PILLAR] = 2
        shape[terrain == Terrain.LOW_WALL] = 3
        shape[terrain == Terrain.VOID] = 4
        rng = np.random.default_rng(scene.seed + 17)
        variant = rng.integers(0, N_VARIANTS, (H, W)).astype(np.uint8)
        rot = rng.integers(0, 4, (H, W)).astype(np.uint8)
        # underlay for pillars / low walls
        under = np.full((H, W), int(Material.STONE_FLOOR), np.uint8)
        for y, x in zip(*np.nonzero((shape == 2) | (shape == 3))):
            cand = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                xx, yy = x + dx, y + dy
                if 0 <= xx < W and 0 <= yy < H and shape[yy, xx] == 0:
                    cand.append(mat[yy, xx])
            if cand:
                under[y, x] = max(set(cand), key=cand.count)
        cellmap = np.stack([mat, shape, under, variant], -1).astype(np.uint8)
        self.cellmap_tex = ctx.texture((W, H), 4, cellmap.tobytes())
        self.cellmap_tex.filter = (mgl.NEAREST, mgl.NEAREST)
        self.cellmap_tex.repeat_x = self.cellmap_tex.repeat_y = False
        ys, xs = np.nonzero(shape != 4)
        inst = np.stack([xs, ys, mat[ys, xs], variant[ys, xs], rot[ys, xs], shape[ys, xs]], -1).astype(F4)
        self.n_cells = len(inst)
        self.cell_vbo = ctx.buffer(inst.tobytes())
        self.prog_terrain = self.shaders.get("terrain")
        self.vao_terrain = ctx.vertex_array(self.prog_terrain, [(self.quad.vbo, "2f", "in_pos"), (self.cell_vbo, "2f 4f/i", "in_cell", "in_m")])
        # --- textures ------------------------------------------------------------------------------
        alb, nrm = build_texture_arrays(self.quality["tex"])
        res = alb.shape[1]
        self.tex_alb = ctx.texture_array((res, res, alb.shape[0]), 4, np.ascontiguousarray(alb).tobytes())
        self.tex_nrm = ctx.texture_array((res, res, nrm.shape[0]), 4, np.ascontiguousarray(nrm).tobytes())
        for t in (self.tex_alb, self.tex_nrm):
            t.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
            t.anisotropy = 4.0
            t.build_mipmaps()
        mt = np.zeros((32, 4), F4)
        for m, v in MAT_TABLE.items():
            mt[int(m)] = v
        self.mat_table = mt
        # --- occluder / info ----------------------------------------------------------------------------
        self.terrain = terrain
        self.sdf_np = build_occluder(terrain, [], s)
        self.sdf_tex = self._tex_r(self.sdf_np, "f2")
        info = build_info(mat, terrain, s)
        self.info_tex = ctx.texture((W * s, H * s), 4, info.tobytes())
        self.info_tex.filter = (mgl.LINEAR, mgl.LINEAR)
        self.info_tex.repeat_x = self.info_tex.repeat_y = False
        self.dark_tex = ctx.texture((W * 2, H * 2), 1, np.zeros((H * 2, W * 2), np.uint8).tobytes())
        self.dark_tex.filter = (mgl.LINEAR, mgl.LINEAR)
        self.dark_tex.repeat_x = self.dark_tex.repeat_y = False
        self._dark_sig = None
        # --- sprites ---------------------------------------------------------------------------------
        self.atlas = build_atlas()
        self.pa_tex = ctx.texture(self.atlas.alb.shape[1::-1], 4, np.ascontiguousarray(self.atlas.alb).tobytes())
        self.pn_tex = ctx.texture(self.atlas.nrm.shape[1::-1], 4, np.ascontiguousarray(self.atlas.nrm).tobytes())
        for t in (self.pa_tex, self.pn_tex):
            t.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
            t.anisotropy = 2.0
            t.build_mipmaps()
        self.prog_sprite = self.shaders.get("sprite")
        self.sprite_dyn_vbo = ctx.buffer(reserve=4096 * 16 * 4, dynamic=True)
        self.vao_sprite_dyn = ctx.vertex_array(self.prog_sprite, [(self.quad.vbo, "2f", "in_pos"),
                                               (self.sprite_dyn_vbo, "2f 2f 4f 4f 4f/i", "in_c", "in_size", "in_r", "in_uv", "in_col")])
        self._build_static_decor(scene)
        # --- lights ------------------------------------------------------------------------------------
        lights = self._collect_lights(scene)
        self.lights = LightSet(lights, self.sdf_np, s)
        self.prog_light = self.shaders.get("light")
        self.light_vbo = ctx.buffer(reserve=160 * 10 * 4, dynamic=True)
        self.vao_light = ctx.vertex_array(self.prog_light, [(self.quad.vbo, "2f", "in_pos"), (self.light_vbo, "2f 4f 4f/i", "in_lpos", "in_lrad", "in_lcol")])
        # --- fog ---------------------------------------------------------------------------------------
        self.fog = Fog(ctx, W, H, 4)
        # --- other passes / targets --------------------------------------------------------------------
        self._init_passes()
        self._resize_targets()
        from emberdeep.render.postfx import GRADES
        self.grade = GRADES.get(scene.theme, GRADES["dungeon"])
        self.camera.frame_points([(W / 2, H / 2)], 0, snap=True)
        self.camera.zoom = self.camera.tzoom = self.camera._fit_zoom()
        self.zones.clear()
        self.vents.clear()
        self.frame = None
        self.particles.clear()
        self.fx.decals.clear(); self.fx.lights.clear()
        self.pov_options = ["omniscient"] + list(scene.agent_ids)
        try:
            self.hud.set_scene(scene)
        except Exception:  # noqa: BLE001
            log.exception("hud.set_scene failed")
        self.lava_cells = np.argwhere(mat == int(Material.LAVA))[:, ::-1].astype(F4)
        self._rgb_dirty = True

    def _tex_r(self, arr, dtype):
        t = self.ctx.texture((arr.shape[1], arr.shape[0]), 1, np.ascontiguousarray(arr, dtype="f4" if dtype == "f4" else "f2").tobytes(), dtype=dtype)
        t.filter = (self._mgl.LINEAR, self._mgl.LINEAR)
        t.repeat_x = t.repeat_y = False
        return t

    def _release_scene(self):
        for name in ("cellmap_tex", "cell_vbo", "vao_terrain", "tex_alb", "tex_nrm", "sdf_tex", "info_tex", "dark_tex", "pa_tex", "pn_tex",
                     "sprite_dyn_vbo", "vao_sprite_dyn", "static_vbo", "vao_static", "light_vbo", "vao_light"):
            o = getattr(self, name, None)
            if o is not None:
                try:
                    o.release()
                except Exception:  # noqa: BLE001
                    pass
                setattr(self, name, None)
        if getattr(self, "fog", None) is not None:
            self.fog.release()
            self.fog = None
        for z in self.zones.values():
            if z.tex is not None:
                z.tex.release()
        self.zones = {}

    def _collect_lights(self, scene):
        from emberdeep.render.api import LightSpec

        def fix(v):
            return v + 0.5 if abs(v - round(v)) < 1e-6 else v
        out = []
        for l in scene.lights:
            out.append(LightSpec(fix(l.x), fix(l.y), l.bright, l.dim, l.color, l.intensity, l.flicker, l.height, l.kind, l.attached_to))
        have = np.array([[l.x, l.y] for l in out], F4) if out else np.zeros((0, 2), F4)

        def near(x, y, r=1.3):
            return len(have) and bool((np.hypot(have[:, 0] - x, have[:, 1] - y) < r).any())
        extra = {"brazier": (4, 4, "brazier", (1.0, 0.7, 0.4), 0.3), "campfire": (4, 5, "fire", (1.0, 0.6, 0.25), 0.3),
                 "torch_sconce": (4, 4, "torch", (1.0, 0.72, 0.4), 0.15), "candles": (1.5, 2.5, "candle", (1.0, 0.8, 0.5), 0.2),
                 "crystal": (2.5, 2.5, "crystal", (0.4, 0.7, 1.0), 0.05)}
        for d in scene.decor:
            if d.kind in extra and not near(fix(d.x), fix(d.y)):
                b, dm, k, c, f = extra[d.kind]
                out.append(LightSpec(fix(d.x), fix(d.y), b, dm, c, 1.0, f, 1.2, k))
        # lava glow
        lava = np.argwhere(self.mat == int(Material.LAVA))
        if len(lava):
            taken = set()
            for y, x in lava:
                key = (x // 3, y // 3)
                if key in taken:
                    continue
                taken.add(key)
                out.append(LightSpec(x + 0.5, y + 0.5, 2.0, 4.0, (1.0, 0.38, 0.1), 0.9, 0.35, 0.8, "lava"))
        return out

    def _build_static_decor(self, scene):
        order = {"rug": 0, "puddle": 0, "blood_stain": 1, "moss_patch": 1, "cobweb": 1, "glyph": 1, "chains": 1, "grate": 1, "bones": 2, "skull": 2,
                 "rubble": 2, "stairs_down": 0, "stairs_up": 0, "mushrooms": 3, "tree": 9}
        rows = []
        order["bush"] = 4
        for d in sorted(scene.decor, key=lambda d: (order.get(d.kind, 5), d.y)):
            if d.kind not in __import__("emberdeep.render.props", fromlist=["SPRITES"]).SPRITES:
                continue
            x, y = d.x, d.y
            if abs(x - round(x)) < 1e-6 and abs(y - round(y)) < 1e-6:
                x, y = x + 0.5, y + 0.5
            kind, scl = d.kind, d.scale
            cx_, cy_ = int(min(max(x, 0), self.W - 1)), int(min(max(y, 0), self.H - 1))
            if kind == "tree":
                if self.terrain[cy_, cx_] not in (Terrain.WALL, Terrain.VOID):
                    kind, scl = "bush", 1.15
                else:
                    scl = min(scl, 0.8)
            u0, v0, u1, v1, w, h = self.atlas.lookup(kind, d.variant)
            rows.append([x, y, w * scl, h * scl, d.rotation, 0, 0, 0, u0, v0, u1, v1, 1, 1, 1, 1])
        rb = np.random.default_rng(scene.seed + 5)
        for by, bx in np.argwhere(self.terrain == Terrain.BRUSH):
            if rb.random() < 0.7:
                u0, v0, u1, v1, w, h = self.atlas.lookup("bush", int(rb.integers(0, 2)))
                sc = 0.55 + 0.4 * rb.random()
                rows.append([bx + 0.3 + 0.4 * rb.random(), by + 0.3 + 0.4 * rb.random(), w * sc, h * sc, rb.random() * 6.28, 0, 0, 0, u0, v0, u1, v1, 0.85, 1, 0.85, 0.92])
        arr = np.array(rows, F4).reshape(-1, 16)
        self.n_static = len(arr)
        self.static_vbo = self.ctx.buffer(arr.tobytes() if len(arr) else b"\0" * 64)
        self.vao_static = self.ctx.vertex_array(self.prog_sprite, [(self.quad.vbo, "2f", "in_pos"),
                                                (self.static_vbo, "2f 2f 4f 4f 4f/i", "in_c", "in_size", "in_r", "in_uv", "in_col")])

    def _init_passes(self):
        sh, ctx, q = self.shaders, self.ctx, self.quad
        self.prog_ambient = sh.get("ambient")
        self.vao_ambient = ctx.vertex_array(self.prog_ambient, [(q.fs_vbo, "2f", "in_pos")])
        self.prog_fog = sh.get("ambient", "fog")
        self.vao_fog = ctx.vertex_array(self.prog_fog, [(q.fs_vbo, "2f", "in_pos")])
        self.prog_ent = sh.get("entity")
        self.ent_vbo = ctx.buffer(reserve=1024 * 20 * 4, dynamic=True)
        self.vao_ent = ctx.vertex_array(self.prog_ent, [(q.vbo, "2f", "in_pos"), (self.ent_vbo, "4f 4f 4f 4f 4f/i", "in_c", "in_col", "in_p", "in_q", "in_m")])
        self.prog_zone = sh.get("zone")
        self.vao_zone = ctx.vertex_array(self.prog_zone, [(q.vbo, "2f", "in_pos")])
        if not hasattr(self, "_blank_mask"):
            self._blank_mask = ctx.texture((2, 2), 1, b"\xff" * 4)

    def _resize_targets(self):
        ctx, mgl = self.ctx, self._mgl
        rs = self.quality["scale"]
        self.rw, self.rh = max(16, int(self.fw * rs)), max(16, int(self.fh * rs))
        for n in ("g_alb", "g_nrm", "g_emi", "g_fbo", "light_target"):
            o = getattr(self, n, None)
            if o is not None:
                o.release()
        rw, rh = self.rw, self.rh
        self.g_alb = ctx.texture((rw, rh), 4, dtype="f1")
        self.g_nrm = ctx.texture((rw, rh), 4, dtype="f1")
        self.g_emi = ctx.texture((rw, rh), 4, dtype="f2")
        for t in (self.g_alb, self.g_nrm, self.g_emi):
            t.filter = (mgl.NEAREST, mgl.NEAREST)
            t.repeat_x = t.repeat_y = False
        self.g_fbo = ctx.framebuffer([self.g_alb, self.g_nrm, self.g_emi])
        self.post.resize(rw, rh, self.fw, self.fh)
        self._ui_surface = None

    # ------------------------------------------------------------------------------------------
    # frame application (authoritative state)
    # ------------------------------------------------------------------------------------------
    def _apply_frame(self, frame: FrameState, instant: bool = False) -> None:
        from emberdeep.render.vfx.effects import VEnt
        first = self.frame is None
        self.frame = frame
        seen = set()
        for e in frame.entities:
            ve = self.vents.get(e.id)
            if ve is None:
                ve = VEnt(e.id, e.kind, e.name, float(e.x), float(e.y), float(e.z), e.size, e.faction, e.icon, tuple(e.tint))
                ve.bob_phase = (e.id * 1.37) % 6.28
                self.vents[e.id] = ve
                ve.facing = e.facing
                if not first and e.kind == "creature":
                    ve.spawn_scale = 1.0
            seen.add(e.id)
            ve.kind, ve.name, ve.size, ve.faction, ve.icon, ve.tint = e.kind, e.name, e.size, e.faction, e.icon, tuple(e.tint)
            if not ve.moving:
                ve.x, ve.y = float(e.x), float(e.y)
            ve.z = float(e.z)
            ve.conditions = set(e.conditions)
            ve.effects = set(e.effects)
            ve.statuses = set(e.statuses)
            ve.hidden = bool(e.hidden) or ("hidden" in e.statuses)
            ve.invisible = "invisible" in e.conditions
            ve.downed, ve.dead = bool(e.downed), bool(e.dead)
            ve.state = e.state
            ve.light = e.light
            ve.hp, ve.max_hp = e.hp, e.max_hp
            ve.gone = "escaped" in e.statuses or (e.kind == "creature" and not e.alive and not e.dead and not e.downed and False)
            if instant or abs(ve.facing - e.facing) > 1e-3 and not ve.moving:
                ve.facing = e.facing
            ve.tilt = 0.35 if ve.downed else (0.0 if not ve.dead else ve.tilt)
            if ve.dead:
                ve.alpha = 0.0
            elif not ve.downed or True:
                ve.alpha = 1.0 if ve.alpha > 0 or not ve.dead else ve.alpha
        for eid in list(self.vents):
            if eid not in seen:
                del self.vents[eid]
        # POV options & fog
        keys = ["omniscient"] + [a for a in (self.scene.agent_ids if self.scene else [])]
        for k in frame.visible:
            if k not in keys:
                keys.append(k)
        self.pov_options = keys
        self._update_fog_target(instant or first)
        self._frame_party(frame, first)
        self._update_doors(frame)
        self._reconcile_zones(frame, instant or first)
        try:
            self.hud.on_frame(frame)
        except Exception:  # noqa: BLE001
            log.exception("hud.on_frame failed")
        self._rgb_dirty = True

    def _frame_party(self, frame, first):
        pf = {f.index for f in self.scene.factions if f.is_player_side}
        pts = [(e.x + e.size * 0.5, e.y + e.size * 0.5) for e in frame.entities if e.kind == "creature" and e.faction in pf and not e.dead]
        if not pts:
            pts = [(e.x + e.size * 0.5, e.y + e.size * 0.5) for e in frame.entities if e.kind == "creature" and not e.dead]
        if not pts:
            return
        if first:
            self.camera.frame_points(pts, 6.0, snap=True, zmax=1.25)
        elif frame.mode == "exploration" and self.camera.follow and self.active_vis is None:
            a = np.array(pts, F4)
            self.camera.focus(float(a[:, 0].mean()), float(a[:, 1].mean()))

    def _update_fog_target(self, instant=False):
        f = self.frame
        if f is None or not hasattr(self, "fog") or self.fog is None:
            return
        if self.pov == "omniscient" or self.pov not in f.visible:
            self.fog.set_target(None, None)
            return
        self.fog.set_target(f.visible[self.pov], f.explored.get(self.pov), instant)

    def _update_doors(self, frame):
        doors = dict(frame.doors)
        for e in frame.entities:
            if e.kind == "door":
                doors.setdefault((e.x, e.y), e.state or "closed")
        t = self.terrain
        for (x, y), st in doors.items():
            d = self.door_vis.get((x, y))
            orient = "h" if (0 <= x - 1 and x + 1 < self.W and t[y, x - 1] in (Terrain.WALL, Terrain.PILLAR) and t[y, x + 1] in (Terrain.WALL, Terrain.PILLAR)) else "v"
            if d is None:
                self.door_vis[(x, y)] = dict(state=st, ang=1.5 if st in ("open",) else 0.0, target=1.5 if st == "open" else 0.0, orient=orient)
            elif d["state"] != st and not d.get("anim_locked"):
                d["state"] = st
                d["target"] = 1.5 if st == "open" else 0.0
        closed = sorted(k for k, v in self.door_vis.items() if v["state"] in ("closed", "locked"))
        if closed != self._closed_sig:
            from emberdeep.render.lighting import build_occluder
            self._closed_sig = closed
            self.sdf_np = build_occluder(self.terrain, closed, self.S)
            self.sdf_tex.write(np.ascontiguousarray(self.sdf_np, dtype="f2").tobytes())

    def _reconcile_zones(self, frame, instant=False):
        ids = set()
        for z in frame.zones:
            ids.add(z.id)
            zl = self.zones.get(z.id)
            if zl is None:
                zl = ZoneLayer(z.id, z.kind, z.cells, z.center, z.radius, z.attached_to, z.harmful)
                zl.alpha = 1.0 if instant else 0.0
                self.zones[z.id] = zl
            else:
                zl.set_cells(z.cells, z.center, z.radius)
            zl.target = 1.0
        for zid, zl in self.zones.items():
            if zid not in ids and not getattr(zl, "pinned", False):
                zl.target = 0.0

    # ------------------------------------------------------------------------------------------
    # protocol
    # ------------------------------------------------------------------------------------------
    def push(self, frame: FrameState, events: list[dict]) -> None:
        if self.scene is None:
            raise RuntimeError("set_scene() must be called before push()")
        self.director.push(frame, events)

    def is_open(self) -> bool:
        return not self._closed

    def play_until_idle(self, max_seconds: float | None = None) -> None:
        t0 = time.time()
        sim = 0.0
        limit = max_seconds if max_seconds is not None else (120.0 if self.mode == "rgb_array" else 1e9)
        while not self._closed:
            if self.director.idle and not (self.paused and self.mode == "human" and False):
                break
            if self.mode == "human":
                if time.time() - t0 > limit:
                    break
                self.run_frame()
            else:
                if sim > limit:
                    break
                self.step_sim(1.0 / 60.0)
                sim += 1.0 / 60.0
        if self.mode == "human" and not self._closed:
            self.run_frame()

    def capture(self) -> np.ndarray:
        if self.mode == "rgb_array" or self._rgb_dirty:
            self.render()
        return self.post.read()

    def request_human_action(self, agent_id: str, actions: list[ActionDescriptor]) -> int:
        if self.mode != "human" or not actions:
            return actions[0].index if actions else 0
        self.human_pending, self.human_result, self.human_actions = True, None, actions
        try:
            self.hud.begin_human_choice(agent_id, actions)
        except Exception:  # noqa: BLE001
            log.exception("begin_human_choice failed")
        while self.human_result is None and not self._closed:
            self.run_frame()
        self.human_pending = False
        try:
            self.hud.end_human_choice()
        except Exception:  # noqa: BLE001
            pass
        if self.human_result is None:
            return actions[-1].index
        return self.human_result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.audio.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.mode == "human":
                self._pg.display.quit()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------------------------------
    # frame loop
    # ------------------------------------------------------------------------------------------
    def run_frame(self) -> None:
        """One interactive frame (human mode): input, simulation, render, present."""
        dt = min(self.clock.tick(self.fps) / 1000.0, 0.1)
        self._handle_input()
        if self._closed:
            return
        sim_dt = 0.0 if self.paused else dt * self.speed
        if self._step_request:
            self._step_request = False
            self.director.step_once()
        self.step_sim(sim_dt, real_dt=dt)
        self.render()
        self._pg.display.flip()

    def step_sim(self, dt: float, real_dt: float | None = None) -> None:
        real_dt = dt if real_dt is None else real_dt
        t0 = time.perf_counter()
        self.time += real_dt
        hs = self.fx.hitstop
        if hs > 0:
            self.fx.hitstop = max(0.0, hs - real_dt)
            dt *= 0.05
        self.director.update(dt)
        self.fx.update(dt)
        self._update_visuals(dt, real_dt)
        self.particles.update(dt)
        self._ambient_fx(real_dt)
        self.camera.update(real_dt)
        self.fog.update(real_dt)
        self._update_zones(real_dt)
        try:
            self.hud.update(real_dt, self.view_info())
        except Exception:  # noqa: BLE001
            log.exception("hud.update failed")
        try:
            self.audio.update(real_dt)
        except Exception:  # noqa: BLE001
            pass
        self._rgb_dirty = True
        self.timings["sim"] = time.perf_counter() - t0

    # ------------------------------------------------------------------------------------------
    def _update_visuals(self, dt, real_dt):
        k = 1.0 - math.exp(-real_dt * 9.0)
        for ve in self.vents.values():
            if ve.kind != "creature":
                continue
            target = 1.0 if self._creature_visible(ve) else 0.0
            ve.vis_alpha += (target - ve.vis_alpha) * k
            ve.flash = max(0.0, ve.flash - real_dt * 5.0)
            ve.spawn_scale += (1.0 - ve.spawn_scale) * (1 - math.exp(-real_dt * 10.0))
            ve.ox *= math.exp(-real_dt * 14.0) if not ve.moving else 1.0
            ve.oy *= math.exp(-real_dt * 14.0) if not ve.moving else 1.0
            ve.glow = max(0.0, ve.glow - real_dt * 2.0)
        for d in self.door_vis.values():
            d["ang"] += (d["target"] - d["ang"]) * (1 - math.exp(-real_dt * 9.0))

    def _update_zones(self, dt):
        dead = []
        for zid, z in self.zones.items():
            z.alpha += (z.target - z.alpha) * (1 - math.exp(-dt * 4.0))
            if z.target == 0.0 and z.alpha < 0.01:
                dead.append(zid)
            if z.attached >= 0 and z.attached in self.vents:
                ve = self.vents[z.attached]
                z.offset = (ve.cx - z.base_center[0], ve.cy - z.base_center[1])
            else:
                z.offset = (0.0, 0.0)
        for zid in dead:
            z = self.zones.pop(zid)
            if z.tex is not None:
                z.tex.release()

    def _ambient_fx(self, dt):
        """Fire / ember / dust particles for static light sources and lava."""
        self._ambient_acc += dt
        if self._ambient_acc < 1 / 30:
            return
        dt = self._ambient_acc
        self._ambient_acc = 0.0
        if self.scene is None or self.lights.n == 0:
            return
        x0, y0, x1, y1 = self.camera.visible_rect(1.0)
        pos = self.lights.pos
        inview = (pos[:, 0] > x0) & (pos[:, 0] < x1) & (pos[:, 1] > y0) & (pos[:, 1] < y1)
        rng = self._rng
        P = self.particles
        for i in np.nonzero(inview)[0]:
            kind = self.lights.kind[i]
            if kind not in ("torch", "brazier", "fire", "candle"):
                continue
            n_exp = {"torch": 9, "brazier": 16, "fire": 20, "candle": 3}[kind] * dt
            n = int(n_exp) + (1 if rng.random() < n_exp % 1 else 0)
            if n == 0:
                continue
            px, py = pos[i]
            sc = 0.6 if kind in ("torch", "candle") else 1.0
            ang = rng.random(n) * 6.28
            r = rng.random(n) * 0.12 * sc
            p0 = np.stack([px + np.cos(ang) * r, py + np.sin(ang) * r], 1)
            vel = np.stack([rng.normal(0, 0.12, n), -0.5 - rng.random(n) * 0.6], 1) * sc
            P.emit(n, p0, vel, life=0.5 + rng.random(n) * 0.35, size=(0.11 * sc, 0.02), c0=(1.0, 0.62, 0.2, 0.85), c1=(1.0, 0.25, 0.05, 0.0), shape="glow", add=1.0, drag=0.5)
            if rng.random() < n_exp * 0.3:
                P.emit(1, p0[:1], vel[:1] * 1.5 + np.array([rng.normal(0, 0.4), 0], F4), life=1.4 + rng.random() * 0.8, size=(0.03, 0.01), c0=(1.0, 0.7, 0.3, 1.0), c1=(1.0, 0.3, 0.1, 0.0), shape="dot", add=1.0, drag=0.2, grav=(0.0, -0.15))
            if kind in ("brazier", "fire") and rng.random() < n_exp * 0.12:
                P.emit(1, p0[:1], (0.0, -0.45), life=2.2, size=(0.14, 0.55), c0=(0.35, 0.33, 0.33, 0.22), c1=(0.2, 0.2, 0.22, 0.0), shape="smoke", add=0.0, drag=0.3)
        if len(self.lava_cells):
            c = self.lava_cells
            inl = c[(c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1)]
            if len(inl):
                n = int(min(len(inl) * 0.4 * dt * 1.2, 6) + rng.random())
                if n:
                    sel = inl[rng.integers(0, len(inl), n)] + rng.random((n, 2)).astype(F4)
                    P.emit(n, sel, np.stack([rng.normal(0, 0.08, n), -0.3 - rng.random(n) * 0.4], 1), life=1.0 + rng.random(n), size=(0.025, 0.008), c0=(1.0, 0.5, 0.12, 1.0), c1=(1.0, 0.2, 0.05, 0.0), shape="dot", add=1.0)
        # glints on loot / chests
        if rng.random() < dt * 6:
            cand = [v for v in self.vents.values() if v.kind in ("loot", "chest") and v.state != "open"]
            if cand:
                v = cand[int(rng.integers(len(cand)))]
                if self._object_visible(v):
                    P.emit(1, (v.x + 0.2 + rng.random() * 0.6, v.y + 0.2 + rng.random() * 0.6), (0, 0), life=0.7, size=(0.01, 0.2), c0=(1.0, 0.9, 0.5, 1.0), c1=(1.0, 0.9, 0.5, 0.0), shape="star", add=1.0)

    # ------------------------------------------------------------------------------------------
    # visibility
    # ------------------------------------------------------------------------------------------
    def _pov_faction(self):
        if self.pov.startswith("faction:"):
            try:
                return int(self.pov.split(":")[1])
            except ValueError:
                return -2
        for v in self.frame.entities if self.frame else ():
            if v.agent_id == self.pov:
                return v.faction
        return -2

    def _creature_visible(self, ve) -> bool:
        if self.pov == "omniscient" or not self.fog.enabled:
            return True
        if ve.hidden and ve.faction != self._pov_faction():
            return False
        for dy in range(ve.size):
            for dx in range(ve.size):
                if self.fog.visible_at(int(round(ve.x)) + dx, int(round(ve.y)) + dy):
                    return True
        return False

    def _object_visible(self, ve) -> bool:
        if self.pov == "omniscient" or not self.fog.enabled:
            return True
        return self.fog.explored_at(int(round(ve.x)), int(round(ve.y)))

    # ------------------------------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------------------------------
    def view_info(self) -> ViewInfo:
        cam = self.camera
        ents = {}
        vis = set()
        c = cam.cell_px
        for ve in self.vents.values():
            if ve.kind == "creature" and not ve.gone and ve.vis_alpha > 0.05 and ve.alpha > 0.02:
                sx, sy = cam.world_to_screen(ve.cx, ve.cy)
                ents[ve.eid] = (sx, sy, ve.size * 0.5 * 0.94 * c)
                vis.add(ve.eid)
            elif ve.kind != "creature" and self._object_visible(ve):
                vis.add(ve.eid)
        return ViewInfo(screen_size=(self.fw, self.fh), world_to_screen=cam.world_to_screen, screen_to_world=cam.screen_to_world,
                        cell_px=c, pov=self.pov, pov_options=list(self.pov_options), entity_screen=ents, visible_entities=vis,
                        hovered_eid=self.hover_eid, hovered_cell=self.hover_cell, selected_eid=self.selected_eid, paused=self.paused,
                        speed=self.speed, time=self.time, human_turn=self.human_pending, options=dict(self.options))

    def pick(self, sx, sy):
        wx, wy = self.camera.screen_to_world(sx, sy)
        cell = (int(math.floor(wx)), int(math.floor(wy)))
        if not (0 <= cell[0] < self.W and 0 <= cell[1] < self.H):
            cell = None
        best, bd = -1, 1e9
        for ve in self.vents.values():
            if ve.kind == "creature" and ve.vis_alpha > 0.3 and not ve.gone and ve.alpha > 0.1:
                d = math.hypot(wx - ve.cx, wy - ve.cy)
                if d < ve.size * 0.5 and d < bd:
                    best, bd = ve.eid, d
        if best < 0 and cell is not None:
            for ve in self.vents.values():
                if ve.kind != "creature" and int(round(ve.x)) == cell[0] and int(round(ve.y)) == cell[1] and self._object_visible(ve):
                    best = ve.eid
        return cell, best

    def _mouse(self, pos):
        ww, wh = max(1, self.win_size[0]), max(1, self.win_size[1])
        return pos[0] * self.fw / ww, pos[1] * self.fh / wh

    def _handle_input(self):
        pg = self._pg
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                self._closed = True
                return
            if ev.type in (pg.VIDEORESIZE, getattr(pg, "WINDOWRESIZED", -1), getattr(pg, "WINDOWSIZECHANGED", -1)):
                surf = pg.display.get_surface()
                if surf is not None and (surf.get_width(), surf.get_height()) != (self.fw, self.fh):
                    self.fw, self.fh = surf.get_size()
                    self.win_size = pg.display.get_window_size()
                    self.ui_scale = float(min(max(self.fh / 900.0, 0.6), 2.5))
                    self.camera.screen = (self.fw, self.fh)
                    self.camera.ui_scale = self.ui_scale
                    self._set_pads()
                    self._resize_targets()
                continue
            # mouse events are given to the HUD in framebuffer coordinates
            try:
                cmd = self.hud.handle_event(ev, self.view_info())
            except Exception:  # noqa: BLE001
                log.exception("hud.handle_event failed")
                cmd = None
            if cmd is not None:
                self._apply_command(cmd)
                continue
            if self.has_hud and ev.type == pg.KEYDOWN and ev.key != pg.K_f:
                continue
            if ev.type == pg.MOUSEWHEEL:
                mx, my = self._mouse(pg.mouse.get_pos())
                self.camera.zoom_at(mx, my, 1.12 ** ev.y)
            elif ev.type == pg.MOUSEBUTTONDOWN:
                mx, my = self._mouse(ev.pos)
                if ev.button in (2, 3):
                    self._drag = (ev.button, mx, my, False)
                elif ev.button == 1:
                    cell, eid = self.pick(mx, my)
                    cmd = None
                    try:
                        cmd = self.hud.world_click(cell, eid, 1, self.view_info()) if cell is not None else None
                    except Exception:  # noqa: BLE001
                        log.exception("hud.world_click failed")
                    if cmd is None and self.human_pending:
                        cmd = self._fallback_click(cell, eid)
                    if cmd is not None:
                        self._apply_command(cmd)
                    elif eid >= 0:
                        self.selected_eid = eid
            elif ev.type == pg.MOUSEBUTTONUP:
                if ev.button in (2, 3) and self._drag:
                    if ev.button == 3 and not self._drag[3] and self.human_pending:
                        pass
                    self._drag = None
            elif ev.type == pg.MOUSEMOTION:
                mx, my = self._mouse(ev.pos)
                self.hover_cell, self.hover_eid = self.pick(mx, my)
                if self._drag:
                    b, ox, oy, moved = self._drag
                    dx, dy = mx - ox, my - oy
                    if moved or abs(dx) + abs(dy) > 3:
                        self.camera.pan_pixels(dx, dy)
                        self._drag = (b, mx, my, True)
            elif ev.type == pg.KEYDOWN:
                self._key(ev)
        keys = pg.key.get_pressed()
        dx = (keys[pg.K_d] or keys[pg.K_RIGHT]) - (keys[pg.K_a] or keys[pg.K_LEFT])
        dy = (keys[pg.K_s] or keys[pg.K_DOWN]) - (keys[pg.K_w] or keys[pg.K_UP])
        if dx or dy:
            sp = 14.0 / max(self.camera.zoom, 0.4) * (1 / 60)
            self.camera.pan_cells(dx * sp, dy * sp)
            self.camera.cx, self.camera.cy = self.camera.tx, self.camera.ty

    def _fallback_click(self, cell, eid):
        for a in self.human_actions:
            if a.kind in ("end_turn", "say", "decision"):
                continue
            if (a.target_eid >= 0 and a.target_eid == eid) or (cell is not None and a.target_cell == cell):
                return UICommand("choose_action", a.index)
        return None

    def _key(self, ev):
        pg = self._pg
        k = ev.key
        if k == pg.K_f:
            self.camera.follow = not self.camera.follow
            if self.camera.follow and self.frame and self.frame.active_id in self.vents:
                v = self.vents[self.frame.active_id]
                self.camera.focus(v.cx, v.cy)
        elif k == pg.K_SPACE:
            self.paused = not self.paused
        elif k in (pg.K_EQUALS, pg.K_PLUS, pg.K_KP_PLUS):
            self.speed = min(16.0, self.speed * 2)
        elif k in (pg.K_MINUS, pg.K_KP_MINUS):
            self.speed = max(0.25, self.speed / 2)
        elif k == pg.K_n:
            self._step_request = True
        elif k == pg.K_TAB:
            self.cycle_pov()
        elif k == pg.K_h:
            self.hud_visible = not getattr(self, "hud_visible", True)
        elif k == pg.K_g:
            self.options["grid"] = not self.options.get("grid", False)
        elif k == pg.K_F12:
            self.screenshot()
        elif k == pg.K_ESCAPE and self.mode == "human" and False:
            self._closed = True

    def cycle_pov(self, pov: str | None = None):
        opts = self.pov_options or ["omniscient"]
        if pov is None:
            i = opts.index(self.pov) if self.pov in opts else -1
            pov = opts[(i + 1) % len(opts)]
        self.pov = pov
        self._update_fog_target(True)

    def screenshot(self, path: str | None = None) -> str:
        from PIL import Image
        img = self.capture()
        path = path or os.path.join(os.getcwd(), time.strftime("emberdeep_%Y%m%d_%H%M%S.png"))
        Image.fromarray(img).save(path)
        return path

    def _apply_command(self, cmd: UICommand):
        k, v = cmd.kind, cmd.value
        if k == "noop":
            return
        if k == "toggle_pause":
            self.paused = not self.paused
        elif k == "pause":
            self.paused = True
        elif k == "resume":
            self.paused = False
        elif k == "step":
            self._step_request = True
        elif k == "speed":
            self.speed = float(min(16.0, max(0.25, v if v else 1.0)))
        elif k == "pov":
            self.cycle_pov(v if isinstance(v, str) else None)
        elif k == "select":
            self.selected_eid = int(v) if v is not None else -1
        elif k == "focus":
            if v in self.vents:
                ve = self.vents[v]
                self.camera.focus(ve.cx, ve.cy)
            elif isinstance(v, (tuple, list)):
                self.camera.focus(v[0], v[1])
        elif k == "choose_action":
            self.human_result = int(v)
        elif k == "toggle":
            if v is not None:
                if v == "mute":
                    try:
                        self.audio.muted = not getattr(self.audio, "muted", False)
                    except Exception:  # noqa: BLE001
                        pass
                self.options[v] = not self.options.get(v, False)
        elif k == "skip_anim":
            self.director.skip()
        elif k == "quit":
            self._closed = True
        elif k == "zoom":
            self.camera.tzoom = min(max(self.camera.tzoom * float(v or 1.0), self.camera.ZMIN), self.camera.ZMAX)
        elif k == "screenshot":
            self.screenshot()

    # ------------------------------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------------------------------
    def _cam_uniforms(self, prog):
        from emberdeep.render.gl.util import set_uniform
        cx, cy = self.camera.view_center()
        set_uniform(prog, "u_center", (cx, cy))
        set_uniform(prog, "u_scale", self.camera.cell_px * self.quality["scale"])
        set_uniform(prog, "u_screen", (float(self.rw), float(self.rh)))
        set_uniform(prog, "u_time", float(self.time))

    def render(self) -> None:
        from emberdeep.render.gl.util import set_uniform
        if self.scene is None:
            return
        ctx, mgl = self.ctx, self._mgl
        tm = {}
        t0 = time.perf_counter()
        W, H = self.W, self.H
        mapsize = (float(W), float(H))
        rw, rh = self.rw, self.rh
        # --- G buffer -------------------------------------------------------------------------------
        self.g_fbo.use()
        ctx.viewport = (0, 0, rw, rh)
        ctx.disable(mgl.BLEND)
        self.g_fbo.clear(0.0, 0.0, 0.0, 0.0)
        p = self.prog_terrain
        self._cam_uniforms(p)
        self.tex_alb.use(0); self.tex_nrm.use(1); self.cellmap_tex.use(2); self.sdf_tex.use(3); self.info_tex.use(4)
        p["u_alb"].value = 0; p["u_nrm"].value = 1; p["u_cellmap"].value = 2; p["u_sdf"].value = 3; p["u_info"].value = 4
        set_uniform(p, "u_mapsize", mapsize)
        p["u_mat"].write(self.mat_table.tobytes())
        set_uniform(p, "u_grade_tint", (1.0, 1.0, 1.0))
        self.vao_terrain.render(mgl.TRIANGLE_STRIP, instances=self.n_cells)
        # sprites (static decor + dynamic)
        ctx.enable(mgl.BLEND)
        ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA, mgl.ZERO, mgl.ONE
        sp = self.prog_sprite
        self._cam_uniforms(sp)
        self.pa_tex.use(5); self.pn_tex.use(6)
        sp["u_pa"].value = 5; sp["u_pn"].value = 6
        if self.n_static:
            self.vao_static.render(mgl.TRIANGLE_STRIP, instances=self.n_static)
        dyn = self._build_dyn_sprites()
        if len(dyn):
            self.sprite_dyn_vbo.write(dyn.tobytes())
            self.vao_sprite_dyn.render(mgl.TRIANGLE_STRIP, instances=len(dyn))
        tm["gbuf"] = time.perf_counter() - t0
        # --- lighting -------------------------------------------------------------------------------
        t1 = time.perf_counter()
        self.post.scene_fbo.use()
        ctx.viewport = (0, 0, rw, rh)
        ctx.disable(mgl.BLEND)
        self._update_dark()
        amb = self._ambient()
        a = self.prog_ambient
        self._cam_uniforms(a)
        self.g_alb.use(0); self.g_emi.use(1); self.dark_tex.use(2); self.g_nrm.use(3)
        a["u_galb"].value = 0; a["u_gemi"].value = 1; a["u_dark"].value = 2; a["u_gnrm"].value = 3
        set_uniform(a, "u_mapsize", mapsize); set_uniform(a, "u_ambient", amb)
        self.vao_ambient.render(mgl.TRIANGLE_STRIP)
        rows = self._gather_lights()
        self._light_rows = rows
        if len(rows):
            ctx.enable(mgl.BLEND)
            ctx.blend_func = mgl.ONE, mgl.ONE
            lp = self.prog_light
            self._cam_uniforms(lp)
            self.g_alb.use(0); self.g_nrm.use(1); self.sdf_tex.use(3); self.dark_tex.use(4)
            lp["u_galb"].value = 0; lp["u_gnrm"].value = 1; lp["u_sdf"].value = 3; lp["u_dark"].value = 4
            set_uniform(lp, "u_mapsize", mapsize)
            data = np.empty((len(rows), 10), F4)
            data[:, 0:2] = rows[:, 0:2]
            data[:, 2:6] = rows[:, 2:6] * np.array([1, 1, 1, 0], F4)
            data[:, 6:10] = rows[:, 6:10]
            self.light_vbo.write(data.tobytes())
            self.vao_light.render(mgl.TRIANGLE_STRIP, instances=len(rows))
        tm["light"] = time.perf_counter() - t1
        # --- fog --------------------------------------------------------------------------------------
        t2 = time.perf_counter()
        if self.fog.enabled:
            ctx.enable(mgl.BLEND)
            ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
            fp = self.prog_fog
            self._cam_uniforms(fp)
            self.g_alb.use(0); self.g_emi.use(1); self.fog.tex.use(2)
            fp["u_galb"].value = 0; fp["u_gemi"].value = 1; fp["u_fog"].value = 2
            set_uniform(fp, "u_mapsize", mapsize)
            set_uniform(fp, "u_mist", (0.010, 0.012, 0.020))
            self.vao_fog.render(mgl.TRIANGLE_STRIP)
        # --- zones / entities / particles ----------------------------------------------------------
        ctx.enable(mgl.BLEND)
        ctx.blend_func = mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA
        self._draw_zones(ground=True)
        self._draw_entities()
        self._draw_zones(ground=False)
        pp = self.particles.prog
        self._cam_uniforms(pp)
        self.fog.tex.use(2)
        pp["u_fog"].value = 2
        set_uniform(pp, "u_mapsize", mapsize)
        set_uniform(pp, "u_usefog", 1.0 if self.fog.enabled else 0.0)
        ctx.blend_func = mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA
        self.particles.draw({})
        ctx.disable(mgl.BLEND)
        tm["world"] = time.perf_counter() - t2
        # --- post -------------------------------------------------------------------------------------
        t3 = time.perf_counter()
        ex, lift, gain, sat, con, vig, bloom, grain = self.grade
        self.post.bloom(0.85)
        self.post.composite(dict(u_bloom_amt=bloom, u_exposure=ex, u_lift=lift, u_gain=gain, u_sat=sat, u_contrast=con,
                                 u_vignette=vig, u_grain=grain, u_chroma=0.0025 * self.fx.chroma + 0.0004, u_flash=self.fx.flash,
                                 u_flash_col=self.fx.flash_col, u_time=float(self.time)))
        tm["post"] = time.perf_counter() - t3
        # --- HUD ---------------------------------------------------------------------------------------
        t4 = time.perf_counter()
        if not isinstance(self.hud, NullHUD) and getattr(self, "hud_visible", True):
            self.post.draw_ui(self._draw_hud())
        if self.screen_fbo is not None:
            self.post.present(self.screen_fbo, (self.fw, self.fh))
        tm["hud"] = time.perf_counter() - t4
        tm["total"] = time.perf_counter() - t0
        self.timings.update(tm)
        self._rgb_dirty = False
        self._frames_rendered += 1

    def _draw_hud(self):
        pg = self._pg
        if self._ui_surface is None or self._ui_surface.get_size() != (self.fw, self.fh):
            self._ui_surface = pg.Surface((self.fw, self.fh), pg.SRCALPHA, 32)
        self._ui_surface.fill((0, 0, 0, 0))
        try:
            self.hud.draw(self._ui_surface, self.view_info())
        except Exception:  # noqa: BLE001
            log.exception("hud.draw failed")
            return None
        return self._ui_surface.get_buffer().raw

    THEME_MIN_AMBIENT = {"dungeon": (0.13, 0.13, 0.17), "crypt": (0.10, 0.125, 0.15), "cave": (0.13, 0.15, 0.17), "lair": (0.20, 0.15, 0.14),
                         "sewer": (0.10, 0.12, 0.11), "temple": (0.13, 0.12, 0.13), "tavern": (0.14, 0.12, 0.10)}

    def _ambient(self):
        a = np.array(self.scene.ambient, F4)
        mn = np.array(self.THEME_MIN_AMBIENT.get(self.scene.theme, (0.1, 0.1, 0.13)), F4)
        return np.maximum(a, mn) if a.max() < 0.3 else a * {"arena": 0.5, "forest": 0.95, "castle": 0.7}.get(self.scene.theme, 0.7)

    def _update_dark(self):
        cells = []
        for z in self.zones.values():
            if ZONE_KINDS.get(z.kind, 15) == 3 and z.alpha > 0.05:
                o = getattr(z, "offset", (0, 0))
                cells.append((z.id, len(z.cells), round(z.alpha, 1), round(o[0]), round(o[1])))
        sig = tuple(cells)
        if sig == self._dark_sig:
            return
        self._dark_sig = sig
        m = np.zeros((self.H, self.W), F4)
        for z in self.zones.values():
            if ZONE_KINDS.get(z.kind, 15) == 3 and len(z.cells):
                o = getattr(z, "offset", (0, 0))
                xs = np.clip(z.cells[:, 0] + int(round(o[0])), 0, self.W - 1)
                ys = np.clip(z.cells[:, 1] + int(round(o[1])), 0, self.H - 1)
                m[ys, xs] = max(m.max(), z.alpha)
        up = np.repeat(np.repeat(m, 2, 0), 2, 1)
        up = _box_blur_np(_box_blur_np(up, 1), 1)
        self.dark_tex.write(np.clip(up * 255, 0, 255).astype(np.uint8).tobytes())

    def _gather_lights(self):
        t = self.time
        rows = [self.lights.frame(t)]
        # carried lights
        att = []
        for ve in self.vents.values():
            if ve.light and ve.kind == "creature" and not ve.gone and ve.alpha > 0.1 and ve.vis_alpha > 0.1 and not ve.dead:
                b, d = ve.light
                r = np.zeros(10, F4)
                r[0] = ve.cx + 0.2 * math.cos(t * 3 + ve.eid) * 0.1; r[1] = ve.cy - 0.1
                r[2] = b * 0.9; r[3] = (b + d) * 0.9; r[4] = 1.4
                r[6:9] = np.array((1.0, 0.78, 0.5), F4) * 2.0 * (1.0 + 0.1 * math.sin(t * 9 + ve.eid * 3))
                r[9] = 7.0
                att.append(r)
        # faint personal glow for player-side creatures in dark scenes keeps darkness readable
        if float(np.max(self.scene.ambient)) < 0.3:
            pf = {f.index for f in self.scene.factions if f.is_player_side}
            for ve in self.vents.values():
                if ve.kind == "creature" and ve.faction in pf and not ve.light and not ve.dead and not ve.gone and ve.vis_alpha > 0.1:
                    r = np.zeros(10, F4)
                    r[0], r[1] = ve.cx, ve.cy - 0.1
                    r[2], r[3], r[4] = 1.6, 4.2, 1.6
                    r[6:9] = np.array((1.0, 0.86, 0.68), F4) * 0.9
                    r[9] = 7.0
                    att.append(r)
        if att:
            rows.append(np.array(att, F4))
        dyn = self.fx.light_array()
        if len(dyn):
            rows.append(dyn)
        # fire / lava zones
        zl = []
        for z in self.zones.values():
            k = ZONE_KINDS.get(z.kind, 15)
            if k in (11, 12, 13, 5, 4) and z.alpha > 0.05:
                o = getattr(z, "offset", (0, 0))
                col = {11: (1.0, 0.45, 0.12), 12: (1.0, 0.6, 0.2), 13: (1.0, 0.95, 0.75), 5: (0.6, 0.75, 1.0), 4: (1.0, 0.88, 0.5)}[k]
                r = np.zeros(10, F4)
                r[0], r[1] = z.center[0] + o[0], z.center[1] + o[1]
                rad = max(z.radius, 1.5)
                r[2], r[3], r[4] = rad * 0.5, rad * 1.4, 1.5
                r[6:9] = np.array(col, F4) * (0.45 if k in (4, 5) else (0.6 if k == 11 else 0.9)) * z.alpha * (0.9 + 0.1 * math.sin(t * 7))
                r[9] = 5.0
                zl.append(r)
        if zl:
            rows.append(np.array(zl, F4))
        allr = np.concatenate(rows) if rows else np.zeros((0, 10), F4)
        if not len(allr):
            return allr
        x0, y0, x1, y1 = self.camera.visible_rect(0.0)
        rad = allr[:, 3]
        keep = (allr[:, 0] + rad > x0) & (allr[:, 0] - rad < x1) & (allr[:, 1] + rad > y0) & (allr[:, 1] - rad < y1) & (allr[:, 6:9].max(1) > 0.003)
        allr = allr[keep]
        mx = self.quality["max_lights"]
        if len(allr) > mx:
            cx, cy = self.camera.cx, self.camera.cy
            score = allr[:, 6:9].max(1) * allr[:, 3] / (1.0 + np.hypot(allr[:, 0] - cx, allr[:, 1] - cy) * 0.15)
            allr = allr[np.argsort(-score)[:mx]]
        return allr

    def _light_at(self, pts: np.ndarray) -> np.ndarray:
        """CPU approximation of incoming light colour at points [N,2] (no shadows)."""
        rows = self._light_rows
        amb = np.array(self.scene.ambient, F4)
        out = np.tile(amb * 2.0, (len(pts), 1))
        if len(rows) and len(pts):
            d = np.hypot(pts[:, None, 0] - rows[None, :, 0], pts[:, None, 1] - rows[None, :, 1])
            b = rows[None, :, 2]; tot = np.maximum(rows[None, :, 3], 1e-3)
            t1 = np.clip((d - b * 0.35) / np.maximum(b * 0.65, 1e-3), 0, 1); t2 = np.clip((d - b * 0.8) / np.maximum(tot - b * 0.8, 1e-3), 0, 1)
            att = (1 - 0.6 * t1) * (1 - t2)
            out += np.einsum("nl,lc->nc", att * att, rows[:, 6:9])
        return out

    def _build_dyn_sprites(self) -> np.ndarray:
        rows = []
        at = self.atlas
        # decals
        for d in self.fx.decals:
            u0, v0, u1, v1, w, h = at.lookup(d["kind"], d["variant"])
            rows.append([d["x"], d["y"], w * d["scale"], h * d["scale"], d["rot"], 0, 0, 0, u0, v0, u1, v1, 1, 1, 1, d["alpha"]])
        # doors
        for (x, y), d in self.door_vis.items():
            if not self._door_seen(x, y):
                continue
            cx, cy = x + 0.5, y + 0.5
            base = 0.0 if d["orient"] == "h" else math.pi / 2
            u0, v0, u1, v1, w, h = at.lookup("door_frame", 0)
            rows.append([cx, cy, w, h, base, 0, 0, 0, u0, v0, u1, v1, 1, 1, 1, 1])
            if d["state"] == "broken":
                u0, v0, u1, v1, w, h = at.lookup("door_broken", 0)
                rows.append([cx, cy, w, h, base, 0, 0, 0, u0, v0, u1, v1, 1, 1, 1, 1])
                continue
            u0, v0, u1, v1, w, h = at.lookup("door_leaf", 1 if d["state"] == "locked" else 0)
            ang = -d["ang"]
            c, s = math.cos(base), math.sin(base)
            # hinge at local (-0.5, 0); leaf centre rotates about it
            lx, ly = -0.5 + 0.5 * math.cos(ang), 0.5 * math.sin(ang)
            wx, wy = cx + lx * c - ly * s, cy + lx * s + ly * c
            rows.append([wx, wy, w, h, base + ang, 0, 0, 0, u0, v0, u1, v1, 1, 1, 1, 1])
        # objects
        for ve in self.vents.values():
            if ve.kind in ("creature", "door") or ve.gone or not self._object_visible(ve):
                continue
            kind = {"chest": "chest_open" if ve.state == "open" else "chest", "loot": "loot", "lever": "lever_on" if ve.state == "on" else "lever_off",
                    "trap": "trap", "exit": "exit", "altar": "altar"}.get(ve.kind)
            if kind is None or (ve.kind == "trap" and ve.hidden):
                continue
            u0, v0, u1, v1, w, h = at.lookup(kind, ve.eid)
            rows.append([ve.x + ve.size * 0.5, ve.y + ve.size * 0.5, w * ve.size, h * ve.size, 0.0, 0, 0, 0, u0, v0, u1, v1, 1, 1, 1, 1])
        if not rows:
            return np.zeros((0, 16), F4)
        return np.array(rows[:4096], F4)

    def _door_seen(self, x, y):
        if self.pov == "omniscient" or not self.fog.enabled:
            return True
        return self.fog.explored_at(x, y)

    def _draw_zones(self, ground: bool):
        if not self.zones:
            return
        from emberdeep.render.gl.util import set_uniform
        ctx, mgl = self.ctx, self._mgl
        p = self.prog_zone
        self._cam_uniforms(p)
        for z in sorted(self.zones.values(), key=lambda z: z.id):
            k = ZONE_KINDS.get(z.kind, 15)
            if (k in GROUND_ZONES) != ground or z.alpha < 0.01 or not len(z.cells):
                continue
            if z.dirty or z.tex is None:
                self._build_zone_mask(z)
            o = getattr(z, "offset", (0.0, 0.0))
            set_uniform(p, "u_box", z.box)
            set_uniform(p, "u_off", (float(o[0]), float(o[1])))
            set_uniform(p, "u_kind", k)
            set_uniform(p, "u_alpha", float(z.alpha))
            set_uniform(p, "u_zc", (z.center[0], z.center[1]))
            set_uniform(p, "u_zr", float(z.radius if z.radius > 0 else max(1.0, math.sqrt(len(z.cells) / math.pi))))
            z.tex.use(7)
            p["u_mask"].value = 7
            if self.fog.enabled:
                pass
            self.vao_zone.render(mgl.TRIANGLE_STRIP)

    def _build_zone_mask(self, z):
        mgl = self._mgl
        pad = 1
        x0, y0 = z.cells.min(0) - pad
        x1, y1 = z.cells.max(0) + 1 + pad
        s = 8
        m = np.zeros((y1 - y0, x1 - x0), F4)
        m[z.cells[:, 1] - y0, z.cells[:, 0] - x0] = 1.0
        up = np.repeat(np.repeat(m, s, 0), s, 1)
        up = _box_blur_np(_box_blur_np(up, 4), 4)
        up = np.clip(up * 1.6 - 0.05, 0, 1)
        if z.tex is not None:
            z.tex.release()
        z.tex = self.ctx.texture((up.shape[1], up.shape[0]), 1, (up * 255).astype(np.uint8).tobytes())
        z.tex.filter = (mgl.LINEAR, mgl.LINEAR)
        z.tex.repeat_x = z.tex.repeat_y = False
        z.box = (float(x0), float(y0), float(x1), float(y1))
        z.dirty = False

    def _draw_entities(self):
        rows = self._build_entities()
        if not len(rows):
            return
        mgl = self._mgl
        self.tokens.finalize()
        p = self.prog_ent
        self._cam_uniforms(p)
        self.tokens.tex.use(8)
        p["u_tok"].value = 8
        n = min(len(rows), 1024)
        self.ent_vbo.write(np.ascontiguousarray(rows[:n]).tobytes())
        self.ctx.blend_func = mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA
        self.vao_ent.render(mgl.TRIANGLE_STRIP, instances=n)

    def _build_entities(self) -> np.ndarray:
        if self.frame is None:
            return np.zeros((0, 20), F4)
        t = self.time
        fac = {f.index: f for f in self.scene.factions}
        cre = [v for v in self.vents.values() if v.kind == "creature" and not v.gone and v.vis_alpha > 0.02 and v.alpha > 0.01]
        if not cre:
            return np.zeros((0, 20), F4)
        cre.sort(key=lambda v: (v.z > 0, v.y + v.size))
        pts = np.array([[v.cx, v.cy] for v in cre], F4)
        light = self._light_at(pts)
        shadows, bodies, overlays = [], [], []
        active = self.active_vis if self.active_vis is not None else self.frame.active_id
        for i, ve in enumerate(cre):
            f = fac.get(ve.faction)
            fcol = f.color if f else (170, 170, 170)
            layer = self.tokens.get(ve.icon, ve.tint, fcol, bool(f and f.is_player_side))
            z = ve.z
            bob = (0.05 * math.sin(t * 2.0 + ve.bob_phase) + 0.02 * z) if z > 0 else 0.0
            rad = ve.size * 0.5 * 0.94 * ve.scale * ve.spawn_scale * (1.0 + 0.07 * z)
            cx, cy = ve.cx, ve.cy - bob - 0.10 * z
            lit = np.clip(light[i] * 1.2, 0.0, 1.6) * 0.62 + 0.38
            lit = lit * (1 - 0.8 * self._dark_at(cx, cy))
            a = ve.alpha * ve.vis_alpha * (0.35 if ve.invisible else 1.0)
            if ve.dead:
                a = 0.0
            desat = 0.0
            tint = np.zeros(4, F4)
            if ve.downed:
                desat = 0.7
                lit = lit * 0.8
            conds = ve.conditions
            if "petrified" in conds:
                desat, tint = 1.0, np.array([0.58, 0.58, 0.56, 0.45], F4)
            elif "poisoned" in conds:
                tint = np.array([0.35, 0.85, 0.25, 0.18], F4)
            elif "frightened" in conds:
                tint = np.array([0.55, 0.35, 0.9, 0.16], F4)
            shimmer = 1.0 if ve.hidden else 0.0
            sy = 1.0 if not ve.downed else 0.9
            sh_off = 0.07 * rad + 0.10 * z
            shadows.append([cx + sh_off * 0.6, ve.cy + sh_off + 0.1 * z, rad * (1.0 - 0.05 * z), 0.0, 0.0, 0.0, 0.0, 0.55 * ve.alpha * ve.vis_alpha / (1 + 0.5 * z), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            bodies.append([cx, cy, rad, ve.tilt, lit[0], lit[1], lit[2], a, desat, ve.flash, shimmer, sy, tint[0], tint[1], tint[2], tint[3], 0, layer, 0, 0])
            ov = []
            if ve.eid == active:
                ov.append((0, (1.0, 0.78, 0.3), 1.0, 0))
            if ve.eid == self.selected_eid:
                ov.append((2, (0.5, 0.9, 1.0), 1.0, 0))
            elif ve.eid == self.hover_eid:
                ov.append((1, (1.0, 1.0, 1.0), 0.8, 0))
            for c in conds:
                if c in COND_OVERLAYS:
                    ty, col = COND_OVERLAYS[c]
                    ov.append((ty, col, 0.8, 0))
            seen_ty = set()
            for e in ve.effects:
                if e in EFFECT_OVERLAYS:
                    ty, col = EFFECT_OVERLAYS[e]
                    if ty not in seen_ty:
                        seen_ty.add(ty)
                        ov.append((ty, col, 0.7, 0))
            if "raging" in ve.statuses and 8 not in seen_ty:
                ov.append((8, (1.0, 0.25, 0.08), 0.7, 0))
            if "concentrating" in ve.statuses or getattr(ve, "conc", False):
                ov.append((14, (0.6, 0.9, 1.0), 0.6, 0))
            if "dying" in ve.statuses and not ve.dead:
                ov.append((17, (1.0, 0.15, 0.1), 0.8, 0))
            if ve.glow > 0.02:
                ov.append((13, ve.glow_col, min(1.0, ve.glow), 0))
            if not ve.downed and not ve.dead and a > 0.5 and ve.facing is not None:
                ov.append((16, (1.0, 0.95, 0.8), 0.7, ve.facing))
            for ty, col, st, aux in ov:
                overlays.append([cx, cy, rad, 0.0, col[0], col[1], col[2], st * min(1.0, a * 1.5), 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, ty, aux])
        return np.array(shadows + bodies + overlays, F4)

    def _dark_at(self, x, y):
        sig = self._dark_sig
        if not sig:
            return 0.0
        for z in self.zones.values():
            if ZONE_KINDS.get(z.kind, 15) == 3 and len(z.cells):
                o = getattr(z, "offset", (0, 0))
                if ((z.cells[:, 0] + round(o[0]) == int(x)) & (z.cells[:, 1] + round(o[1]) == int(y))).any():
                    return z.alpha
        return 0.0
