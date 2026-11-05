# Emberdeep world renderer (`emberdeep/render`)

OpenGL 3.3 core (moderngl) + pygame-ce. `Renderer(mode='human'|'rgb_array', width, height, quality, fps, hud, audio, pov)`
implements `RendererProtocol`. `import emberdeep.render` never imports pygame/moderngl (`Renderer` is lazy).

## Pipeline

```
StaticScene --compile once--> cell instances + cellmap/SDF/info textures, prop atlas, light set
FrameState + events --> Director (groups/tracks) --> VEnt visual state, particles, dynamic lights, decals, zones
per frame:
  G-buffer (MRT: albedo+rough | normal+wallflag+ao | emissive)   terrain instanced quads -> sprites (decor, doors, objects, decals)
  ambient + emissive  -> HDR RGBA16F
  lights (1 instanced draw): normal-mapped N.L, SDF sphere-traced soft shadows, bright/dim falloff, flicker, darkness mask
  fog of war composite (visible / explored memory / mist)       [skipped for 'omniscient']
  ground zones -> tokens (shadow, body, overlays) -> air zones -> particles
  bloom (dual filter) -> ACES + theme grade + vignette + grain + chromatic pulse -> HUD (pygame RGBA) -> screen / capture
```

Files: `renderer.py` (loop, input, passes), `director.py` (events -> timed groups), `vfx/recipes.py` (every `VFX_KEYS` key),
`vfx/effects.py` (VEnt, FXContext, primitives), `particles.py`, `tokens.py`, `props.py`, `textures/`, `lighting.py`, `fog.py`,
`camera.py`, `postfx.py`, `shaders/*.glsl|vert|frag`, `mock.py`.

## Controls

Wheel zoom to cursor, right/middle drag or WASD/arrows pan, `F` follow toggle. With the HUD present it owns Space, N, Tab, V/G/H,
F12 etc. (see the HUD help, `F1`); without a HUD the renderer handles Space (pause), +/- (speed), N (step), Tab (POV), H, G, F12.

## Quality presets

| preset | texture px | SDF texels/cell | shadow steps | max lights | particles | bloom levels | render scale |
|---|---|---|---|---|---|---|---|
| low | 128 | 3 | 12 | 20 | 5k | 3 | 0.75 |
| medium | 128 | 4 | 20 | 40 | 12k | 4 | 1.0 |
| high | 192 | 4 | 28 | 64 | 24k | 5 | 1.0 |

## Adding things

* **VFX recipe**: in `vfx/recipes.py` write `@recipe("my_key") def f(g, c): ... return impact_time`. `g.at(t, fn)` schedules a
  callback, `g.track(delay, dur, fn(u, dt), end)` a timed animation; `c` has `src`, `targets`, `point`, `dir`, `radius`, `cells`,
  `hit`, `crit`, `t0`. Use primitives (`burst`, `ring`, `glow`, `projectile`, `beam`, `cone`, `aura`). Add the key to `VFX_KEYS`
  (contract) via a contract request if it is new.
* **Material**: add a generator in `textures/materials.py` (returns albedo, height, roughness, emissive), register it in
  `GENERATORS`, add a row to `MAT_TABLE` in `renderer.py` (structured?, ragged?, kind, tile size), bump `TEX_VERSION`.
* **Decor**: write a painter function in `props.py` (`_name(P, variant)`), register in `SPRITES`, bump `PROPS_VERSION`.
  Shapes carry height, so normals/shadows are derived automatically; `em` paints emissive.
* **Zone overlay**: add a branch to `shaders/zone.frag` and a name in `ZONE_KINDS` (`renderer.py`).

## Notes / conventions

Decor/light coordinates that are integers are treated as cell indices (+0.5). Door orientation is inferred from adjacent walls.
Generated textures and sprites are cached in `render.assets.cache_dir()` (`textures_v*_*.npz`, `props_v*.npz`).
