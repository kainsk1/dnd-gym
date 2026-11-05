# Emberdeep HUD and audio (render/ui, render/audio.py)

`from emberdeep.render.ui import HUD` implements `HUDProtocol`. The HUD draws onto a cleared SRCALPHA surface; all art is procedural (gilded slate panels, filigree corners, medallions) and cached. UI scale is screen height / 900 (clamped 0.7-3.2).

## Layout
- Left: party frames (collapsible; auto-compact when short on height).
- Top centre: initiative ribbon (round counter, medallions, HP slivers) or the compact Exploring plate.
- Top right: minimap (fog for the current POV, camera rectangle, click/drag to focus).
- Right: research overlay (optional), chronicle (combat log, All/Combat/Social filters, wheel scroll), playback controls.
- Docked left of the right column: inspector for the selected creature.
- Bottom centre (human turns): action bar, target mode prompt, reaction modal.
- Overlays: banners, floating numbers, nameplates, speech bubbles, tooltips, scoreboard, F1 help.

## Controls
Space pause, N or . step, [ ] speed, Tab skip animations, V cycle POV, G grid, H threat, L LOS, B nameplates, C log, M minimap, F2 research, F4 mute, F1 help, F12 screenshot, Home focus active. Human turn: 1-0 and Q-P abilities, Enter end turn, numpad move, click token/tile, Esc or right-click cancels targeting.

## Integration
`hud.set_scene`, `on_frame` (authoritative state), `on_event(event, frame, view)` (when the animation lands), `update(dt, view)`, `draw(surface, view)`, then `handle_event` and, if it returns None, `world_click`. Set `hud.audio = AudioEngine()` to get click sounds and mute toggling. See `contract_requests/render-ui.md`.

## Audio
`AudioEngine(enabled=True, volume=0.7)` synthesises about 30 SFX plus ambient and combat loops with numpy, caches them as `sfx_v3.npz`, and is a silent no-op without a mixer. Call `on_event(event, pan)` and `update(dt)`.

## Demo
`python -m emberdeep.render.ui.demo [--state ...]`; headless: `--screenshot docs/screenshots/hud_<state>.png --state combat|human|human_aoe|inspector|scoreboard|tooltip|exploration|banner|research|help|reaction`.

## Theming
Palette, damage colours, fonts and panel art live in `render/ui/theme.py`.
