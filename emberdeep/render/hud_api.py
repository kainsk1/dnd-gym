"""Interface between the GL world renderer (render/renderer.py) and the HUD layer
(render/ui/*). The HUD draws with pygame onto a transparent RGBA surface that
the renderer composites on top of the post-processed world image.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from emberdeep.render.api import ActionDescriptor, FrameState, StaticScene


@dataclass
class ViewInfo:
    """Camera + interaction state handed to the HUD every frame."""
    screen_size: tuple[int, int]
    world_to_screen: Callable[[float, float], tuple[float, float]]   # squares -> pixels (top-left of cell (x,y) at (x,y))
    screen_to_world: Callable[[float, float], tuple[float, float]]
    cell_px: float                                                   # current zoomed pixel size of one square
    pov: str = "omniscient"                                          # visibility key in FrameState.visible
    pov_options: list[str] = field(default_factory=list)
    entity_screen: dict[int, tuple[float, float, float]] = field(default_factory=dict)  # eid -> (cx, cy, radius_px) animated
    visible_entities: set[int] = field(default_factory=set)          # eids drawn this frame (fog-filtered)
    hovered_eid: int = -1
    hovered_cell: tuple[int, int] | None = None
    selected_eid: int = -1
    paused: bool = False
    speed: float = 1.0
    time: float = 0.0                                                # seconds since start (for UI animation)
    human_turn: bool = False                                         # awaiting a human action
    options: dict[str, bool] = field(default_factory=dict)           # toggles: grid, threat, los, labels, log, minimap...


@dataclass
class UICommand:
    """Returned by HUD.handle_event to the renderer."""
    kind: str        # "pause" | "resume" | "toggle_pause" | "step" | "speed" | "pov" | "select" | "focus" |
                     # "choose_action" | "toggle" | "skip_anim" | "quit" | "zoom" | "screenshot" |
                     # "noop" (input consumed by the HUD, nothing to do — renderer must not treat it as unhandled)
                     # "focus" value: eid (int) or (x, y) map-square floats (minimap click)
    value: Any = None


class HUDProtocol:
    """Implemented by emberdeep.render.ui.hud.HUD."""

    def set_scene(self, scene: StaticScene) -> None: ...
    def on_frame(self, frame: FrameState) -> None: ...          # new authoritative state (after animations of a step finish)
    def on_event(self, event: dict, frame: FrameState, view: ViewInfo) -> None: ...  # at the moment its animation lands
    def update(self, dt: float, view: ViewInfo) -> None: ...
    # Optional: ``dirty`` (bool attribute, valid after ``update``). When False the last drawn surface is still
    # exact, so the renderer may skip clear + draw + texture upload and reuse the previous HUD texture.
    dirty: bool = True
    def draw(self, surface, view: ViewInfo) -> None: ...        # surface: pygame.Surface (SRCALPHA, screen size), cleared by caller
    def handle_event(self, ev, view: ViewInfo) -> UICommand | None: ...  # pygame event; None = not consumed
    def begin_human_choice(self, agent_id: str, actions: list[ActionDescriptor]) -> None: ...
    def end_human_choice(self) -> None: ...
    def world_click(self, cell: tuple[int, int], eid: int, button: int, view: ViewInfo) -> UICommand | None: ...
