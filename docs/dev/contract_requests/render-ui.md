# Contract requests from render-ui

1. `UICommand("noop")`: returned by `HUD.handle_event` / `world_click` when the HUD consumed an input without a command (clicks on panels, wheel over the log). Renderer should ignore it. `HUD.mouse_over_panel()` can also be polled.
2. `UICommand("focus", value)`: value is an eid (int) from party frames and an `(x, y)` float tuple (map squares) from the minimap.
3. `ViewInfo.options`: HUD reads keys grid, threat, los, labels, log, minimap, research, mute and falls back to `HUD.DEFAULT_OPTIONS`. Renderer should initialise from it and flip on `toggle`.
4. `ActionDescriptor`: optional `enabled: bool` (+ `disabled_reason`). HUD reads `getattr(d, "enabled", True)` and greys the button.
5. `EntityView`: ability scores. HUD reads `stats["str"|"dex"|"con"|"int"|"wis"|"cha"]` for the inspector. Stats prefixed `r_` or `reward_` are shown as reward components.
