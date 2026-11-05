# Contract notes from `mapgen`

No contract changes are required; these are clarifications the builder / render-adapter engineers should confirm.

1. **Coordinate convention.** `render.api.LightSpec` / `Decor` and `mapdata.LightData` say "cell centre = x + 0.5" without
   saying whether the stored float is the continuous position or the cell index. `mapgen` / `ascii_map` / `decorate` store
   **continuous positions**: the centre of cell `(cx, cy)` is `(cx + 0.5, cy + 0.5)`. The render adapter should copy
   `LightData.x/y` and `Decor.x/y` into `LightSpec`/`Decor` unchanged.
2. **Wall-mounted lights** sit on the floor side of the wall (about 0.35 squares from the wall face) so that shadow
   ray-marching does not occlude them inside the wall cell.
3. **Multi-cell props.** A sarcophagus spanning 2-3 `LOW_WALL` cells is one `Decor(kind="sarcophagus")` at the footprint
   centre, `scale` = length in squares, `rotation` 0 or pi/2.
4. **Braziers** (`LOW_WALL` + `Decor("brazier")` + `LightData(kind="brazier")`) are the only decor that decorate may
   turn into terrain; they are placed only where all 8 neighbours are walkable.
5. **Light kinds produced** (`LightData.kind`): `torch`, `brazier`, `lava`, `crystal`, `magic` (glowing mushrooms), `candle`,
   `fire` (campfire / hoard glow). The render adapter should map them onto `LightSpec.kind` (`candle`, `fire`, `crystal`,
   `magic` exist there).
6. **Altar named "Fountain of Life"** yields `Decor("fountain")` instead of `Decor("altar")`.
7. **Key items.** Generated keyed doors use `DoorData.key_id` (`iron_key_N`, `crypt_key`, `sacristy_key`); a chest with
   `params["items"] == [key_id]` is guaranteed to exist and to be reachable without that door.
