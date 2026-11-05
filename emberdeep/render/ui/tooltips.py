"""Tooltip rendering and the rules-text glossary (short, own-words SRD-style descriptions)."""
from __future__ import annotations

import pygame

from emberdeep.render.ui import theme as T
from emberdeep.render.ui.combat_log import humanize
from emberdeep.render.ui.widgets import Tip

CONDITIONS: dict[str, tuple[str, str, str]] = {
    # id: (icon, colour-name, description)
    "blinded": ("blindfold", "", "Cannot see. Attacks against it have advantage; its own attacks have disadvantage."),
    "charmed": ("chained-heart", "", "Cannot attack or harm its charmer, who has advantage on social checks against it."),
    "deafened": ("mute", "", "Cannot hear, and fails any check that depends on hearing."),
    "frightened": ("terror", "", "Disadvantage on checks and attacks while the source of fear is in sight; cannot "
                                  "willingly move closer to it."),
    "grappled": ("grab", "", "Speed drops to 0 until released or broken free."),
    "incapacitated": ("sleepy", "", "Cannot take actions or reactions."),
    "invisible": ("invisible-face", "", "Cannot be seen without special senses. Attacks against it have disadvantage; "
                                         "its attacks have advantage."),
    "paralyzed": ("frozen-body", "", "Cannot move or act. Attacks against it have advantage, and nearby hits are "
                                      "critical."),
    "petrified": ("stone-pile", "", "Turned to stone: incapacitated, unaware, and resistant to all damage."),
    "poisoned": ("poison-bottle", "", "Disadvantage on attack rolls and ability checks."),
    "prone": ("falling-leaf", "", "On the ground. Melee attacks against it have advantage, ranged ones disadvantage. "
                                  "Standing up costs half its movement."),
    "restrained": ("rope-coil", "", "Speed 0. Attacks against it have advantage; its attacks and DEX saves have "
                                    "disadvantage."),
    "stunned": ("knockout", "", "Cannot act or move and fails STR/DEX saves. Attacks against it have advantage."),
    "unconscious": ("sleepy", "", "Helpless and prone, unaware of surroundings. Nearby hits are critical."),
    "dying": ("heart-beats", "", "At 0 HP and making death saving throws. Three failures mean death."),
    "stable": ("heart-plus", "", "Unconscious but no longer dying."),
    "hidden": ("hooded-figure", "", "Unseen by enemies until it attacks or is found."),
    "fleeing": ("run", "", "Panicked and running from the fight."),
    "surrendered": ("dove", "", "Has yielded and will not attack unless provoked."),
    "surprised": ("eye-target", "", "Caught off guard: cannot act on its first turn."),
    "concentrating": ("spiral-bloom", "", "Maintaining a spell; damage can break it."),
}
EFFECTS: dict[str, tuple[str, str]] = {
    "bless": ("sparkles", "Adds a d4 to attack rolls and saving throws."),
    "bane": ("skull-bolt", "Subtracts a d4 from attack rolls and saving throws."),
    "rage": ("angry-eyes", "Bonus melee damage, resistance to weapon damage, advantage on STR checks and saves."),
    "dodge": ("shield-bounces", "Attacks against it have disadvantage until its next turn."),
    "hunters_mark": ("crosshair", "Marked prey takes extra damage from the caster's weapon attacks."),
    "hex": ("cursed-star", "Cursed: takes extra necrotic damage and has disadvantage on one ability's checks."),
    "shield_of_faith": ("shield", "A shimmering field grants +2 AC."),
    "shield": ("shield", "Magical barrier: +5 AC until the start of its next turn."),
    "mage_armor": ("robe", "Protective force sets its AC to 13 + DEX modifier."),
    "haste": ("sprint", "Doubled speed, +2 AC, extra action; lethargy follows when it ends."),
    "slow": ("hourglass", "Halved speed, -2 AC and DEX saves, fewer actions."),
    "invisibility": ("invisible-face", "Cannot be seen until it attacks or casts."),
    "mirror_image": ("double-ringed-orb", "Illusory duplicates cause attacks to miss."),
    "blur": ("aura", "Attackers have disadvantage against this creature."),
    "heroism": ("laurels", "Immune to fear and gains temporary hit points each turn."),
    "sanctuary": ("holy-symbol", "Attackers must first resist a ward before targeting it."),
    "barkskin": ("oak", "Its skin hardens: AC cannot be lower than 16."),
    "aid": ("heart-plus", "Maximum and current hit points are increased."),
    "second_wind": ("heart-plus", "Recovered a burst of hit points."),
    "spiritual_weapon": ("magic-swirl", "A floating weapon strikes at the caster's command."),
}
STATUS_ICON = {"dying": "heart-beats", "hidden": "hooded-figure", "fleeing": "run", "surrendered": "dove",
               "surprised": "eye-target", "stable": "heart-plus"}

RESOURCE_INFO: dict[str, tuple[str, str, str]] = {
    # id: (label, icon, description)
    "ki": ("Ki", "meditation", "Monk energy spent on special techniques."),
    "rage": ("Rages", "angry-eyes", "Uses of Rage remaining."),
    "sorcery_points": ("Sorcery", "magic-swirl", "Spent to shape spells or create slots."),
    "channel_divinity": ("Channel Div.", "holy-symbol", "Divine power for a once-per-rest feature."),
    "second_wind": ("Second Wind", "heart-plus", "Bonus action to recover hit points."),
    "action_surge": ("Action Surge", "lightning-shout", "One extra action this rest."),
    "potions": ("Potions", "health-potion", "Healing potions carried."),
    "arcane_recovery": ("Arcane Recovery", "spell-book", "Regain some spell slots on a short rest."),
    "lay_on_hands": ("Lay on Hands", "hand-of-god", "Pool of healing."),
    "wild_shape": ("Wild Shape", "bear-head", "Beast transformations remaining."),
    "bardic_inspiration": ("Inspiration", "musical-notes", "Dice to grant allies."),
    "sneak_attack": ("Sneak Attack", "backstab", "Extra damage when conditions are met."),
}


def cond_tip(name: str) -> Tip:
    n = name.lower()
    icon, _, desc = CONDITIONS.get(n, ("", "", ""))
    if not desc:
        icon, desc = EFFECTS.get(n, ("magic-swirl", "A lingering magical effect."))
    return Tip(humanize(n), "Condition" if n in CONDITIONS else "Effect", [(desc, T.TEXT_DIM)], icon or "magic-swirl",
               T.ORANGE if n in CONDITIONS else T.CYAN)


def effect_tip(eid: str) -> Tip:
    icon, desc = EFFECTS.get(eid, ("magic-swirl", "A lingering magical effect."))
    return Tip(humanize(eid), "Effect", [(desc, T.TEXT_DIM)], icon, T.CYAN)


def icon_for_cond(name: str) -> str:
    n = name.lower()
    if n in CONDITIONS:
        return CONDITIONS[n][0]
    if n in EFFECTS:
        return EFFECTS[n][0]
    return "magic-swirl"


class TipLayer:
    DELAY = 0.22

    def __init__(self, hud) -> None:
        self.hud = hud
        self._key = None
        self._since = 0.0
        self._surf: pygame.Surface | None = None
        self._alpha = 0.0

    def _build(self, tip: Tip) -> pygame.Surface:
        ui = self.hud.ui
        pad = ui.s(12)
        maxw = ui.s(tip.width or 280)
        inner_w = maxw - 2 * pad
        rows: list[tuple[str, object]] = []
        h = pad
        ic = ui.s(26) if tip.icon else 0
        head_w = inner_w - (ic + ui.s(8) if ic else 0)
        title = ui.text(ui.ellipsize(tip.title, "title", 14.5, head_w), "title", 14.5, T.GOLD_HI) if tip.title else None
        sub = ui.text(tip.subtitle, "body_italic", 12.5, T.TEXT_DIM) if tip.subtitle else None
        head_h = max(ic, (title.get_height() if title else 0) + (sub.get_height() if sub else 0))
        h += head_h
        body: list[tuple[pygame.Surface, int]] = []
        if tip.hp is not None:
            h += ui.s(8) + ui.s(16)
        if tip.lines:
            h += ui.s(8)
        for text, col in tip.lines:
            for ln in ui.wrap(text, "body", 13.5, inner_w):
                s = ui.text(ln, "body", 13.5, col)
                body.append((s, h))
                h += s.get_height()
            h += ui.s(2)
        h += pad
        w = maxw
        if not tip.lines and tip.hp is None:
            w = max((title.get_width() if title else 0), (sub.get_width() if sub else 0)) + ic + ui.s(8 if ic else 0) + 2 * pad
            w = max(w, ui.s(110))
        surf = pygame.Surface((w + 2, h + 2), pygame.SRCALPHA)
        r = pygame.Rect(1, 1, w, h)
        ui.draw_panel(surf, r, "tip", tip.accent)
        x = r.x + pad
        y = r.y + pad
        if tip.icon:
            ui.blit_icon(surf, tip.icon, (x + ic // 2, y + ic // 2 + 1), ic, tip.icon_color)
            x += ic + ui.s(8)
        if title:
            surf.blit(title, (x, y - ui.s(1)))
        if sub:
            surf.blit(sub, (x, y + (title.get_height() if title else 0) - ui.s(2)))
        yy = r.y + pad + head_h
        if tip.hp is not None:
            frac, lab = tip.hp
            br = pygame.Rect(r.x + pad, yy + ui.s(8), inner_w, ui.s(14))
            col = T.hp_color(frac)
            ui.bar(surf, br, frac, col)
            lt = ui.text(lab, "mono", 10.5, T.TEXT, shadow=True)
            surf.blit(lt, lt.get_rect(center=br.center))
            yy = br.bottom
        if tip.lines:
            ui.divider(surf, r.x + pad, r.right - pad, yy + ui.s(6), T.GOLD_DIM, 90)
        for s, by in body:
            surf.blit(s, (r.x + pad, r.y + by))
        return surf

    def draw(self, surf: pygame.Surface) -> None:
        hud = self.hud
        tip = hud.current_tip()
        if tip is None:
            self._key = None
            self._alpha = max(0.0, self._alpha - 0.2)
            return
        key = (tip.title, tip.subtitle, tuple(tip.lines), tip.hp, tip.icon, hud.ui.scale)
        if key != self._key:
            self._key = key
            self._since = hud.t
            self._surf = self._build(tip)
            self._alpha = 0.0
        if hud.t - self._since < self.DELAY or self._surf is None:
            return
        self._alpha = min(1.0, self._alpha + 0.18)
        s = self._surf
        mx, my = hud.mouse
        W, H = surf.get_size()
        off = hud.ui.s(18)
        x = mx + off
        y = my + off
        if x + s.get_width() > W - 6:
            x = mx - off - s.get_width()
        if y + s.get_height() > H - 6:
            y = my - off - s.get_height()
        x, y = max(4, x), max(4, y)
        if self._alpha < 1:
            s = s.copy()
            s.set_alpha(int(255 * self._alpha))
        surf.blit(s, (x, y))
