"""Combat log: converts event dicts into rich entries and draws the scrollable panel."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import pygame

from emberdeep.core.events import EventType as ET
from emberdeep.render.ui import theme as T
from emberdeep.render.ui.theme import Color
from emberdeep.render.ui.widgets import Tip, button, layout_rich

log = logging.getLogger(__name__)

COND_NAMES = ("Blinded", "Charmed", "Deafened", "Frightened", "Grappled", "Incapacitated", "Invisible", "Paralyzed",
              "Petrified", "Poisoned", "Prone", "Restrained", "Stunned", "Unconscious")
SKILLS = ("Acrobatics", "Animal Handling", "Arcana", "Athletics", "Deception", "History", "Insight", "Intimidation",
          "Investigation", "Medicine", "Nature", "Perception", "Performance", "Persuasion", "Religion",
          "Sleight of Hand", "Stealth", "Survival")
ABIL_FULL = ("Strength", "Dexterity", "Constitution", "Intelligence", "Wisdom", "Charisma")
DOOR_STATES = ("opens", "closes", "unlocks", "breaks")
OK = T.GREEN
BAD = T.RED
D20 = "dice-twenty-faces-twenty"


def humanize(s: str) -> str:
    if not s:
        return ""
    fixed = {"hunters_mark": "Hunter's Mark", "thieves_tools": "Thieves' Tools"}
    if s in fixed:
        return fixed[s]
    return " ".join(w.capitalize() if w not in ("of", "the", "and") else w for w in str(s).replace("_", " ").split())


def dmg_name(kind) -> str:
    try:
        return T.DAMAGE_NAMES[int(kind)]
    except Exception:  # noqa: BLE001
        return "force"


def ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


@dataclass
class LogEntry:
    cat: str                      # combat | social | flow
    tokens: list[tuple]
    style: str = "normal"         # normal | indent | header | alert | dim
    color: Color | None = None
    born: float = 0.0
    _cache: dict = field(default_factory=dict)


class Formatter:
    def __init__(self, hud) -> None:
        self.hud = hud

    # token helpers
    def nm(self, eid: int) -> tuple:
        return self.hud.name_token(eid)

    @staticmethod
    def t(text: str, color: Color = T.TEXT, role: str = "body") -> tuple:
        return ("t", text, color, role)

    def dice(self, roll: int, rolls: list[int], bonus: int, total: int, adv: int = 0) -> list[tuple]:
        toks: list[tuple] = [("i", D20, T.GOLD)]
        rc = T.GREEN if roll == 20 else T.RED if roll == 1 else T.TEXT
        if rolls and len(rolls) > 1:
            kept = False
            for r in rolls:
                if r == roll and not kept:
                    toks.append(("d", str(r), rc, False))
                    kept = True
                else:
                    toks.append(("d", str(r), T.TEXT_FAINT, True))
        else:
            toks.append(("d", str(roll), rc, False))
        if bonus:
            toks.append(("d", f"{'+' if bonus >= 0 else '-'} {abs(bonus)}", T.TEXT_DIM, False))
        toks.append(("d", f"= {total}", T.TEXT, False))
        return toks

    def fmt(self, d: dict) -> list[LogEntry]:  # noqa: C901 - one branch per event type
        h = self.hud
        et = d.get("t")
        data = d.get("data") or {}
        src, dst = d.get("src", -1), d.get("dst", -1)
        amt = d.get("amount", 0)
        kind = d.get("kind", -1)
        t = self.t
        nm = self.nm
        E = LogEntry
        try:
            if et == ET.ATTACK:
                hit, crit, fumble = data.get("hit"), data.get("crit"), data.get("fumble")
                toks = [nm(src), t(" attacks "), nm(dst)]
                w = data.get("name")
                if w:
                    toks += [t(" with "), t(str(w), T.TEXT_DIM, "body_italic")]
                if data.get("opportunity"):
                    toks.append(t(" (opportunity)", T.TEXT_FAINT, "body_italic"))
                toks.append(("nl",))
                roll = int(data.get("roll", 0) or 0)
                toks += self.dice(roll, list(data.get("rolls") or [roll]), int(data.get("bonus", 0) or 0),
                                  int(data.get("total", 0) or 0), int(data.get("adv", 0) or 0))
                if data.get("ac") is not None:
                    toks.append(t(f"vs AC {data['ac']}", T.TEXT_DIM))
                toks.append(t("—", T.TEXT_FAINT))
                if crit:
                    toks.append(t("Critical hit!", T.GOLD_HI, "body_bold"))
                elif fumble:
                    toks.append(t("Fumble", T.RED, "body_bold"))
                elif hit:
                    toks.append(t("Hit", T.GREEN, "body_bold"))
                else:
                    toks.append(t("Miss", T.RED, "body_bold"))
                return [E("combat", toks)]
            if et == ET.MISS:
                return []
            if et == ET.DAMAGE:
                dn = dmg_name(kind)
                col = T.DAMAGE_COLORS.get(dn, T.TEXT)
                toks = [nm(dst), t(" takes "), t(str(amt), col, "body_bold"), ("i", T.DAMAGE_ICONS[dn], col),
                        t(f"{dn} damage", col)]
                mod = data.get("mod")
                if mod:
                    toks.append(t(f"({mod})", T.TEXT_DIM, "body_italic"))
                if data.get("crit"):
                    toks.append(t("critical", T.GOLD_HI, "body_italic"))
                if data.get("temp_absorbed"):
                    toks.append(t(f"({data['temp_absorbed']} absorbed by temp HP)", T.CYAN, "body_italic"))
                return [E("combat", toks, "indent")]
            if et == ET.HEAL:
                toks = [nm(dst), t(" recovers "), t(f"{amt}", T.GREEN, "body_bold"), ("i", "heart-plus", T.GREEN),
                        t("HP", T.GREEN)]
                if src not in (-1, dst):
                    toks += [t("from"), nm(src)]
                return [E("combat", toks, "indent")]
            if et == ET.TEMP_HP:
                return [E("combat", [nm(dst), t(" gains "), t(f"{amt} temporary HP", T.CYAN, "body_bold")], "indent")]
            if et in (ET.SAVE, ET.CHECK):
                ok = data.get("success")
                if et == ET.SAVE:
                    ab = ABIL_FULL[int(kind)] if 0 <= int(kind) < 6 else "Saving"
                    head = [nm(dst), t(f" {'resists' if ok else 'fails'} — "), t(f"{ab} save", T.TEXT_DIM, "body_italic")]
                else:
                    k = int(kind)
                    nmx = ABIL_FULL[k - 100] + " check" if k >= 100 and k - 100 < 6 else (
                        SKILLS[k] if 0 <= k < 18 else "Check")
                    head = [nm(dst), t(f" — {nmx}", T.TEXT_DIM, "body_italic")]
                    if data.get("purpose"):
                        head.append(t(f"({humanize(data['purpose'])})", T.TEXT_FAINT, "body_italic"))
                roll = int(data.get("roll", 0) or 0)
                tot = int(data.get("total", roll) or 0)
                toks = head + [("nl",)] + self.dice(roll, [roll], tot - roll, tot) + [
                    t(f"vs DC {data.get('dc', '?')}", T.TEXT_DIM), t("—", T.TEXT_FAINT),
                    t("Success" if ok else "Failure", OK if ok else BAD, "body_bold")]
                return [E("combat", toks, "indent")]
            if et == ET.SPELL_CAST:
                lvl = int(amt or 0)
                toks = [nm(src), t(" casts "), ("i", "sparkles", T.PURPLE),
                        t(str(data.get("name") or humanize(data.get("spell", "a spell"))), T.PURPLE, "body_bold")]
                if lvl:
                    toks.append(t(f"({ordinal(lvl)} level)", T.TEXT_FAINT))
                tg = [x for x in (data.get("targets") or []) if x != src]
                if len(tg) == 1:
                    toks += [t("on"), nm(tg[0])]
                elif len(tg) > 1:
                    toks.append(t(f"on {len(tg)} targets", T.TEXT_DIM))
                return [E("combat", toks)]
            if et == ET.ABILITY_USE:
                toks = [nm(src), t(" uses "), t(str(data.get("name") or humanize(data.get("ability", ""))),
                                                T.YELLOW, "body_bold")]
                if dst not in (-1, src):
                    toks += [t("on"), nm(dst)]
                return [E("combat", toks)]
            if et == ET.EFFECT_START:
                nme = data.get("name") or humanize(data.get("effect", ""))
                toks = [nm(dst), t(" gains "), t(str(nme), T.CYAN, "body_bold")]
                if data.get("duration"):
                    toks.append(t(f"({data['duration']} rounds)", T.TEXT_FAINT))
                return [E("combat", toks, "indent")]
            if et == ET.EFFECT_END:
                return [E("combat", [t(humanize(data.get("effect", "Effect")), T.CYAN), t(" fades from "), nm(dst)],
                          "dim")]
            if et == ET.CONDITION_ADD:
                cn = COND_NAMES[int(kind)] if 0 <= int(kind) < 14 else "a condition"
                return [E("combat", [nm(dst), t(" is "), t(cn.lower(), T.ORANGE, "body_bold")], "indent")]
            if et == ET.CONDITION_REMOVE:
                cn = COND_NAMES[int(kind)] if 0 <= int(kind) < 14 else "condition"
                return [E("combat", [nm(dst), t(" is no longer "), t(cn.lower(), T.TEXT_DIM)], "dim")]
            if et == ET.CONCENTRATION_CHECK:
                ok = data.get("success")
                roll = int(data.get("roll", 0) or 0)
                tot = int(data.get("total", roll) or 0)
                toks = [nm(dst), t(" concentration on "), t(humanize(data.get("spell", "")), T.PURPLE), ("nl",)]
                toks += self.dice(roll, [roll], tot - roll, tot) + [t(f"vs DC {data.get('dc', '?')}", T.TEXT_DIM),
                                                                     t("—", T.TEXT_FAINT),
                                                                     t("Held" if ok else "Lost", OK if ok else BAD,
                                                                       "body_bold")]
                return [E("combat", toks, "indent")]
            if et == ET.CONCENTRATION_END:
                return [E("combat", [nm(dst), t(" loses concentration on "), t(humanize(data.get("spell", "")), T.PURPLE)],
                          "dim")]
            if et == ET.DOWNED:
                return [E("combat", [nm(dst), t(" falls unconscious!", T.ORANGE, "body_bold")], "alert", T.ORANGE)]
            if et == ET.DEATH:
                return [E("combat", [("i", "skull-crossed-bones", T.RED), nm(dst), t(" dies.", T.RED, "body_bold")],
                          "alert", T.RED)]
            if et == ET.DEATH_SAVE:
                ok = data.get("success")
                return [E("combat", [nm(dst), t(" death save: "), ("d", str(data.get("roll", "?")), T.TEXT, False),
                                     t("—", T.TEXT_FAINT), t("success" if ok else "failure", OK if ok else BAD, "body_bold"),
                                     t(f"({data.get('successes', 0)}✓ {data.get('failures', 0)}✗)", T.TEXT_FAINT)],
                          "indent")]
            if et == ET.STABILIZED:
                return [E("combat", [nm(dst), t(" is stabilised", T.GREEN)])]
            if et == ET.REVIVED:
                return [E("combat", [nm(dst), t(" is back on their feet", T.GREEN, "body_bold")])]
            if et == ET.EXHAUSTION:
                return [E("combat", [nm(dst), t(f" suffers exhaustion (level {amt})", T.ORANGE)], "indent")]
            if et == ET.SURRENDER:
                return [E("social", [nm(src), t(" surrenders.", T.TEXT_DIM, "body_italic")])]
            if et == ET.FLEE:
                return [E("combat", [nm(src), t(" flees in panic.", T.ORANGE, "body_italic")])]
            if et == ET.MOVE:
                return []
            if et == ET.FORCED_MOVE:
                return [E("combat", [nm(dst), t(f" is shoved {amt} squares"), t("by"), nm(src)], "dim")]
            if et == ET.TELEPORT:
                return [E("combat", [nm(src), t(" teleports.", T.PURPLE, "body_italic")], "dim")]
            if et == ET.FALL:
                return [E("combat", [nm(dst), t(f" falls {data.get('height_ft', '?')} ft and takes {amt} damage",
                                                 T.ORANGE)], "indent")]
            if et == ET.STAND_UP:
                return [E("combat", [nm(src), t(" stands up.", T.TEXT_DIM)], "dim")]
            if et == ET.ESCAPE:
                return [E("combat", [nm(src), t(" escapes!", T.GREEN, "body_bold")])]
            if et == ET.ZONE_START:
                return [E("combat", [t(humanize(data.get("kind", "A zone")), T.PURPLE, "body_bold"),
                                     t(" fills the area"), t("(" + ("harmful" if data.get("harmful", True) else "benign") + ")",
                                                             T.TEXT_FAINT)], "dim")]
            if et == ET.SUMMON:
                return [E("combat", [nm(src), t(" summons "), nm(dst)])]
            if et == ET.TRANSFORM:
                return [E("combat", [nm(dst), t(" transforms.", T.PURPLE, "body_italic")], "dim")]
            if et == ET.REACTION:
                return [E("combat", [nm(src), t(" reacts: "), t(str(data.get("name", "Reaction")), T.PURPLE, "body_bold")])]
            if et == ET.COUNTERSPELL:
                ok = data.get("success")
                return [E("combat", [nm(src), t(" counterspells "), t(humanize(data.get("spell", "")), T.PURPLE),
                                     t("—"), t("countered" if ok else "fizzles", OK if ok else BAD, "body_bold")])]
            if et == ET.RECHARGE:
                return [E("combat", [nm(src), t(f" {humanize(data.get('ability', ''))} recharges" if data.get("success")
                                                else " fails to recharge", T.TEXT_DIM)], "dim")]
            if et == ET.PICKUP:
                toks = [nm(src), t(" collects loot")]
                if amt:
                    toks += [("i", "coins", T.YELLOW), t(f"{amt} gp", T.YELLOW, "body_bold")]
                return [E("social", toks)]
            if et == ET.GIVE:
                toks = [nm(src), t(" gives "), nm(dst)]
                if amt:
                    toks += [("i", "coins", T.YELLOW), t(f"{amt} gp", T.YELLOW, "body_bold")]
                return [E("social", toks)]
            if et == ET.USE_ITEM:
                return [E("combat", [nm(src), t(" uses "), t(humanize(data.get("item", "an item")), T.YELLOW)])]
            if et == ET.DROP:
                return [E("combat", [nm(src), t(" drops loot", T.TEXT_DIM)], "dim")]
            if et == ET.EQUIP:
                return [E("combat", [nm(src), t(" equips "), t(humanize(data.get("item", "")), T.TEXT_DIM)], "dim")]
            if et == ET.DOOR:
                return [E("combat", [nm(src), t(" forces the door" if data.get("forced") else " works the door"
                                                 ), ], "dim")]
            if et == ET.CHEST_OPEN:
                return [E("social", [nm(src), t(" opens a chest.", T.YELLOW)])]
            if et == ET.LOCKPICK:
                ok = data.get("success")
                return [E("combat", [nm(src), t(" picks the lock — "), t("success" if ok else "failure",
                                                                           OK if ok else BAD, "body_bold")])]
            if et == ET.TRAP_TRIGGER:
                return [E("combat", [t("A trap springs!", T.ORANGE, "body_bold"), t("Victim:"), nm(dst)], "alert",
                          T.ORANGE)]
            if et == ET.TRAP_DETECT:
                return [E("combat", [nm(src), t(" spots a trap.", T.YELLOW)])]
            if et == ET.TRAP_DISARM:
                ok = data.get("success")
                return [E("combat", [nm(src), t(" disarms the trap." if ok else " fails to disarm the trap.",
                                                 OK if ok else BAD)])]
            if et == ET.LEVER:
                return [E("combat", [nm(src), t(f" throws a lever ({data.get('state', '')})", T.TEXT_DIM)], "dim")]
            if et == ET.INTERACT:
                return [E("combat", [nm(src), t(f" interacts with the {data.get('kind', 'object')}", T.TEXT_DIM)], "dim")]
            if et == ET.GOLD_CHANGE:
                if not amt:
                    return []
                col = T.YELLOW if amt > 0 else T.ORANGE
                return [E("social", [nm(dst), t(" gold "), ("i", "coins", col), t(f"{'+' if amt > 0 else ''}{amt} gp", col,
                                                                                  "body_bold")], "dim")]
            if et == ET.HIDE:
                return [E("combat", [nm(src), t(" slips into the shadows." if data.get("success") else
                                                 " fails to hide.", T.TEXT_DIM, "body_italic")], "dim")]
            if et == ET.REVEALED:
                return [E("combat", [nm(dst), t(" is revealed!", T.ORANGE)], "dim")]
            if et == ET.SPOTTED:
                return [E("combat", [nm(src), t(" notices "), nm(dst)], "dim")]
            if et == ET.SEARCH:
                return [E("combat", [nm(src), t(" searches"), t("— found something!" if data.get("found") else
                                                                 "— nothing.", T.TEXT_DIM)], "dim")]
            if et == ET.SAY:
                vocab = h.scene.comm_vocab if h.scene else []
                k = int(kind)
                msg = vocab[k] if 0 <= k < len(vocab) else f"<{k}>"
                return [E("social", [nm(src), t(":", T.TEXT_DIM), t(f"“{msg}”", T.TEXT, "body_italic")], "normal")]
            if et == ET.TRUCE_OFFER:
                return [E("social", [("i", "dove", T.GOLD_HI), nm(src), t(" offers a truce to "), nm(dst)], "alert",
                          T.GOLD)]
            if et == ET.TRUCE_ACCEPT:
                return [E("social", [("i", "dove", T.GREEN), nm(src), t(" accepts the truce with "), nm(dst),
                                     t("— hostilities pause.", T.GREEN, "body_bold")], "alert", T.GREEN)]
            if et == ET.RELATION_CHANGE:
                rel = ("Hostile", "Neutral", "Truce", "Allied")
                a, b = int(data.get("a", 0)), int(data.get("b", 0))
                new = int(data.get("new", 0))
                return [E("social", [t(h.faction_name(a), h.faction_color(a), "body_bold"), t("and"),
                                     t(h.faction_name(b), h.faction_color(b), "body_bold"),
                                     t("are now"), t(rel[new] if 0 <= new < 4 else "?", T.GOLD_HI, "body_bold")])]
            if et == ET.BETRAYAL:
                return [E("social", [("i", "dagger-rose", T.RED), t("BETRAYAL! ", T.RED, "body_bold"), nm(src),
                                     t(" strikes "), nm(dst), t(" despite the "
                                                                + ("alliance" if data.get("relation") == 3 else "truce") + "!")],
                          "alert", T.RED)]
            if et == ET.PARLEY:
                ok = data.get("success")
                return [E("social", [nm(src), t(" parleys with "), nm(dst), t("—"),
                                     t("persuaded" if ok else "rebuffed", OK if ok else BAD, "body_bold")])]
            if et == ET.REST_VOTE:
                return [E("social", [nm(src), t(" votes to rest." if amt else " withdraws the rest vote."
                                                 , T.TEXT_DIM)], "dim")]
            if et == ET.REST:
                return [E("social", [("i", "campfire", T.ORANGE), t(f"The party takes a {data.get('kind', 'short')} rest.",
                                                                       T.GREEN, "body_bold")], "alert", T.GREEN)]
            if et == ET.OBJECTIVE:
                ok = data.get("success", True)
                return [E("social", [("i", "laurels", T.GOLD_HI), t("Objective " + ("complete: " if ok else "failed: "),
                                                                     T.GOLD_HI if ok else T.RED, "body_bold"),
                                     t(str(data.get("name", "")), T.TEXT)], "alert", T.GOLD if ok else T.RED)]
            if et == ET.LOG:
                return [E("combat", [t(str(data.get("text", "")), T.TEXT_DIM, "body_italic")], "dim")]
            if et == ET.COMBAT_START:
                return [E("flow", [t("Roll for initiative!", T.RED, "body_bold")], "header", T.RED)]
            if et == ET.COMBAT_END:
                return [E("flow", [t("Combat ends.", T.TEXT_DIM, "body_italic")], "header", T.GOLD_DIM)]
            if et == ET.ROUND_START:
                return [E("flow", [t(f"Round {amt}", T.GOLD, "title")], "header", T.GOLD)]
            if et == ET.TURN_START:
                return [E("flow", [t("▸", T.GOLD_DIM), nm(src), t("'s turn" if True else "", T.TEXT_FAINT)], "dim")]
            if et == ET.EPISODE_START:
                return [E("flow", [t(h.scene.name if h.scene else "Episode begins", T.GOLD, "title")], "header", T.GOLD)]
            if et == ET.EPISODE_END:
                return [E("flow", [t("Episode over: " + str(data.get("reason", "")), T.GOLD_HI, "body_bold")], "header",
                          T.GOLD)]
        except Exception:  # noqa: BLE001
            log.debug("log format failed for %s", d, exc_info=True)
        return []


class CombatLog:
    FILTERS = ("All", "Combat", "Social")

    def __init__(self, hud) -> None:
        self.hud = hud
        self.fmt = Formatter(hud)
        self.entries: list[LogEntry] = []
        self.filter = "All"
        self.collapsed = False
        self.scroll = 0.0              # pixels above the bottom (0 = follow)
        self._unseen = 0
        self._pend_round = None
        self._pend_turn = None
        self.rect = pygame.Rect(0, 0, 0, 0)
        self._max_scroll = 0

    def _push(self, e: LogEntry) -> None:
        e.born = self.hud.t
        self.entries.append(e)
        if self.scroll > 1:
            self._unseen += 1

    def add(self, d: dict) -> None:
        """Append formatted entries. Round headers and turn markers are held back until something
        actually happens, so quiet rounds do not spam the chronicle."""
        et = d.get("t")
        for e in self.fmt.fmt(d):
            if et == ET.ROUND_START:
                rn = d.get("amount", 0)
                if self._pend_round is None:
                    self._pend_round = (rn, rn, e)
                else:
                    self._pend_round = (self._pend_round[0], rn, e)
                continue
            if et == ET.TURN_START:
                self._pend_turn = e
                continue
            if e.cat != "flow" or et in (ET.COMBAT_START, ET.COMBAT_END, ET.EPISODE_START, ET.EPISODE_END):
                self._flush_pending()
            self._push(e)
        if len(self.entries) > 600:
            del self.entries[:100]

    def _flush_pending(self) -> None:
        pr, self._pend_round = self._pend_round, None
        pt, self._pend_turn = self._pend_turn, None
        if pr is not None:
            first, last, e = pr
            if first != last:
                self._push(LogEntry("flow", [("t", (f"Rounds {first}\u2013{last - 1} passed quietly" if last - 1 > first else f"Round {first} passed quietly"), T.TEXT_FAINT,
                                                "body_italic")], "dim"))
            self._push(e)
        if pt is not None:
            self._push(pt)

    def clear(self) -> None:
        self.entries.clear()
        self._pend_round = self._pend_turn = None
        self.scroll = 0

    def _visible(self, e: LogEntry) -> bool:
        if self.filter == "All":
            return True
        if self.filter == "Combat":
            return e.cat in ("combat", "flow")
        return e.cat in ("social",) or (e.cat == "flow" and e.style == "header")

    def _render(self, e: LogEntry, w: int) -> pygame.Surface:
        ui = self.hud.ui
        s = e._cache.get(w)
        if s is not None:
            return s
        pad = ui.s(10)
        ind = ui.s(14) if e.style == "indent" else 0
        if e.style == "header":
            inner = layout_rich(ui, e.tokens, w - 2 * pad, 13.5)
            h = inner.get_height() + ui.s(16)
            s = pygame.Surface((w, h), pygame.SRCALPHA)
            col = e.color or T.GOLD
            tw = inner.get_width()
            txt_w = max(self._text_extent(inner), 1)
            s.blit(inner, ((w - txt_w) // 2, ui.s(8)))
            ym = h // 2
            x1 = (w - txt_w) // 2 - ui.s(10)
            x2 = (w + txt_w) // 2 + ui.s(10)
            ui.divider(s, pad, x1, ym, col, 130)
            ui.divider(s, x2, w - pad, ym, col, 130)
        else:
            col_dim = e.style == "dim"
            toks = e.tokens
            inner = layout_rich(ui, toks, w - 2 * pad - ind, 13 if col_dim else 13.5, 0, ui.s(10))
            h = inner.get_height() + ui.s(5)
            s = pygame.Surface((w, h), pygame.SRCALPHA)
            if e.style == "alert":
                c = e.color or T.GOLD
                bg = pygame.Surface((w - 2 * ui.s(4), h - ui.s(2)), pygame.SRCALPHA)
                pygame.draw.rect(bg, (*T.scale_color(c, 0.35), 90), bg.get_rect(), border_radius=ui.s(4))
                pygame.draw.rect(bg, (*c, 170), (0, 0, ui.s(3), bg.get_height()), border_radius=ui.s(2))
                s.blit(bg, (ui.s(4), ui.s(1)))
            elif e.style == "indent":
                pygame.draw.line(s, (*T.GOLD_DIM, 120), (pad + ui.s(4), 2), (pad + ui.s(4), h - 3), 1)
            if col_dim:
                inner.set_alpha(190)
            s.blit(inner, (pad + ind, ui.s(2)))
        e._cache[w] = s
        return s

    @staticmethod
    def _text_extent(surf: pygame.Surface) -> int:
        r = surf.get_bounding_rect()
        return r.right

    def draw(self, surf: pygame.Surface, rect: pygame.Rect) -> None:
        hud, ui = self.hud, self.hud.ui
        self.rect = rect
        ui.draw_panel(surf, rect, "panel")
        hud.hot(rect, block=True)
        pad = ui.s(10)
        head_h = ui.s(34)
        # header: title + filter tabs + collapse
        ui.blit_text(surf, "CHRONICLE", (rect.x + pad + ui.s(2), rect.y + head_h // 2 + 1), "title", 12.5, T.GOLD,
                     anchor="midleft", spacing=1)
        fx = rect.right - pad - ui.s(24)
        bw = {f: ui.text_w(f, "body_bold", 12) + ui.s(14) for f in self.FILTERS}
        fx -= sum(bw.values()) + ui.s(4) * (len(bw) - 1)
        x = fx
        for f in self.FILTERS:
            br = pygame.Rect(x, rect.y + ui.s(8), bw[f], head_h - ui.s(14))
            button(hud, surf, br, label=f, active=(f == self.filter), pt=12, click=lambda f=f: self._set_filter(f),
                   tip=Tip(f"{f} events", lines=[("Filter the chronicle.", T.TEXT_DIM)]) if False else None)
            x += bw[f] + ui.s(4)
        cb = pygame.Rect(rect.right - pad - ui.s(22), rect.y + ui.s(8), ui.s(22), head_h - ui.s(14))
        button(hud, surf, cb, icon="@chev_down" if not self.collapsed else "@chev_up", click=self._toggle_collapse,
               tip=Tip("Collapse log" if not self.collapsed else "Expand log", lines=[("Hotkey: C", T.TEXT_FAINT)]))
        ui.divider(surf, rect.x + pad, rect.right - pad, rect.y + head_h, T.GOLD, 110)
        if self.collapsed:
            return
        body = pygame.Rect(rect.x + ui.s(2), rect.y + head_h + ui.s(4), rect.w - ui.s(4), rect.h - head_h - ui.s(8))
        hud.hot(body, wheel=self._wheel)
        vis = [e for e in self.entries if self._visible(e)]
        # lay out from newest backwards
        clip = surf.get_clip()
        surf.set_clip(body.clip(clip))
        y = body.bottom + int(self.scroll)
        total = 0
        blits = []
        for e in reversed(vis):
            s = self._render(e, body.w)
            y -= s.get_height()
            total += s.get_height()
            if y + s.get_height() >= body.y and y <= body.bottom:
                age = hud.t - e.born
                blits.append((s, y, age, e))
            if y < body.y - ui.s(80) and total > body.h + ui.s(80):
                # keep measuring total only roughly for scroll bounds
                pass
        for s, yy, age, e in blits:
            if age < 0.5:
                s = s.copy()
                s.set_alpha(int(255 * T.ease_out_cubic(age / 0.5)))
                xo = int((1 - T.ease_out_cubic(age / 0.5)) * ui.s(10))
            else:
                xo = 0
            if age < 1.2 and e.style in ("normal", "indent", "alert"):
                fl = pygame.Surface((s.get_width(), s.get_height()), pygame.SRCALPHA)
                fl.fill((*T.GOLD_HI, int(34 * (1 - age / 1.2))))
                surf.blit(fl, (body.x, yy))
            surf.blit(s, (body.x + xo, yy))
        self._max_scroll = max(0, total - body.h + ui.s(4))
        self.scroll = T.clamp(self.scroll, 0, self._max_scroll)
        # fade edge at top
        fk = ("logfade", body.w, ui.scale)
        fade = ui._misc.fetch(fk)
        if fade is None:
            fade = pygame.Surface((body.w, ui.s(14)), pygame.SRCALPHA)
            for i in range(fade.get_height()):
                fade.fill((12, 13, 18, int(190 * (1 - i / fade.get_height()))), (0, i, body.w, 1))
            ui._misc.put(fk, fade)
        surf.blit(fade, body.topleft)
        surf.set_clip(clip)
        if self.scroll > ui.s(8):
            lab = "▼ Latest" + (f"  ({self._unseen})" if self._unseen else "")
            bw2 = ui.text_w(lab, "body_bold", 12) + ui.s(20)
            br = pygame.Rect(0, 0, bw2, ui.s(22))
            br.midbottom = (body.centerx, body.bottom - ui.s(4))
            button(hud, surf, br, label=lab, pt=12, click=self._to_bottom)
        elif self._max_scroll > 0:
            self._unseen = 0
        if not vis:
            ui.blit_text(surf, "The chronicle is empty.", body.center, "body_italic", 14, T.TEXT_FAINT,
                         anchor="center")

    def _set_filter(self, f: str):
        self.filter = f
        self.scroll = 0

    def _toggle_collapse(self):
        self.collapsed = not self.collapsed

    def _to_bottom(self):
        self.scroll = 0
        self._unseen = 0

    def _wheel(self, dy: int):
        step = self.hud.ui.s(44)
        self.scroll = T.clamp(self.scroll + dy * step, 0, self._max_scroll)
        if self.scroll <= 1:
            self._unseen = 0
