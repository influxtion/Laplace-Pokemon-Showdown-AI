r"""Convert a live poke-env battle into a poke-engine `State`, and sample (determinize)
the opponent's hidden set so the engine can search a concrete game.

Background. The strongest open-source Showdown bot, Foul Play, doesn't hand-roll a forward
simulator (the dead end this project hit in deep_search.py). It feeds the position into
`poke-engine` -- a fast Rust battle engine with a real MCTS / expectiminimax search -- and
handles hidden information by *determinization*: sample several full opponent teams consistent
with what's been revealed, search each, and pool the results. This module is the bridge:
poke-env `Battle`/`Pokemon` objects in, a poke-engine `State` out.

What's known vs. guessed:
  * Our side is fully known (stats, moves, item, ability) from the request the server sends us.
  * The opponent's species and revealed moves are known; the rest of their set (the other
    moves, ability, tera type, item) and any not-yet-seen bench Pokemon are *sampled* from the
    gen9 randbats set sheet (server/data/random-battles/gen9/sets.json) -- the same public pool
    knowledge.py already uses. Items aren't in that sheet, so they're guessed by role/moves.

The engine's Pokemon constructor takes explicit stats; we pass the real ones for our side and
the standard randbats estimate (base + level, 85 EVs / 31 IVs / neutral) for the opponent --
exactly what knowledge.estimate uses, so damage numbers stay consistent across the project.
"""

import json
import os
import random

from poke_env.data import to_id_str, GenData

from knowledge import _estimate_stat

from poke_engine import (
    State, Side, SideConditions, VolatileStatusDurations, Pokemon as PEPokemon, Move as PEMove,
)

GEN = 9
_POKEDEX = GenData.from_gen(GEN).pokedex

_SETS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "server", "data", "random-battles", "gen9", "sets.json",
)

# --- status / weather / terrain / hazard name maps (poke-env -> poke-engine strings) --------

_STATUS = {"BRN": "Burn", "PSN": "Poison", "TOX": "Toxic",
           "PAR": "Paralyze", "SLP": "Sleep", "FRZ": "Freeze"}

_WEATHER = {"SUNNYDAY": "sun", "DESOLATELAND": "harshsun",
            "RAINDANCE": "rain", "PRIMORDIALSEA": "heavyrain",
            "SANDSTORM": "sand", "SNOWSCAPE": "snow", "SNOW": "snow", "HAIL": "hail"}

_TERRAIN = {"ELECTRIC_TERRAIN": "electricterrain", "GRASSY_TERRAIN": "grassyterrain",
            "MISTY_TERRAIN": "mistyterrain", "PSYCHIC_TERRAIN": "psychicterrain"}


def _status_str(mon):
    if mon is None or mon.status is None:
        return "None"
    return _STATUS.get(mon.status.name, "None")


def _types_tuple(mon):
    """Current types (post-Tera if terastallized), padded to 2 with 'typeless'."""
    ts = [t.name.lower() for t in mon.types if t is not None]
    if not ts:
        ts = ["normal"]
    if len(ts) == 1:
        ts.append("typeless")
    return (ts[0], ts[1])


def _base_types_tuple(mon):
    """The species' original (pre-Tera) types, from the dex, padded to 2."""
    entry = _POKEDEX.get(to_id_str(mon.species)) or {}
    ts = [t.lower() for t in entry.get("types", [])]
    if not ts:
        ts = [t.name.lower() for t in mon.types if t is not None] or ["normal"]
    if len(ts) == 1:
        ts.append("typeless")
    return (ts[0], ts[1])


def _stat(mon, key, own):
    """A single stat: the real value for our side (from the request), else the randbats
    estimate used everywhere else in the project."""
    if own:
        real = (getattr(mon, "stats", None) or {}).get(key)
        if real:
            return int(real)
    return int(_estimate_stat(mon, key))


def _maxhp(mon, own):
    if own and getattr(mon, "max_hp", 0):
        return int(mon.max_hp)
    return int(_estimate_stat(mon, "hp"))


def _hp(mon, maxhp):
    frac = mon.current_hp_fraction
    if mon.fainted or frac is None:
        return 0 if mon.fainted else maxhp
    return max(0, min(maxhp, round(maxhp * frac)))


# --- opponent set determinization -----------------------------------------------------------

class _SetSheet:
    """The gen9 randbats sets, with level / ordered movepool / abilities / tera types kept
    (knowledge.py drops these). Used to sample a concrete opponent set."""

    def __init__(self, path=_SETS_PATH):
        self.by_species = {}      # species_id -> {"level": int, "sets": [ {...} ]}
        try:
            raw = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            return
        for species, info in raw.items():
            sets = []
            for s in info.get("sets", []):
                sets.append({
                    "moves": [to_id_str(m) for m in s.get("movepool", [])],
                    "abilities": [to_id_str(a) for a in s.get("abilities", [])],
                    "tera": [t.lower() for t in s.get("teraTypes", [])] or ["normal"],
                    "role": s.get("role", ""),
                })
            self.by_species[to_id_str(species)] = {"level": info.get("level", 100), "sets": sets}

    def level(self, species_id):
        return (self.by_species.get(species_id) or {}).get("level", 100)

    def sample_set(self, species_id, revealed_ids):
        """Pick a set consistent with the revealed moves (uniform over the survivors)."""
        info = self.by_species.get(species_id)
        if not info or not info["sets"]:
            return None
        revealed = set(revealed_ids)
        survivors = [s for s in info["sets"] if revealed <= set(s["moves"])] or info["sets"]
        return random.choice(survivors)

    @staticmethod
    def sample_moves(revealed_ids, pool, k=4):
        """A plausible concrete moveset: the revealed moves plus a random draw from the rest of
        the role's movepool. 56% of gen9 randbats movepools exceed 4 moves, so truncating the
        pool (what this used to do) fixed the opponent's unrevealed moves to the same sheet-order
        prefix in every determinization -- the search could never anticipate the rest."""
        moves = list(dict.fromkeys(revealed_ids))[:k]
        rest = [m for m in pool if m not in moves]
        need = k - len(moves)
        if need > 0 and rest:
            moves += random.sample(rest, min(need, len(rest)))
        return moves

    def random_species(self, exclude):
        choices = [sp for sp in self.by_species if sp not in exclude and self.by_species[sp]["sets"]]
        return random.choice(choices) if choices else None


SETS = _SetSheet()


# --- joint set data (complete sets with observed counts) ------------------------------------
# Aggregated real generated randbats sets: "level,item,ability,move1..move4,teraType" -> count
# (from foul-play's public dataset, data.foulplay.cc). Sampling complete JOINT sets weighted
# by count preserves the move/item/ability/tera correlations that marginal probabilities lose
# and matches the true generator distribution. Validated need: at compute-matched settings we
# lost 1-11 to Foul Play, whose only relevant structural edge was this sampling.
_JOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data_joint_sets_gen9.json")


class _JointSets:
    def __init__(self, path=_JOINT_PATH):
        self.by_species = {}
        try:
            raw = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            return
        for sp, sets in raw.items():
            parsed = []
            for key, count in sets.items():
                parts = key.split(",")
                if len(parts) < 5:
                    continue
                try:
                    level = int(parts[0])
                except ValueError:
                    continue
                parsed.append({
                    "level": level,
                    "item": to_id_str(parts[1]),
                    "ability": to_id_str(parts[2]),
                    "tera": parts[-1].lower(),
                    "moves": [to_id_str(m) for m in parts[3:-1]],
                    "count": int(count),
                })
            if parsed:
                self.by_species[to_id_str(sp)] = parsed

    def sample(self, species_id, revealed_moves=(), item=None, ability=None,
               item_exclude=(), force_scarf=False):
        """One complete set consistent with the evidence, weighted by observed count.

        Filters are applied as a ladder that never empties the pool: each constraint
        narrows the candidates only if survivors remain (a stale dataset must degrade
        to 'less informed', never to 'no set')."""
        cands = self.by_species.get(species_id)
        if not cands:
            return None
        revealed = set(revealed_moves)

        def narrow(pool, pred):
            kept = [s for s in pool if pred(s)]
            return kept or pool

        cands = narrow(cands, lambda s: revealed <= set(s["moves"]))
        if item:
            cands = narrow(cands, lambda s: s["item"] == item)
        elif force_scarf:
            cands = narrow(cands, lambda s: s["item"] == "choicescarf")
        elif item_exclude:
            cands = narrow(cands, lambda s: s["item"] not in item_exclude)
        if ability:
            cands = narrow(cands, lambda s: s["ability"] == ability)
        weights = [s["count"] for s in cands]
        return random.choices(cands, weights=weights)[0]


JOINT = _JointSets()


# Real per-role item/ability/tera probabilities from the pkmn randbats *stats* feed (the set
# sheet has no items, so we previously guessed -> under-estimated Choice/Life Orb/weather damage
# and stayed in fatal matchups). Cached locally; refresh with the curl in the module docstring.
_STATS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data_randbats_stats_gen9.json")


def _weighted(dist):
    """Sample a key from {key: probability}, or None if empty."""
    if not dist:
        return None
    total = sum(dist.values()) or 1.0
    r = random.random() * total
    c = 0.0
    for k, p in dist.items():
        c += p
        if r <= c:
            return k
    return next(iter(dist))


class _StatsFeed:
    """pkmn randbats stats: species_id -> role -> {items/abilities/tera: {id: prob}} (+ a
    species-level marginal fallback for when the role isn't listed)."""

    def __init__(self, path=_STATS_PATH):
        self.by_species = {}
        try:
            raw = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            return
        for name, info in raw.items():
            roles = {}
            for role, ri in info.get("roles", {}).items():
                roles[role] = {
                    "items": {to_id_str(k): v for k, v in ri.get("items", {}).items()},
                    "abilities": {to_id_str(k): v for k, v in ri.get("abilities", {}).items()},
                    "tera": {k.lower(): v for k, v in ri.get("teraTypes", {}).items()},
                }
            self.by_species[to_id_str(name)] = {
                "roles": roles,
                "items": {to_id_str(k): v for k, v in info.get("items", {}).items()},
                "abilities": {to_id_str(k): v for k, v in info.get("abilities", {}).items()},
                "tera": {k.lower(): v for k, v in info.get("teraTypes", {}).items()},
            }

    def _bucket(self, species_id, role):
        sp = self.by_species.get(species_id)
        if not sp:
            return None
        return sp["roles"].get(role) or sp        # role-specific, else species marginal

    def item(self, species_id, role, exclude=()):
        b = self._bucket(species_id, role)
        if not b:
            return None
        dist = b["items"]
        if exclude:
            dist = {k: v for k, v in dist.items() if k not in exclude}
        return _weighted(dist)

    def ability(self, species_id, role):
        b = self._bucket(species_id, role)
        return _weighted(b["abilities"]) if b else None

    def tera(self, species_id, role):
        b = self._bucket(species_id, role)
        return _weighted(b["tera"]) if b else None


STATS = _StatsFeed()


def _guess_item(set_dict, ability_id):
    """Cheap item prior from role / ability (the set sheet has no items). Most gen9 randbats
    leads run Heavy-Duty Boots, so that's the default."""
    role = set_dict.get("role", "")
    if ability_id in ("protosynthesis", "quarkdrive"):
        return "boosterenergy"
    if role == "AV Pivot":
        return "assaultvest"
    if role in ("Wallbreaker",):
        return "lifeorb"
    if role in ("Bulky Support", "Fast Support", "Bulky Attacker"):
        return "leftovers"
    return "heavydutyboots"


# --- Pokemon construction -------------------------------------------------------------------

_CHOICE_ITEMS = ("choiceband", "choicespecs", "choicescarf")


def _moves_from_objs(move_objs, limit=4):
    pem = []
    for mv in list(move_objs)[:limit]:
        pp = getattr(mv, "current_pp", None)
        pem.append(PEMove(id=mv.id, pp=int(pp) if pp else 16, disabled=False))
    while len(pem) < 4:
        pem.append(PEMove(id="none", pp=0, disabled=True))
    return pem


def _moves_from_ids(move_ids, limit=4, only_enabled=None):
    """PEMove list; if only_enabled is a move id, every other move is disabled (choice lock --
    poke-engine does NOT enforce the lock from item + last_used_move, verified empirically, so
    we encode it in the move flags the search respects)."""
    pem = [PEMove(id=mid, pp=16, disabled=(only_enabled is not None and mid != only_enabled))
           for mid in list(move_ids)[:limit]]
    while len(pem) < 4:
        pem.append(PEMove(id="none", pp=0, disabled=True))
    return pem


def _last_used_str(mon, pe_moves):
    """poke-engine Side.last_used_move ('move:<index>' -- the engine panics on move names)."""
    last = getattr(mon, "last_move", None)
    if last is None:
        return "move:none"
    for i, m in enumerate(pe_moves):
        if m.id == last.id:
            return f"move:{i}"
    return "move:none"


def _own_pokemon(mon, moves_override=None):
    """Build an engine Pokemon for our side (everything known)."""
    maxhp = _maxhp(mon, own=True)
    moves = moves_override if moves_override is not None else _moves_from_objs(mon.moves.values())
    item = to_id_str(mon.item) if mon.item else "none"
    ability = to_id_str(mon.ability) if mon.ability else "none"
    tera = mon.tera_type.name.lower() if getattr(mon, "tera_type", None) else "typeless"
    return PEPokemon(
        id=to_id_str(mon.species), level=mon.level or 100,
        types=_types_tuple(mon), base_types=_base_types_tuple(mon),
        hp=_hp(mon, maxhp), maxhp=maxhp,
        ability=ability, base_ability=ability, item=item,
        nature="serious", evs=(85,) * 6,
        attack=_stat(mon, "atk", True), defense=_stat(mon, "def", True),
        special_attack=_stat(mon, "spa", True), special_defense=_stat(mon, "spd", True),
        speed=_stat(mon, "spe", True),
        status=_status_str(mon), moves=moves,
        tera_type=tera, terastallized=bool(getattr(mon, "is_terastallized", False)),
    )


def _opp_pokemon_determinized(mon, use_stats=True, used_since_switch=None, speed_hint=None,
                              use_joint=False):
    """Build an engine Pokemon for a revealed opponent mon, sampling its hidden set.
    use_stats toggles the real randbats item/ability/tera feed (for A/B testing it).
    used_since_switch: move ids this mon has used since it last switched in (active mon only) --
    2+ distinct moves rules out a Choice item, 1 move + a Choice item means it's locked.
    speed_hint: 'scarf'/'noscarf' verdict from observed turn order (randbats speed spreads are
    fixed, so outspeeding its known raw speed means Scarf) -- overrides item sampling.
    use_joint: sample a complete counted joint set (see _JointSets) instead of composing
    the set from marginals; falls through to the marginal path when no joint data exists."""
    species = to_id_str(mon.species)
    # Formes that appear mid-battle (Mimikyu-Busted, Ogerpon-*-Tera, etc.) aren't keys in the
    # randbats sheet/stats feed; fall back to the base species so we still get real sets.
    # (Observed live: mimikyubusted missed the sheet, got the no-set fallback with a 'typeless'
    # tera, and the engine's opponent-tera branches then erased its Ghost immunity.)
    sheet_id = species if species in SETS.by_species else to_id_str(mon.base_species)
    revealed = list(mon.moves.keys())
    # A mon that used two different moves without leaving the field can't be Choice-locked.
    multi_moved = bool(used_since_switch) and len(used_since_switch) >= 2
    item_exclude = _CHOICE_ITEMS if multi_moved else \
        ("choicescarf",) if speed_hint == "noscarf" else ()

    if use_joint:
        joint_id = species if species in JOINT.by_species else to_id_str(mon.base_species)
        js = JOINT.sample(
            joint_id, revealed_moves=revealed,
            item=to_id_str(mon.item) if mon.item else None,
            ability=to_id_str(mon.ability) if mon.ability else None,
            item_exclude=item_exclude,
            force_scarf=(speed_hint == "scarf" and not mon.item and not multi_moved))
        if js is not None:
            move_ids = (list(dict.fromkeys(revealed))
                        + [m for m in js["moves"] if m not in revealed])[:4]
            ability = to_id_str(mon.ability) if mon.ability else js["ability"]
            tera = (mon.tera_type.name.lower() if getattr(mon, "tera_type", None)
                    else js["tera"])
            item = to_id_str(mon.item) if mon.item else js["item"]
            if speed_hint == "scarf" and not mon.item and not multi_moved:
                item = "choicescarf"
            locked = None
            last = getattr(mon, "last_move", None)
            if (last is not None and not multi_moved and last.id in move_ids
                    and (item in _CHOICE_ITEMS or ability == "gorillatactics")):
                locked = last.id
            maxhp = _maxhp(mon, own=False)
            return PEPokemon(
                id=species, level=mon.level or js["level"],
                types=_types_tuple(mon), base_types=_base_types_tuple(mon),
                hp=_hp(mon, maxhp), maxhp=maxhp,
                ability=ability, base_ability=ability, item=item,
                nature="serious", evs=(85,) * 6,
                attack=_stat(mon, "atk", False), defense=_stat(mon, "def", False),
                special_attack=_stat(mon, "spa", False),
                special_defense=_stat(mon, "spd", False),
                speed=_stat(mon, "spe", False),
                status=_status_str(mon), moves=_moves_from_ids(move_ids, only_enabled=locked),
                tera_type=tera, terastallized=bool(getattr(mon, "is_terastallized", False)),
            )

    s = SETS.sample_set(sheet_id, revealed)
    if s is not None:
        role = s.get("role", "")
        st_item = STATS.item(sheet_id, role, exclude=item_exclude) if use_stats else None
        st_ability = STATS.ability(sheet_id, role) if use_stats else None
        st_tera = STATS.tera(sheet_id, role) if use_stats else None
        move_ids = SETS.sample_moves(revealed, s["moves"])
        ability = (to_id_str(mon.ability) if mon.ability
                   else st_ability or (random.choice(s["abilities"]) if s["abilities"] else "none"))
        tera = (mon.tera_type.name.lower() if getattr(mon, "tera_type", None)
                else st_tera or random.choice(s["tera"]))
        # Real item probability (Choice/Life Orb/etc.) -> correct incoming-damage estimate.
        item = to_id_str(mon.item) if mon.item else st_item or _guess_item(s, ability)
    else:
        move_ids = revealed
        ability = to_id_str(mon.ability) if mon.ability else "none"
        # Never guess 'typeless' tera: the engine explores opponent-tera branches, and a
        # typeless tera strips the mon's real typing (and its immunities) in those lines.
        tera = (mon.tera_type.name.lower() if getattr(mon, "tera_type", None)
                else _types_tuple(mon)[0])
        item = to_id_str(mon.item) if mon.item else "heavydutyboots"

    # Turn-order evidence: outsped its known raw speed -> model it as Scarf'd in every world
    # (unless the item is revealed, or it used 2+ moves without switching -- then the speed came
    # from an ability we don't model, and a Choice item is impossible anyway).
    if speed_hint == "scarf" and not mon.item and not multi_moved:
        item = "choicescarf"

    # Choice lock: if this world's item is a Choice item (or the ability is Gorilla Tactics) and
    # the mon has committed to a move since switching in, only that move is selectable.
    locked = None
    last = getattr(mon, "last_move", None)
    if (last is not None and not multi_moved and last.id in move_ids
            and (item in _CHOICE_ITEMS or ability == "gorillatactics")):
        locked = last.id

    maxhp = _maxhp(mon, own=False)
    return PEPokemon(
        id=species, level=mon.level or 100,
        types=_types_tuple(mon), base_types=_base_types_tuple(mon),
        hp=_hp(mon, maxhp), maxhp=maxhp,
        ability=ability, base_ability=ability, item=item,
        nature="serious", evs=(85,) * 6,
        attack=_stat(mon, "atk", False), defense=_stat(mon, "def", False),
        special_attack=_stat(mon, "spa", False), special_defense=_stat(mon, "spd", False),
        speed=_stat(mon, "spe", False),
        status=_status_str(mon), moves=_moves_from_ids(move_ids, only_enabled=locked),
        tera_type=tera, terastallized=bool(getattr(mon, "is_terastallized", False)),
    )


def _sampled_unrevealed_pokemon(species_id, use_stats=True, use_joint=False):
    """A wholly-unseen opponent bench slot: pick a random set for this species."""
    if use_joint:
        js = JOINT.sample(species_id)
        if js is not None:
            entry = _POKEDEX.get(species_id) or {}
            base = entry.get("baseStats", {})

            class _Shim:
                def __init__(s_):
                    s_.base_stats = {k: base.get(k, 80)
                                     for k in ("hp", "atk", "def", "spa", "spd", "spe")}
                    s_.level = js["level"]
                    s_.stats = {}
            shim = _Shim()
            maxhp = int(_estimate_stat(shim, "hp"))
            types = [t.lower() for t in entry.get("types", ["normal"])]
            if len(types) == 1:
                types.append("typeless")
            return PEPokemon(
                id=species_id, level=js["level"],
                types=(types[0], types[1]), base_types=(types[0], types[1]),
                hp=maxhp, maxhp=maxhp,
                ability=js["ability"], base_ability=js["ability"], item=js["item"],
                nature="serious", evs=(85,) * 6,
                attack=int(_estimate_stat(shim, "atk")),
                defense=int(_estimate_stat(shim, "def")),
                special_attack=int(_estimate_stat(shim, "spa")),
                special_defense=int(_estimate_stat(shim, "spd")),
                speed=int(_estimate_stat(shim, "spe")),
                status="None", moves=_moves_from_ids(js["moves"]),
                tera_type=js["tera"], terastallized=False,
            )
    info = SETS.by_species.get(species_id)
    if not info or not info["sets"]:
        return PEPokemon(id=species_id, level=SETS.level(species_id))
    s = random.choice(info["sets"])
    role = s.get("role", "")
    st_ability = STATS.ability(species_id, role) if use_stats else None
    st_tera = STATS.tera(species_id, role) if use_stats else None
    ability = st_ability or (random.choice(s["abilities"]) if s["abilities"] else "none")
    tera = st_tera or random.choice(s["tera"])
    entry = _POKEDEX.get(species_id) or {}
    base = entry.get("baseStats", {})
    lvl = info["level"]

    class _Shim:   # _estimate_stat only needs base_stats / level / stats
        def __init__(s_):
            s_.base_stats = {k: base.get(k, 80) for k in ("hp", "atk", "def", "spa", "spd", "spe")}
            s_.level = lvl
            s_.stats = {}
    shim = _Shim()
    maxhp = int(_estimate_stat(shim, "hp"))
    types = [t.lower() for t in entry.get("types", ["normal"])]
    if len(types) == 1:
        types.append("typeless")
    return PEPokemon(
        id=species_id, level=lvl, types=(types[0], types[1]), base_types=(types[0], types[1]),
        hp=maxhp, maxhp=maxhp, ability=ability, base_ability=ability,
        item=(STATS.item(species_id, role) if use_stats else None) or _guess_item(s, ability),
        nature="serious", evs=(85,) * 6,
        attack=int(_estimate_stat(shim, "atk")), defense=int(_estimate_stat(shim, "def")),
        special_attack=int(_estimate_stat(shim, "spa")), special_defense=int(_estimate_stat(shim, "spd")),
        speed=int(_estimate_stat(shim, "spe")),
        status="None", moves=_moves_from_ids(SETS.sample_moves([], s["moves"])),
        tera_type=tera, terastallized=False,
    )


# --- side / state assembly ------------------------------------------------------------------

def _dummy():
    return PEPokemon(id="pikachu", level=1, hp=0, maxhp=0)


def _remaining(start_turn, now, base, extended):
    """Turns remaining for a timed condition set on `start_turn`. If it has outlived its
    base duration it must be item-extended (Light Clay / weather rocks / Terrain Extender),
    so switch to the extended clock; never report less than 1 while it's still active."""
    left = base - (now - start_turn)
    if left < 1:
        left = extended - (now - start_turn)
    return max(1, left)


def _side_conditions(sc, now, protect=0):
    """poke-env side_conditions dict -> engine SideConditions. Hazards carry layer counts;
    timed conditions (screens/tailwind) carry real turns-remaining derived from the start
    turn poke-env records (they used to be hard-coded as freshly set, so a Reflect about to
    expire looked 5 turns strong). `protect` is the active mon's consecutive-protect count
    (poke-env mon.protect_counter): the engine halves... rather, cubes the success odds per
    consecutive use (verified: 100%/33%/11%), but only if it's told -- it was always 0, so
    back-to-back Protects looked guaranteed (7 failed Protects mined in one 30-game run)."""
    layers = {}
    starts = {}
    for cond, value in sc.items():
        name = cond.name
        if name in ("SPIKES", "TOXIC_SPIKES"):
            layers[name] = value
        else:
            starts[name] = value    # start turn for non-stackable conditions
    def timed(name, base, extended=None):
        if name not in starts:
            return 0
        return _remaining(starts[name], now, base, extended or base)
    return SideConditions(
        stealth_rock=1 if "STEALTH_ROCK" in starts else 0,
        spikes=layers.get("SPIKES", 0),
        toxic_spikes=layers.get("TOXIC_SPIKES", 0),
        sticky_web=1 if "STICKY_WEB" in starts else 0,
        tailwind=timed("TAILWIND", 4),
        reflect=timed("REFLECT", 5, 8),
        light_screen=timed("LIGHT_SCREEN", 5, 8),
        aurora_veil=timed("AURORA_VEIL", 5, 8),
        safeguard=timed("SAFEGUARD", 5),
        mist=timed("MIST", 5),
        protect=max(0, int(protect)),
    )


def _boosts(mon):
    b = mon.boosts
    return dict(attack_boost=b["atk"], defense_boost=b["def"], special_attack_boost=b["spa"],
                special_defense_boost=b["spd"], speed_boost=b["spe"],
                accuracy_boost=b.get("accuracy", 0), evasion_boost=b.get("evasion", 0))


# poke-env Effect name -> poke-engine volatile-status string. Curated to persistent effects that
# change the value of a position (chip, passive heal, trap, immunity). Taunt/Encore are handled
# separately below because the engine also needs their duration counters (and Encore has a hard
# consistency invariant); yawn/confusion stay excluded for now.
_VOLATILE_MAP = {
    "SUBSTITUTE": "substitute", "LEECH_SEED": "leechseed", "SALT_CURE": "saltcure",
    "CURSE": "curse", "AQUA_RING": "aquaring", "INGRAIN": "ingrain", "MAGNET_RISE": "magnetrise",
    "TORMENT": "torment", "ATTRACT": "attract", "HEAL_BLOCK": "healblock",
    "NIGHTMARE": "nightmare", "DESTINY_BOND": "destinybond", "FOCUS_ENERGY": "focusenergy",
    "PARTIALLY_TRAPPED": "partiallytrapped",
}


def _volatile_set(mon):
    return {s for e in mon.effects if (s := _VOLATILE_MAP.get(e.name))}


def _duration_volatiles(mon, last_used):
    """(extra volatiles, VolatileStatusDurations) for Taunt/Encore on the active mon.

    poke-env keeps a turns-elapsed counter for both (end_turn increments turn-countable
    effects), and poke-engine's durations count up the same way, expiring at 3 -- so the
    counter passes through, clamped so a stale count can't make the engine expire it on
    the wrong side of a turn. HARD INVARIANT (verified: the engine panics otherwise):
    'encore' may only be set when last_used_move is a real 'move:<i>'."""
    vols = set()
    kw = {}
    for e, count in mon.effects.items():
        if e.name == "TAUNT":
            vols.add("taunt")
            kw["taunt"] = max(0, min(int(count), 2))
        elif e.name == "ENCORE" and last_used.startswith("move:") and last_used != "move:none":
            vols.add("encore")
            kw["encore"] = max(0, min(int(count), 2))
    return vols, VolatileStatusDurations(**kw)


def _sub_health(mon, maxhp):
    """Substitute is made at 1/4 max HP; we can't see its current HP, so use that as the
    estimate (Foul Play does the same when it hasn't seen the sub take a hit)."""
    if any(e.name == "SUBSTITUTE" for e in mon.effects):
        return max(1, maxhp // 4)
    return 0


def _delayed(pending, side, now):
    """Engine (wish, future_sight) tuples for one side from the tracked pending effects.
    Wish heals at the end of the turn it 'lands on'; Future Sight hits one turn later."""
    pending = pending or {}
    wish_turn, wish_amt = pending.get(f"{side}_wish", (0, 0))
    wish = (1, int(wish_amt)) if wish_turn == now else (0, 0)
    fs_turn = pending.get(f"{side}_fs", 0)
    fs = (max(0, min(2, fs_turn - now + 1)), "0") if fs_turn >= now else (0, "0")
    return wish, fs


def _our_side(battle, pending=None):
    active = battle.active_pokemon
    active_moves = _moves_from_objs(battle.available_moves)
    active_pe = _own_pokemon(active, moves_override=active_moves)
    bench = [m for m in battle.team.values() if m is not active]
    pkmn = [active_pe] + [_own_pokemon(m) for m in bench]
    while len(pkmn) < 6:
        pkmn.append(_dummy())
    wish, fs = _delayed(pending, battle.player_role or "p1", battle.turn)
    last_used = _last_used_str(active, active_moves)
    dur_vols, durations = _duration_volatiles(active, last_used)
    return Side(
        pokemon=pkmn[:6],
        side_conditions=_side_conditions(battle.side_conditions, battle.turn,
                                         protect=getattr(active, "protect_counter", 0)),
        active_index="0", volatile_status_durations=durations,
        wish=wish, future_sight=fs, volatile_statuses=_volatile_set(active) | dur_vols,
        substitute_health=_sub_health(active, _maxhp(active, own=True)),
        last_used_move=last_used,
        switch_out_move_second_saved_move="none",
        force_switch=bool(getattr(battle, "force_switch", False)),
        force_trapped=bool(getattr(battle, "trapped", False)),
        **_boosts(active),
    )


def _opp_side(battle, use_stats=True, opp_used_since_switch=None, opp_speed_hints=None,
              pending=None, use_joint=False):
    hints = opp_speed_hints or {}
    active = battle.opponent_active_pokemon
    active_pe = _opp_pokemon_determinized(active, use_stats, used_since_switch=opp_used_since_switch,
                                          speed_hint=hints.get(to_id_str(active.species)),
                                          use_joint=use_joint)
    bench = [m for m in battle.opponent_team.values() if m is not active]
    pkmn = [active_pe] + [_opp_pokemon_determinized(m, use_stats,
                                                    speed_hint=hints.get(to_id_str(m.species)),
                                                    use_joint=use_joint)
                          for m in bench]

    # Fill unseen bench up to 6 with sampled species (so endgame / faint-count eval is sane).
    seen = {to_id_str(active.species)} | {to_id_str(m.species) for m in bench}
    while len(pkmn) < 6:
        sp = SETS.random_species(seen)
        if sp is None:
            pkmn.append(_dummy())
            continue
        seen.add(sp)
        pkmn.append(_sampled_unrevealed_pokemon(sp, use_stats, use_joint=use_joint))
    opp_role = "p2" if (battle.player_role or "p1") == "p1" else "p1"
    wish, fs = _delayed(pending, opp_role, battle.turn)
    last_used = _last_used_str(active, active_pe.moves)
    dur_vols, durations = _duration_volatiles(active, last_used)
    return Side(
        pokemon=pkmn[:6],
        side_conditions=_side_conditions(battle.opponent_side_conditions, battle.turn,
                                         protect=getattr(active, "protect_counter", 0)),
        active_index="0", volatile_status_durations=durations,
        wish=wish, future_sight=fs, volatile_statuses=_volatile_set(active) | dur_vols,
        substitute_health=_sub_health(active, _maxhp(active, own=False)),
        last_used_move=last_used,
        switch_out_move_second_saved_move="none",
        **_boosts(active),
    )


def _weather_str(battle):
    """(engine weather name, turns remaining). Ability weathers (Desolate Land etc.) don't
    time out; item extensions are inferred by outliving the base 5 turns."""
    for w, start in battle.weather.items():
        s = _WEATHER.get(w.name)
        if s:
            if s in ("harshsun", "heavyrain"):
                return s, -1
            return s, _remaining(start, battle.turn, 5, 8)
    return "none", -1


def _terrain_str(battle):
    for f, start in battle.fields.items():
        s = _TERRAIN.get(f.name)
        if s:
            return s, _remaining(start, battle.turn, 5, 8)
    return "none", 0


def _trick_room(battle):
    for f, start in battle.fields.items():
        if f.name == "TRICK_ROOM":
            return True, _remaining(start, battle.turn, 5, 5)
    return False, 0


def build_state(battle, use_stats=True, opp_used_since_switch=None, opp_speed_hints=None,
                pending=None, use_joint=False):
    """A determinized poke-engine State for the current position. Call repeatedly to get
    different opponent-set samples. use_stats toggles the real randbats item/ability/tera feed.
    opp_used_since_switch: move ids the opponent's active has used since switching in (the
    caller tracks this across turns) -- drives Choice-lock inference.
    opp_speed_hints: {species_id: 'scarf'|'noscarf'} turn-order verdicts (see engine_search).
    pending: tracked Wish/Future Sight effects ({'p1_wish': (turn, amt), 'p2_fs': turn, ...}).
    use_joint: sample complete counted joint sets (distributionally correct) instead of
    composing sets from marginals."""
    weather, weather_left = _weather_str(battle)
    terrain, terrain_left = _terrain_str(battle)
    tr, tr_left = _trick_room(battle)
    return State(
        side_one=_our_side(battle, pending),
        side_two=_opp_side(battle, use_stats, opp_used_since_switch, opp_speed_hints, pending,
                           use_joint=use_joint),
        weather=weather, weather_turns_remaining=weather_left,
        terrain=terrain, terrain_turns_remaining=terrain_left,
        trick_room=tr, trick_room_turns_remaining=tr_left, team_preview=False,
    )
