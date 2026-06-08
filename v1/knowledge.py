"""Opponent set prediction + damage estimation -- the "Pokemon understanding" layer.

A strong player doesn't just react to what the opponent has *revealed*; they know the
common sets. See a Haxorus and you assume Outrage / Dragon Dance before it clicks them.
This module gives the agent that same prior.

WHY THIS IS TRACTABLE (and not cheating):
  * gen9randombattle sets are drawn from a FIXED, PUBLIC pool shipped with the Showdown
    server (data/random-battles/gen9/sets.json). We read the same set sheet every human
    ladder player has memorised -- a prior over the distribution, NOT the opponent's
    actual hidden choices.
  * The damage estimate uses only: the public gen-9 damage formula, OUR OWN known stats,
    and ESTIMATED opponent stats (from public base stats + level + standard randbats
    spread). poke-env never even exposes the opponent's true hidden stats, so we *can't*
    peek. It's just a sharper version of the base_power x type_multiplier proxy the
    observation already carries; the agent still has to learn the actual policy.

DESIGN NOTE -- swappable for the future OU bot:
  Everything funnels through `MovesetPredictor.predict_moves(species, revealed)`. For
  random battles that's a table lookup (`RandbatsKnowledge`). The planned `ppo_competitive`
  OU bot has no premade sets, so it will provide a different predictor (priors from typing
  / base stats / usage data) behind this same interface. The env code below doesn't care
  which one it gets.
"""

import json
import os
from functools import lru_cache

from poke_env.battle import Move
from poke_env.data import to_id_str

GEN = 9

# Fixed role vocabulary from gen9 randbats (stable order -> stable feature indices).
ROLE_NAMES = [
    "AV Pivot", "Bulky Attacker", "Bulky Setup", "Bulky Support", "Fast Attacker",
    "Fast Bulky Setup", "Fast Support", "Setup Sweeper", "Tera Blast user", "Wallbreaker",
]

TYPE_NAMES = [
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE", "FIGHTING", "POISON",
    "GROUND", "FLYING", "PSYCHIC", "BUG", "ROCK", "GHOST", "DRAGON", "DARK", "STEEL", "FAIRY",
]

HAZARD_CONDS = {"STEALTH_ROCK", "SPIKES", "TOXIC_SPIKES", "STICKY_WEB"}
PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "chillyreception"}

_SETS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "server", "data", "random-battles", "gen9", "sets.json",
)

N_THREAT_FLAGS = 6  # priority, recovery, hazard, setup, status, pivot


@lru_cache(maxsize=None)
def get_move(move_id):
    """Build (and cache) a Move object from its id, or None if the id is unknown."""
    try:
        return Move(move_id, gen=GEN)
    except Exception:
        return None


def safe_priority(move):
    """move.priority, but 0 for pseudo-moves (Struggle/recharge) whose data omits it."""
    try:
        return move.priority
    except (KeyError, AttributeError):
        return 0


def _estimate_stat(mon, key):
    """A stat value: the real one if known (our mons), else estimated from base+level.

    Estimate assumes the randbats-ish convention of 31 IVs, 85 EVs, neutral nature -- good
    enough for a feature the network learns from (it just needs to be consistent)."""
    known = (mon.stats or {}).get(key) if hasattr(mon, "stats") else None
    if known:
        return known
    base = mon.base_stats[key]
    lvl = mon.level or 100
    point = (2 * base + 31 + 85 // 4) * lvl // 100
    if key == "hp":
        return point + lvl + 10
    return point + 5


def _max_hp(mon):
    if getattr(mon, "max_hp", 0):
        return mon.max_hp
    return _estimate_stat(mon, "hp")


def estimate_damage_fraction(attacker, move, defender):
    """Estimated damage of `move` from `attacker` into `defender`, as a fraction of the
    defender's max HP (0 for status/immune; clamped to 1.5 for overkill).

    Wrapped defensively: it runs in the hot loop on live battle moves, and a single odd
    move's data shouldn't crash a multi-hour training run -- a missed estimate just reads 0.
    """
    try:
        return _estimate_damage_fraction(attacker, move, defender)
    except Exception:
        return 0.0


def _estimate_damage_fraction(attacker, move, defender):
    if attacker is None or defender is None or move is None or move.base_power <= 0:
        return 0.0
    physical = move.category.name == "PHYSICAL"
    a = _estimate_stat(attacker, "atk" if physical else "spa")
    d = _estimate_stat(defender, "def" if physical else "spd")
    if d <= 0:
        return 0.0
    lvl = attacker.level or 100
    base = ((2 * lvl / 5 + 2) * move.base_power * a / d) / 50 + 2

    type_mult = defender.damage_multiplier(move)            # type chart (0 if immune)
    stab = 1.5 if move.type in [t for t in attacker.types if t] else 1.0
    dmg = base * type_mult * stab * 0.925                   # average damage roll

    # A burned physical attacker hits for half (unless Guts, which we don't track here).
    if physical and attacker.status is not None and attacker.status.name == "BRN":
        dmg *= 0.5

    maxhp = _max_hp(defender)
    return min(dmg / maxhp, 1.5) if maxhp else 0.0


class RandbatsKnowledge:
    """Set predictor backed by the gen9 randbats sets file."""

    def __init__(self, path=_SETS_PATH):
        self._sets = {}      # species_id -> list of {moves:set, role:str}
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            for species, info in raw.items():
                parsed = []
                for s in info.get("sets", []):
                    parsed.append({
                        "moves": {to_id_str(m) for m in s.get("movepool", [])},
                        "role": s.get("role"),
                        "abilities": {to_id_str(a) for a in s.get("abilities", [])},
                    })
                self._sets[to_id_str(species)] = parsed
        except (OSError, ValueError) as e:
            print(f"[knowledge] could not load randbats sets ({e}); predictions disabled.")

    # --- prediction ---------------------------------------------------------

    def predict_sets(self, species, revealed_ids):
        """The sets consistent with the moves revealed so far (set-narrowing).

        Once Haxorus shows Outrage, only the set containing Outrage survives -> its other
        moves become the prediction. If nothing matches (odd data), keep all sets."""
        sets = self._sets.get(to_id_str(species or ""), [])
        if not sets:
            return []
        revealed = set(revealed_ids)
        consistent = [s for s in sets if revealed <= s["moves"]]
        return consistent or sets

    def predict_moves(self, species, revealed_ids):
        """Union of the candidate moves across the still-possible sets (as ids)."""
        ids = set()
        for s in self.predict_sets(species, revealed_ids):
            ids |= s["moves"]
        ids |= set(revealed_ids)  # always include what we've actually seen
        return ids

    # --- feature vectors for the opponent's ACTIVE Pokemon ------------------
    #
    # These are PROBABILISTIC, not binary: a value is the fraction of the still-possible
    # sets that have the trait. Before Haxorus reveals anything that's 0.5 for an Outrage
    # (1 of its 2 sets) but 1.0 for Close Combat (both sets). Revealing a move narrows the
    # surviving sets, which is exactly the Bayesian update -- P(Outrage) jumps to 1.0 once
    # we see it, Set-B-only moves drop to 0. (We treat each set's listed movepool as its
    # moveset; a few sets list >4 options, a second-order nuance we don't model.)

    def _surviving_sets(self, opp):
        if opp is None:
            return []
        return self.predict_sets(opp.species, set(opp.moves.keys()))

    def move_probs(self, opp):
        """move_id -> probability the opponent's active is carrying it."""
        sets = self._surviving_sets(opp)
        if not sets:
            return {mid: 1.0 for mid in (opp.moves.keys() if opp else [])}
        n = len(sets)
        probs = {}
        for s in sets:
            for mid in s["moves"]:
                probs[mid] = probs.get(mid, 0.0) + 1.0 / n
        for mid in opp.moves.keys():     # moves we've actually seen are certain
            probs[mid] = 1.0
        return probs

    def predicted_abilities(self, opp):
        """ability_id -> probability the opponent's active has it.

        1.0 on the revealed ability once known. Otherwise it's read off the still-possible
        randbats sets (each set lists its candidate abilities), narrowed by revealed moves --
        so an Orthworm, whose every set runs Earth Eater, reads as a CERTAIN Ground immunity
        even before it's shown. Falls back to the dex's possible abilities (uniform) for a
        species we have no set data for."""
        if opp is None:
            return {}
        if opp.ability:                                  # revealed -> certain
            return {to_id_str(opp.ability): 1.0}
        sets = self._surviving_sets(opp)
        if not sets:
            poss = [to_id_str(a) for a in (opp.possible_abilities or [])]
            return {a: 1.0 / len(poss) for a in poss} if poss else {}
        n = len(sets)
        probs = {}
        for s in sets:
            ab = s.get("abilities") or set()
            if not ab:
                continue
            w = 1.0 / (n * len(ab))                      # split each set's mass over its candidates
            for a in ab:
                probs[a] = probs.get(a, 0.0) + w
        return probs

    def predicted_coverage(self, opp):
        """18-d: probability the opponent carries an attacking move of each type."""
        vec = [0.0] * 18
        sets = self._surviving_sets(opp)
        if not sets:
            return vec
        n = len(sets)
        for s in sets:
            types_in_set = set()
            for mid in s["moves"]:
                mv = get_move(mid)
                if mv is not None and mv.base_power > 0 and mv.type is not None:
                    types_in_set.add(mv.type.name)
            for name in types_in_set:
                if name in TYPE_NAMES:
                    vec[TYPE_NAMES.index(name)] += 1.0 / n
        return vec

    def role_flags(self, opp):
        """Probability of each role across the still-possible sets."""
        vec = [0.0] * len(ROLE_NAMES)
        sets = self._surviving_sets(opp)
        if not sets:
            return vec
        n = len(sets)
        for s in sets:
            role = s.get("role")
            if role in ROLE_NAMES:
                vec[ROLE_NAMES.index(role)] += 1.0 / n
        return vec

    @staticmethod
    def _set_threats(moves):
        """Booleans [priority, recovery, hazard, setup, status, pivot] for one set."""
        prio = recov = hazard = setup = status = pivot = False
        for mid in moves:
            mv = get_move(mid)
            if mv is None:
                continue
            if mv.base_power > 0 and safe_priority(mv) > 0:
                prio = True
            if mv.heal and mv.heal > 0:
                recov = True
            if mv.side_condition is not None and mv.side_condition.name in HAZARD_CONDS:
                hazard = True
            if mv.boosts and any(v > 0 for v in mv.boosts.values()):
                setup = True
            if mv.status is not None:
                status = True
            if mid in PIVOT_MOVES:
                pivot = True
        return [prio, recov, hazard, setup, status, pivot]

    def threat_flags(self, opp):
        """Probability of each threat category across the still-possible sets."""
        sets = self._surviving_sets(opp)
        if not sets:
            return [0.0] * N_THREAT_FLAGS
        n = len(sets)
        acc = [0.0] * N_THREAT_FLAGS
        for s in sets:
            for i, present in enumerate(self._set_threats(s["moves"])):
                if present:
                    acc[i] += 1.0 / n
        return acc

    def predicted_incoming(self, opp, my_active, immune_types=None):
        """Scariest *probability-weighted* incoming hit (HP fraction): a 50%-likely nuke
        counts as a real-but-discounted threat, not a guaranteed one.

        immune_types (a set of type NAMEs) lets a caller drop move types our active is immune
        to via its ABILITY -- e.g. a Levitate mon shouldn't fear Ground. Default None keeps the
        original behaviour exactly, so the observation features built on this are unchanged."""
        if opp is None or my_active is None:
            return 0.0
        worst = 0.0
        for mid, p in self.move_probs(opp).items():
            mv = get_move(mid)
            if mv is not None and mv.base_power > 0:
                if immune_types and mv.type is not None and mv.type.name in immune_types:
                    continue
                worst = max(worst, p * estimate_damage_fraction(opp, mv, my_active))
        return worst


# Singleton: load the set sheet once at import.
KNOWLEDGE = RandbatsKnowledge()
