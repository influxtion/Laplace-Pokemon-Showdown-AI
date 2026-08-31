"""Opponent set prediction and damage estimation.

A good player knows the common sets rather than waiting to be shown: you see a Haxorus and
assume Outrage / Dragon Dance before it clicks them. This is that prior.

Not cheating, for the record. The gen9randombattle sets come from the public pool shipped
with the server (data/random-battles/gen9/sets.json) that any ladder player has seen, so
this is a prior over the distribution, not the opponent's hidden choices. The damage
estimate uses only the public gen-9 formula, our own known stats, and opponent stats
estimated from public base stats + level + the standard randbats spread -- poke-env never
exposes the opponent's real stats anyway.

Everything goes through RandbatsKnowledge, which is a table lookup. A format without
premade sets can swap in a different predictor (priors from typing / base stats / usage)
behind the same interface without touching callers.
"""

import json
import os
from functools import lru_cache

from poke_env.battle import Move
from poke_env.data import to_id_str

from laplace import paths

GEN = 9

# Role vocabulary from gen9 randbats. Order is FIXED -- feature indices depend on it.
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

_SETS_PATH = paths.RANDBATS_SETS

N_THREAT_FLAGS = 6  # priority, recovery, hazard, setup, status, pivot


@lru_cache(maxsize=None)
def get_move(move_id):
    """Build (and cache) a Move from its id. None if the id is unknown."""
    try:
        return Move(move_id, gen=GEN)
    except Exception:
        return None


def safe_priority(move):
    """move.priority, but 0 for pseudo-moves (Struggle, recharge) whose data omits it."""
    try:
        return move.priority
    except (KeyError, AttributeError):
        return 0


def _estimate_stat(mon, key):
    """Real value if known (our mons), else estimated from base + level.

    Assumes 31 IVs / 85 EVs / neutral nature -- the randbats convention. Being consistent
    everywhere matters more than being exactly right."""
    known = (mon.stats or {}).get(key) if hasattr(mon, "stats") else None
    if known:
        return known
    base = mon.base_stats[key]
    lvl = mon.level or 100
    point = (2 * base + 31 + 85 // 4) * lvl // 100
    if key == "hp":
        return point + lvl + 10
    return point + 5


def _max_hp(mon, knowledge=None):
    if getattr(mon, "max_hp", 0):
        return mon.max_hp
    return (knowledge or KNOWLEDGE).estimate_stat(mon, "hp")


# Abilities that nullify a whole attacking type (immunity or absorb-heal). damage_multiplier
# misses all of these -- an Earth Eater Orthworm reads as taking full Earthquake damage --
# and the engine only prices the absorb inside its own worlds, so when visit shares flatten
# in a bad position the pooled argmax can land on a move that literally heals the target.
_ABSORB_ABILITY_TYPE = {
    "levitate": "GROUND", "eartheater": "GROUND",
    "waterabsorb": "WATER", "dryskin": "WATER", "stormdrain": "WATER",
    "voltabsorb": "ELECTRIC", "lightningrod": "ELECTRIC", "motordrive": "ELECTRIC",
    "flashfire": "FIRE", "wellbakedbody": "FIRE",
    "sapsipper": "GRASS",
}

# Mold Breaker-class attackers punch through every ability above.
_ABILITY_IGNORING = {"moldbreaker", "teravolt", "turboblaze"}

# Moves whose effective type/interaction breaks the simple table: Tera Blast's type is the
# user's tera type, Thousand Arrows hits through Ground immunities.
_ABSORB_CHECK_SKIP = {"terablast", "thousandarrows"}


# Abilities that move a Pokemon out of its move's printed priority bracket, and the moves
# they apply to. Move.priority is the move's OWN priority and knows nothing about these, so
# two moves that read as an equal bracket can still have been ordered by ability rather than
# by speed -- which is exactly the assumption engine_search._infer_scarf runs on.
_PRIORITY_ABILITIES = {
    "prankster":     lambda mv: mv.category.name == "STATUS",       # +1
    "galewings":     lambda mv: mv.type is not None and mv.type.name == "FLYING",
    "triage":        lambda mv: (mv.heal or 0) > 0,                 # +3
    "myceliummight": lambda mv: mv.category.name == "STATUS",       # -1 (moves last)
    "quickdraw":     lambda mv: mv.category.name != "STATUS",       # 30% of the time
    "stall":         lambda mv: True,                               # always last
}


def priority_ability_possible(mon, move, knowledge=None):
    """True if `mon` might be moving out of `move`'s printed priority bracket by ability.

    POSSIBILITY, not certainty, on purpose -- the opposite standard from _ability_absorbs.
    That one gates a veto, where acting on a guess is the expensive mistake; this one gates
    whether a turn-order observation is INTERPRETABLE, where the expensive mistake is
    reading a Prankster Klefki's Spikes as evidence of a Choice Scarf and then handing every
    determinized world a Scarf (and a Choice lock) for the rest of the game. Anything short
    of "no set here has a priority ability that applies" means the observation says nothing
    about speed."""
    if mon is None or move is None:
        return False
    if mon.ability:
        test = _PRIORITY_ABILITIES.get(to_id_str(mon.ability))
        return bool(test and test(move))
    for ability, p in (knowledge or KNOWLEDGE).predicted_abilities(mon).items():
        test = _PRIORITY_ABILITIES.get(ability)
        if p > 0 and test and test(move):
            return True
    return False


def _ability_absorbs(attacker, move, defender, knowledge=None):
    """True iff the defender's ability CERTAINLY nullifies this move: revealed, or the set
    prior puts >= 0.99 on an ability that absorbs the move's type. Any uncertainty -> False.
    Never veto on a guess.

    What "certain" means is the PRIOR's job, and the two priors reach it differently: Random
    Battle can say so once every still-possible generated set agrees, because the generator
    is ground truth, while OU only has usage statistics and reserves certainty for a species
    with a single legal ability. See laplace.agent.metagame for the contract."""
    if move.id in _ABSORB_CHECK_SKIP or move.type is None:
        return False
    if attacker is not None and to_id_str(attacker.ability or "") in _ABILITY_IGNORING:
        return False
    mtype = move.type.name
    if defender.ability:
        return _ABSORB_ABILITY_TYPE.get(to_id_str(defender.ability)) == mtype
    probs = (knowledge or KNOWLEDGE).predicted_abilities(defender)
    absorbed = sum(p for a, p in probs.items() if _ABSORB_ABILITY_TYPE.get(a) == mtype)
    return absorbed >= 0.99


def move_nullified(attacker, move, defender, knowledge=None):
    """True iff an attacking move is CERTAIN to deal zero: type-chart immunity on the
    revealed typing, or an absorb ability per _ability_absorbs. Deterministic hard
    knowledge only, which is what makes it safe for a guard to demote on."""
    try:
        if move is None or defender is None or move.base_power <= 0:
            return False
        if move.id in _ABSORB_CHECK_SKIP:
            return False
        if defender.damage_multiplier(move) == 0:
            return True
        return _ability_absorbs(attacker, move, defender, knowledge)
    except Exception:
        return False


def estimate_damage_fraction(attacker, move, defender, knowledge=None):
    """Estimated damage as a fraction of the defender's max HP. 0 for status / immune /
    certainly-absorbed, clamped to 1.5 for overkill.

    Runs in the hot loop on live moves, so it swallows errors: one odd move's data
    shouldn't crash a multi-hour run, and a missed estimate just reads 0.
    """
    try:
        if _ability_absorbs(attacker, move, defender, knowledge):
            return 0.0
        return _estimate_damage_fraction(attacker, move, defender, knowledge)
    except Exception:
        return 0.0


def _estimate_damage_fraction(attacker, move, defender, knowledge=None):
    if attacker is None or defender is None or move is None or move.base_power <= 0:
        return 0.0
    kb = knowledge or KNOWLEDGE
    physical = move.category.name == "PHYSICAL"
    a = kb.estimate_stat(attacker, "atk" if physical else "spa")
    d = kb.estimate_stat(defender, "def" if physical else "spd")
    if d <= 0:
        return 0.0
    lvl = attacker.level or 100
    base = ((2 * lvl / 5 + 2) * move.base_power * a / d) / 50 + 2

    type_mult = defender.damage_multiplier(move)            # type chart (0 if immune)
    stab = 1.5 if move.type in [t for t in attacker.types if t] else 1.0
    dmg = base * type_mult * stab * 0.925                   # average roll

    # Burn halves physical damage. Guts not modelled.
    if physical and attacker.status is not None and attacker.status.name == "BRN":
        dmg *= 0.5

    maxhp = _max_hp(defender, kb)
    return min(dmg / maxhp, 1.5) if maxhp else 0.0


class RandbatsKnowledge:
    """Set predictor backed by the gen9 randbats sets file."""

    def __init__(self, path=_SETS_PATH):
        self._sets = {}      # species_id -> [{moves: set, role: str, abilities: set}]
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

    # --- the Knowledge protocol (see laplace.agent.metagame) ------------------

    @staticmethod
    def estimate_stat(mon, key):
        """One stat: real if we know it, else the randbats convention (85 EVs / 31 IVs /
        neutral). Public name for _estimate_stat, which the rest of this file and the
        adapter still call directly."""
        return _estimate_stat(mon, key)

    @staticmethod
    def speed_bounds(mon):
        """(slowest, fastest) unboosted Speed -- a single point in Random Battle.

        Every generated set runs the same spread, so 'could it have outsped without a Choice
        Scarf?' has one answer and the bound is degenerate. It exists so the Scarf inference
        can be written once and stay correct in a format where spreads are chosen."""
        spe = int(_estimate_stat(mon, "spe"))
        return spe, spe

    # --- prediction -----------------------------------------------------------

    def predict_sets(self, species, revealed_ids):
        """The sets consistent with the moves revealed so far.

        Once Haxorus shows Outrage, only sets containing Outrage survive and their other
        moves become the prediction. If nothing matches (odd data), keep all sets."""
        sets = self._sets.get(to_id_str(species or ""), [])
        if not sets:
            return []
        revealed = set(revealed_ids)
        consistent = [s for s in sets if revealed <= s["moves"]]
        return consistent or sets

    def predict_moves(self, species, revealed_ids):
        """Union of candidate moves across the still-possible sets."""
        ids = set()
        for s in self.predict_sets(species, revealed_ids):
            ids |= s["moves"]
        ids |= set(revealed_ids)  # always include what we've actually seen
        return ids

    # --- feature vectors for the opponent's active ----------------------------
    #
    # Probabilities, not flags: each value is the fraction of still-possible sets with the
    # trait. Before Haxorus reveals anything, Outrage reads 0.5 (1 of its 2 sets) and Close
    # Combat 1.0 (both). Revealing a move narrows the surviving sets -- the Bayesian update
    # -- so P(Outrage) jumps to 1.0 once seen. We treat each set's movepool as its moveset;
    # a few list >4 options, which we don't model.

    def _surviving_sets(self, opp):
        if opp is None:
            return []
        return self.predict_sets(opp.species, set(opp.moves.keys()))

    def move_probs(self, opp):
        """move_id -> P(the opponent's active is carrying it)."""
        sets = self._surviving_sets(opp)
        if not sets:
            return {mid: 1.0 for mid in (opp.moves.keys() if opp else [])}
        n = len(sets)
        probs = {}
        for s in sets:
            for mid in s["moves"]:
                probs[mid] = probs.get(mid, 0.0) + 1.0 / n
        for mid in opp.moves.keys():     # seen moves are certain
            probs[mid] = 1.0
        return probs

    def predicted_abilities(self, opp):
        """ability_id -> P(the opponent's active has it).

        1.0 once revealed. Otherwise read off the still-possible sets narrowed by revealed
        moves, so an Orthworm -- always Earth Eater -- reads as a certain Ground immunity
        before it shows. Falls back to a uniform split over the dex's possible abilities
        for a species with no set data."""
        if opp is None:
            return {}
        if opp.ability:                                  # revealed, so certain
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
            w = 1.0 / (n * len(ab))                      # split each set's mass over its abilities
            for a in ab:
                probs[a] = probs.get(a, 0.0) + w
        return probs

    def predicted_coverage(self, opp):
        """18-d: P(the opponent carries an attacking move of each type)."""
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
        """P(each role) across the still-possible sets."""
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
        """P(each threat category) across the still-possible sets."""
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
        """Scariest probability-weighted incoming hit, as an HP fraction. A 50%-likely nuke
        counts as a discounted threat, not a guaranteed one.

        immune_types drops move types our active is immune to via its ability -- a Levitate
        mon shouldn't fear Ground. Default None preserves the old behaviour, so the
        observation features built on this are unchanged."""
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


# Load the set sheet once at import.
KNOWLEDGE = RandbatsKnowledge()
