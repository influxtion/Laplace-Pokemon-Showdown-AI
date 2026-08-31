r"""poke-env Battle -> poke-engine State for Gen 9 OU, with the opponent's set determinized.

Same contract as `poke_engine_adapter.build_state`, and the same interface the search calls
N times a turn: every call returns one complete, legal position consistent with everything
the server has revealed, and the search pools the results. Everything that merely encodes
the board comes from `agent.state_common` and is shared with the Random Battle adapter; what
is rewritten here is the guessing.

Three things change, and only three:

1. THE BENCH IS KNOWN. Team preview names all six opposing Pokemon before the first turn.
   The randbats adapter has to invent unseen bench slots from the whole dex -- the single
   largest source of noise in its worlds -- and here that guess disappears entirely. What
   stays hidden is each Pokemon's set, not its identity.

2. SETS ARE CHOSEN, NOT GENERATED. There is no generator sheet to sample from, so the prior
   is the metagame itself: Smogon analysis sets first (complete and correlated), usage
   marginals as the fallback. See laplace.ou.usage.

3. STATS ARE PART OF THE SET. Random Battle fixes the spread, so its adapter estimates every
   stat from base + level. An OU Pokemon's EVs and nature are the player's choice and swing
   a stat by ~60%, so the spread is sampled WITH the set and the stats are computed from it.
   Getting this wrong is not a rounding error: it decides whether a 2HKO is a 2HKO.

Everything the randbats adapter infers from observation -- Choice locks, the Scarf verdict
read off turn order, Heavy-Duty Boots read off hazard chip, real turn counters on screens
and Wish -- applies here unchanged and is threaded through the same kwargs.
"""

import random

from poke_env.data import GenData, to_id_str

from poke_engine import Side, State, Pokemon as PEPokemon

from laplace.agent.state_common import (
    _CHOICE_ITEMS, _boosts, _duration_volatiles, _hp, _item_evidence, _last_used_str,
    _moves_from_ids, _observed_pp, _our_side, _resolve_item, _side_conditions,
    _sleep_turns, _status_str, _sub_health, _terrain_str, _toxic_count, _trick_room,
    _types_tuple, _base_types_tuple, _volatile_set, _weather_str, _delayed, _dummy,
)
from laplace.ou.knowledge import KNOWLEDGE, full_learnset
from laplace.ou.usage import CURATED, USAGE, DEFAULT_IV, LEVEL, raw_stat

GEN = 9
_POKEDEX = GenData.from_gen(GEN).pokedex

_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")

# Illusion is the one hidden-information problem team preview does NOT solve: the disguised
# Pokemon shows a teammate's species, and the teammate is on the previewed list too. Both
# formes are OU-legal.
_ILLUSION_FORMES = ("zoroarkhisui", "zoroark")
_TELL_IGNORED_MOVES = {"struggle"}


# Share of worlds drawn from the curated sets rather than the usage marginals. The two
# priors fail in opposite directions and neither dominates:
#
#   curated    correlated and coherent -- the item, nature and moves came from one set
#              somebody would really build -- but NARROW. Measured against the same month's
#              usage feed, curated sets can express only ~85% of the item mass their species
#              actually run, and for some the gap is huge: Rocky Helmet is 34% of real
#              Zapdos and is in no Zapdos set, so a curated-only sampler cannot draw it in
#              any world, ever. A hypothesis absent from every world is one the search is
#              structurally unable to prepare for.
#
#   marginal   broad, and the item/tera/spread frequencies are the real ones, but the
#              fields are drawn INDEPENDENTLY, so it also builds Pokemon nobody would bring
#              (the incoherent-world pathology that cost the randbats bot its
#              compute-matched benchmark).
#
# Measured on the 2026-07 @ 1825 feed, top 25 curated species, usage-weighted. "fidelity"
# is the overlap between the sampled item distribution and the real one; "incoherent" is the
# share of drawn Pokemon carrying a combination the game makes nonsense (a Choice item plus
# a setup move, an Assault Vest plus a status move):
#
#                     fidelity   incoherent
#     curated only      78.0%        0.94%
#     marginal only     98.7%        7.67%
#     50/50 mix         88.8%        4.31%
#
# Neither column is the objective on its own, and each source is optimal at exactly the one
# it was built for -- which is why "marginals score higher on fidelity" is not an argument
# for dropping the curated sets. Determinization is the mechanism for hedging between two
# priors: the worlds are pooled, so a mixture reaches hypotheses the curated file cannot
# express while most worlds stay internally coherent.
#
# Half and half is the neutral starting point, NOT a tuned value. It is a knob on
# OUMetagame precisely so it can be moved once there is evidence to move it with, and the
# evidence would have to be a head-to-head, not either column above.
CURATED_FRACTION = 0.5


def _use_curated_here(use_curated):
    """Per-world coin flip between the two priors. `use_curated=False` forces marginals."""
    if not use_curated:
        return False
    return random.random() < (use_curated if isinstance(use_curated, float)
                              else CURATED_FRACTION)


# --- spreads ------------------------------------------------------------------------------

def _ev_list(evs):
    """A {stat: value} EV dict -> the six-slot list the stat formula wants."""
    return [int(evs.get(k, 0)) for k in _STAT_KEYS]


def _stats_from_spread(dex_id, nature, evs, ivs=None, level=LEVEL):
    """{stat: value} for a concrete spread. None if the species has no dex entry."""
    base = (_POKEDEX.get(dex_id) or {}).get("baseStats")
    if not base:
        return None
    ivs = ivs or {}
    return {k: raw_stat(base, k, int((evs or {}).get(k, 0)),
                        int(ivs.get(k, DEFAULT_IV)), level, nature or "serious")
            for k in _STAT_KEYS}


def _sample_spread(usage_id, dex_id):
    """(nature, ev dict) drawn from the species' observed spreads.

    The fallback is a neutral 84-EV-everywhere spread rather than an empty one: a Pokemon
    with no usage entry is a rare pick, not a Pokemon that ran no EVs, and handing the
    engine a 0-EV body would systematically under-price it in every world."""
    if usage_id:
        drawn = USAGE.spread(usage_id)
        if drawn:
            nature, evs = drawn
            return nature, dict(zip(_STAT_KEYS, evs))
    return "serious", {k: 84 for k in _STAT_KEYS}


# --- Illusion -----------------------------------------------------------------------------

class _ZoroShim:
    """poke-env-mon lookalike that re-skins a disguised opponent as the Zoroark forme.

    Identical in purpose to the randbats adapter's shim: Illusion copies the teammate's
    species in the protocol, but HP%, status and move usage belong to the Zoroark, so
    species-derived values are read off the Zoroark forme while observed state proxies
    through. The OU version differs only in that the level is always 100 and the base stats
    come straight from the dex."""

    __slots__ = ("_mon", "species", "base_species", "level", "base_stats", "stats", "types")

    def __init__(self, mon, zoro_id):
        entry = _POKEDEX.get(zoro_id) or {}
        base = entry.get("baseStats", {})
        self._mon = mon
        self.species = zoro_id
        self.base_species = "zoroark"
        self.level = LEVEL
        self.base_stats = {k: base.get(k, 80) for k in _STAT_KEYS}
        self.stats = {}
        self.types = tuple(_FakeType(t) for t in entry.get("types", ["Dark"]))

    def __getattr__(self, name):
        return getattr(self._mon, name)


class _FakeType:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name


def _illusion_candidates(battle):
    """Zoroark formes on the opponent's previewed team that have not shown themselves yet.

    Team preview makes this near-exact, which the randbats tell can never be: there, any
    Pokemon might be a Zoroark and the only evidence is an off-movepool click. Here the
    disguise is impossible unless a Zoroark was previewed, and it is over once one has been
    seen under its own name."""
    previewed = {to_id_str(m.species) for m in (battle.teampreview_opponent_team or [])}
    candidates = [z for z in _ILLUSION_FORMES if z in previewed]
    if not candidates:
        return ()
    revealed = {to_id_str(m.species) for m in battle.opponent_team.values()}
    return tuple(z for z in candidates if z not in revealed)


def _illusion_tell(mon, candidates):
    """The Zoroark forme this Pokemon really is, or None.

    Illusion fakes the species but cannot fake MOVE USAGE, so a revealed move outside the
    displayed species' legal gen-9 learnset means the display is a lie. Gated on `candidates`
    (a previewed, still-unseen Zoroark), on the learnset rather than a usage list -- a rare
    tech move is not evidence of anything -- and never on Ditto, which shows foreign moves
    legitimately."""
    if not candidates:
        return None
    species = to_id_str(mon.species)
    if species in _ILLUSION_FORMES or species == "ditto":
        return None
    revealed = set(mon.moves.keys()) - _TELL_IGNORED_MOVES
    if not revealed or "transform" in revealed:
        return None
    if revealed <= full_learnset(KNOWLEDGE.dex_key(mon)):
        return None
    fits = [z for z in candidates if revealed <= full_learnset(z)]
    return random.choice(fits) if fits else None


# --- one opponent Pokemon -----------------------------------------------------------------

def _opp_pokemon(mon, *, revealed_moves=None, used_since_switch=None, speed_hint=None,
                 item_hint=None, use_curated=True, illusion_candidates=(), full_hp=False):
    """Engine Pokemon for one opposing Pokemon, sampling everything still hidden.

    revealed_moves        -- override for the observed movepool; None reads mon.moves.
    used_since_switch     -- move ids used since switch-in. 2+ distinct rules out a Choice
                             item; exactly 1 plus a Choice item means locked.
    speed_hint            -- 'scarf'/'noscarf' from observed turn order.
    item_hint             -- 'boots'/'noboots' from observed hazard damage. An OBSERVATION,
                             so it outranks anything sampled.
    use_curated           -- prefer a complete Smogon set over composed marginals.
    illusion_candidates   -- previewed, still-unseen Zoroark formes.
    full_hp               -- this Pokemon has never been on the field, so the protocol has
                             told us nothing about its HP; poke-env reports 0%. Never let
                             that reach the engine as a fainted bench.
    """
    zoro = _illusion_tell(mon, illusion_candidates)
    if zoro:
        mon = _ZoroShim(mon, zoro)

    species = to_id_str(mon.species)
    dex_id = KNOWLEDGE.dex_key(mon)
    usage_id = USAGE.resolve(mon)
    revealed = list(revealed_moves if revealed_moves is not None else mon.moves.keys())

    item_state, known_item = _item_evidence(mon)
    multi_moved = bool(used_since_switch) and len(used_since_switch) >= 2
    item_exclude = _CHOICE_ITEMS if multi_moved else \
        ("choicescarf",) if speed_hint == "noscarf" else ()
    if item_hint == "noboots":
        item_exclude = tuple(item_exclude) + ("heavydutyboots",)
    force_scarf = (speed_hint == "scarf" and item_state == "unknown" and not multi_moved)
    known_ability = to_id_str(mon.ability) if mon.ability else None

    set_id = species if CURATED.has(species) else to_id_str(mon.base_species)
    cs = CURATED.sample(set_id, revealed_moves=revealed, item=known_item,
                        ability=known_ability, item_exclude=item_exclude,
                        force_scarf=force_scarf) if _use_curated_here(use_curated) else None

    if cs is not None:
        move_ids = cs["moves"]
        ability = known_ability or cs["ability"] or USAGE.ability(usage_id) or "none"
        item = _resolve_item(item_state, known_item, cs["item"])
        tera = cs["tera"]
        nature, evs, ivs = cs["nature"], cs["evs"], cs["ivs"]
    else:
        move_ids = USAGE.moves(usage_id, revealed) if usage_id else list(revealed)
        ability = known_ability or (USAGE.ability(usage_id) if usage_id else None) or "none"
        sampled_item = (USAGE.item(usage_id, exclude=item_exclude) if usage_id else None)
        if force_scarf:
            sampled_item = "choicescarf"
        item = _resolve_item(item_state, known_item, sampled_item or "leftovers")
        tera = USAGE.tera(usage_id) if usage_id else None
        nature, evs = _sample_spread(usage_id, dex_id)
        ivs = {}

    # Observations beat samples, in the order the randbats adapter settled on: hazard
    # evidence reads the item slot directly, while the speed verdict cannot tell a Choice
    # Scarf from a speed ability, so Boots wins when both fire.
    if item_state == "unknown":
        if force_scarf:
            item = "choicescarf"
        if item_hint == "boots":
            item = "heavydutyboots"
    if item in ("nothing", ""):
        item = "none"

    # A revealed Tera is a fact; an unrevealed one is a guess, and 'typeless' is never an
    # acceptable guess -- the engine explores opponent-Tera branches, and a typeless Tera
    # would strip the Pokemon's real typing and its immunities inside them.
    if getattr(mon, "tera_type", None) is not None:
        tera = mon.tera_type.name.lower()
    if not tera:
        tera = _types_tuple(mon)[0]

    stats = _stats_from_spread(dex_id, nature, evs, ivs) or {}
    maxhp = int(stats.get("hp") or 0) or int(KNOWLEDGE.estimate_stat(mon, "hp"))
    hp = maxhp if full_hp else _hp(mon, maxhp)

    def stat(key):
        return int(stats.get(key) or KNOWLEDGE.estimate_stat(mon, key))

    # Choice lock: this world gave it a Choice item (or Gorilla Tactics) and it has already
    # committed to a move since switching in, so only that move is selectable. poke-engine
    # does not derive the lock from item + last_used_move, so we set the flags ourselves.
    locked = None
    last = getattr(mon, "last_move", None)
    if (last is not None and not multi_moved and last.id in move_ids
            and (item in _CHOICE_ITEMS or ability == "gorillatactics")):
        locked = last.id

    return PEPokemon(
        id=species, level=getattr(mon, "level", None) or LEVEL,
        types=_types_tuple(mon), base_types=_base_types_tuple(mon),
        hp=hp, maxhp=maxhp,
        ability=ability, base_ability=ability, item=item,
        nature="serious", evs=(85,) * 6,      # cosmetic: the real stats are passed below
        attack=stat("atk"), defense=stat("def"),
        special_attack=stat("spa"), special_defense=stat("spd"), speed=stat("spe"),
        status=_status_str(mon) if not full_hp else "None",
        sleep_turns=_sleep_turns(mon) if not full_hp else 0,
        moves=_moves_from_ids(move_ids, only_enabled=locked, pp_by_id=_observed_pp(mon)),
        tera_type=tera, terastallized=bool(getattr(mon, "is_terastallized", False)),
    )


# --- the opposing side --------------------------------------------------------------------

def _roster_key(mon):
    """The identity a Pokemon keeps across formes.

    Species Clause makes base species unique on a team, and mid-battle formes are exactly
    what would otherwise break the match between a previewed 'Ogerpon-Wellspring' and the
    'Ogerpon-Wellspring-Tera' standing on the field."""
    return to_id_str(getattr(mon, "base_species", "") or mon.species)


def _split_roster(battle):
    """(revealed Pokemon, previewed-but-unseen Pokemon) for the opponent.

    poke-env's `opponent_team` silently falls back to the team-preview list while nothing
    has been revealed, and those are the SAME objects as `teampreview_opponent_team`. Telling
    the two apart by identity is what keeps a turn-1 state from listing all six as revealed
    -- each of which would then be built at 0% HP, because a previewed Pokemon reports no
    HP at all."""
    preview = list(battle.teampreview_opponent_team or [])
    preview_ids = {id(m) for m in preview}
    revealed = [m for m in battle.opponent_team.values() if id(m) not in preview_ids]
    placed = {_roster_key(m) for m in revealed}
    unseen = [m for m in preview if _roster_key(m) not in placed]
    return revealed, unseen


def _opp_side(battle, *, opp_used_since_switch=None, opp_speed_hints=None,
              opp_item_hints=None, pending=None, use_curated=True):
    speed_hints = opp_speed_hints or {}
    item_hints = opp_item_hints or {}
    illusion = _illusion_candidates(battle)
    revealed, unseen = _split_roster(battle)
    active = battle.opponent_active_pokemon

    def build(mon, is_active):
        key = to_id_str(mon.species)
        return _opp_pokemon(
            mon,
            used_since_switch=opp_used_since_switch if is_active else None,
            speed_hint=speed_hints.get(key), item_hint=item_hints.get(key),
            use_curated=use_curated, illusion_candidates=illusion)

    # No guard on `active is None`: the randbats adapter does not have one either, and the
    # search already treats a failed build as one lost world (diag['det_fail']) rather than
    # a lost turn. A silent half-built side would be worse than a counted failure.
    active_pe = build(active, True)
    pkmn = [active_pe]
    pkmn += [build(m, False) for m in revealed if m is not active]
    pkmn += [_opp_pokemon(m, revealed_moves=[], use_curated=use_curated,
                          illusion_candidates=illusion, full_hp=True) for m in unseen]

    # No team preview data at all (an unrated challenge with preview off, or a protocol we
    # did not see the start of). Fall back to the randbats behaviour -- invent bench slots --
    # but draw them from the active's usual teammates rather than uniformly from the dex.
    seen = {_roster_key(m) for m in revealed} | {_roster_key(m) for m in unseen}
    anchor = USAGE.resolve(active) if active is not None else None
    while len(pkmn) < 6:
        sid = (USAGE.teammate(anchor, exclude=seen) if anchor else None) \
            or USAGE.popular(exclude=seen)
        if sid is None:
            pkmn.append(_dummy())
            continue
        seen.add(sid)
        pkmn.append(_sampled_species(sid, use_curated=use_curated))

    opp_role = "p2" if (battle.player_role or "p1") == "p1" else "p1"
    wish, fs = _delayed(pending, opp_role, battle.turn)
    last_used = _last_used_str(active, active_pe.moves)
    dur_vols, durations = _duration_volatiles(active, last_used)
    return Side(
        pokemon=pkmn[:6],
        side_conditions=_side_conditions(battle.opponent_side_conditions, battle.turn,
                                         protect=getattr(active, "protect_counter", 0),
                                         toxic_count=_toxic_count(active)),
        active_index="0", volatile_status_durations=durations,
        wish=wish, future_sight=fs, volatile_statuses=_volatile_set(active) | dur_vols,
        substitute_health=_sub_health(active, pkmn[0].maxhp),
        last_used_move=last_used,
        switch_out_move_second_saved_move="none",
        **_boosts(active),
    )


def _sampled_species(species_id, use_curated=True):
    """A whole Pokemon invented from nothing but its species. Preview-less games only."""
    entry = _POKEDEX.get(species_id) or {}
    types = [t.lower() for t in entry.get("types", ["normal"])] or ["normal"]
    if len(types) == 1:
        types.append("typeless")
    cs = CURATED.sample(species_id) if _use_curated_here(use_curated) else None
    if cs is not None:
        moves, item = cs["moves"], cs["item"]
        ability = cs["ability"] or USAGE.ability(species_id) or "none"
        tera = cs["tera"] or types[0]
        nature, evs, ivs = cs["nature"], cs["evs"], cs["ivs"]
    else:
        moves = USAGE.moves(species_id, [])
        item = USAGE.item(species_id) or "leftovers"
        ability = USAGE.ability(species_id) or "none"
        tera = USAGE.tera(species_id) or types[0]
        nature, evs = _sample_spread(species_id, species_id)
        ivs = {}
    stats = _stats_from_spread(species_id, nature, evs, ivs) or {}
    maxhp = int(stats.get("hp") or 300)
    if item in ("nothing", ""):
        item = "none"
    return PEPokemon(
        id=species_id, level=LEVEL, types=(types[0], types[1]),
        base_types=(types[0], types[1]), hp=maxhp, maxhp=maxhp,
        ability=ability, base_ability=ability, item=item,
        nature="serious", evs=(85,) * 6,
        attack=int(stats.get("atk") or 200), defense=int(stats.get("def") or 200),
        special_attack=int(stats.get("spa") or 200),
        special_defense=int(stats.get("spd") or 200),
        speed=int(stats.get("spe") or 200),
        status="None", moves=_moves_from_ids(moves), tera_type=tera, terastallized=False,
    )


# --- entry point --------------------------------------------------------------------------

def build_state(battle, use_stats=True, opp_used_since_switch=None, opp_speed_hints=None,
                pending=None, use_joint=True, opp_item_hints=None):
    """A determinized State for the current OU position. Call repeatedly for more samples.

    The signature matches `poke_engine_adapter.build_state` exactly so the search does not
    branch on format. `use_stats` is accepted and ignored -- there is no marginal-free mode
    in OU, the marginals ARE one of the two priors.

    `use_joint` draws the same distinction it draws in Random Battle -- complete sets vs
    composed marginals -- but as a MIXTURE rather than a preference: True mixes them at
    CURATED_FRACTION, a float sets that share explicitly, False forces marginals only. See
    CURATED_FRACTION for why neither source is allowed to win outright."""
    weather, weather_left = _weather_str(battle)
    terrain, terrain_left = _terrain_str(battle)
    tr, tr_left = _trick_room(battle)
    return State(
        side_one=_our_side(battle, pending),
        side_two=_opp_side(battle, opp_used_since_switch=opp_used_since_switch,
                           opp_speed_hints=opp_speed_hints, opp_item_hints=opp_item_hints,
                           pending=pending, use_curated=use_joint),
        weather=weather, weather_turns_remaining=weather_left,
        terrain=terrain, terrain_turns_remaining=terrain_left,
        trick_room=tr, trick_room_turns_remaining=tr_left, team_preview=False,
    )
