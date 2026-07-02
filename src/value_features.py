r"""Featurize a poke-engine `State` for the value network.

The value net V(state) ~ P(side_one wins) must score states that only exist inside the
engine -- positions reached by rolling candidate moves one ply forward with
`generate_instructions` / `apply_instructions` -- so it reads engine State objects, not
poke-env battles. At data-generation time we featurize the same determinized states the
search runs on, so train and inference distributions match (including the sampled hidden
sets' noise).

Layout (side_one = us, fixed by the adapter):
  global   : weather one-hot(4) + turns, terrain one-hot(4) + turns, trick room flag + turns
  per side : team aggregates, side conditions, active block (stats/status/boosts/items/
             best damage vs opposing active/speed race), 5 bench blocks sorted canonically
             (alive, hp, stats, status, item flags, STAB effectiveness both directions)

Damage features use poke_engine.calculate_damage on the actives (4 diagonal calls per
state); bench matchups use the gen-9 type chart, which is cheap and switch-relevant.
"""

import numpy as np

from poke_engine import calculate_damage

from poke_env.data import GenData

_TYPE_CHART = GenData.from_gen(9).type_chart   # chart[DEFENDER][ATTACKER] -> multiplier

_STATUSES = ("burn", "poison", "toxic", "paralyze", "sleep", "freeze")

# Item groups that change how hard a mon hits / takes hits.
_ITEM_FLAGS = (
    ("choiceband", "choicespecs"),
    ("choicescarf",),
    ("lifeorb",),
    ("assaultvest",),
    ("heavydutyboots",),
    ("leftovers", "blacksludge"),
)

_VOLATILES = ("substitute", "leechseed", "saltcure", "curse")

_WEATHERS = (("sun", "harshsun"), ("rain", "heavyrain"), ("sand",), ("hail", "snow"))
_TERRAINS = ("electricterrain", "grassyterrain", "mistyterrain", "psychicterrain")

_BOOST_KEYS = ("attack_boost", "defense_boost", "special_attack_boost",
               "special_defense_boost", "speed_boost", "accuracy_boost", "evasion_boost")

# Ability groups that swing a position's value (v2). Grouped by effect, not name, so one
# flag generalizes across e.g. all the hit-absorbing immunity abilities.
_ABILITY_FLAGS = (
    ("levitate",),
    ("unaware",),
    ("magicguard",),
    ("regenerator",),
    ("intimidate",),
    ("speedboost", "protosynthesis", "quarkdrive"),
    ("waterabsorb", "voltabsorb", "flashfire", "stormdrain", "sapsipper", "eartheater",
     "dryskin", "lightningrod"),
    ("thickfat", "multiscale", "furcoat", "icescales"),
    ("moldbreaker", "teravolt", "turboblaze"),
    ("hugepower", "purepower", "gorillatactics"),
)

_RECOVERY_MOVES = {"roost", "recover", "softboiled", "slackoff", "synthesis", "moonlight",
                   "morningsun", "shoreup", "milkdrink", "rest", "strengthsap", "wish"}

# global 12 + 2 sides x (2 agg + 11 conditions + 45 active + 5*24 bench)
N_VALUE_FEATURES = 12 + 2 * (2 + 11 + 45 + 5 * 24)


def _eff(atk_types, def_types):
    """Best STAB effectiveness of atk_types attacking def_types (typeless = neutral)."""
    best = 0.0
    for at in atk_types:
        at = at.upper()
        mult = 1.0
        for dt in def_types:
            dt = dt.upper()
            row = _TYPE_CHART.get(dt)
            if row is None:        # typeless / unknown
                continue
            mult *= row.get(at, 1.0)
        best = max(best, mult)
    return best


def _status_onehot(status):
    s = (status or "").lower()
    return [1.0 if key in s else 0.0 for key in _STATUSES]


def _item_flags(item):
    it = (item or "").lower()
    return [1.0 if it in group else 0.0 for group in _ITEM_FLAGS]


def _ability_flags(ability):
    ab = (ability or "").lower()
    return [1.0 if ab in group else 0.0 for group in _ABILITY_FLAGS]


def _has_recovery(mon):
    return 1.0 if any(m.id in _RECOVERY_MOVES and not m.disabled and m.pp > 0
                      for m in mon.moves) else 0.0


def _mon_alive(mon):
    return mon.hp > 0 and mon.id != "none"


def _stat_block(mon):
    return [mon.hp / max(mon.maxhp, 1), mon.level / 100.0,
            mon.attack / 500.0, mon.defense / 500.0, mon.special_attack / 500.0,
            mon.special_defense / 500.0, mon.speed / 500.0]


def _boost_mult(stages):
    return (2 + stages) / 2 if stages >= 0 else 2 / (2 - stages)


def _effective_speed(mon, side):
    spe = mon.speed * _boost_mult(side.speed_boost)
    if "paralyze" in (mon.status or "").lower():
        spe *= 0.5
    if (mon.item or "").lower() == "choicescarf":
        spe *= 1.5
    if side.side_conditions.tailwind > 0:
        spe *= 2
    return spe


def _active_damage_fracs(state):
    """Best damage fraction each active deals to the other, via the engine's damage calc.
    Diagonal move pairing: one call per move slot yields both sides' rolls for that slot."""
    s1, s2 = state.side_one, state.side_two
    a1 = s1.pokemon[int(s1.active_index)]
    a2 = s2.pokemon[int(s2.active_index)]
    m1 = [m.id for m in a1.moves if m.id != "none" and not m.disabled] or ["splash"]
    m2 = [m.id for m in a2.moves if m.id != "none" and not m.disabled] or ["splash"]
    best1 = best2 = 0.0
    for i in range(max(len(m1), len(m2))):
        try:
            r1, r2 = calculate_damage(state, m1[min(i, len(m1) - 1)],
                                      m2[min(i, len(m2) - 1)], True)
        except Exception:
            continue
        if r1:
            best1 = max(best1, (sum(r1) / len(r1)) / max(a2.maxhp, 1))
        if r2:
            best2 = max(best2, (sum(r2) / len(r2)) / max(a1.maxhp, 1))
    return min(best1, 2.0), min(best2, 2.0)


def _side_features(side, opp_side, best_dmg_frac, opp_best_dmg_frac):
    active = side.pokemon[int(side.active_index)]
    opp_active = opp_side.pokemon[int(opp_side.active_index)]
    mons = list(side.pokemon)

    alive = [m for m in mons if _mon_alive(m)]
    feats = [len(alive) / 6.0, sum(m.hp / max(m.maxhp, 1) for m in alive) / 6.0]

    sc = side.side_conditions
    wish = side.wish
    fs = side.future_sight
    feats += [
        float(sc.stealth_rock), sc.spikes / 3.0, sc.toxic_spikes / 2.0, float(sc.sticky_web),
        sc.reflect / 5.0, sc.light_screen / 5.0, sc.aurora_veil / 5.0, sc.tailwind / 4.0,
        min(sc.toxic_count, 8) / 8.0,
        1.0 if wish[0] > 0 else 0.0,
        1.0 if (fs[0] if isinstance(fs, tuple) else 0) > 0 else 0.0,
    ]

    # Active block.
    feats += _stat_block(active)
    feats += _status_onehot(active.status)
    feats += [getattr(side, k) / 6.0 for k in _BOOST_KEYS]
    feats.append(side.substitute_health / max(active.maxhp, 1))
    vols = set(side.volatile_statuses)
    feats += [1.0 if v in vols else 0.0 for v in _VOLATILES]
    feats += _item_flags(active.item)
    feats += _ability_flags(active.ability)
    feats.append(_has_recovery(active))
    feats.append(best_dmg_frac)
    my_spe = _effective_speed(active, side)
    opp_spe = _effective_speed(opp_active, opp_side)
    feats.append(1.0 if my_spe > opp_spe else 0.0)
    feats.append(min(my_spe / max(opp_spe, 1.0), 3.0))

    # Bench, canonically sorted so slot order doesn't matter.
    bench = [m for i, m in enumerate(mons) if i != int(side.active_index)]
    bench.sort(key=lambda m: (-_mon_alive(m), -(m.hp / max(m.maxhp, 1)), -m.speed, m.id))
    opp_types = [t for t in opp_active.types if t and t != "typeless"]
    for i in range(5):
        if i < len(bench) and _mon_alive(bench[i]):
            m = bench[i]
            my_types = [t for t in m.types if t and t != "typeless"]
            feats += [1.0] + _stat_block(m) + _status_onehot(m.status) + _item_flags(m.item)
            feats += [_eff(my_types, opp_types) / 4.0, _eff(opp_types, my_types) / 4.0]
            feats.append(1.0 if m.speed > opp_spe else 0.0)   # wins the speed race if sent in
            feats.append(_has_recovery(m))
        else:
            feats += [0.0] * 24
    return feats


def featurize(state):
    """State -> float32 vector of length N_VALUE_FEATURES, from side_one's perspective."""
    feats = []
    w = (state.weather or "none").lower()
    feats += [1.0 if w in group else 0.0 for group in _WEATHERS]
    wt = state.weather_turns_remaining
    feats.append((8 if wt < 0 else min(wt, 8)) / 8.0 if w != "none" else 0.0)
    t = (state.terrain or "none").lower()
    feats += [1.0 if t == name else 0.0 for name in _TERRAINS]
    tt = state.terrain_turns_remaining
    feats.append((8 if tt < 0 else min(tt, 8)) / 8.0 if t != "none" else 0.0)
    feats.append(1.0 if state.trick_room else 0.0)
    feats.append(min(max(state.trick_room_turns_remaining, 0), 5) / 5.0 if state.trick_room else 0.0)

    d1, d2 = _active_damage_fracs(state)
    feats += _side_features(state.side_one, state.side_two, d1, d2)
    feats += _side_features(state.side_two, state.side_one, d2, d1)
    return np.asarray(feats, dtype=np.float32)
