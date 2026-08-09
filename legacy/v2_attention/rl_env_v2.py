"""A richer, structured observation, laid out for the attention network.

The normal version flattens everything into one long list of numbers. That's fine for a
plain network, but it throws away the shape of a battle, which is really twelve Pokemon,
six a side, plus the state of the field. The hope was that a network which knows about that
structure would learn better than one rediscovering "these 64 numbers are one Pokemon".

So the observation is a fixed grid instead:

    [ our six | their six | the field ]

Our own team is always fully known. Theirs is filled in as Pokemon are revealed, and the
slots we haven't seen yet are left as zeroes.

The network folds the first part back into a twelve-by-something grid, which is why the
layout has to stay fixed and is spelled out here.
"""

import numpy as np
from gymnasium.spaces import Box

from poke_env.environment.singles_env import SinglesEnv

# --- the vocabulary ----------------------------------------------------------

TYPE_NAMES = [
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE", "FIGHTING", "POISON",
    "GROUND", "FLYING", "PSYCHIC", "BUG", "ROCK", "GHOST", "DRAGON", "DARK", "STEEL", "FAIRY",
]
STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]
BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
STATUS_NAMES = ["BRN", "PAR", "SLP", "FRZ", "PSN", "TOX"]

# Items that change how a turn plays out, grouped by what they do.
ITEM_CATEGORIES = [
    {"leftovers", "blacksludge"},                 # heals a bit every turn
    {"choiceband", "choicespecs", "choicescarf"}, # more power or speed, locked to one move
    {"heavydutyboots"},                           # walks over hazards
    {"lifeorb"},                                  # hits harder, hurts itself
    {"assaultvest"},                              # tanky, but no status moves
    {"focussash"},                               # survives one hit from full health
    {"rockyhelmet"},                             # hurts anything that touches it
    {"eviolite"},                                # bulk for the unevolved
    {"boosterenergy"},                           # switches on a stat boost immediately
    {"weaknesspolicy"},                          # gets stronger after a big hit
]

# Same idea for abilities.
ABILITY_CATEGORIES = [
    {"levitate"},                                       # can't be hit by Ground
    {"flashfire", "wellbakedbody"},                     # can't be hit by Fire
    {"waterabsorb", "stormdrain", "dryskin"},           # can't be hit by Water
    {"voltabsorb", "lightningrod", "motordrive"},       # can't be hit by Electric
    {"sapsipper"},                                      # can't be hit by Grass
    {"multiscale", "shadowshield"},                     # takes half damage at full health
    {"intimidate"},                                     # weakens whatever it faces
    {"regenerator"},                                    # heals every time it switches out
    {"magicguard"},                                     # ignores poison, hazards, weather
    {"unaware"},                                        # ignores the opponent's boosts
    {"hugepower", "purepower"},                         # double attack
    {"speedboost", "protosynthesis", "quarkdrive",      # gets faster, one way or another
     "swiftswim", "chlorophyll", "sandrush", "unburden"},
]

N_MON = 12          # six each side
# What we record per Pokemon: whether it's out, whether it's fainted, its health, whether
# we've seen it at all, then typing, base stats, boosts, status, items, abilities and Tera.
PER_MON = 1 + 1 + 1 + 1 + 18 + 6 + 7 + 6 + len(ITEM_CATEGORIES) + len(ABILITY_CATEGORIES) + 1  # = 64


def _norm(name):
    return name.lower().replace(" ", "").replace("-", "") if name else ""


def _onehot(name, vocab):
    vec = [0.0] * len(vocab)
    if name in vocab:
        vec[vocab.index(name)] = 1.0
    return vec


def _category_flags(value_id, categories, possible_ids=None):
    """Full marks if we know it's in a category, half if it merely could be."""
    vec = [0.0] * len(categories)
    if value_id:
        for i, ids in enumerate(categories):
            if value_id in ids:
                vec[i] = 1.0
    elif possible_ids:
        for i, ids in enumerate(categories):
            if possible_ids & ids:
                vec[i] = 0.5
    return vec


def _mon_token(mon, is_active):
    """Everything we record about one Pokemon. All zeroes if we've never seen it."""
    if mon is None:
        return [0.0] * PER_MON

    types = [t.name for t in mon.types if t is not None]
    type_vec = [1.0 if name in types else 0.0 for name in TYPE_NAMES]

    base_stats = [mon.base_stats[k] / 200.0 for k in STAT_KEYS]
    boosts = [mon.boosts.get(k, 0) / 6.0 for k in BOOST_KEYS]
    status_vec = _onehot(mon.status.name, STATUS_NAMES) if mon.status is not None else [0.0] * 6

    item_vec = _category_flags(_norm(mon.item), ITEM_CATEGORIES)
    possible = {_norm(a) for a in (mon.possible_abilities or [])} if not mon.ability else None
    ability_vec = _category_flags(_norm(mon.ability), ABILITY_CATEGORIES, possible)

    token = (
        [1.0 if is_active else 0.0]
        + [1.0 if mon.fainted else 0.0]
        + [mon.current_hp_fraction]
        + [1.0]                                  # yes, there's really a Pokemon here
        + type_vec + base_stats + boosts + status_vec
        + item_vec + ability_vec
        + [1.0 if mon.is_terastallized else 0.0]
    )
    return token


def _eff_speed(mon):
    """Roughly how fast something is, counting boosts and paralysis."""
    if mon is None:
        return 0.0
    boost = mon.boosts.get("spe", 0)
    mult = (2 + boost) / 2 if boost >= 0 else 2 / (2 - boost)
    spe = mon.base_stats["spe"] * mult
    if mon.status is not None and mon.status.name == "PAR":
        spe *= 0.5
    return spe


def _hazards(side):
    sr = spikes = tspikes = 0.0
    for cond, value in side.items():
        n = cond.name
        if n == "STEALTH_ROCK":
            sr = 1.0
        elif n == "SPIKES":
            spikes = value / 3.0
        elif n == "TOXIC_SPIKES":
            tspikes = value / 2.0
    return [sr, spikes, tspikes]


def _screens(side):
    refl = ls = av = tw = 0.0
    for cond in side:
        n = cond.name
        if n == "REFLECT":
            refl = 1.0
        elif n == "LIGHT_SCREEN":
            ls = 1.0
        elif n == "AURORA_VEIL":
            av = 1.0
        elif n == "TAILWIND":
            tw = 1.0
    return [refl, ls, av, tw]


def _terrain(battle):
    elec = grass = psy = mist = 0.0
    for f in battle.fields:
        n = f.name
        if n == "ELECTRIC_TERRAIN":
            elec = 1.0
        elif n == "GRASSY_TERRAIN":
            grass = 1.0
        elif n == "PSYCHIC_TERRAIN":
            psy = 1.0
        elif n == "MISTY_TERRAIN":
            mist = 1.0
    none = 0.0 if (elec or grass or psy or mist) else 1.0
    return [elec, grass, psy, mist, none]


def _trick_room(battle):
    return any(f.name == "TRICK_ROOM" for f in battle.fields)


def _weather(battle):
    sun = rain = sand = snow = 0.0
    for w in battle.weather:
        n = w.name
        if n in ("SUNNYDAY", "DESOLATELAND"):
            sun = 1.0
        elif n in ("RAINDANCE", "PRIMORDIALSEA"):
            rain = 1.0
        elif n == "SANDSTORM":
            sand = 1.0
        elif n in ("SNOW", "HAIL"):
            snow = 1.0
    none = 0.0 if battle.weather else 1.0
    return [sun, rain, sand, snow, none]


def _tera_type_vec(mon):
    if mon is not None and mon.tera_type is not None:
        return _onehot(mon.tera_type.name, TYPE_NAMES)
    return [0.0] * 18


def _global_features(battle):
    me = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    tr = _trick_room(battle)

    # The moves we can actually click this turn.
    move_power = [0.0] * 4
    move_mult = [0.0] * 4
    move_acc = [0.0] * 4
    move_phys = [0.0] * 4
    move_status = [0.0] * 4
    for i, mv in enumerate(battle.available_moves[:4]):
        move_power[i] = mv.base_power / 100.0
        if opp is not None:
            move_mult[i] = opp.damage_multiplier(mv) / 4.0
        acc = mv.accuracy
        move_acc[i] = 1.0 if acc is True else float(acc)
        cat = mv.category.name
        move_phys[i] = 1.0 if cat == "PHYSICAL" else 0.0
        move_status[i] = 1.0 if cat == "STATUS" else 0.0

    my_remaining = sum(1 for m in battle.team.values() if not m.fainted) / 6.0
    opp_remaining = (6 - sum(1 for m in battle.opponent_team.values() if m.fainted)) / 6.0

    faster = _eff_speed(me) > _eff_speed(opp)
    if tr:                      # Trick Room turns the speed order upside down
        faster = not faster

    feats = (
        _weather(battle) + _terrain(battle) + [1.0 if tr else 0.0]
        + _hazards(battle.side_conditions) + _hazards(battle.opponent_side_conditions)
        + _screens(battle.side_conditions) + _screens(battle.opponent_side_conditions)
        + [my_remaining, opp_remaining]
        + [1.0 if battle.can_tera else 0.0]
        + _tera_type_vec(me) + _tera_type_vec(opp)
        + move_power + move_mult + move_acc + move_phys + move_status
        + [1.0 if faster else 0.0]
        + [min(battle.turn / 30.0, 1.0)]
    )
    return feats


GLOBAL = 11 + 6 + 8 + 2 + 1 + 36 + 20 + 1 + 1
N_FEATURES = N_MON * PER_MON + GLOBAL


def structured_observation(battle):
    """Assemble the whole thing: twelve Pokemon, then the field."""
    tokens = []
    my_team = list(battle.team.values())[:6]
    for mon in my_team:
        tokens += _mon_token(mon, mon is battle.active_pokemon)
    tokens += [0.0] * PER_MON * (6 - len(my_team))

    opp_team = list(battle.opponent_team.values())[:6]
    for mon in opp_team:
        tokens += _mon_token(mon, mon is battle.opponent_active_pokemon)
    tokens += [0.0] * PER_MON * (6 - len(opp_team))

    return np.array(tokens + _global_features(battle), dtype=np.float32)


class ShowdownTeamEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        obs_space = Box(low=-1.0, high=4.0, shape=(N_FEATURES,), dtype=np.float32)
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}

    def embed_battle(self, battle):
        return structured_observation(battle)

    def calc_reward(self, battle):
        # A big win bonus pushes towards winning, but make it too big and nothing learns.
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, status_value=0.1, victory_value=40.0,
        )
