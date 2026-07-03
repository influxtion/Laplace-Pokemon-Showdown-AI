"""Phase 1 (v2): a structured, richer observation built for a team-attention network.

The baseline (rl_env.py) flattens everything into one 141-number vector. Fine for a plain
MLP, but it hides the structure of a battle: really 12 Pokemon (your 6 + their 6) plus global
field state. A network that respects that -- a shared encoder per Pokemon, then attention
across the team -- can learn better than a flat MLP rediscovering "these 64 numbers are one
Pokemon."

So the observation is a fixed grid:

    [ mon_0 | mon_1 | ... | mon_11 | global ]
      <-- 12 tokens of PER_MON each -->  <- GLOBAL ->

mon_0..5  = your team (always known)
mon_6..11 = opponent's team (revealed mons filled, the rest zeros)

team_net reshapes the first 12*PER_MON numbers back into a (12, PER_MON) grid and runs
attention over it, which is why the layout is fixed and documented here. Carries the
win-weighted reward from the baseline.
"""

import numpy as np
from gymnasium.spaces import Box

from poke_env.environment.singles_env import SinglesEnv

# --- vocabularies ------------------------------------------------------------

TYPE_NAMES = [
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE", "FIGHTING", "POISON",
    "GROUND", "FLYING", "PSYCHIC", "BUG", "ROCK", "GHOST", "DRAGON", "DARK", "STEEL", "FAIRY",
]  # 18
STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]            # 6
BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]  # 7
STATUS_NAMES = ["BRN", "PAR", "SLP", "FRZ", "PSN", "TOX"]        # 6

# Curated high-impact items, grouped into categories (one flag each).
ITEM_CATEGORIES = [
    {"leftovers", "blacksludge"},                 # passive healing
    {"choiceband", "choicespecs", "choicescarf"}, # locked into one move (power/speed)
    {"heavydutyboots"},                           # ignores entry hazards
    {"lifeorb"},                                  # power boost + recoil
    {"assaultvest"},                              # special bulk, no status moves
    {"focussash"},                               # survive a KO at full HP
    {"rockyhelmet"},                             # punishes contact
    {"eviolite"},                                # bulk for not-fully-evolved
    {"boosterenergy"},                           # triggers Proto/Quark
    {"weaknesspolicy"},                          # boosts after super-effective hit
]  # 10

# Curated high-impact ability categories (one flag each).
ABILITY_CATEGORIES = [
    {"levitate"},                                       # immune to Ground
    {"flashfire", "wellbakedbody"},                     # immune to Fire
    {"waterabsorb", "stormdrain", "dryskin"},           # immune to Water
    {"voltabsorb", "lightningrod", "motordrive"},       # immune to Electric
    {"sapsipper"},                                      # immune to Grass
    {"multiscale", "shadowshield"},                     # halves damage at full HP
    {"intimidate"},                                     # drops attack on switch-in
    {"regenerator"},                                    # heals on switch
    {"magicguard"},                                     # no indirect damage
    {"unaware"},                                        # ignores stat boosts
    {"hugepower", "purepower"},                         # doubles attack
    {"speedboost", "protosynthesis", "quarkdrive",      # speed enablers
     "swiftswim", "chlorophyll", "sandrush", "unburden"},
]  # 12

N_MON = 12          # 6 ours + 6 theirs
# PER_MON layout: active(1) fainted(1) hp(1) known(1) types(18) base_stats(6)
#                 boosts(7) status(6) items(10) abilities(12) terastallized(1)
PER_MON = 1 + 1 + 1 + 1 + 18 + 6 + 7 + 6 + len(ITEM_CATEGORIES) + len(ABILITY_CATEGORIES) + 1  # = 64


def _norm(name):
    return name.lower().replace(" ", "").replace("-", "") if name else ""


def _onehot(name, vocab):
    vec = [0.0] * len(vocab)
    if name in vocab:
        vec[vocab.index(name)] = 1.0
    return vec


def _category_flags(value_id, categories, possible_ids=None):
    """1.0 if value_id is in a category; else 0.5 if any possible_id is (for hidden info)."""
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
    """Fixed-length PER_MON feature block for one Pokemon (all zeros if unknown)."""
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
        + [1.0]                                  # "known": this slot holds a real Pokemon
        + type_vec + base_stats + boosts + status_vec
        + item_vec + ability_vec
        + [1.0 if mon.is_terastallized else 0.0]
    )
    return token


def _eff_speed(mon):
    """Rough effective speed: base * boost multiplier, halved if paralyzed."""
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

    # The agent's available moves (action-relevant).
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
    if tr:                      # Trick Room reverses the speed order
        faster = not faster

    feats = (
        _weather(battle) + _terrain(battle) + [1.0 if tr else 0.0]                       # 11
        + _hazards(battle.side_conditions) + _hazards(battle.opponent_side_conditions)   # 6
        + _screens(battle.side_conditions) + _screens(battle.opponent_side_conditions)   # 8
        + [my_remaining, opp_remaining]                                                  # 2
        + [1.0 if battle.can_tera else 0.0]                                              # 1
        + _tera_type_vec(me) + _tera_type_vec(opp)                                       # 36
        + move_power + move_mult + move_acc + move_phys + move_status                    # 20
        + [1.0 if faster else 0.0]                                                       # 1
        + [min(battle.turn / 30.0, 1.0)]                                                 # 1
    )
    return feats


GLOBAL = 11 + 6 + 8 + 2 + 1 + 36 + 20 + 1 + 1   # = 86
N_FEATURES = N_MON * PER_MON + GLOBAL           # 12*64 + 86 = 854


def structured_observation(battle):
    """Build the full [12 Pokemon tokens | global] observation vector."""
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
        # High victory_value encourages winning, but too high prevents learning
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, status_value=0.1, victory_value=40.0,
        )
