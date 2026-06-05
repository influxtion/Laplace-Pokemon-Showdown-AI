"""The reinforcement-learning environment (Stage 2).

This is where "training an AI" actually happens. We don't write rules for which
move to pick (that was Stage 1). Instead we describe the battle to a neural
network as a list of numbers, hand back a reward when things go well, and let a
learning algorithm (PPO) figure out a good policy through trial and error.

poke-env does the heavy lifting: `SinglesEnv` already knows how to talk to the
Showdown server, what actions are legal, and how to translate a chosen action
number into an in-game move. We only have to supply the two things that are
specific to *our* AI:

  * embed_battle()  -> turn the battle state into numbers (the "observation")
  * calc_reward()   -> say how good the current state is (the "reward")

The observation is deliberately detailed. An early 12-number version plateaued
against a strong opponent because it was blind to most of the game. This version
adds the things that actually drive decisions: stat boosts, status, the bench
(for switching), move details, field/terrain/screens, hazards, weather, the
opponent's revealed moves, and high-impact abilities.
"""

import numpy as np
from gymnasium.spaces import Box

from poke_env.environment.singles_env import SinglesEnv

# --- lookup tables -----------------------------------------------------------

STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]   # base-stat dict keys
BOOST_KEYS = ["atk", "def", "spa", "spd", "spe"]        # in-battle stat-stage keys
STATUS_NAMES = ["BRN", "PAR", "SLP", "FRZ", "PSN", "TOX"]  # status conditions we encode
EFFECT_NAMES = ["SUBSTITUTE", "LEECH_SEED", "TAUNT", "CONFUSION"]  # volatile effects

# High-impact abilities, grouped into strategic categories. Each category becomes one
# flag per active Pokemon. Encoding categories (not the hundreds of raw abilities) keeps
# this compact while capturing the abilities that actually change a decision.
ABILITY_CATEGORIES = [
    {"levitate"},                                      # immune to Ground
    {"flashfire", "wellbakedbody"},                    # immune to Fire
    {"waterabsorb", "stormdrain", "dryskin"},          # immune to Water
    {"voltabsorb", "lightningrod", "motordrive"},      # immune to Electric
    {"sapsipper"},                                     # immune to Grass
    {"multiscale", "shadowshield"},                    # halves damage at full HP
    {"intimidate"},                                    # drops attack on switch-in
    {"regenerator"},                                   # heals on switch
    {"magicguard"},                                    # no indirect damage
    {"unaware"},                                       # ignores stat boosts
    {"hugepower", "purepower"},                        # doubles attack
    {"speedboost", "protosynthesis", "quarkdrive",     # speed enablers
     "swiftswim", "chlorophyll", "sandrush", "unburden"},
]


# --- per-Pokemon feature helpers ---------------------------------------------

def _base_stats(mon):
    """6 base stats scaled to roughly 0-1 (0 if the Pokemon is unknown)."""
    if mon is None:
        return [0.0] * 6
    return [mon.base_stats[k] / 200.0 for k in STAT_KEYS]


def _boosts(mon):
    """5 stat-stage boosts in -1..1 (a +6 Swords Dance reads as +1 attack)."""
    if mon is None:
        return [0.0] * 5
    return [mon.boosts[k] / 6.0 for k in BOOST_KEYS]


def _status_onehot(mon):
    """One-hot of the active status condition (all zeros = healthy)."""
    vec = [0.0] * len(STATUS_NAMES)
    if mon is not None and mon.status is not None and mon.status.name in STATUS_NAMES:
        vec[STATUS_NAMES.index(mon.status.name)] = 1.0
    return vec


def _effects(mon):
    """Flags for a few important volatile effects (substitute, leech seed, etc.)."""
    vec = [0.0] * len(EFFECT_NAMES)
    if mon is not None:
        names = {e.name for e in mon.effects}
        for i, nm in enumerate(EFFECT_NAMES):
            if nm in names:
                vec[i] = 1.0
    return vec


def _norm_ability(name):
    return name.lower().replace(" ", "").replace("-", "")


def _ability_flags(mon, infer_possible=False):
    """One flag per high-impact ability category.

    1.0 = the Pokemon is confirmed to have an ability in this category.
    0.5 = the ability is unknown (opponent, not yet revealed) but the species COULD
          have one in this category (from `possible_abilities`).
    0.0 = none / unknown with no possibility.
    """
    vec = [0.0] * len(ABILITY_CATEGORIES)
    if mon is None:
        return vec
    if mon.ability:  # confirmed (always true for our own Pokemon)
        aid = _norm_ability(mon.ability)
        for i, ids in enumerate(ABILITY_CATEGORIES):
            if aid in ids:
                vec[i] = 1.0
    elif infer_possible:  # opponent whose ability hasn't shown yet
        possibles = {_norm_ability(a) for a in (mon.possible_abilities or [])}
        for i, ids in enumerate(ABILITY_CATEGORIES):
            if possibles & ids:
                vec[i] = 0.5
    return vec


def _matchup(me, opp):
    """Best type effectiveness I have into them, and they have into me (0-1 each)."""
    if me is None or opp is None:
        return [0.0, 0.0]
    offense = max((opp.damage_multiplier(t) for t in me.types if t is not None), default=1.0)
    defense = max((me.damage_multiplier(t) for t in opp.types if t is not None), default=1.0)
    return [offense / 4.0, defense / 4.0]


# --- battle-wide feature helpers ---------------------------------------------

def _hazards(side_conditions):
    """Entry hazards on one side: [stealth rock, spikes layers, toxic spikes layers]."""
    sr, spikes, tspikes = 0.0, 0.0, 0.0
    for cond, value in side_conditions.items():
        name = cond.name
        if name == "STEALTH_ROCK":
            sr = 1.0
        elif name == "SPIKES":
            spikes = value / 3.0
        elif name == "TOXIC_SPIKES":
            tspikes = value / 2.0
    return [sr, spikes, tspikes]


def _screens(side_conditions):
    """Screen-type conditions on one side: [reflect, light screen, aurora veil, tailwind]."""
    refl = ls = av = tw = 0.0
    for cond in side_conditions:
        name = cond.name
        if name == "REFLECT":
            refl = 1.0
        elif name == "LIGHT_SCREEN":
            ls = 1.0
        elif name == "AURORA_VEIL":
            av = 1.0
        elif name == "TAILWIND":
            tw = 1.0
    return [refl, ls, av, tw]


def _terrain(battle):
    """One-hot of the active terrain: [electric, grassy, psychic, misty, none]."""
    elec = grass = psy = mist = 0.0
    for f in battle.fields:
        name = f.name
        if name == "ELECTRIC_TERRAIN":
            elec = 1.0
        elif name == "GRASSY_TERRAIN":
            grass = 1.0
        elif name == "PSYCHIC_TERRAIN":
            psy = 1.0
        elif name == "MISTY_TERRAIN":
            mist = 1.0
    none = 0.0 if (elec or grass or psy or mist) else 1.0
    return [elec, grass, psy, mist, none]


def _trick_room(battle):
    """Whether Trick Room is active (it reverses speed order, so the 'faster' flag flips)."""
    return [1.0 if any(f.name == "TRICK_ROOM" for f in battle.fields) else 0.0]


def _weather(battle):
    """One-hot of the weather: [sun, rain, sand, snow, none]."""
    sun = rain = sand = snow = 0.0
    for w in battle.weather:
        name = w.name
        if name in ("SUNNYDAY", "DESOLATELAND"):
            sun = 1.0
        elif name in ("RAINDANCE", "PRIMORDIALSEA"):
            rain = 1.0
        elif name == "SANDSTORM":
            sand = 1.0
        elif name in ("SNOW", "HAIL"):
            snow = 1.0
    none = 0.0 if battle.weather else 1.0
    return [sun, rain, sand, snow, none]


def _bench(battle):
    """For up to 5 non-active team members: [hp fraction, offense, defense] vs opponent."""
    opp = battle.opponent_active_pokemon
    feats = []
    benched = [m for m in battle.team.values() if m is not battle.active_pokemon][:5]
    for mon in benched:
        feats += [mon.current_hp_fraction] + _matchup(mon, opp)
    while len(feats) < 15:
        feats.append(0.0)
    return feats


def _opponent_moves(battle):
    """Opponent's revealed moves: up to 4 base powers, plus their effectiveness on us."""
    me = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    power = [0.0] * 4
    effectiveness = [0.0] * 4
    if opp is not None:
        for i, move in enumerate(list(opp.moves.values())[:4]):
            power[i] = move.base_power / 100.0
            if me is not None:
                effectiveness[i] = me.damage_multiplier(move) / 4.0
    return power + effectiveness


# Total length of the observation vector (see embed_battle for the layout).
N_FEATURES = 141


class ShowdownSinglesEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Negative low because stat boosts can be negative.
        obs_space = Box(low=-1.0, high=4.0, shape=(N_FEATURES,), dtype=np.float32)
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}

    def embed_battle(self, battle):
        """Describe the current battle to the network as N_FEATURES numbers."""
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        # Our active Pokemon's (up to 4) moves.
        move_power = [0.0] * 4
        move_multiplier = [0.0] * 4
        move_accuracy = [0.0] * 4
        move_physical = [0.0] * 4
        move_status = [0.0] * 4
        for i, move in enumerate(battle.available_moves[:4]):
            move_power[i] = move.base_power / 100.0
            if opp is not None:
                move_multiplier[i] = opp.damage_multiplier(move) / 4.0
            acc = move.accuracy
            move_accuracy[i] = 1.0 if acc is True else float(acc)
            category = move.category.name
            move_physical[i] = 1.0 if category == "PHYSICAL" else 0.0
            move_status[i] = 1.0 if category == "STATUS" else 0.0

        my_hp = me.current_hp_fraction if me else 0.0
        opp_hp = opp.current_hp_fraction if opp else 0.0
        my_remaining = sum(1 for m in battle.team.values() if not m.fainted) / 6.0
        opp_fainted = sum(1 for m in battle.opponent_team.values() if m.fainted)
        opp_remaining = (6 - opp_fainted) / 6.0
        faster = 1.0 if (me and opp and me.base_stats["spe"] > opp.base_stats["spe"]) else 0.0

        features = (
            move_power + move_multiplier + move_accuracy + move_physical + move_status  # 20
            + [my_hp, opp_hp, my_remaining, opp_remaining]          # 4
            + _base_stats(me) + _base_stats(opp)                    # 12
            + _boosts(me) + _boosts(opp)                            # 10
            + _status_onehot(me) + _status_onehot(opp)              # 12
            + _matchup(me, opp) + [faster]                          # 3
            + _effects(me) + _effects(opp)                          # 8
            + _ability_flags(me) + _ability_flags(opp, infer_possible=True)  # 24
            + _bench(battle)                                        # 15
            + _opponent_moves(battle)                               # 8
            + _hazards(battle.side_conditions)                      # 3
            + _hazards(battle.opponent_side_conditions)             # 3
            + _terrain(battle) + _trick_room(battle)                # 6
            + _screens(battle.side_conditions)                      # 4
            + _screens(battle.opponent_side_conditions)             # 4
            + _weather(battle)                                      # 5
        )
        return np.array(features, dtype=np.float32)

    def calc_reward(self, battle):
        """Reward = change in how good our position is since last turn.

        Winning is worth a lot (+30). In between, the agent gets small rewards for
        making the opponent faint / lose HP and small penalties for the reverse,
        so it gets useful feedback long before the battle ends.
        """
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,   # each faint swing is worth 2
            hp_value=1.0,        # reward chipping HP
            status_value=0.2,    # small bonus for inflicting status
            victory_value=30.0,  # winning dominates everything else
        )
