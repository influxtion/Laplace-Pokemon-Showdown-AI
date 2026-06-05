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

The observation here is deliberately rich (60 numbers). An earlier 12-number
version plateaued against a strong opponent because it couldn't "see" things
like stat boosts, status, hazards, or the opponent's stats. The features below
add that missing information so the agent has enough to learn real strategy.
"""

import numpy as np
from gymnasium.spaces import Box

from poke_env.environment.singles_env import SinglesEnv

# --- helpers -----------------------------------------------------------------

STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]   # base-stat dictionary keys
BOOST_KEYS = ["atk", "def", "spa", "spd", "spe"]        # in-battle stat-stage keys
STATUS_NAMES = ["BRN", "PAR", "SLP", "FRZ", "PSN", "TOX"]  # the status conditions we encode


def _base_stats(mon):
    """6 base stats, scaled to roughly 0-1 (0 if the Pokemon is unknown)."""
    if mon is None:
        return [0.0] * 6
    return [mon.base_stats[k] / 200.0 for k in STAT_KEYS]


def _boosts(mon):
    """5 stat-stage boosts, each in -1..1 (a +6 Swords Dance reads as +1 attack)."""
    if mon is None:
        return [0.0] * 5
    return [mon.boosts[k] / 6.0 for k in BOOST_KEYS]


def _status_onehot(mon):
    """One-hot of the active status condition (all zeros = healthy)."""
    vec = [0.0] * len(STATUS_NAMES)
    if mon is not None and mon.status is not None and mon.status.name in STATUS_NAMES:
        vec[STATUS_NAMES.index(mon.status.name)] = 1.0
    return vec


def _matchup(me, opp):
    """Best type effectiveness I have into them, and they have into me (0-1 each)."""
    if me is None or opp is None:
        return [0.0, 0.0]
    offense = max((opp.damage_multiplier(t) for t in me.types if t is not None), default=1.0)
    defense = max((me.damage_multiplier(t) for t in opp.types if t is not None), default=1.0)
    return [offense / 4.0, defense / 4.0]


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


# Total length of the observation vector built by embed_battle below.
#   moves 8 + hp 2 + team 2 + base stats 12 + boosts 10 + status 12
#   + matchup 2 + speed 1 + hazards 6 + weather 5  =  60
N_FEATURES = 60


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

        # Our active Pokemon's (up to 4) moves: power and type effectiveness.
        move_power = [0.0] * 4
        move_multiplier = [0.0] * 4
        for i, move in enumerate(battle.available_moves[:4]):
            move_power[i] = move.base_power / 100.0
            if opp is not None:
                move_multiplier[i] = opp.damage_multiplier(move) / 4.0

        # Current HP of whoever is out right now.
        my_hp = me.current_hp_fraction if me else 0.0
        opp_hp = opp.current_hp_fraction if opp else 0.0

        # How many Pokemon each side has left.
        my_remaining = sum(1 for m in battle.team.values() if not m.fainted) / 6.0
        opp_fainted = sum(1 for m in battle.opponent_team.values() if m.fainted)
        opp_remaining = (6 - opp_fainted) / 6.0

        # Is our active faster (by base speed)?
        faster = 1.0 if (me and opp and me.base_stats["spe"] > opp.base_stats["spe"]) else 0.0

        features = (
            move_power                              # 4
            + move_multiplier                       # 4
            + [my_hp, opp_hp]                        # 2
            + [my_remaining, opp_remaining]         # 2
            + _base_stats(me) + _base_stats(opp)    # 12
            + _boosts(me) + _boosts(opp)            # 10
            + _status_onehot(me) + _status_onehot(opp)  # 12
            + _matchup(me, opp)                     # 2
            + [faster]                              # 1
            + _hazards(battle.side_conditions)      # 3
            + _hazards(battle.opponent_side_conditions)  # 3
            + _weather(battle)                      # 5
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
