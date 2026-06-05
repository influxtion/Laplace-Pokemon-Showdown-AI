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
"""

import numpy as np
from gymnasium.spaces import Box

from poke_env.environment.singles_env import SinglesEnv

# Size of our observation vector (see embed_battle below).
N_FEATURES = 12


class ShowdownSinglesEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Tell the framework what our observation looks like: N_FEATURES floats.
        # (The base class automatically bundles an "action_mask" alongside this,
        # so the agent also knows which actions are currently legal.)
        obs_space = Box(low=0.0, high=4.0, shape=(N_FEATURES,), dtype=np.float32)
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}

    def embed_battle(self, battle):
        """Describe the current battle to the network as N_FEATURES numbers."""
        opponent = battle.opponent_active_pokemon

        # Our active Pokemon's (up to 4) moves: their power and type effectiveness.
        move_power = np.zeros(4, dtype=np.float32)
        move_multiplier = np.zeros(4, dtype=np.float32)
        for i, move in enumerate(battle.available_moves[:4]):
            move_power[i] = move.base_power / 100.0  # ~0-2.5
            if opponent is not None:
                move_multiplier[i] = opponent.damage_multiplier(move) / 4.0  # 0-1

        # How many Pokemon each side has left (a sense of who's "ahead").
        my_remaining = sum(1 for m in battle.team.values() if not m.fainted) / 6.0
        opp_fainted = sum(1 for m in battle.opponent_team.values() if m.fainted)
        opp_remaining = (6 - opp_fainted) / 6.0

        # Current HP of whoever is out right now.
        my_hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
        opp_hp = opponent.current_hp_fraction if opponent else 0.0

        return np.concatenate(
            [move_power, move_multiplier, [my_remaining, opp_remaining, my_hp, opp_hp]]
        ).astype(np.float32)

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
