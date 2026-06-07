"""Self-play opponent: a poke-env Player that picks its moves with a trained model.

`SingleAgentWrapper` drives the opponent by calling `opponent.choose_move(battle)` once per
turn and expects a `BattleOrder` straight back (not a coroutine). So this player:

  1. builds the exact same observation the learning agent sees (`rl_env.build_observation`),
  2. runs the model to pick a legal action (using the action mask), and
  3. converts that action index into a move/switch order.

The opponent is a *snapshot* of the agent. `sync_from` copies the learner's current weights
into it; the training script calls that every so often so the opponent slowly tracks the
agent instead of being a fixed target.
"""

import os
import sys

# This file lives in v1/selfplay/, but reuses modules in v1/ (rl_env, train_rl). Put the
# parent v1/ directory on the import path so those bare imports resolve when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from poke_env.player import Player
from poke_env.environment.singles_env import SinglesEnv

from rl_env import build_observation


class ModelPlayer(Player):
    def __init__(self, model=None, deterministic=False, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.deterministic = deterministic

    def sync_from(self, learner):
        """Copy the learner's current network weights into this opponent."""
        self.model.policy.load_state_dict(learner.policy.state_dict())

    def choose_move(self, battle):
        # Fall back to a random legal move if the model isn't ready or anything goes wrong;
        # a single bad turn should never crash a multi-hour training run.
        if self.model is None:
            return self.choose_random_move(battle)
        try:
            mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
            obs = {"observation": build_observation(battle), "action_mask": mask}
            action, _ = self.model.predict(
                obs, action_masks=mask, deterministic=self.deterministic
            )
            action = np.int64(np.asarray(action).flatten()[0])
            return SinglesEnv.action_to_order(action, battle, strict=False)
        except Exception:
            return self.choose_random_move(battle)
