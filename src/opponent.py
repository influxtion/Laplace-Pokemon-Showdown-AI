"""Self-play opponent: a poke-env Player that picks moves with a trained model.

SingleAgentWrapper calls choose_move(battle) once per turn and expects a BattleOrder back
(not a coroutine), so this player builds the same observation the learner sees
(rl_env.build_observation), runs the model to pick a legal action via the mask, and converts
that action index into a move/switch order.

It's a snapshot of the agent: sync_from copies the learner's current weights in, which the
training script does periodically so the opponent tracks the agent rather than being fixed.
"""

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
        """Copy the learner's current weights into this opponent."""
        self.model.policy.load_state_dict(learner.policy.state_dict())

    def choose_move(self, battle):
        # Random legal move if the model isn't ready or anything goes wrong; one bad turn
        # shouldn't crash a multi-hour run.
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
