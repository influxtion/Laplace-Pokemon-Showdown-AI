"""An opponent that plays using a trained network. This is what self-play runs against.

Each turn it builds the same picture of the battle the learner sees, runs the network to
pick one of the legal actions, and turns that into an actual move or switch.

It's a copy of the agent frozen in time. The training script periodically copies the
learner's current weights across, so the opponent improves alongside the agent instead of
staying a fixed target.
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
        """Bring this opponent up to date with the learner's current weights."""
        self.model.policy.load_state_dict(learner.policy.state_dict())

    def choose_move(self, battle):
        # Fall back to a random legal move if the network isn't ready or anything goes
        # wrong. One bad turn shouldn't bring down a training run that's hours deep.
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
