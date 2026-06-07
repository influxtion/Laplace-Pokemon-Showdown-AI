r"""Test-time search: shallow (1-ply) lookahead layered on the trained policy.

WHAT THIS IS (and isn't): true MCTS needs a forward simulator to roll the game out from
hypothetical states many times per move. We don't have one -- Pokemon's dynamics live in the
Showdown server, the opponent's set/HP/item are hidden, and there's no cheap clone-and-sim.
Building a full battle simulator is a separate large project, and a bad one would hurt.

So this does the tractable, useful thing instead: the trained policy PROPOSES its few most
likely actions, and a one-turn lookahead built on our damage calculator + opponent
set-prediction (knowledge.py) DISPOSES -- picking the candidate that actually KOs / avoids
being KO'd. Lookahead + an explicit damage model on top of a learned policy reliably beats
the raw policy, even though it's not a full game-tree search.

Use it at TEST time (it's a poke-env Player): see eval_search.py.
"""

import os
import sys

# This file lives in v1/v3/ but reuses v1/ modules (rl_env, knowledge). Put the parent v1/
# directory on the import path so those bare imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from poke_env.player import Player
from poke_env.battle import Move, Pokemon
from poke_env.environment.singles_env import SinglesEnv

from rl_env import build_observation, _eff_speed
from knowledge import KNOWLEDGE, estimate_damage_fraction


class SearchPlayer(Player):
    """Plays with `model`, but re-ranks the policy's top-k actions by a 1-turn damage check."""

    def __init__(self, model, top_k=4, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.top_k = top_k

    # --- the learned prior --------------------------------------------------

    def _policy_probs(self, obs, mask):
        """Action probabilities from the trained (masked) policy, or None on any failure."""
        try:
            obs_t, _ = self.model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_t, action_masks=mask)
                return dist.distribution.probs.cpu().numpy().flatten()
        except Exception:
            return None

    # --- the 1-turn lookahead score ----------------------------------------

    def _move_score(self, me, move, opp):
        """Higher = better immediate exchange. KO-before-they-act is rewarded; trading into a
        likely KO on us is penalised."""
        if me is None or opp is None:
            return 0.0
        dmg = estimate_damage_fraction(me, move, opp)        # fraction of opp's HP
        faster = _eff_speed(me) > _eff_speed(opp)
        incoming = KNOWLEDGE.predicted_incoming(opp, me)     # worst predicted hit to us
        score = min(dmg, 1.0)
        if dmg >= opp.current_hp_fraction and faster:
            score += 1.0                                     # we KO before they move
        elif incoming >= me.current_hp_fraction:
            score -= 0.5                                     # we'd likely faint first
        return score

    @staticmethod
    def _switch_score(mon, opp):
        """Mild preference for switching into a favourable type matchup (costs a turn)."""
        if mon is None or opp is None:
            return 0.0
        offense = max((opp.damage_multiplier(t) for t in mon.types if t), default=1.0)
        defense = max((mon.damage_multiplier(t) for t in opp.types if t), default=1.0)
        return 0.3 * (offense - defense)

    def _action_score(self, action, battle, me, opp):
        order = SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
        target = getattr(order, "order", None)
        if isinstance(target, Move):
            return self._move_score(me, target, opp)
        if isinstance(target, Pokemon):
            return self._switch_score(target, opp)
        return 0.0

    # --- decision -----------------------------------------------------------

    def choose_move(self, battle):
        try:
            me = battle.active_pokemon
            opp = battle.opponent_active_pokemon
            mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
            legal = list(np.nonzero(mask)[0])
            if not legal:
                return self.choose_random_move(battle)

            obs = {"observation": build_observation(battle), "action_mask": mask}
            probs = self._policy_probs(obs, mask)

            # Policy proposes its top-k; if probs are unavailable, consider all legal actions.
            if probs is not None:
                candidates = sorted(legal, key=lambda i: probs[i], reverse=True)[:self.top_k]
            else:
                candidates = legal

            # Among candidates, pick the best 1-turn outcome; break ties by policy prob.
            best = max(
                candidates,
                key=lambda i: (self._action_score(i, battle, me, opp),
                               probs[i] if probs is not None else 0.0),
            )
            return SinglesEnv.action_to_order(np.int64(best), battle, strict=False)
        except Exception:
            return self.choose_random_move(battle)
