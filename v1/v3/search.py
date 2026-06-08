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
from knowledge import KNOWLEDGE, estimate_damage_fraction, safe_priority


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
        """Higher = better 1-turn exchange. Models the turn as a net HP swing (damage we deal
        minus the probability-weighted hit we take back), then adds bonuses/penalties for who
        secures the KO given turn order. This is richer than rewarding KOs alone: it also
        values chip damage and correctly prefers, e.g., a non-KO move that out-speeds a faster
        threat over a KO move that lets us faint first."""
        if me is None or opp is None:
            return 0.0
        dmg = min(estimate_damage_fraction(me, move, opp), 1.0)   # we deal (frac of opp HP)
        incoming = KNOWLEDGE.predicted_incoming(opp, me)          # prob-weighted hit we take
        # Turn order: our move's priority, else raw speed. (We can't see the opponent's chosen
        # move, so this ignores opponent priority -- an acknowledged 1-ply approximation.)
        i_move_first = safe_priority(move) > 0 or _eff_speed(me) > _eff_speed(opp)
        i_ko = dmg >= opp.current_hp_fraction
        they_ko = incoming >= me.current_hp_fraction

        score = dmg - incoming                               # base: net HP traded this turn
        if i_ko and i_move_first:
            score += 1.5                                     # clean KO, we take nothing back
        elif i_ko:
            score += 1.0                                     # KO, but we eat a hit first...
            if they_ko:
                score -= 1.0                                 # ...and might be KO'd before it lands
        elif they_ko and not i_move_first:
            score -= 1.0                                     # we get KO'd without securing one
        return score

    @staticmethod
    def _switch_score(mon, opp):
        """Score a switch: reward a favourable type matchup, but subtract the hit the switch-in
        eats on the way in and a flat tempo cost (switching gives up the turn)."""
        if mon is None or opp is None:
            return 0.0
        offense = max((opp.damage_multiplier(t) for t in mon.types if t), default=1.0)
        defense = max((mon.damage_multiplier(t) for t in opp.types if t), default=1.0)
        matchup = 0.3 * (offense - defense)
        incoming = KNOWLEDGE.predicted_incoming(opp, mon)    # what the switch-in takes next turn
        return matchup - 0.5 * incoming - 0.15               # -0.15: lost tempo

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
