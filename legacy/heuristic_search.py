r"""The one-turn searcher, plus the handful of things it was missing.

The first attempt at this was a fresh scoring function written from scratch, and it was
much worse: 33% against the existing searcher's 57%. It had quietly re-derived, badly, all
the matchup and turn-order tuning that was already in there. So this version leaves that
alone and only adds what's genuinely absent, each piece testable on its own:

  - Future Sight and Doom Desire no longer count as damage this turn. The old scorer saw
    their raw power and read them as an instant kill, so a Slowbro would click Future Sight
    into a healthy target over and over, doing nothing while it got chipped down.
  - Moves that do something other than damage now get credit: inflicting status, setting up,
    laying hazards, and healing (worth more the more hurt we are).
  - Each option is scored on its own, so one option throwing an error can't drop the whole
    turn to a random move.

Used at test time by eval_search.py and play.py.
"""

import numpy as np

from poke_env.battle import Move, Pokemon
from poke_env.environment.singles_env import SinglesEnv

from rl_env import build_observation
from knowledge import KNOWLEDGE, estimate_damage_fraction, HAZARD_CONDS
from search import SearchPlayer, _own_immune_types


class HeuristicSearchPlayer(SearchPlayer):
    """The one-turn searcher, with delayed moves fixed and side effects given credit."""

    # These land two turns later, so this turn they do nothing at all. We give partial
    # credit for the eventual hit, since the opponent may well switch or kill us first.
    DELAYED_MOVES = {"futuresight", "doomdesire"}
    DELAYED_DISCOUNT = 0.5

    # What the non-damage effects are worth, in the same health units as the main score.
    STATUS_VALUE = 0.25      # a burn or paralysis on something healthy wins long games
    SETUP_VALUE = 0.13       # per stat stage gained
    HAZARD_VALUE = 0.30      # a fresh layer of hazards
    RECOVERY_VALUE = 0.6     # scaled by how much we heal and how hurt we already are

    def __init__(self, model, debug=False, **kwargs):
        super().__init__(model, **kwargs)
        self.debug = debug
        self._cur_battle = None      # kept so the bonus terms can look at the field

    # ----------------------------- changes to the scoring ------------------------

    def _move_score(self, me, move, opp, tera=False, seeded=False):
        """The usual score, with delayed moves handled properly and side effects added."""
        if me is None or opp is None:
            return 0.0
        if getattr(move, "id", None) in self.DELAYED_MOVES:
            score = self._delayed_move_score(me, move, opp)
        else:
            score = super()._move_score(me, move, opp, tera=tera, seeded=seeded)
        return score + self._side_effect_bonus(me, move, opp)

    def _delayed_move_score(self, me, move, opp):
        """Future Sight and friends: nothing lands this turn, we just take a hit, with
        partial credit for the damage arriving later."""
        incoming = KNOWLEDGE.predicted_incoming(opp, me, immune_types=_own_immune_types(me))
        future = min(estimate_damage_fraction(me, move, opp), 1.0)
        return self.DELAYED_DISCOUNT * future - incoming

    def _side_effect_bonus(self, me, move, opp):
        """Credit for everything a move does that isn't damage."""
        extra = 0.0
        try:
            if (move.category.name == "STATUS" and getattr(move, "status", None) is not None
                    and opp.status is None and self._status_lands(move, opp)):
                extra += self.STATUS_VALUE
        except Exception:
            pass

        for v in (getattr(move, "boosts", None) or {}).values():   # setting up
            if v > 0:
                extra += self.SETUP_VALUE * v

        sc = getattr(move, "side_condition", None)
        if sc is not None and sc.name in HAZARD_CONDS and not self._hazard_maxed(sc):
            extra += self.HAZARD_VALUE

        heal = getattr(move, "heal", 0) or 0       # healing, worth more the more hurt we are
        if heal > 0 and me is not None:
            extra += self.RECOVERY_VALUE * heal * (1.0 - me.current_hp_fraction)
        return extra

    @staticmethod
    def _status_lands(move, opp):
        """Can this status actually stick? Stops us congratulating ourselves for burning a
        Fire type or paralysing an Electric one."""
        try:
            st = move.status.name
            types = {t.name for t in opp.types if t}
        except Exception:
            return True
        if st == "BRN" and "FIRE" in types:
            return False
        if st == "PAR" and "ELECTRIC" in types:
            return False
        if st in ("PSN", "TOX") and ("POISON" in types or "STEEL" in types):
            return False
        if st == "FRZ" and "ICE" in types:
            return False
        if getattr(move, "id", None) == "thunderwave" and "GROUND" in types:
            return False     # Thunder Wave is Electric, and Ground types ignore it
        return True

    def _hazard_maxed(self, side_condition):
        """Is this hazard already maxed out on their side, so another layer does nothing?"""
        opp_side = getattr(self._cur_battle, "opponent_side_conditions", None) or {}
        layers = opp_side.get(side_condition, 0)
        name = side_condition.name
        return ((name == "STEALTH_ROCK" and layers >= 1)
                or (name == "SPIKES" and layers >= 3)
                or (name == "TOXIC_SPIKES" and layers >= 2))

    # ------------------------------ picking, and explaining -----------------------

    def _describe(self, action, battle):
        try:
            order = SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
            target = getattr(order, "order", None)
            if isinstance(target, Pokemon):
                return f"switch {target.species}"
            if isinstance(target, Move):
                return f"{target.id}{' +tera' if getattr(order, 'terastallize', False) else ''}"
        except Exception:
            pass
        return f"action{action}"

    def _log_decision(self, battle, scored):
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        mine = f"{me.species} {me.current_hp_fraction:.0%}" if me else "-"
        theirs = f"{opp.species} {opp.current_hp_fraction:.0%}" if opp else "-"
        print(f"[T{battle.turn}] {mine}  vs  {theirs}", flush=True)
        for value, i in scored[:5]:
            print(f"    {value:+.3f}  {self._describe(i, battle)}", flush=True)

    def choose_move(self, battle):
        try:
            self._cur_battle = battle
            me = battle.active_pokemon
            opp = battle.opponent_active_pokemon
            mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
            legal = list(np.nonzero(mask)[0])
            if not legal:
                return self.choose_random_move(battle)

            obs = {"observation": build_observation(battle), "action_mask": mask}
            probs = self._policy_probs(obs, mask)

            # Score each option separately, so one blowing up doesn't cost us the turn.
            scored = []
            for i in legal:
                try:
                    v = self._action_score(i, battle, me, opp)
                except Exception:
                    v = float("-inf")
                prior = probs[i] if probs is not None else 0.0
                scored.append((v + self.POLICY_PRIOR_WEIGHT * prior, i))
            scored.sort(key=lambda t: t[0], reverse=True)

            if self.debug:
                self._log_decision(battle, scored)
            return SinglesEnv.action_to_order(np.int64(scored[0][1]), battle, strict=False)
        except Exception:
            return self.choose_random_move(battle)
