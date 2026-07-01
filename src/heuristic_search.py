r"""HeuristicSearchPlayer: SearchPlayer's proven 1-ply scoring, plus the heuristics it lacks.

A from-scratch static-evaluation bot was tried first and regressed (33% vs SearchPlayer's
57%) -- it re-derived, worse, the matchup/turn-order tuning SearchPlayer already encodes (the
seven replay fixes, tera, ability immunity, Leech Seed). So instead of replacing that, this
*subclasses* SearchPlayer and adds only what's genuinely missing, each piece measurable on its
own against the 57% baseline:

  - Delayed moves (Future Sight / Doom Desire) no longer count as this-turn damage. The base
    scorer read their 120 base power as an immediate KO, so a Slowbro would click Future Sight
    into a full-HP target turn after turn, doing nothing while it got chipped.
  - Move side-effects the max-damage core ignores get credit: inflicting status, setup boosts,
    laying entry hazards, and recovery (valued more the more hurt we are).
  - choose_move scores each action independently, so one action raising an exception can't drop
    the whole turn to a random move (the inherited version evaluates them inside one max()).

Still to come on this base: entry-hazard-aware switching and a 1-turn expectiminimax over the
opponent's move distribution (knowledge.py already provides it). Build + benchmark each.

A poke-env Player, used at test time -- see eval_search.py / play.py.
"""

import numpy as np

from poke_env.battle import Move, Pokemon
from poke_env.environment.singles_env import SinglesEnv

from rl_env import build_observation
from knowledge import KNOWLEDGE, estimate_damage_fraction, HAZARD_CONDS
from search import SearchPlayer, _own_immune_types


class HeuristicSearchPlayer(SearchPlayer):
    """SearchPlayer + delayed-move fix + status/setup/hazard/recovery valuation."""

    # Future Sight / Doom Desire land ~2 turns later; this turn they deal nothing. We value the
    # eventual hit at a discount (the opponent may switch or KO us first) minus the hit we eat now.
    DELAYED_MOVES = {"futuresight", "doomdesire"}
    DELAYED_DISCOUNT = 0.5

    # Side-effect bonuses, in the same HP-fraction units as the base score (~[-2, 2.5]).
    STATUS_VALUE = 0.25      # burn/para/tox/sleep on a healthy opponent -> attrition wins
    SETUP_VALUE = 0.13       # per net positive stat stage from a setup move
    HAZARD_VALUE = 0.30      # laying a fresh layer of entry hazards
    RECOVERY_VALUE = 0.6     # scaled by heal amount * how much HP we're missing

    def __init__(self, model, debug=False, **kwargs):
        super().__init__(model, **kwargs)
        self.debug = debug
        self._cur_battle = None      # set each turn so bonus terms can read side conditions

    # ----------------------------- scoring overrides -----------------------------

    def _move_score(self, me, move, opp, tera=False, seeded=False):
        """SearchPlayer's score, but with delayed moves corrected and side-effects credited."""
        if me is None or opp is None:
            return 0.0
        if getattr(move, "id", None) in self.DELAYED_MOVES:
            score = self._delayed_move_score(me, move, opp)
        else:
            score = super()._move_score(me, move, opp, tera=tera, seeded=seeded)
        return score + self._side_effect_bonus(me, move, opp)

    def _delayed_move_score(self, me, move, opp):
        """Future Sight / Doom Desire: no damage this turn, just the hit we take, plus a
        discounted credit for the delayed strike."""
        incoming = KNOWLEDGE.predicted_incoming(opp, me, immune_types=_own_immune_types(me))
        future = min(estimate_damage_fraction(me, move, opp), 1.0)
        return self.DELAYED_DISCOUNT * future - incoming

    def _side_effect_bonus(self, me, move, opp):
        """Value of a move's secondary effects the raw damage model misses."""
        extra = 0.0
        try:
            if (move.category.name == "STATUS" and getattr(move, "status", None) is not None
                    and opp.status is None and self._status_lands(move, opp)):
                extra += self.STATUS_VALUE
        except Exception:
            pass

        for v in (getattr(move, "boosts", None) or {}).values():   # setup (self stat raises)
            if v > 0:
                extra += self.SETUP_VALUE * v

        sc = getattr(move, "side_condition", None)
        if sc is not None and sc.name in HAZARD_CONDS and not self._hazard_maxed(sc):
            extra += self.HAZARD_VALUE

        heal = getattr(move, "heal", 0) or 0       # recovery, worth more the more HP we've lost
        if heal > 0 and me is not None:
            extra += self.RECOVERY_VALUE * heal * (1.0 - me.current_hp_fraction)
        return extra

    @staticmethod
    def _status_lands(move, opp):
        """Whether a status move's status can actually afflict this target (type immunities),
        so we don't credit burning a Fire type or paralysing an Electric one."""
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
            return False     # Ground is immune to the Electric-type Thunder Wave
        return True

    def _hazard_maxed(self, side_condition):
        """True if this hazard is already at its max layers on the opponent's side (so a fresh
        layer would do nothing)."""
        opp_side = getattr(self._cur_battle, "opponent_side_conditions", None) or {}
        layers = opp_side.get(side_condition, 0)
        name = side_condition.name
        return ((name == "STEALTH_ROCK" and layers >= 1)
                or (name == "SPIKES" and layers >= 3)
                or (name == "TOXIC_SPIKES" and layers >= 2))

    # ------------------------------ decision + debug ------------------------------

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

            # Score each action independently: a failure on one must not randomise the turn.
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
