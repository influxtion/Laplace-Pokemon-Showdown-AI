r"""Test-time search: a 1-ply lookahead evaluator layered on the trained policy.

WHAT THIS IS (and isn't): true MCTS needs a forward simulator to roll the game out from
hypothetical states many times per move. We don't have one -- Pokemon's dynamics live in the
Showdown server, the opponent's set/HP/item are hidden, and there's no cheap clone-and-sim.
Building a full battle simulator is a separate large project, and a bad one would hurt.

So this does the tractable, useful thing instead: for EVERY legal action it computes a
one-turn outcome from our damage calculator + opponent set-prediction (knowledge.py) -- net
HP swing, who KOs first, whether a switch escapes a losing matchup -- and picks the best.
The trained policy is kept only as a small PRIOR (a tie-breaker among comparable actions), so
the damage model can override a clearly bad policy choice instead of being trapped by it.

WHY EVALUATE ALL ACTIONS (this was the big fix): the old version only re-ranked the policy's
top-k actions. If the policy never proposed a switch, the searcher literally could not switch,
so it would sit in a losing matchup and click a resisted move. Scoring every legal action
fixes that -- it can switch, finish with priority, and avoid bad attacks on its own.

It also understands TERASTALLIZATION (action ids 22-25 are "move + terastallize"): tera changes
your typing, which changes your move's STAB and what hits you super-effectively. The plain
score ignored that, so the agent would tera into a fresh weakness for no gain. Here a tera
action is scored under the tera typing and is only preferred when it actually helps.

Use it at TEST time (it's a poke-env Player): see eval_search.py / play.py.
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
from poke_env.data import to_id_str
from poke_env.environment.singles_env import SinglesEnv

from rl_env import build_observation, _eff_speed
from knowledge import KNOWLEDGE, estimate_damage_fraction, safe_priority, TYPE_NAMES

# Type chart for reasoning about Terastallization (effectiveness of an attacking type against
# an arbitrary defending type). Guarded so an API change degrades to "no tera modelling"
# rather than crashing the agent: the all-action evaluation below is the main win regardless.
try:
    from poke_env.battle import PokemonType
    from poke_env.data import GenData

    _TYPE_CHART = GenData.from_gen(9).type_chart
    _TYPE_BY_INDEX = [PokemonType[name] for name in TYPE_NAMES]
except Exception:
    _TYPE_CHART = None
    _TYPE_BY_INDEX = None


def _type_eff(attacking_type, defending_types):
    """Effectiveness multiplier of one attacking type against a list of defending types."""
    defs = [t for t in defending_types if t is not None]
    if attacking_type is None or _TYPE_CHART is None or not defs:
        return 1.0
    try:
        return attacking_type.damage_multiplier(*defs, type_chart=_TYPE_CHART)
    except Exception:
        return 1.0


# Abilities that grant a full immunity to one attacking TYPE (ability id -> type name). Clicking
# such a move into a holder does nothing -- and several of these heal or boost the holder, so
# it's actively bad. Ids are to_id_str form, matching knowledge.predicted_abilities keys.
IMMUNITY_ABILITIES = {
    "levitate": "GROUND", "eartheater": "GROUND",
    "flashfire": "FIRE", "wellbakedbody": "FIRE",
    "waterabsorb": "WATER", "stormdrain": "WATER", "dryskin": "WATER",
    "voltabsorb": "ELECTRIC", "lightningrod": "ELECTRIC", "motordrive": "ELECTRIC",
    "sapsipper": "GRASS",
}


def _immunity_chance(opp, move_type):
    """Probability the opponent's ability makes it immune to this move's type. Reads the
    set-narrowed ability prediction (knowledge.predicted_abilities), so an unrevealed Orthworm
    (always Earth Eater) reads ~1.0 against Ground and a coin-flip ability reads ~0.5."""
    if opp is None or move_type is None:
        return 0.0
    chance = sum(p for aid, p in KNOWLEDGE.predicted_abilities(opp).items()
                 if IMMUNITY_ABILITIES.get(aid) == move_type.name)
    return min(chance, 1.0)


def _own_immune_types(mon):
    """The type name our OWN Pokemon is immune to via its (known) ability, as a set -- so the
    opponent's moves of that type shouldn't scare us (Levitate vs Ground, etc). None if no such
    immunity. Type-based immunities (e.g. Flying vs Ground) already read 0 from the type chart."""
    if mon is None or not getattr(mon, "ability", None):
        return None
    t = IMMUNITY_ABILITIES.get(to_id_str(mon.ability))
    return {t} if t else None


class SearchPlayer(Player):
    """Plays by scoring every legal action with a 1-turn lookahead, using the policy as a prior."""

    # How much the trained policy's probability counts. Action scores are in HP-fraction units
    # (~[-2, 2.5]); at 0.3 a confident policy (prob ~1) only shifts ties, so a clear matchup
    # difference (>1.0) still wins -- the policy advises, the damage model decides.
    POLICY_PRIOR_WEIGHT = 0.3
    # Terastallization is a one-per-game resource. A tera action must beat its plain twin by at
    # least this much (extra damage + reduced incoming, in HP fractions) or it's penalised, so
    # the agent saves tera instead of wasting it -- and never tera's into a weakness for nothing.
    TERA_MIN_BENEFIT = 0.10
    TERA_COST = 0.50
    # Extra penalty (on top of the lost damage) for a move the opponent's ability MIGHT nullify,
    # scaled by that probability: a wasted turn, and many immunity abilities heal/boost the
    # holder. Makes an immunity-type move "too risky" unless the chance is low / nothing better.
    IMMUNITY_RISK = 0.60
    # How much a switch-in's best damage into the opponent counts. Lets the agent switch for an
    # OFFENSIVE advantage -- bail on a Pokemon whose moves are all resisted and bring in one that
    # actually threatens the opponent -- discounted because that damage only lands next turn.
    SWITCH_OFFENSE_WEIGHT = 0.7

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

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

    # --- Terastallization-aware damage helpers ------------------------------

    @staticmethod
    def _tera_offense(me, move, opp, tera_type):
        """Our move's damage if we Terastallize: same as normal but with the tera STAB.

        After tera you keep 1.5x STAB on moves matching your ORIGINAL types and gain 1.5x on
        the tera type (2x if the tera type matches an original type). So a Flying move on a
        Psychic/Flying mon that tera's to Fighting gets no boost -- which is the point."""
        base = estimate_damage_fraction(me, move, opp)
        if base <= 0 or move.type is None:
            return base
        orig_types = [t for t in me.types if t]
        orig_stab = 1.5 if move.type in orig_types else 1.0
        if move.type == tera_type and move.type in orig_types:
            tera_stab = 2.0
        elif move.type == tera_type or move.type in orig_types:
            tera_stab = 1.5
        else:
            tera_stab = 1.0
        return base / orig_stab * tera_stab

    def _tera_incoming(self, me, opp, tera_type, base_incoming):
        """Scariest incoming hit if our defensive typing became just the tera type. Scales the
        normal estimate by how the opponent's predicted coverage hits tera vs our current types,
        so tera-ing into a weakness (e.g. Fighting vs a Fairy attacker) reads as MORE danger."""
        cur_vuln = self._coverage_vuln(opp, [t for t in me.types if t])
        tera_vuln = self._coverage_vuln(opp, [tera_type])
        if cur_vuln <= 0:
            return base_incoming
        return base_incoming * (tera_vuln / cur_vuln)

    @staticmethod
    def _coverage_vuln(opp, defending_types):
        """Expected effectiveness of the opponent's predicted attacking coverage into the given
        defending types (probability-weighted over the types it likely carries)."""
        if _TYPE_BY_INDEX is None:
            return 1.0
        total = 0.0
        for i, p in enumerate(KNOWLEDGE.predicted_coverage(opp)):
            if p > 0:
                total += p * _type_eff(_TYPE_BY_INDEX[i], defending_types)
        return total

    @staticmethod
    def _self_drop_penalty(move):
        """Small penalty for moves that lower the USER's own stats (Overheat, Draco Meteor,
        Close Combat...): real cost the raw damage number doesn't capture."""
        drops = 0
        for v in (getattr(move, "self_boost", None) or {}).values():
            if v < 0:
                drops += -v
        return 0.08 * drops

    # --- the 1-turn lookahead score ----------------------------------------

    def _move_score(self, me, move, opp, tera=False):
        """Higher = better 1-turn exchange. Net HP swing (damage we deal minus the
        probability-weighted hit we take back) plus bonuses for who secures the KO given turn
        order, a penalty for self-stat-drops, and -- for tera actions -- the tera typing and a
        cost for spending tera without a clear payoff."""
        if me is None or opp is None:
            return 0.0
        base_dmg = min(estimate_damage_fraction(me, move, opp), 1.0)
        base_incoming = KNOWLEDGE.predicted_incoming(opp, me, immune_types=_own_immune_types(me))

        dmg, incoming = base_dmg, base_incoming
        tera_type = getattr(me, "tera_type", None)
        if tera and tera_type is not None:
            dmg = min(self._tera_offense(me, move, opp, tera_type), 1.0)
            incoming = self._tera_incoming(me, opp, tera_type, base_incoming)

        # Ability immunity: discount the damage by the chance it does nothing (so it also can't
        # "KO"), and penalise the risk separately. Tera doesn't change the MOVE's type, so this
        # is keyed off move.type either way.
        immunity = _immunity_chance(opp, move.type)
        dmg *= (1.0 - immunity)

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
        score -= self._self_drop_penalty(move)
        score -= immunity * self.IMMUNITY_RISK               # risk of clicking into an immunity

        if tera and tera_type is not None:
            benefit = (dmg - base_dmg) + (base_incoming - incoming)
            if benefit <= self.TERA_MIN_BENEFIT:
                score -= self.TERA_COST     # don't waste tera / tera into a weakness for nothing
        return score

    @staticmethod
    def _switch_score(mon, opp):
        """Score a switch by the matchup the switch-in gives us: the best damage IT could deal to
        the opponent (its moves are known to us, since it's our own bench) minus the hit it eats
        coming in and a flat tempo cost. This is what lets the agent bail on a Pokemon that can't
        hurt the opponent and bring in one that can."""
        if mon is None or opp is None:
            return 0.0
        moves = list(getattr(mon, "moves", {}).values())
        if moves:
            best_offense = max(
                (min(estimate_damage_fraction(mon, mv, opp) * (1.0 - _immunity_chance(opp, mv.type)), 1.0)
                 for mv in moves),
                default=0.0,
            )
        else:
            # No move data yet: fall back to a STAB type-effectiveness proxy (1x->0.25 ... 4x->1.0).
            eff = max((opp.damage_multiplier(t) for t in mon.types if t), default=1.0)
            best_offense = 0.25 * eff
        incoming = KNOWLEDGE.predicted_incoming(opp, mon, immune_types=_own_immune_types(mon))
        return SearchPlayer.SWITCH_OFFENSE_WEIGHT * best_offense - 0.5 * incoming - 0.15

    def _action_score(self, action, battle, me, opp):
        order = SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
        target = getattr(order, "order", None)
        if isinstance(target, Pokemon):
            return self._switch_score(target, opp)
        if isinstance(target, Move):
            return self._move_score(me, target, opp, tera=bool(getattr(order, "terastallize", False)))
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

            # Score EVERY legal action with the 1-ply model; add the policy prob as a small
            # prior so it only tips comparable actions (and breaks ties), never traps us in a
            # losing line the policy happens to like.
            def total(i):
                prior = probs[i] if probs is not None else 0.0
                return self._action_score(i, battle, me, opp) + self.POLICY_PRIOR_WEIGHT * prior

            best = max(legal, key=total)
            return SinglesEnv.action_to_order(np.int64(best), battle, strict=False)
        except Exception:
            return self.choose_random_move(battle)
