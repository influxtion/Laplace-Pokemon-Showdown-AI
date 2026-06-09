r"""Test-time search: a 1-ply lookahead layered on the trained policy.

This isn't MCTS. True MCTS needs a forward simulator to roll the game out, and we don't have
one: the dynamics live in the server, the opponent's set/HP/item are hidden, and there's no
cheap clone-and-sim. Building a battle simulator is a separate project, and a bad one would
hurt.

So instead, for every legal action we compute a one-turn outcome from our damage calculator
and opponent set-prediction (knowledge.py) -- net HP swing, who KOs first, whether a switch
escapes a losing matchup -- and pick the best. The policy is kept only as a small prior to
break ties among comparable actions, so the damage model can override a clearly bad choice.

Scoring every action matters. The old version only re-ranked the policy's top-k, so if the
policy never proposed a switch the searcher couldn't switch and would sit in a losing matchup
clicking a resisted move. Evaluating all legal actions lets it switch, finish with priority,
and skip bad attacks on its own.

It also handles Terastallization (action ids 22-25 are move + terastallize). Tera changes your
typing, hence your STAB and what hits you super-effectively; a tera action is scored under the
tera typing and only preferred when it actually helps, so the agent won't tera into a weakness.

It's a poke-env Player, used at test time -- see eval_search.py / play.py.
"""

import numpy as np
import torch

from poke_env.player import Player
from poke_env.battle import Move, Pokemon
from poke_env.data import to_id_str
from poke_env.environment.singles_env import SinglesEnv

from rl_env import build_observation, _eff_speed
from knowledge import KNOWLEDGE, estimate_damage_fraction, safe_priority, TYPE_NAMES

# Type chart for Tera reasoning (an attacking type's effectiveness against a defending type).
# Guarded so a poke-env API change degrades to no tera modelling instead of crashing; the
# all-action evaluation is the main win regardless.
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


# Abilities granting a full immunity to one attacking type (ability id -> type name). Clicking
# such a move into a holder does nothing, and several of these heal or boost it, so it's
# actively bad. Ids are to_id_str form, matching knowledge.predicted_abilities keys.
IMMUNITY_ABILITIES = {
    "levitate": "GROUND", "eartheater": "GROUND",
    "flashfire": "FIRE", "wellbakedbody": "FIRE",
    "waterabsorb": "WATER", "stormdrain": "WATER", "dryskin": "WATER",
    "voltabsorb": "ELECTRIC", "lightningrod": "ELECTRIC", "motordrive": "ELECTRIC",
    "sapsipper": "GRASS",
}


def _immunity_chance(opp, move_type):
    """Probability the opponent's ability makes it immune to this move's type. Reads the
    set-narrowed prediction (knowledge.predicted_abilities), so an unrevealed Orthworm (always
    Earth Eater) reads ~1.0 against Ground and a coin-flip ability reads ~0.5."""
    if opp is None or move_type is None:
        return 0.0
    chance = sum(p for aid, p in KNOWLEDGE.predicted_abilities(opp).items()
                 if IMMUNITY_ABILITIES.get(aid) == move_type.name)
    return min(chance, 1.0)


def _own_immune_types(mon):
    """The type our own mon is immune to via its known ability, as a set, so the opponent's
    moves of that type don't scare us (Levitate vs Ground). None if no such immunity. Type-based
    immunities (Flying vs Ground) already read 0 from the type chart."""
    if mon is None or not getattr(mon, "ability", None):
        return None
    t = IMMUNITY_ABILITIES.get(to_id_str(mon.ability))
    return {t} if t else None


class SearchPlayer(Player):
    """Scores every legal action with a 1-turn lookahead, using the policy as a prior."""

    # How much the policy's probability counts. Action scores are in HP-fraction units
    # (~[-2, 2.5]); at 0.3 a confident policy only shifts ties, so a clear matchup difference
    # (>1.0) still wins. The policy advises, the damage model decides.
    POLICY_PRIOR_WEIGHT = 0.3
    # Tera is a one-per-game resource. A tera action must beat its plain twin by at least this
    # much (extra damage + reduced incoming, in HP fractions) or it's penalised, so the agent
    # saves tera instead of wasting it or tera-ing into a weakness.
    TERA_MIN_BENEFIT = 0.10
    TERA_COST = 0.50
    # Extra penalty (on top of lost damage) for a move the opponent's ability might nullify,
    # scaled by that probability: a wasted turn, and many immunity abilities heal/boost the
    # holder. Makes an immunity-type move too risky unless the chance is low or nothing's better.
    IMMUNITY_RISK = 0.60
    # How much a switch-in's best damage into the opponent counts. Lets the agent switch for
    # offense -- bail on a mon whose moves are all resisted -- discounted since that damage only
    # lands next turn.
    SWITCH_OFFENSE_WEIGHT = 0.7

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    # --- the learned prior --------------------------------------------------

    def _policy_probs(self, obs, mask):
        """Action probabilities from the masked policy, or None on any failure."""
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
        """Our move's damage if we Terastallize: as normal but with the tera STAB.

        After tera you keep 1.5x STAB on moves matching your original types and gain 1.5x on
        the tera type (2x if the tera type matches an original type). So a Flying move on a
        Psychic/Flying mon that tera's to Fighting gets no boost."""
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
        """Scariest incoming hit if our typing became just the tera type. Scales the normal
        estimate by how the opponent's predicted coverage hits tera vs our current types, so
        tera-ing into a weakness (Fighting vs a Fairy attacker) reads as more danger."""
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
        """Small penalty for moves that lower the user's own stats (Overheat, Draco Meteor,
        Close Combat): a real cost the raw damage number misses."""
        drops = 0
        for v in (getattr(move, "self_boost", None) or {}).values():
            if v < 0:
                drops += -v
        return 0.08 * drops

    # --- the 1-turn lookahead score ----------------------------------------

    def _move_score(self, me, move, opp, tera=False):
        """Higher = better 1-turn exchange. Net HP swing (damage we deal minus the
        probability-weighted hit back), plus bonuses for who secures the KO given turn order, a
        penalty for self-stat-drops, and for tera actions the tera typing and a cost for
        spending tera without a clear payoff."""
        if me is None or opp is None:
            return 0.0
        base_dmg = min(estimate_damage_fraction(me, move, opp), 1.0)
        base_incoming = KNOWLEDGE.predicted_incoming(opp, me, immune_types=_own_immune_types(me))

        dmg, incoming = base_dmg, base_incoming
        tera_type = getattr(me, "tera_type", None)
        if tera and tera_type is not None:
            dmg = min(self._tera_offense(me, move, opp, tera_type), 1.0)
            incoming = self._tera_incoming(me, opp, tera_type, base_incoming)

        # Ability immunity: discount damage by the chance it does nothing (so it can't "KO")
        # and penalise the risk separately. Tera doesn't change the move's type, so this is
        # keyed off move.type either way.
        immunity = _immunity_chance(opp, move.type)
        dmg *= (1.0 - immunity)

        # Turn order: our move's priority, else raw speed. We can't see the opponent's move, so
        # this ignores their priority -- a 1-ply approximation.
        i_move_first = safe_priority(move) > 0 or _eff_speed(me) > _eff_speed(opp)
        i_ko = dmg >= opp.current_hp_fraction
        they_ko = incoming >= me.current_hp_fraction

        score = dmg - incoming                               # net HP traded this turn
        if i_ko and i_move_first:
            score += 1.5                                     # clean KO, take nothing back
        elif i_ko:
            score += 1.0                                     # KO, but we eat a hit first
            if they_ko:
                score -= 1.0                                 # and might be KO'd before it lands
        elif they_ko and not i_move_first:
            score -= 1.0                                     # KO'd without securing one
        score -= self._self_drop_penalty(move)
        score -= immunity * self.IMMUNITY_RISK               # risk of clicking into an immunity

        if tera and tera_type is not None:
            benefit = (dmg - base_dmg) + (base_incoming - incoming)
            if benefit <= self.TERA_MIN_BENEFIT:
                score -= self.TERA_COST     # don't waste tera or tera into a weakness
        return score

    @staticmethod
    def _switch_score(mon, opp):
        """Score a switch by the matchup it gives us: the switch-in's best damage into the
        opponent (its moves are known, it's our bench) minus the hit it eats coming in and a flat
        tempo cost. Lets the agent bail on a mon that can't hurt the opponent for one that can."""
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
            # No move data yet: fall back to a type-effectiveness proxy (1x->0.25 ... 4x->1.0).
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

            # Score every legal action with the 1-ply model; add the policy prob as a small
            # prior so it only tips comparable actions and breaks ties, never trapping us in a
            # losing line the policy happens to like.
            def total(i):
                prior = probs[i] if probs is not None else 0.0
                return self._action_score(i, battle, me, opp) + self.POLICY_PRIOR_WEIGHT * prior

            best = max(legal, key=total)
            return SinglesEnv.action_to_order(np.int64(best), battle, strict=False)
        except Exception:
            return self.choose_random_move(battle)
