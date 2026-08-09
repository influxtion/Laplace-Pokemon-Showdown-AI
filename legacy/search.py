r"""Think one turn ahead, using the trained network only as a hint.

This is not MCTS. Real MCTS needs to be able to play the game forward in its head, and at
this point in the project we couldn't: the rules live on the server, the opponent's set and
item are hidden, and there's no cheap way to clone a battle and simulate it. Writing a
battle engine is a whole project of its own, and a bad one is worse than none.

So instead, for every legal thing we could do, we work out roughly how the turn goes using
our own damage calculator and set predictions: how much health each side loses, who lands a
kill first, whether switching escapes a bad matchup. Then we take the best one. The trained
network only nudges between options that are already close.

Scoring every option matters. An earlier version only re-ranked the network's top few
suggestions, which meant that if the network never suggested switching, the bot simply
couldn't switch, and would sit in a hopeless matchup clicking a resisted move forever.

Terastallizing is handled too. Tera changes your typing, so it changes both what you hit
hard and what hits you hard. A Tera option is scored under the new typing and only wins if
it genuinely helps, so the bot won't Tera itself into a weakness.

Used at test time by eval_search.py and play.py.
"""

import numpy as np
import torch

from poke_env.player import Player
from poke_env.battle import Move, Pokemon
from poke_env.data import to_id_str
from poke_env.environment.singles_env import SinglesEnv

from rl_env import build_observation, _eff_speed
from knowledge import KNOWLEDGE, estimate_damage_fraction, safe_priority, TYPE_NAMES

# The type chart, for reasoning about Tera. Wrapped in a try so that if the library changes
# underneath us we just lose the Tera modelling rather than crashing outright.
try:
    from poke_env.battle import PokemonType
    from poke_env.data import GenData

    _TYPE_CHART = GenData.from_gen(9).type_chart
    _TYPE_BY_INDEX = [PokemonType[name] for name in TYPE_NAMES]
except Exception:
    _TYPE_CHART = None
    _TYPE_BY_INDEX = None


def _type_eff(attacking_type, defending_types):
    """How effective one attacking type is against a defending typing."""
    defs = [t for t in defending_types if t is not None]
    if attacking_type is None or _TYPE_CHART is None or not defs:
        return 1.0
    try:
        return attacking_type.damage_multiplier(*defs, type_chart=_TYPE_CHART)
    except Exception:
        return 1.0


# Abilities that make something completely immune to a type. Attacking into one of these
# doesn't just waste the turn; several of them heal or boost the target, so it's worse than
# doing nothing.
IMMUNITY_ABILITIES = {
    "levitate": "GROUND", "eartheater": "GROUND",
    "flashfire": "FIRE", "wellbakedbody": "FIRE",
    "waterabsorb": "WATER", "stormdrain": "WATER", "dryskin": "WATER",
    "voltabsorb": "ELECTRIC", "lightningrod": "ELECTRIC", "motordrive": "ELECTRIC",
    "sapsipper": "GRASS",
}


def _immunity_chance(opp, move_type):
    """How likely their ability is to make this move do nothing. Based on which sets they
    could still be running, so an Orthworm, which always has Earth Eater, reads as a certain
    Ground immunity even before it shows, while a fifty-fifty ability reads as a half."""
    if opp is None or move_type is None:
        return 0.0
    chance = sum(p for aid, p in KNOWLEDGE.predicted_abilities(opp).items()
                 if IMMUNITY_ABILITIES.get(aid) == move_type.name)
    return min(chance, 1.0)


def _own_immune_types(mon):
    """Any type our own Pokemon's ability makes it immune to, so we don't flinch at moves
    that can't touch us. Ordinary type immunities already read as zero from the chart; this
    is only for the ability-based ones like Levitate."""
    if mon is None or not getattr(mon, "ability", None):
        return None
    t = IMMUNITY_ABILITIES.get(to_id_str(mon.ability))
    return {t} if t else None


def _is_seeded(mon):
    """Is this Pokemon Leech Seeded? Switching out is the only way to shake it off."""
    return mon is not None and any(e.name == "LEECH_SEED" for e in mon.effects)


class SearchPlayer(Player):
    """Scores everything we could legally do this turn, with the network as a tiebreaker."""

    # How much the network's opinion counts. Scores are measured in health, roughly -2 to
    # 2.5, so at 0.3 even a very confident network only shifts genuinely close calls. A
    # clear matchup difference still wins. The network advises; the damage model decides.
    POLICY_PRIOR_WEIGHT = 0.3
    # Tera is once per game. A Tera option has to beat its plain version by at least this
    # much, or it gets charged for the privilege, so the bot saves it rather than throwing
    # it away or Teraing into a weakness.
    TERA_MIN_BENEFIT = 0.10
    TERA_COST = 0.50
    # An extra penalty, on top of the lost damage, for attacking into a possible immunity.
    # It's a wasted turn, and plenty of those abilities heal or boost their holder.
    IMMUNITY_RISK = 0.60
    # How much a switch-in's own offence counts. This is what lets the bot bail out of a
    # Pokemon whose every move is resisted. Discounted, since that damage only lands next turn.
    SWITCH_OFFENSE_WEIGHT = 0.7
    # Leech Seed drains us every turn and heals them, and switching is the only cure. So we
    # lean towards switching when seeded, unless staying in lands a kill or a big hit, in
    # which case finishing the job beats running away.
    LEECH_SEED_SWITCH_BONUS = 0.35
    LEECH_SEED_STAY_PENALTY = 0.35
    LEECH_SEED_STAY_DMG = 0.5

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    # --- asking the network -----------------------------------------------

    def _policy_probs(self, obs, mask):
        """What the trained network would do here, or None if anything goes wrong."""
        try:
            obs_t, _ = self.model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_t, action_masks=mask)
                return dist.distribution.probs.cpu().numpy().flatten()
        except Exception:
            return None

    # --- damage, with Tera taken into account -------------------------------

    @staticmethod
    def _tera_offense(me, move, opp, tera_type):
        """What this move would do if we Terastallized first.

        After Teraing you keep the usual bonus on moves matching your original types and
        gain one on your Tera type, doubled up if they're the same type. So a Flying move on
        a Psychic/Flying Pokemon that Teras into Fighting loses its bonus entirely."""
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
        """How hard we'd get hit if our typing became the Tera type. Compares how well their
        likely attacks line up against the new typing versus the old one, so Teraing into
        Fighting in front of a Fairy attacker correctly reads as more dangerous."""
        cur_vuln = self._coverage_vuln(opp, [t for t in me.types if t])
        tera_vuln = self._coverage_vuln(opp, [tera_type])
        if cur_vuln <= 0:
            return base_incoming
        return base_incoming * (tera_vuln / cur_vuln)

    @staticmethod
    def _coverage_vuln(opp, defending_types):
        """How well their likely attacks line up against a given typing, weighted by how
        likely they are to actually have each of them."""
        if _TYPE_BY_INDEX is None:
            return 1.0
        total = 0.0
        for i, p in enumerate(KNOWLEDGE.predicted_coverage(opp)):
            if p > 0:
                total += p * _type_eff(_TYPE_BY_INDEX[i], defending_types)
        return total

    @staticmethod
    def _self_drop_penalty(move):
        """A nudge against moves that weaken you when you use them, like Overheat or Close
        Combat. That's a real cost the damage number alone doesn't show."""
        drops = 0
        for v in (getattr(move, "self_boost", None) or {}).values():
            if v < 0:
                drops += -v
        return 0.08 * drops

    # --- scoring one turn ahead ---------------------------------------------

    def _move_score(self, me, move, opp, tera=False, seeded=False):
        """How good this move looks over the coming turn. Higher is better.

        Mostly it's the health we take off them minus the health we expect to lose, with
        bonuses for landing a kill (bigger if we move first), a nudge against moves that
        weaken us, the Tera accounting described above, and a push to switch out if we're
        being drained by Leech Seed and this move isn't going to end things."""
        if me is None or opp is None:
            return 0.0
        base_dmg = min(estimate_damage_fraction(me, move, opp), 1.0)
        base_incoming = KNOWLEDGE.predicted_incoming(opp, me, immune_types=_own_immune_types(me))

        dmg, incoming = base_dmg, base_incoming
        tera_type = getattr(me, "tera_type", None)
        if tera and tera_type is not None:
            dmg = min(self._tera_offense(me, move, opp, tera_type), 1.0)
            incoming = self._tera_incoming(me, opp, tera_type, base_incoming)

        # Scale the damage down by the chance their ability just eats it, so a move that
        # probably does nothing can't be credited with a kill, then charge for the risk
        # separately. Teraing doesn't change the move's type, so this applies either way.
        immunity = _immunity_chance(opp, move.type)
        dmg *= (1.0 - immunity)

        # Who goes first: our priority if we have any, otherwise speed. We can't see their
        # move, so their priority is invisible to us here.
        i_move_first = safe_priority(move) > 0 or _eff_speed(me) > _eff_speed(opp)
        i_ko = dmg >= opp.current_hp_fraction
        they_ko = incoming >= me.current_hp_fraction

        score = dmg - incoming                               # net health traded this turn
        if i_ko and i_move_first:
            score += 1.5                                     # clean kill, no damage taken
        elif i_ko:
            score += 1.0                                     # kill, but we take a hit first
            if they_ko:
                score -= 1.0                                 # and we might not survive it
        elif they_ko and not i_move_first:
            score -= 1.0                                     # we die and get nothing for it
        score -= self._self_drop_penalty(move)
        score -= immunity * self.IMMUNITY_RISK

        if tera and tera_type is not None:
            benefit = (dmg - base_dmg) + (base_incoming - incoming)
            if benefit <= self.TERA_MIN_BENEFIT:
                score -= self.TERA_COST     # not worth burning Tera on

        # Seeded, and this move won't finish things, so staying just feeds the seed.
        if seeded and not (i_ko or dmg >= self.LEECH_SEED_STAY_DMG):
            score -= self.LEECH_SEED_STAY_PENALTY
        return score

    @staticmethod
    def _switch_score(mon, opp, seeded=False):
        """How good a matchup this switch gets us: the best damage the incoming Pokemon can
        deal, minus the hit it takes on the way in, minus a flat cost for spending the turn.
        This is what lets the bot abandon something that can't hurt anything. If our current
        Pokemon is Leech Seeded, switching also gets a bonus, since that's the cure."""
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
            # We haven't seen its moves yet, so fall back to raw type effectiveness.
            eff = max((opp.damage_multiplier(t) for t in mon.types if t), default=1.0)
            best_offense = 0.25 * eff
        incoming = KNOWLEDGE.predicted_incoming(opp, mon, immune_types=_own_immune_types(mon))
        score = SearchPlayer.SWITCH_OFFENSE_WEIGHT * best_offense - 0.5 * incoming - 0.15
        if seeded:
            score += SearchPlayer.LEECH_SEED_SWITCH_BONUS   # switching shakes off the seed
        return score

    def _action_score(self, action, battle, me, opp):
        order = SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
        target = getattr(order, "order", None)
        seeded = _is_seeded(me)
        if isinstance(target, Pokemon):
            return self._switch_score(target, opp, seeded)
        if isinstance(target, Move):
            return self._move_score(me, target, opp,
                                    tera=bool(getattr(order, "terastallize", False)), seeded=seeded)
        return 0.0

    # --- picking the move ---------------------------------------------------

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

            # Score everything, then add a little of the network's opinion. Kept small on
            # purpose, so it only settles close calls and can never drag us into a losing
            # line just because the network happens to like it.
            def total(i):
                prior = probs[i] if probs is not None else 0.0
                return self._action_score(i, battle, me, opp) + self.POLICY_PRIOR_WEIGHT * prior

            best = max(legal, key=total)
            return SinglesEnv.action_to_order(np.int64(best), battle, strict=False)
        except Exception:
            return self.choose_random_move(battle)
