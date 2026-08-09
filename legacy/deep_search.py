r"""Looking two turns ahead, using a small home-made simulator.

Everything before this looked exactly one turn ahead: score the immediate exchange, stop.
That's blind to anything whose payoff arrives a turn later. A Dragon Dance looks like "deal
no damage, take a hit" and gets rejected, even when the sweep it sets up wins the game
outright.

So this one looks two turns ahead. We have no real engine, so we simulate on a deliberately
tiny model: just the two Pokemon currently out, their health, boosts, status and speed,
using the same damage estimates the rest of the project uses.

    for everything I could do this turn:
        play out this turn against their most likely attack
        play out one more turn, my best follow-up against their best reply
        see how the position looks
    do whichever came out best.

The simulator keeps the things that make looking ahead worthwhile: boosts raise damage and
speed, so setup actually pays off in the rollout; killing something stops it hitting back;
recovery and self-inflicted stat drops are applied.

It deliberately keeps the opponent simple. They just click their best attack: no setup, no
switching, no status, and the bench is ignored entirely. An over-ambitious simulator that
gets things wrong is worse than a small honest one.

Used at test time by eval_search.py and play.py.
"""

import numpy as np

from poke_env.battle import Move, Pokemon
from poke_env.environment.singles_env import SinglesEnv

from knowledge import KNOWLEDGE, estimate_damage_fraction, get_move, safe_priority
from search import SearchPlayer, _immunity_chance, _own_immune_types
from heuristic_search import HeuristicSearchPlayer

_BAD_STATUS = {"BRN", "PSN", "TOX", "PAR", "SLP", "FRZ"}


def _boost_mult(stage):
    """What a stat boost is worth: +1 is one and a half times, -1 is two thirds."""
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


class _Snap:
    """A throwaway snapshot of one Pokemon. This is everything the simulator tracks."""
    __slots__ = ("mon", "hp", "boosts", "status", "tera")

    def __init__(self, mon, hp=None, boosts=None, status=False, tera=False):
        self.mon = mon
        self.hp = (mon.current_hp_fraction if hp is None else hp)
        self.boosts = (dict(mon.boosts) if boosts is None else boosts)
        # False here means "copy whatever the real Pokemon has"; None means "nothing".
        self.status = ((mon.status.name if mon.status else None) if status is False else status)
        self.tera = tera

    def copy(self):
        return _Snap(self.mon, self.hp, dict(self.boosts), self.status, self.tera)

    @property
    def fainted(self):
        return self.hp <= 1e-6


class DeepSearchPlayer(HeuristicSearchPlayer):
    """Scores each option by playing two turns out on the small simulator."""

    DEPTH = 2                 # this turn plus the follow-up
    DEPTH_DISCOUNT = 0.95     # slight preference for good things happening sooner
    FAINT_VALUE = 1.6
    POSITION_WEIGHT = 0.45    # how much the resulting matchup counts
    BOOST_ASSET = 0.04        # a little credit for boosts we're still holding at the end
    STATUS_SWING = 0.10       # worth of having them statused, or the cost of being statused

    # ------------------------------ the simulator ------------------------------

    def _sim_speed(self, s):
        spe = s.mon.base_stats["spe"] * _boost_mult(s.boosts.get("spe", 0))
        return spe * 0.5 if s.status == "PAR" else spe

    def _sim_damage(self, atk, move, deff):
        """Damage inside the simulation, accounting for boosts, Tera, and whether the
        attacker is burned in the simulated world rather than the real one."""
        try:
            if move is None or move.base_power <= 0:
                return 0.0
            if atk.tera and getattr(atk.mon, "tera_type", None) is not None:
                base = SearchPlayer._tera_offense(atk.mon, move, deff.mon, atk.mon.tera_type)
            else:
                base = estimate_damage_fraction(atk.mon, move, deff.mon)
            if base <= 0:
                return 0.0
            physical = move.category.name == "PHYSICAL"
            o, d = ("atk", "def") if physical else ("spa", "spd")
            base *= _boost_mult(atk.boosts.get(o, 0)) / _boost_mult(deff.boosts.get(d, 0))
            # The estimate already halved for a burn on the real Pokemon. Undo or apply that
            # so it matches the burn status inside this simulation instead.
            if physical:
                real_brn = atk.mon.status is not None and atk.mon.status.name == "BRN"
                sim_brn = atk.status == "BRN"
                if real_brn and not sim_brn:
                    base *= 2.0
                elif sim_brn and not real_brn:
                    base *= 0.5
            return min(base, 1.5)
        except Exception:
            return 0.0

    def _opp_damage(self, opp, me):
        """The hardest they could plausibly hit us, ignoring anything our ability shrugs off."""
        immune = _own_immune_types(me.mon)
        worst = 0.0
        for mid, p in KNOWLEDGE.move_probs(opp.mon).items():
            mv = get_move(mid)
            if mv is None or mv.base_power <= 0:
                continue
            if immune and mv.type is not None and mv.type.name in immune:
                continue
            worst = max(worst, p * self._sim_damage(opp, mv, me))
        return min(worst, 1.5)

    def _my_potential(self, me, opp):
        """The hardest we could hit them next turn, discounted for possible immunities."""
        best = 0.0
        for mv in me.mon.moves.values():
            d = self._sim_damage(me, mv, opp) * (1.0 - _immunity_chance(opp.mon, mv.type))
            best = max(best, min(d, 1.0))
        return best

    @staticmethod
    def _apply_self_effects(s, move):
        """Whatever the move does to the Pokemon using it: healing, boosts, stat drops."""
        heal = getattr(move, "heal", 0) or 0
        if heal > 0:
            s.hp = min(s.hp + heal, 1.0)
        if move.category.name == "STATUS":                 # setup moves boost their user
            for stat, v in (getattr(move, "boosts", None) or {}).items():
                s.boosts[stat] = max(-6, min(6, s.boosts.get(stat, 0) + v))
        for stat, v in (getattr(move, "self_boost", None) or {}).items():   # Close Combat, etc.
            s.boosts[stat] = max(-6, min(6, s.boosts.get(stat, 0) + v))

    def _sim_move_turn(self, me, opp, move, tera):
        """Play out one turn: we use this move, they use their best attack."""
        m, o = me.copy(), opp.copy()
        if tera:
            m.tera = True
        my_dmg = self._sim_damage(m, move, o)
        opp_dmg = self._opp_damage(o, m)
        i_first = safe_priority(move) > 0 or self._sim_speed(m) > self._sim_speed(o)
        if i_first:
            o.hp -= my_dmg
            if not o.fainted:                              # dead things don't hit back
                m.hp -= opp_dmg
        else:
            m.hp -= opp_dmg
            if not m.fainted:
                o.hp -= my_dmg
        if not m.fainted:
            self._apply_self_effects(m, move)
        return m, o

    def _sim_switch_turn(self, switch_in, opp):
        """Play out one turn where we switch, and the incoming Pokemon eats their attack.
        Entry hazards aren't modelled here, same as everywhere else in this era."""
        m = _Snap(switch_in)
        m.hp -= self._opp_damage(opp, m)
        return m, opp.copy()

    # ------------------------------ judging a position ------------------------

    def _static_eval(self, me, opp):
        """How good this position looks for us, once we stop looking further ahead."""
        value = me.hp - opp.hp
        value += self.FAINT_VALUE * ((1.0 if opp.fainted else 0.0) - (1.0 if me.fainted else 0.0))
        if not me.fainted and not opp.fainted:
            value += self.POSITION_WEIGHT * (self._my_potential(me, opp) - self._opp_damage(opp, me))
            pos = lambda s: sum(v for v in s.boosts.values() if v > 0)
            value += self.BOOST_ASSET * (pos(me) - pos(opp))
            value -= self.STATUS_SWING if me.status in _BAD_STATUS else 0.0
            value += self.STATUS_SWING if opp.status in _BAD_STATUS else 0.0
        return value

    def _rollout(self, me, opp, depth):
        """The best we can do from here, looking a few more turns ahead. We take our best
        move each turn; they answer with their best attack."""
        if depth <= 0 or me.fainted or opp.fainted:
            return self._static_eval(me, opp)
        best = None
        for mv in me.mon.moves.values():                   # follow-ups are moves only, no switches
            m2, o2 = self._sim_move_turn(me, opp, mv, tera=me.tera)
            v = self._rollout(m2, o2, depth - 1)
            best = v if best is None else max(best, v)
        if best is None:                                   # nothing to try, so judge it as it is
            return self._static_eval(me, opp)
        return self.DEPTH_DISCOUNT * best

    def _action_score(self, action, battle, me, opp):
        """Score an option by simulating two turns instead of guessing at one."""
        if me is None or opp is None:
            return 0.0
        order = SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
        target = getattr(order, "order", None)
        me_s, opp_s = _Snap(me), _Snap(opp)
        if isinstance(target, Pokemon):
            m1, o1 = self._sim_switch_turn(target, opp_s)
        elif isinstance(target, Move):
            m1, o1 = self._sim_move_turn(me_s, opp_s, target,
                                         tera=bool(getattr(order, "terastallize", False)))
        else:
            return 0.0
        return self._rollout(m1, o1, self.DEPTH - 1)
