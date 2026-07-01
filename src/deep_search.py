r"""DeepSearchPlayer: a 2-ply search over a small forward simulator.

Every agent so far looks one turn ahead -- it scores the immediate exchange and stops. That
can't see anything whose payoff is a turn away: a Dragon Dance reads as "deal nothing, take a
hit" at 1-ply and gets rejected, even when the +1/+1 sweep next turn wins the game; and the
agent will happily trade into a move that KOs it back.

This looks two turns ahead. We don't have the real engine, so we simulate on a deliberately
small model -- just the two active Pokemon, their HP / stat boosts / status / speed -- using the
same expected-damage calc and opponent set-prediction the rest of the project uses:

    for each of my legal actions (move, tera, switch):
        simulate this turn  (my action vs the opponent's best expected move, turn-order aware)
        then one more turn   (my best follow-up vs their best reply)
        evaluate the resulting position
    play the action with the best 2-turn outcome.

The simulator carries the things that make depth worth having: stat boosts raise damage and
speed (so setup pays off in the rollout), a KO suppresses the victim's hit (turn order), and
recovery / self-drops are applied. It deliberately keeps the opponent model simple (they click
their best expected damaging move; no opponent setup/switch/status) and ignores the benched
team in the rollout -- a bad, over-ambitious simulator would hurt more than help, so this one
stays small and honest. It builds on HeuristicSearchPlayer, inheriting the policy prior, the
robust per-action choose_move and the --debug trace.

A poke-env Player, used at test time -- see eval_search.py / play.py.
"""

import numpy as np

from poke_env.battle import Move, Pokemon
from poke_env.environment.singles_env import SinglesEnv

from knowledge import KNOWLEDGE, estimate_damage_fraction, get_move, safe_priority
from search import SearchPlayer, _immunity_chance, _own_immune_types
from heuristic_search import HeuristicSearchPlayer

_BAD_STATUS = {"BRN", "PSN", "TOX", "PAR", "SLP", "FRZ"}


def _boost_mult(stage):
    """Standard stat-stage multiplier (+1 = 1.5x, -1 = 0.66x, etc.)."""
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


class _Snap:
    """A tiny mutable snapshot of one active Pokemon, all the simulator tracks."""
    __slots__ = ("mon", "hp", "boosts", "status", "tera")

    def __init__(self, mon, hp=None, boosts=None, status=False, tera=False):
        self.mon = mon
        self.hp = (mon.current_hp_fraction if hp is None else hp)
        self.boosts = (dict(mon.boosts) if boosts is None else boosts)
        # status=False sentinel means "read it off the real mon"; None means "no status".
        self.status = ((mon.status.name if mon.status else None) if status is False else status)
        self.tera = tera

    def copy(self):
        return _Snap(self.mon, self.hp, dict(self.boosts), self.status, self.tera)

    @property
    def fainted(self):
        return self.hp <= 1e-6


class DeepSearchPlayer(HeuristicSearchPlayer):
    """Scores each action by simulating two turns ahead on a 2-active model."""

    DEPTH = 2                 # plies (full turns) of lookahead; 2 = this turn + the follow-up
    DEPTH_DISCOUNT = 0.95     # mild preference for value sooner rather than later
    FAINT_VALUE = 1.6
    POSITION_WEIGHT = 0.45    # weight on the leaf matchup (my best hit minus their best hit)
    BOOST_ASSET = 0.04        # small credit per net positive offensive stat stage held
    STATUS_SWING = 0.10       # value of having a status on the opponent (or cost of one on us)

    # ------------------------------ simulator core ------------------------------

    def _sim_speed(self, s):
        spe = s.mon.base_stats["spe"] * _boost_mult(s.boosts.get("spe", 0))
        return spe * 0.5 if s.status == "PAR" else spe

    def _sim_damage(self, atk, move, deff):
        """Expected damage fraction atk -> deff for this move, with stat boosts, tera STAB and
        the simulated (not just the real) burn applied."""
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
            # estimate_damage_fraction halved for the *real* mon's burn; correct it to the sim's.
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
        """The opponent's best expected hit into `me` (HP fraction), over their predicted
        moveset, discounting move types our ability is immune to."""
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
        """My best expected hit into `opp` next turn, immunity-discounted (boost/tera aware)."""
        best = 0.0
        for mv in me.mon.moves.values():
            d = self._sim_damage(me, mv, opp) * (1.0 - _immunity_chance(opp.mon, mv.type))
            best = max(best, min(d, 1.0))
        return best

    @staticmethod
    def _apply_self_effects(s, move):
        """Recovery, setup boosts and self-stat-drops the user gets from its own move."""
        heal = getattr(move, "heal", 0) or 0
        if heal > 0:
            s.hp = min(s.hp + heal, 1.0)
        if move.category.name == "STATUS":                 # setup moves boost the user
            for stat, v in (getattr(move, "boosts", None) or {}).items():
                s.boosts[stat] = max(-6, min(6, s.boosts.get(stat, 0) + v))
        for stat, v in (getattr(move, "self_boost", None) or {}).items():   # e.g. Close Combat
            s.boosts[stat] = max(-6, min(6, s.boosts.get(stat, 0) + v))

    def _sim_move_turn(self, me, opp, move, tera):
        """Resolve one turn where I use `move` and the opponent clicks their best hit. Returns
        new (me, opp) snapshots."""
        m, o = me.copy(), opp.copy()
        if tera:
            m.tera = True
        my_dmg = self._sim_damage(m, move, o)
        opp_dmg = self._opp_damage(o, m)
        i_first = safe_priority(move) > 0 or self._sim_speed(m) > self._sim_speed(o)
        if i_first:
            o.hp -= my_dmg
            if not o.fainted:                              # a KO cancels their hit
                m.hp -= opp_dmg
        else:
            m.hp -= opp_dmg
            if not m.fainted:
                o.hp -= my_dmg
        if not m.fainted:
            self._apply_self_effects(m, move)
        return m, o

    def _sim_switch_turn(self, switch_in, opp):
        """I switch `switch_in` in; the opponent's hit lands on it. (Entry hazards: a later
        increment -- the parent classes don't model them either.)"""
        m = _Snap(switch_in)
        m.hp -= self._opp_damage(opp, m)
        return m, opp.copy()

    # ------------------------------ evaluation ------------------------------

    def _static_eval(self, me, opp):
        """Position value from our perspective at a leaf of the search."""
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
        """Best achievable value from this position, looking `depth` more turns ahead. At each
        node I take my best move; the opponent answers with their best hit (inside _sim_*)."""
        if depth <= 0 or me.fainted or opp.fainted:
            return self._static_eval(me, opp)
        best = None
        for mv in me.mon.moves.values():                   # follow-up considers my moves only
            m2, o2 = self._sim_move_turn(me, opp, mv, tera=me.tera)
            v = self._rollout(m2, o2, depth - 1)
            best = v if best is None else max(best, v)
        if best is None:                                   # no usable moves -> evaluate as is
            return self._static_eval(me, opp)
        return self.DEPTH_DISCOUNT * best

    def _action_score(self, action, battle, me, opp):
        """Override: value an action by its 2-turn simulated outcome instead of a 1-ply guess."""
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
