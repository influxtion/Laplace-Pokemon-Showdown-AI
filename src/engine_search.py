r"""EnginePlayer: a poke-env bot that searches with poke-engine (the Foul Play recipe).

For each of our turns we:
  1. sample N_DETERMINIZATIONS concrete opponent teams consistent with what's revealed
     (poke_engine_adapter.build_state),
  2. run poke-engine's multithreaded MCTS on each for SEARCH_TIME_MS,
  3. pool the visit-weighted root policies across the samples, and
  4. play the move with the highest pooled visit share.

This replaces the hand-rolled 1-/2-ply searchers (search.py / deep_search.py): poke-engine is a
real Gen 9 battle engine, so KO probabilities, hazards, items, abilities, residuals, turn order
and Terastallization are modelled correctly, and MCTS handles the simultaneous-move nature of a
turn (plain minimax lets the opponent "see" our move). Determinization is how we cope with the
opponent's hidden set.

A poke-env Player, used at test time -- see eval_search.py / play.py / ladder.py.
"""

import numpy as np

from poke_env.player import Player
from poke_env.data import to_id_str

from poke_engine import monte_carlo_tree_search

from poke_engine_adapter import build_state


class EnginePlayer(Player):
    """Searches each turn with poke-engine over several determinized opponent teams."""

    N_DETERMINIZATIONS = 6     # opponent-team samples per turn (more = steadier, slower)
    SEARCH_TIME_MS = 120       # MCTS budget per sample
    THREADS = 4                # engine search threads per sample

    # use_stats history: without choice-lock modeling the stats feed LOST its A/B at 39% (real
    # Choice/Life Orb items made the bot too cautious). With lock modeling added (2026-07-01) it
    # A/Bs at exact parity (60/120 mirror), and only the stats feed lets the search assign
    # unrevealed Choice items and exploit locked opponents -- so it ships on. Flip to False to
    # revert to the role-based item guess.
    def __init__(self, *args, n_determinizations=None, search_time_ms=None, threads=None,
                 debug=False, record=False, use_stats=True, **kwargs):
        super().__init__(*args, **kwargs)
        if n_determinizations is not None:
            self.N_DETERMINIZATIONS = n_determinizations
        if search_time_ms is not None:
            self.SEARCH_TIME_MS = search_time_ms
        if threads is not None:
            self.THREADS = threads
        self.debug = debug
        self.use_stats = use_stats   # use the real randbats item/ability/tera feed in determinization
        # Diagnostics: count turns and silent failures so we can tell "outplayed" from "bug".
        # det_fail = a determinization/search that raised (state too weird for poke-engine);
        # empty_pooled = a turn where *every* determinization failed -> dumb fallback move.
        self.diag = {"turns": 0, "det_runs": 0, "det_fail": 0, "empty_pooled": 0, "fallback": 0}
        self.record = record
        self.traces = {}   # battle_tag -> list of per-turn decision dicts (when record=True)
        # battle_tag -> {"species", "used"}: which moves the opponent's ACTIVE mon has used since
        # it last switched in. poke-env's mon.last_move only keeps the latest one; we accumulate
        # them so the adapter can infer Choice locks (1 move + Choice item = locked) and rule
        # Choice items out (2+ distinct moves without switching = can't be Choice).
        self._opp_tracker = {}

    def _opp_used_since_switch(self, battle):
        """Update and return the set of move ids the opponent's active has used since switch-in."""
        opp = battle.opponent_active_pokemon
        t = self._opp_tracker.setdefault(battle.battle_tag, {"species": None, "used": set()})
        if opp is None:
            return set()
        # A new species, or last_move cleared (poke-env clears it on switch-out), means the mon
        # (re-)entered the field: the lock history resets.
        if t["species"] != opp.species or opp.last_move is None:
            t["species"] = opp.species
            t["used"] = set()
        if opp.last_move is not None:
            t["used"].add(opp.last_move.id)
        return t["used"]

    # --- search -------------------------------------------------------------

    def _pooled_policy(self, battle):
        """{move_choice: pooled visit share} over N determinized searches, or {} on failure."""
        pooled = {}
        runs = 0
        opp_used = self._opp_used_since_switch(battle)
        for _ in range(self.N_DETERMINIZATIONS):
            try:
                state = build_state(battle, use_stats=self.use_stats,
                                    opp_used_since_switch=opp_used)
                res = monte_carlo_tree_search(state, self.SEARCH_TIME_MS, threads=self.THREADS)
            except BaseException:
                # poke-engine can *panic* on an inconsistent state (raises PanicException, which
                # is NOT an Exception subclass); skip that sample rather than crash the turn.
                self.diag["det_fail"] += 1
                continue
            self.diag["det_runs"] += 1
            total = res.total_visits or 1
            for opt in res.side_one:
                pooled[opt.move_choice] = pooled.get(opt.move_choice, 0.0) + opt.visits / total
            runs += 1
        if runs:
            pooled = {k: v / runs for k, v in pooled.items()}
        return pooled

    # --- mapping the engine's choice back to a poke-env order ----------------

    def _order_for_choice(self, choice, battle):
        """Translate a poke-engine move_choice string into a poke-env BattleOrder, or None if it
        isn't currently legal (caller falls back)."""
        if choice.startswith("switch "):
            species = choice.split(" ", 1)[1]
            for mon in battle.available_switches:
                if to_id_str(mon.species) == species or to_id_str(mon.base_species) == species:
                    return self.create_order(mon)
            return None

        tera = choice.endswith("-tera")
        move_id = choice[:-5] if tera else choice
        for move in battle.available_moves:
            if move.id == move_id:
                return self.create_order(move, terastallize=tera and bool(battle.can_tera))
        return None

    # --- decision -----------------------------------------------------------

    def choose_move(self, battle):
        try:
            if not battle.available_moves and not battle.available_switches:
                return self.choose_default_move(battle)

            self.diag["turns"] += 1
            pooled = self._pooled_policy(battle)
            if not pooled:
                self.diag["empty_pooled"] += 1
            for choice, _share in sorted(pooled.items(), key=lambda kv: kv[1], reverse=True):
                order = self._order_for_choice(choice, battle)
                if order is not None:
                    if self.debug:
                        self._log(battle, pooled, choice)
                    if self.record:
                        self._record(battle, pooled, choice, fallback=False)
                    return order
            # Engine gave nothing usable -> safe fallback (a dumb move; a bug signal if frequent).
            self.diag["fallback"] += 1
            if self.record:
                self._record(battle, pooled, "<fallback>", fallback=True)
            return self.choose_max_damage_move(battle) if battle.available_moves \
                else self.choose_random_move(battle)
        except Exception:
            self.diag["fallback"] += 1
            return self.choose_random_move(battle)

    def _record(self, battle, pooled, choice, fallback):
        """Append a per-turn decision snapshot for post-hoc loss analysis (record=True)."""
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        top = sorted(pooled.items(), key=lambda kv: kv[1], reverse=True)[:3]
        self.traces.setdefault(battle.battle_tag, []).append({
            "turn": battle.turn,
            "me": f"{me.species} {me.current_hp_fraction:.0%}" if me else "-",
            "opp": f"{opp.species} {opp.current_hp_fraction:.0%}" if opp else "-",
            "my_team_alive": sum(1 for m in battle.team.values() if not m.fainted),
            "opp_team_alive": 6 - sum(1 for m in battle.opponent_team.values() if m.fainted),
            "top": [(c, round(s, 2)) for c, s in top],
            "choice": choice,
            "fallback": fallback,
        })

    def choose_max_damage_move(self, battle):
        if battle.available_moves:
            best = max(battle.available_moves, key=lambda m: m.base_power)
            return self.create_order(best)
        return self.choose_random_move(battle)

    def _log(self, battle, pooled, picked):
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        print(f"[T{battle.turn}] {me.species if me else '-'} vs {opp.species if opp else '-'}",
              flush=True)
        for choice, share in sorted(pooled.items(), key=lambda kv: kv[1], reverse=True)[:5]:
            mark = " <-" if choice == picked else ""
            print(f"    {share*100:5.1f}%  {choice}{mark}", flush=True)
