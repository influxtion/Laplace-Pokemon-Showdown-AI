r"""EnginePlayer -- determinized MCTS over poke-engine. This is the bot.

Per turn:
  1. sample N_DETERMINIZATIONS concrete opponent teams consistent with what's revealed
     (poke_engine_adapter.build_state),
  2. MCTS each for SEARCH_TIME_MS,
  3. pool the root policies,
  4. run the guard/rerank pipeline over the pooled ranking,
  5. play the survivor.

poke-engine is a real Gen 9 engine, so KO probabilities, hazards, residuals, items,
abilities, turn order and Tera are all modelled properly, and MCTS handles the
simultaneous-move turn -- plain minimax lets the opponent see our move first.
Determinization is how we cope with the hidden set.

The guards (steps 4) are the interesting part and most of them exist because of a
specific lost game. They're deliberately deterministic: unlike the value net, they're
allowed to override a confident search score, because they run on hard knowledge
(revealed typing, observed history) rather than estimates.

Optional `observer` callback reports search internals per turn -- see _emit. Pure
instrumentation, nothing read back, so moves are identical with and without one
attached. live_analysis.LiveAnalyzer is the shipped consumer.
"""

import time
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from poke_env.player import Player
from poke_env.data import to_id_str

from poke_engine import monte_carlo_tree_search, generate_instructions

from knowledge import (estimate_damage_fraction, get_move, move_nullified, safe_priority,
                       _estimate_stat)
from poke_engine_adapter import build_state


def _spe_mult(boost):
    """Speed multiplier for a boost stage."""
    return (2 + boost) / 2 if boost >= 0 else 2 / (2 - boost)


# --- parallel determinization search ----------------------------------------------
# poke-engine holds the GIL while searching, so concurrent worlds need PROCESSES, not
# threads. Foul Play does the same; it's why it affords 2x our world count on the same
# move timer. States cross as strings, results as plain tuples -- MctsResult is a Rust
# object and doesn't pickle.

_Opt = namedtuple("_Opt", "move_choice visits")
_Res = namedtuple("_Res", "side_one side_two total_visits")


def _search_world(state_str, time_ms, threads):
    """Worker: rebuild the state, search it, return something picklable. None on panic."""
    from poke_engine import State, monte_carlo_tree_search as _mcts
    try:
        res = _mcts(State.from_string(state_str), time_ms, threads=threads)
        return ([(o.move_choice, o.visits) for o in res.side_one],
                [(o.move_choice, o.visits) for o in res.side_two],
                res.total_visits)
    except BaseException:
        return None


def _shim(raw):
    """Worker tuple -> same shape the sequential path returns, so callers don't branch."""
    s1, s2, total = raw
    return _Res([_Opt(c, v) for c, v in s1], [_Opt(c, v) for c, v in s2], total)


class EnginePlayer(Player):
    """poke-engine MCTS over N determinized opponent teams, pooled, then guarded."""

    N_DETERMINIZATIONS = 6     # worlds per turn; more = steadier vote, less search each
    SEARCH_TIME_MS = 120       # MCTS budget per world
    THREADS = 4                # engine threads per world

    # use_stats history: without choice-lock modelling the stats feed LOST its A/B at 39%
    # -- real Choice/Life Orb items made the bot too cautious. With locks modelled it A/Bs
    # at parity (60/120 mirror), and it's the only thing that lets the search assign
    # unrevealed Choice items and exploit a locked opponent. Ships on.
    def __init__(self, *args, n_determinizations=None, search_time_ms=None, threads=None,
                 debug=False, record=False, use_stats=True, speed_inference=True,
                 value_model_path=None, value_worlds=4, value_opp_moves=2,
                 value_margin=11.0, value_on_force_switch=False, value_boost_margin=0.0,
                 value_case_fix=True,
                 tera_min_wins=1, parallel_worlds=False, use_joint_sets=True,
                 mix_root=False, mix_frac=0.75, mix_collapse_eps=0.03, robust_vote=True,
                 absorb_guard=True, futility_guard=True, gamble_guard=False,
                 observer=None, **kwargs):
        super().__init__(*args, **kwargs)
        if n_determinizations is not None:
            self.N_DETERMINIZATIONS = n_determinizations
        if search_time_ms is not None:
            self.SEARCH_TIME_MS = search_time_ms
        if threads is not None:
            self.THREADS = threads
        self.debug = debug
        self.use_stats = use_stats                 # real randbats item/ability/tera feed
        self.speed_inference = speed_inference     # infer Choice Scarf from turn order
        # Diagnostics -- tells "outplayed" apart from "bug" after a bad run.
        # det_fail: a world the engine panicked on. empty_pooled: every world failed, so we
        # fell back to a dumb move.
        self.diag = {"turns": 0, "det_runs": 0, "det_fail": 0, "empty_pooled": 0, "fallback": 0}
        self.record = record
        self.traces = {}   # battle_tag -> per-turn decision dicts (record=True)
        # battle_tag -> {"species", "used"}: moves the opponent's ACTIVE has used since
        # switch-in. poke-env's mon.last_move only keeps the latest, so we accumulate:
        # 1 move + Choice item = locked; 2+ distinct without switching = not Choice.
        self._opp_tracker = {}
        # battle_tag -> {species_id: "scarf"|"noscarf"} from observed turn order. Randbats
        # spreads are fixed (85 EVs / 31 IVs / neutral), so raw speed is essentially known;
        # outsped when it shouldn't have been => Scarf (or a speed ability -- same model).
        self._opp_speed = {}
        # battle_tag -> delayed effects. '<side>_wish' -> (lands_on_turn, amount),
        # '<side>_fs' -> hits_at_end_of_turn. Sides are 'p1'/'p2' as on the protocol.
        self._pending = {}
        # battle_tag -> per-decision records for the futility guard: what we clicked into
        # which matchup, and where opp HP / our boosts stood. Mined: the #1 loss class is
        # chipping a target whose recovery outpaces us (Iron Head x4 into Soft-Boiled
        # Blissey, Moonblast x10 into Wish/Protect Sylveon) or re-boosting into a Haze user
        # (QD x7 into Milotic). Every one a confident argmax, so the near-tie guards never
        # engage. Observation, not prediction -- fires only after the futility has happened.
        self._futility_hist = {}
        # Learned value head. The engine's leaf eval under-fears opponent setup: measured a
        # guaranteed-fail Substitute at 0.478 vs 0.480 for the best move against a +3/+3/+3
        # sweeper. So near-tied roots get rolled one ply with generate_instructions and the
        # successors scored by a net trained on self-play (train_value / value_features).
        self._value_model = None
        self._worlds = []            # [(state, MctsResult)] from the last turn
        self.value_worlds = value_worlds
        self.value_opp_moves = value_opp_moves
        self.value_margin = value_margin
        self.value_on_force_switch = value_on_force_switch
        self.value_boost_margin = value_boost_margin
        # apply_instructions round-trips states through Rust and returns UPPERCASE ids;
        # adapter-built states (= all training data) are lowercase. Case-sensitive checks
        # silently zeroed has_recovery, the volatile flags and the 'typeless' filter on
        # every state the net actually scores. True = train/inference consistency; the
        # kwarg only exists to A/B the old behaviour.
        self.value_case_fix = value_case_fix
        # Min world-wins before a '-tera' choice is playable (ordinary moves need 1). Tera
        # is once-per-game and irreversible; mining says it's burned on a mon that dies
        # within a turn in 48% of tera-losses vs 12% of tera-wins.
        # HISTORY: 2 A/B'd at 60.0%/60 in mirror but went 17W-23L over 40 ladder games vs
        # 64% ungated. Humans tera aggressively and the mirror can't price the tempo lost
        # holding it in a tera race. Reverted to 1; knob kept for study at higher det counts.
        self.tera_min_wins = tera_min_wins
        # Search all worlds concurrently in a process pool (see module top). Same wall
        # clock buys N_DETERMINIZATIONS x the search.
        self.parallel_worlds = parallel_worlds
        # Sample complete counted joint sets rather than composing from marginals.
        # Distributionally correct; came out of the compute-matched Foul Play diagnosis.
        self.use_joint_sets = use_joint_sets
        # Mixed root strategy -- the second structural FP edge after joint sets.
        # Deterministic argmax is exploitable: humans switch immunity absorbers into our
        # predictable clicks and FP's MCTS found the same holes. Sample score-weighted
        # among un-vetoed world-winners within mix_frac of the top. '-tera' only playable
        # as the argmax; the once-per-game resource is never spent on a die roll.
        self.mix_root = mix_root
        self.mix_frac = mix_frac
        # Collapse-aware mix suppression. In lost-ish positions Q-values flatten and pooled
        # shares go near-uniform (Hydrapple T39: recover/ficklebeam/gigadrain/NP all
        # 0.24-0.26) -- that's argmax noise, not a strategy, and every observed
        # clean-cohort death-gamble was a mix SAMPLE out of such a set. With >=3 eligible
        # and span <= eps, skip mixing and let tiebreak/value order stand; they carry the
        # only real signal left. Two eligibles always mix -- the Sucker Punch 50/50 is a
        # genuine mixed-strategy spot. Same 0.03 share scale as _damage_tiebreak's eps.
        self.mix_collapse_eps = mix_collapse_eps
        # Per-guard toggles for LADDER bisecting. Mirror gates miss tempo effects only
        # humans charge for (see the tera-gate precedent), so each root guard has to be
        # cheaply switchable per arm.
        self.absorb_guard = absorb_guard
        self.futility_guard = futility_guard
        self.gamble_guard = gamble_guard
        self._vetoed = set()     # demoted by a guard this turn; never eligible for the mix
        # robust_vote=False reverts to FP-style plain averaging of visit shares. Joint
        # sampling draws worlds by count, so the uniform average IS the likelihood-weighted
        # one. The robust vote was built when marginal sampling over-represented incoherent
        # worlds (the Scale-Shot-into-Fairy pathology); with joint sets those may no longer
        # occur, and averaging preserves more policy mass for mixing. Downstream scores stay
        # on the same w*10+share scale so the guards' >=10 checks keep meaning.
        self.robust_vote = robust_vote
        # Score floor for a "trusted" candidate in the guards/mix/rerank. Robust: 10.0 = won
        # a world. Averaging: scores are plain 0..1 shares, everything is honestly ranked,
        # so no floor.
        self._win_floor = 10.0 if robust_vote else 0.0
        # observer(event, **payload), called at a handful of points per turn (see _emit).
        # Pure instrumentation -- nothing read back -- so an observed run plays identically.
        self.observer = observer
        self._executor = None
        if value_model_path:
            self._load_value_model(value_model_path)

    def _emit(self, event, **data):
        """Fire the observer if one is attached.

        One attribute read when it's off. An observer that raises is counted and swallowed
        -- a broken commentary renderer must never lose the game it's narrating."""
        if self.observer is None:
            return
        try:
            self.observer(event, **data)
        except Exception:
            self.diag["observer_err"] = self.diag.get("observer_err", 0) + 1

    def _pool(self):
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=min(self.N_DETERMINIZATIONS, 12))
        return self._executor

    def _load_value_model(self, path):
        import torch
        from train_value import ValueNet
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net = ValueNet(ckpt["n_in"], hidden=tuple(ckpt.get("hidden", (256, 128))),
                       dropout=ckpt.get("dropout", 0.1))
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        torch.set_num_threads(1)     # tiny net; don't fight the engine for cores
        self._value_model = net

    # --- speed inference from turn order ---------------------------------------

    async def _handle_battle_message(self, split_messages):
        await super()._handle_battle_message(split_messages)
        if not self.speed_inference:
            return
        try:
            tag = split_messages[0][0].replace(">", "").strip()
            battle = self.battles.get(tag)
            if battle is not None:
                self._scan_speed_evidence(battle, split_messages)
        except Exception:
            self.diag["speed_scan_err"] = self.diag.get("speed_scan_err", 0) + 1

    def _scan_speed_evidence(self, battle, split_messages):
        """Scrape who-moved-first evidence (Scarf) and pending Wish/Future Sight out of one
        message payload.

        Speed comparisons are skipped on turns tainted by a mid-turn switch (pivot) or an
        in-turn speed change: the end-of-payload battle state no longer reflects the
        conditions that decided move order."""
        moves = []          # (side, move_id) in order within the current turn
        tainted = False
        wish_side = fs_side = None      # sides that queued a delayed effect this payload
        pend = self._pending.setdefault(battle.battle_tag, {})
        for msg in split_messages:
            if len(msg) < 2:
                continue
            tag = msg[1]
            if tag == "turn":
                # A turn line anchors anything queued earlier in the payload: Wish heals at
                # the end of the turn we're now deciding, Future Sight one later.
                turn = int(msg[2])
                if wish_side:
                    pend[f"{wish_side}_wish"] = (turn, pend.pop("_wish_amt", 0))
                    wish_side = None
                if fs_side:
                    pend[f"{fs_side}_fs"] = turn + 1
                    fs_side = None
                moves, tainted = [], False
            elif tag in ("switch", "drag") and moves:
                tainted = True
            elif tag in ("-boost", "-unboost") and len(msg) > 3 and msg[3] == "spe":
                tainted = True
            elif tag == "move" and len(msg) > 3:
                if to_id_str(msg[3]) == "wish":
                    wish_side = msg[2][:2]
                    healer = battle.active_pokemon if wish_side == battle.player_role \
                        else battle.opponent_active_pokemon
                    pend["_wish_amt"] = int(_estimate_stat(healer, "hp")) // 2 if healer else 0
                moves.append((msg[2][:2], to_id_str(msg[3])))
                if len(moves) == 2 and not tainted:
                    self._infer_scarf(battle, moves)
            elif tag == "-start" and len(msg) > 3 and msg[3] == "move: Future Sight":
                fs_side = msg[2][:2]

    def _infer_scarf(self, battle, moves):
        """Update the Scarf verdict from one same-priority move pair.

        Conservative: 5% tolerance both ways for state-timing noise, and a 'scarf' verdict
        overrides 'noscarf' but never the reverse."""
        (s1, m1), (s2, m2) = moves
        role = battle.player_role
        if role not in (s1, s2) or s1 == s2:
            return
        mv1, mv2 = get_move(m1), get_move(m2)
        if mv1 is None or mv2 is None or safe_priority(mv1) != safe_priority(mv2):
            return
        if any(f.name == "TRICK_ROOM" for f in battle.fields):
            return
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        if me is None or opp is None or me.fainted or opp.fainted:
            return

        our = float((me.stats or {}).get("spe") or _estimate_stat(me, "spe"))
        our *= _spe_mult(me.boosts.get("spe", 0))
        if me.status is not None and me.status.name == "PAR":
            our *= 0.5
        if me.item and to_id_str(me.item) == "choicescarf":
            our *= 1.5
        if any(c.name == "TAILWIND" for c in battle.side_conditions):
            our *= 2

        their_raw = float(_estimate_stat(opp, "spe"))
        their_mult = _spe_mult(opp.boosts.get("spe", 0))
        if opp.status is not None and opp.status.name == "PAR":
            their_mult *= 0.5
        if any(c.name == "TAILWIND" for c in battle.opponent_side_conditions):
            their_mult *= 2

        species = to_id_str(opp.species)
        hints = self._opp_speed.setdefault(battle.battle_tag, {})
        opp_first = s1 != role
        if opp_first and their_raw * their_mult < our * 0.95:
            hints[species] = "scarf"
            self.diag["scarf_inferred"] = self.diag.get("scarf_inferred", 0) + 1
        elif not opp_first and their_raw * their_mult * 1.5 > our * 1.05:
            if hints.get(species) != "scarf":
                hints[species] = "noscarf"
                self.diag["noscarf_inferred"] = self.diag.get("noscarf_inferred", 0) + 1

    def _opp_used_since_switch(self, battle):
        """Update and return the move ids the opponent's active has used since switch-in."""
        opp = battle.opponent_active_pokemon
        t = self._opp_tracker.setdefault(battle.battle_tag, {"species": None, "used": set()})
        if opp is None:
            return set()
        # New species, or last_move cleared (poke-env clears it on switch-out) => the mon
        # (re-)entered the field, so the lock history resets.
        if t["species"] != opp.species or opp.last_move is None:
            t["species"] = opp.species
            t["used"] = set()
        if opp.last_move is not None:
            t["used"].add(opp.last_move.id)
        return t["used"]

    # --- search -------------------------------------------------------------

    def _pooled_policy(self, battle):
        """Ranked {move_choice: score} over N determinized searches, or {} if all failed.

        Aggregation is a robust VOTE, not a plain average. Averaging has a known
        determinization pathology (observed live: Scale Shot into a revealed Fairy) -- when
        worlds disagree, a hedge move that is every world's mediocre runner-up can top the
        pooled average while being no world's actual pick. So a move is only eligible while
        some world made it the outright winner; rank by worlds won, then pooled share.
        Non-winners keep a scaled-down share purely as legality fallback order."""
        pooled = {}
        wins = {}
        runs = 0
        self._worlds = []
        opp_used = self._opp_used_since_switch(battle)
        speed_hints = self._opp_speed.get(battle.battle_tag, {})

        n = self.N_DETERMINIZATIONS
        self._emit("search_start", worlds=n, time_ms=self.SEARCH_TIME_MS,
                   threads=self.THREADS, parallel=self.parallel_worlds)

        searched = []            # (state, result) from whichever search path ran
        if self.parallel_worlds:
            jobs = []
            for j in range(n):
                try:
                    state = build_state(battle, use_stats=self.use_stats,
                                        opp_used_since_switch=opp_used,
                                        opp_speed_hints=speed_hints,
                                        pending=self._pending.get(battle.battle_tag),
                                        use_joint=self.use_joint_sets)
                    fut = self._pool().submit(_search_world, state.to_string(),
                                              self.SEARCH_TIME_MS, self.THREADS)
                    jobs.append((state, fut))
                except BaseException:
                    self.diag["det_fail"] += 1
                    self._emit("world_fail", index=j, total=n)
            for i, (state, fut) in enumerate(jobs):
                t0 = time.perf_counter()
                try:
                    raw = fut.result(timeout=self.SEARCH_TIME_MS / 1000 * 4 + 10)
                except Exception:
                    raw = None
                if raw is None:
                    self.diag["det_fail"] += 1
                    self._emit("world_fail", index=i, total=n)
                    continue
                res = _shim(raw)
                self._emit("world", index=i, total=n, state=state, res=res,
                           ms=(time.perf_counter() - t0) * 1000)
                searched.append((state, res))
        else:
            for i in range(n):
                t0 = time.perf_counter()
                try:
                    state = build_state(battle, use_stats=self.use_stats,
                                        opp_used_since_switch=opp_used,
                                        opp_speed_hints=speed_hints,
                                        pending=self._pending.get(battle.battle_tag),
                                        use_joint=self.use_joint_sets)
                    res = monte_carlo_tree_search(state, self.SEARCH_TIME_MS,
                                                  threads=self.THREADS)
                except BaseException:
                    # poke-engine PANICS on an inconsistent state, and PanicException is
                    # not an Exception subclass -- hence the bare BaseException. Skip the
                    # world rather than lose the turn.
                    self.diag["det_fail"] += 1
                    self._emit("world_fail", index=i, total=n)
                    continue
                self._emit("world", index=i, total=n, state=state, res=res,
                           ms=(time.perf_counter() - t0) * 1000)
                searched.append((state, res))

        for state, res in searched:
            self.diag["det_runs"] += 1
            self._worlds.append((state, res))
            total = res.total_visits or 1
            best = None
            for opt in res.side_one:
                pooled[opt.move_choice] = pooled.get(opt.move_choice, 0.0) + opt.visits / total
                if best is None or opt.visits > best.visits:
                    best = opt
            if best is not None:
                wins[best.move_choice] = wins.get(best.move_choice, 0) + 1
            runs += 1
        if not runs:
            return {}
        # Score: world-winners sort above everything by (wins, pooled share); the rest keep
        # a small share-proportional score so the legality fallback has a sane order.
        # '-tera' is held to tera_min_wins -- below that it scores like a non-winner, so the
        # plain version of the move outranks it.
        scores = {}
        for choice, share in pooled.items():
            share /= runs
            if not self.robust_vote:
                # FP-style plain average. Joint sampling draws worlds by count, so the
                # uniform average IS likelihood-weighted. tera_min_wins is inert here.
                scores[choice] = share
                continue
            w = wins.get(choice, 0)
            needed = self.tera_min_wins if choice.endswith("-tera") else 1
            scores[choice] = (w * 10.0 + share) if w >= needed else share * 1e-3
        return scores

    # --- move_choice -> poke-env order ---------------------------------------

    def _order_for_choice(self, choice, battle):
        """Translate a poke-engine move_choice into a BattleOrder, or None if it isn't
        currently legal (caller falls through to the next candidate)."""
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

    # --- decision ------------------------------------------------------------

    def choose_move(self, battle):
        try:
            if not battle.available_moves and not battle.available_switches:
                return self.choose_default_move(battle)

            self.diag["turns"] += 1
            self._vetoed = set()
            t_turn = time.perf_counter()
            self._emit("turn_start", battle=battle)
            pooled = self._pooled_policy(battle)
            if not pooled:
                self.diag["empty_pooled"] += 1
            ranked = sorted(pooled.items(), key=lambda kv: kv[1], reverse=True)
            self._emit("pooled", battle=battle, ranked=ranked, worlds=self._worlds)
            # Per-stage attribution: which reranker moved the front-runner this turn. Pure
            # logging, but it lets mining attribute overrides EXACTLY instead of re-deriving
            # guard eligibility post-hoc -- a hand re-derivation misattributed one once.
            # Multiple stages can fire on one turn; keep all of them.
            reorders = []
            top = ranked[0][0] if ranked else None

            def _stage(name, rk):
                nonlocal top
                changed = bool(rk) and top is not None and rk[0][0] != top
                if changed:
                    reorders.append(name)
                self._emit("stage", name=name, ranked=rk, changed=changed,
                           was=top, vetoed=frozenset(self._vetoed))
                top = rk[0][0] if rk else top
                return rk

            if self.absorb_guard:
                ranked = _stage("absorb", self._absorb_demote(battle, ranked))
            if self.futility_guard:
                ranked = _stage("futility", self._futility_demote(battle, ranked))
            ranked = _stage("tiebreak", self._damage_tiebreak(battle, ranked))
            ranked = _stage("value", self._value_rerank(ranked, battle))
            if self.gamble_guard:
                ranked = _stage("gamble", self._gamble_veto(battle, ranked))
            ranked = _stage("noop", self._noop_demote(ranked))
            ranked = _stage("deadlock", self._dead_lock_guard(battle, ranked))
            ranked, mixed = self._mix_sample(ranked)
            for choice, _share in ranked:
                order = self._order_for_choice(choice, battle)
                if order is not None:
                    if choice != (ranked[0][0] if ranked else choice):
                        reorders.append("fallthrough")   # top pick(s) not legal right now
                    if self.debug:
                        self._log(battle, pooled, choice)
                    if self.record:
                        self._record(battle, pooled, choice, fallback=False, mixed=mixed,
                                     reorders=reorders,
                                     ms=(time.perf_counter() - t_turn) * 1000)
                    self._emit("decision", battle=battle, choice=choice, ranked=ranked,
                               mixed=mixed, reorders=list(reorders), fallback=False,
                               ms=(time.perf_counter() - t_turn) * 1000)
                    self._futility_record(battle, choice)
                    return order
            # Nothing usable from the engine -> dumb but safe move. Frequent = bug signal.
            self.diag["fallback"] += 1
            if self.record:
                self._record(battle, pooled, "<fallback>", fallback=True)
            self._emit("decision", battle=battle, choice="<fallback>", ranked=ranked,
                       mixed=False, reorders=list(reorders), fallback=True,
                       ms=(time.perf_counter() - t_turn) * 1000)
            return self.choose_max_damage_move(battle) if battle.available_moves \
                else self.choose_random_move(battle)
        except Exception:
            self.diag["fallback"] += 1
            self._emit("decision", battle=battle, choice="<crash>", ranked=[],
                       mixed=False, reorders=[], fallback=True, ms=0.0)
            return self.choose_random_move(battle)

    def _mix_sample(self, ranked):
        """Sample the final choice among near-tied world-winners (mix_root=True).

        Deterministic argmax is exploitable: humans and FP's MCTS both learn our response
        and switch the counter in. FP's answer is random.choices over moves within 75% of
        the top score; this is our port, constrained so it can never undo a shipped guard:

          - only world-winners (score >= _win_floor) are eligible;
          - the eligible set is a PREFIX of the final ranked order, so the first candidate
            below the threshold or vetoed by a guard ends it -- demoted choices and
            everything after them stay out;
          - '-tera' is eligible only as the argmax (tera_wasted is loss-correlated);
          - weights are the vote scores, so world-win counts dominate.

        Returns (possibly reordered ranked, whether a non-argmax was sampled)."""
        if not self.mix_root or len(ranked) < 2 or ranked[0][1] < self._win_floor:
            return ranked, False
        top_score = ranked[0][1]
        elig = []
        for i, (choice, score) in enumerate(ranked):
            if (score < self._win_floor or score < top_score * self.mix_frac
                    or choice in self._vetoed):
                break
            if i > 0 and choice.endswith("-tera"):
                continue
            elig.append((choice, score))
        if len(elig) < 2:
            return ranked, False
        if (self.mix_collapse_eps and len(elig) >= 3
                and elig[0][1] - elig[-1][1] <= self.mix_collapse_eps):
            # Policy collapse: a >=3-way flat eligible set is Q-noise, not a strategy.
            self.diag["mix_suppressed"] = self.diag.get("mix_suppressed", 0) + 1
            self._emit("mix", eligible=elig, pick=None, suppressed=True)
            return ranked, False
        import random
        pick = random.choices([c for c, _ in elig], weights=[s for _, s in elig], k=1)[0]
        self._emit("mix", eligible=elig, pick=pick, suppressed=False)
        if pick == ranked[0][0]:
            return ranked, False
        self.diag["mix_fired"] = self.diag.get("mix_fired", 0) + 1
        scores = dict(ranked)
        return [(pick, scores[pick])] + [rc for rc in ranked if rc[0] != pick], True

    # --- learned value re-ranking ------------------------------------------------

    @staticmethod
    def _engine_choice(choice):
        """Root move_choice -> the spelling generate_instructions accepts.

        'switch <species>' becomes the bare species; 'No Move' (the opponent's only option
        while we take a post-faint free switch) becomes 'none'; moves and '-tera' pass
        through."""
        if choice.startswith("switch "):
            return choice.split(" ", 1)[1]
        return "none" if choice == "No Move" else choice

    def _value_rerank(self, ranked, battle=None):
        """Re-rank near-tied world-winners by learned state value.

        Per candidate: roll one ply in up to value_worlds worlds against the opponent's top
        MCTS replies (visit-weighted), featurize every chance branch, average the net's win
        probabilities. The search's ordering stands unless the net prefers a different
        near-tied candidate. value_margin is in vote units where one world-win = 10, so the
        default 11 only lets the net overturn candidates within one world-win of the
        leader; the search stays in charge of clear decisions.

        Two states get a WIDER margin because the engine's leaf eval is known-bad there:
        - value_on_force_switch: post-faint the replacement is a pure positional choice
          with no tempo cost, so compare all switch candidates by V.
        - value_boost_margin: while the opponent is visibly boosted (offense/speed total
          >= +2), widen so the net can overrule chip-damage autopilot."""
        if self._value_model is None or len(ranked) < 2 or not self._worlds:
            return ranked
        top_score = ranked[0][1]
        if top_score < self._win_floor:   # top choice won no world: nothing trustworthy
            return ranked
        # Margins are calibrated in vote units (one world-win = 10). Averaging-mode scores
        # are 0..1 shares, so /100 keeps the authority band equivalent (11 -> 0.11).
        margin = self.value_margin if self.robust_vote else self.value_margin / 100.0
        free_switch = (self.value_on_force_switch and battle is not None
                       and bool(getattr(battle, "force_switch", False)))
        if free_switch:
            margin = float("inf")
        elif self.value_boost_margin and battle is not None:
            opp = battle.opponent_active_pokemon
            if opp is not None:
                b = opp.boosts
                if b.get("atk", 0) + b.get("spa", 0) + b.get("spe", 0) >= 2:
                    margin = self.value_boost_margin if self.robust_vote \
                        else self.value_boost_margin / 100.0
        # Same rule as _mix_sample: '-tera' eligible only as the argmax. The net may confirm
        # or displace a tera the search already chose, never PROMOTE one from below --
        # mining: 7/17 teras in losses were rerank promotions of a non-argmax '-tera', vs
        # 0/6 in wins (tera_wasted 0.37/loss, 0/win).
        tied = [c for c, s in ranked if s >= self._win_floor and top_score - s <= margin
                and (c == ranked[0][0] or not c.endswith("-tera"))]
        if len(tied) < 2:
            return ranked
        tied = tied[:5 if free_switch else 3]
        try:
            feats, owners, weights = [], [], []
            from value_features import featurize
            for state, res in self._worlds[:self.value_worlds]:
                opp = sorted(res.side_two, key=lambda o: -o.visits)[:self.value_opp_moves]
                opp = [o for o in opp if o.visits > 0]
                opp_total = sum(o.visits for o in opp) or 1
                for cand_i, cand in enumerate(tied):
                    for o in opp:
                        try:
                            branches = generate_instructions(
                                state, self._engine_choice(cand),
                                self._engine_choice(o.move_choice))
                        except Exception:
                            continue
                        for si in branches:
                            w = (o.visits / opp_total) * (si.percentage / 100.0)
                            if w <= 0.0:
                                continue
                            feats.append(featurize(state.apply_instructions(si),
                                                   self.value_case_fix))
                            owners.append(cand_i)
                            weights.append(w)
            if not feats:
                return ranked
            import numpy as _np
            import torch
            with torch.no_grad():
                probs = torch.sigmoid(
                    self._value_model(torch.from_numpy(_np.stack(feats)))).numpy()
            num = _np.zeros(len(tied))
            den = _np.zeros(len(tied))
            for i, w, p in zip(owners, weights, probs):
                num[i] += w * p
                den[i] += w
            if (den <= 0).any():
                # No valued successor means the rolls CRASHED, not that the candidate is
                # bad. Reordering here silently demotes it below every valued candidate with
                # unbounded gap (observed live: spiritbreak 0.75 -> reflect 0.15 the turn
                # before the mon died). A failed roll is a bug signal, never a verdict.
                # Stand down entirely.
                self.diag["value_roll_fail"] = self.diag.get("value_roll_fail", 0) + 1
                return ranked
            valued = [(c, float(num[i] / den[i])) for i, c in enumerate(tied)]
            valued.sort(key=lambda cv: cv[1], reverse=True)
            self._emit("value", valued=valued, margin=margin, free_switch=free_switch,
                       branches=len(feats))
            if valued[0][0] != ranked[0][0]:
                self.diag["value_rerank"] = self.diag.get("value_rerank", 0) + 1
            order = [c for c, _v in valued]
            rest = [rc for rc in ranked if rc[0] not in set(order)]
            scores = dict(ranked)
            return [(c, scores[c]) for c in order] + rest
        except Exception:
            self.diag["value_err"] = self.diag.get("value_err", 0) + 1
            return ranked

    def _successor_sig(self, state, our_choice, opp_choice):
        """Sorted (probability, state-string) branch signature for one joint move pair."""
        sig = []
        for b in generate_instructions(state, our_choice, opp_choice):
            ns = state.apply_instructions(b)
            sig.append((round(b.percentage, 1), ns.to_string()))
        return sorted(sig)

    def _noop_demote(self, ranked):
        """Demote guaranteed-wasted turns below every choice that does something.

        Mined from live losses: full-HP Slack Off x5 and Stealth Rock onto existing rocks x3
        in a single 30-game run. The engine models the failure correctly, its flat eval just
        keeps clicking anyway, and the value net only argues on near-ties.

        Deterministic check, no model: a candidate is a no-op when, against the opponent's
        top predicted reply, every chance branch of the successor equals the branch you get
        from doing NOTHING ('none'). Rolling against the real reply rather than 'none' keeps
        context-dependent moves honest -- Sucker Punch vs an attacker differs from passing
        and survives; vs a status click it genuinely is a pass and gets demoted. Switches
        never match the pass baseline, so they're checked but effectively always kept."""
        if len(ranked) < 2 or not self._worlds or ranked[0][1] < self._win_floor:
            return ranked
        winners = [c for c, s in ranked if s >= self._win_floor][:3]
        if len(winners) < 2:
            return ranked
        state, res = self._worlds[0]
        opp = max(res.side_two, key=lambda o: o.visits, default=None)
        if opp is None:
            return ranked
        opp_choice = self._engine_choice(opp.move_choice)
        try:
            baseline = self._successor_sig(state, "none", opp_choice)
        except Exception:
            self.diag["noop_err"] = self.diag.get("noop_err", 0) + 1
            return ranked
        noops = set()
        for c in winners:
            try:
                if self._successor_sig(state, self._engine_choice(c), opp_choice) == baseline:
                    noops.add(c)
            except Exception:
                self.diag["noop_err"] = self.diag.get("noop_err", 0) + 1
        if not noops or len(noops) == len(winners):
            return ranked
        # A '-tera' variant always differs from the pass baseline (tera rewrites the state
        # before we act, even if we then die), so it can never be flagged a no-op -- which
        # turned this reorder into a tera PROMOTER: plain move demoted, tera twin bubbles up
        # and burns the resource on a move the search scored 19:1 against (observed live:
        # psyshock 0.11 -> psyshock-tera). Same rule as everywhere else: treat a non-argmax
        # tera as a no-op here too. If that leaves nothing promotable, stand down.
        keep = [c for c in winners
                if c not in noops and (c == ranked[0][0] or not c.endswith("-tera"))]
        if not keep:
            return ranked
        demoted = [c for c in winners if c not in keep]
        noops = set(demoted)
        self._vetoed |= noops
        if ranked[0][0] in noops:
            self.diag["noop_demote"] = self.diag.get("noop_demote", 0) + 1
        scores = dict(ranked)
        order = [c for c in winners if c not in noops] + [c for c in winners if c in noops]
        rest = [rc for rc in ranked if rc[0] not in set(winners)]
        return [(c, scores[c]) for c in order] + rest

    def _dead_lock_guard(self, battle, ranked):
        """Never lock the last mon into a status move.

        Observed live (Ditto-as-Giratina, last mon, Choice Scarf): every line losing, so the
        worlds' votes are noise, and 5/8 picked Will-O-Wisp -- locking the final Pokemon
        into a status move for the rest of the game. Locked status with no bench is strictly
        dominated by locked attack: both are locks, one does damage.

        Fires only when (a) we hold an unlocked Choice item, (b) no legal switches, and
        (c) an attacking choice also won a world. So it swaps two world-winners; it never
        overrides the search's only idea."""
        me = battle.active_pokemon
        if (not ranked or me is None or not me.item
                or to_id_str(me.item) not in ("choiceband", "choicespecs", "choicescarf")
                or battle.available_switches or len(battle.available_moves) <= 1):
            return ranked

        def is_status(choice):
            if choice.startswith("switch "):
                return False
            move_id = choice[:-5] if choice.endswith("-tera") else choice
            mv = get_move(move_id)
            return mv is not None and getattr(mv.category, "name", "") == "STATUS"

        top_choice, top_score = ranked[0]
        if top_score < self._win_floor or not is_status(top_choice):
            return ranked
        for i, (choice, score) in enumerate(ranked[1:], start=1):
            if score >= self._win_floor and not is_status(choice):
                self.diag["dead_lock_guard"] = self.diag.get("dead_lock_guard", 0) + 1
                self._vetoed.add(top_choice)
                order = [ranked[i]] + ranked[:i] + ranked[i + 1:]
                return order
        return ranked

    def _absorb_demote(self, battle, ranked):
        """Demote attacks the standing opponent CERTAINLY nullifies, at any score gap.

        Type immunity or an absorb ability (Earth Eater and friends). The near-tie guards
        structurally can't catch this class: observed live, Metagross clicked Earthquake
        into a revealed-Earth-Eater Orthworm at eq=0.26 vs knockoff=0.21 -- a 0.05 gap,
        outside _damage_tiebreak's eps. Every pipeline layer had the ability right; the
        position was just bad enough that visit shares flattened and argmax-over-noise
        landed on the one move that HEALS the target. Re-running the same search on the
        replay reproduces it: near-uniform policy, EQ at 0.17-0.18 every time.

        move_nullified is hard knowledge (revealed ability/typing, or every surviving
        randbats set agrees), so unlike the value net this may override a confident score.
        The MCTS lines where a nullified click is right are hedges against a predicted
        switch; when the search genuinely prefers one it'll still top worlds afterwards. If
        NOTHING non-nullified remains we stand down rather than veto. Demoted choices join
        _vetoed so the mixer can't resurrect them."""
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        if len(ranked) < 2 or me is None or opp is None:
            return ranked

        def nullified(choice):
            if choice.startswith("switch "):
                return False
            move_id = choice[:-5] if choice.endswith("-tera") else choice
            for mv in battle.available_moves:
                if mv.id == move_id:
                    return move_nullified(me, mv, opp)
            return False

        dead = {c for c, _s in ranked if nullified(c)}
        if not dead or all(c in dead or c.endswith("-tera") for c, _s in ranked):
            return ranked
        if ranked[0][0] in dead:
            self.diag["absorb_demote"] = self.diag.get("absorb_demote", 0) + 1
        self._vetoed |= dead
        return ([rc for rc in ranked if rc[0] not in dead]
                + [rc for rc in ranked if rc[0] in dead])

    # Futility tuning: how many consecutive same-move clicks must be net-erased (chip) or
    # net-reset (boosts) before we demote, and the HP slack that still counts as erased
    # (leftovers-scale jitter shouldn't hide a wall).
    #
    # Chip anchoring, from replaying the mined Blissey game: walls OSCILLATE. Her hp
    # snapshots ran 71->35->50->61->23. "No progress since the run started" anchors on her
    # maximum and never fires; flagging at the 23% snapshot would be wrong. The honest
    # signature is a healed-back stretch ENDING NOW -- current hp at-or-above some snapshot
    # at least FUTILE_CHIP_RUN clicks old (35->61 across two Iron Heads = the last two
    # clicks bought nothing). Slow-but-monotone chip never matches, and one berry pop
    # doesn't either, since progress from the older anchor still shows.
    FUTILE_CHIP_RUN = 2
    FUTILE_BOOST_RUN = 2
    FUTILE_HP_EPS = 0.02

    def _futility_record(self, battle, choice):
        """Append this decision to the battle's futility history (see __init__)."""
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        if me is None or opp is None:
            return
        self._futility_hist.setdefault(battle.battle_tag, []).append({
            "turn": battle.turn,
            "me": me.species,
            "opp": opp.species,
            "choice": choice,
            "opp_hp": opp.current_hp_fraction,
            "my_boosts": sum(v for v in (me.boosts or {}).values() if v > 0),
        })

    def _futility_demote(self, battle, ranked):
        """Demote moves that observed history proves are going nowhere, at any score gap.

        Three independent replay deep-reads (ladder AND Foul Play losses) converged on the
        same #1 loss class: the search treats each chip hit as progress against a target
        whose recovery erases it (Blissey Soft-Boiled x4, Sylveon Wish/Protect x10,
        Regenerator pivot cycles), or re-boosts into a Haze user (QD x7 into Milotic). Every
        such click is a CONFIDENT argmax (roost 0.85, QD 0.36-0.55), so tie-window guards
        structurally cannot engage, and the value net shares the blind spot because
        self-play stalls symmetrically.

        Detection is pure observation -- no recovery prediction, no set guessing:
          * chip futility: our most recent consecutive decisions in this matchup (same
            our-mon, same opp-mon) all clicked this same damaging move, and opp's CURRENT
            hp is at-or-above some run snapshot at least FUTILE_CHIP_RUN clicks old, i.e.
            the last >=2 clicks were net-erased. See the class-constant comment for the
            oscillation anchoring rationale.
          * boost futility: same, for a self-boosting status move over FUTILE_BOOST_RUN
            clicks with our net positive boosts not increased (something Hazed/phazed).

        A run breaks on any different choice or either side switching -- new matchup, fresh
        evidence -- so after a demote forces one different move the loop can't silently
        resume for another full run. Demoted moves join _vetoed (the mixer was re-picking
        the same futile click otherwise). If nothing non-futile remains, stand down; never
        leave the search with no idea."""
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        if len(ranked) < 2 or me is None or opp is None:
            return ranked
        hist = self._futility_hist.get(battle.battle_tag)
        if not hist or len(hist) < self.FUTILE_BOOST_RUN:
            return ranked

        def run_of(choice):
            """Consecutive most-recent decisions that clicked `choice` in this matchup."""
            run = []
            for rec in reversed(hist):
                if (rec["choice"] != choice or rec["me"] != me.species
                        or rec["opp"] != opp.species):
                    break
                run.append(rec)
            return run     # most-recent first; run[-1] is the start of the streak

        def futile(choice):
            if choice.startswith("switch "):
                return False
            move_id = choice[:-5] if choice.endswith("-tera") else choice
            mv = get_move(move_id)
            if mv is None:
                return False
            run = run_of(choice)
            if getattr(mv.category, "name", "") != "STATUS":
                if len(run) < self.FUTILE_CHIP_RUN:
                    return False
                # Anchors at least FUTILE_CHIP_RUN clicks back (run is most-recent-first).
                anchors = [rec["opp_hp"] for rec in run[self.FUTILE_CHIP_RUN - 1:]]
                return opp.current_hp_fraction >= min(anchors) - self.FUTILE_HP_EPS
            boosts = dict(mv.boosts or {})
            boosts.update(mv.self_boost or {})
            if not any(v > 0 for v in boosts.values()):
                return False
            my_now = sum(v for v in (me.boosts or {}).values() if v > 0)
            return (len(run) >= self.FUTILE_BOOST_RUN
                    and my_now <= run[-1]["my_boosts"])

        dead = {c for c, _s in ranked if futile(c)}
        if not dead or all(c in dead or c.endswith("-tera") for c, _s in ranked):
            return ranked
        if ranked[0][0] in dead:
            self.diag["futility_demote"] = self.diag.get("futility_demote", 0) + 1
        self._vetoed |= dead
        return ([rc for rc in ranked if rc[0] not in dead]
                + [rc for rc in ranked if rc[0] in dead])

    def _gamble_veto(self, battle, ranked):
        """Demote a status/setup click that a competent opponent punishes for free.

        Decoupled-UCT MCTS prices the opponent as the average of its exploration mixture,
        not as a best-responder. Re-searching a clean live loss (Scrafty vs 15% Mismagius)
        showed the tree giving the guaranteed-kill Knock Off only 7-20% of opponent visits
        vs 72-89% for Bulk Up -- so our setup clicks get scored against an opponent who
        mostly declines free kills. Humans at 2000+ never decline.

        Imposes best-response reasoning at the root, in two steps, all by simulation
        (generate_instructions), never by rules:

        1. KILL REPLIES: roll the setup candidate against every attacking option the
           determinized opponent has; replies with P(our active faints) >= 0.95 are kills.
           Simulation handles the legitimate-gamble cases natively -- a faster Calm Mind
           that lifts us out of range survives its roll, Sucker Punch fails against a
           status click, Protect/Substitute absorb the hit. No kill -> no veto.
        2. RISK DOMINANCE ("does punishing cost them anything?"): roll our best NON-setup
           candidate against the kill reply and against the tree's preferred reply, score
           both with the value net FROM THEIR SIDE. If the kill is within eps of their best
           even when we don't set up, punishing is free -- assume a competent human clicks
           it and veto our setup. If punishing only pays against our setup, that's a genuine
           mind game (e.g. they should switch out) and the search's gamble stands.

        Demoted choices join _vetoed -- two of three clean-cohort gambles were MIX picks out
        of near-uniform policies. Stands down if no candidate survives its own rolls, so
        last-mon desperation stays legal."""
        if self._value_model is None or len(ranked) < 2 or not self._worlds:
            return ranked
        me = battle.active_pokemon
        if me is None or getattr(battle, "force_switch", False):
            return ranked

        def is_status(choice):
            if choice.startswith("switch "):
                return False
            mv = get_move(choice[:-5] if choice.endswith("-tera") else choice)
            return mv is not None and getattr(mv.category, "name", "") == "STATUS"

        # Whole promotable set, not just top-3: the Plusle case had nastyplot promoted from
        # 4th by the value rerank.
        prefix = [c for c, s in ranked[:5] if s >= self._win_floor]
        setups = [c for c in prefix if is_status(c)]
        alts = [c for c in prefix if not is_status(c)]
        if not setups or not alts:
            return ranked
        alt = alts[0]

        def faint_prob(state, our_choice, opp_choice):
            """P(our active is fainted after one ply), from the engine's branch weights."""
            branches = generate_instructions(
                state, self._engine_choice(our_choice), self._engine_choice(opp_choice))
            die = total = 0.0
            for si in branches:
                total += si.percentage
                st2 = state.apply_instructions(si)
                act = st2.side_one.pokemon[int(str(st2.side_one.active_index))]
                if act.hp <= 0:
                    die += si.percentage
            return die / total if total else 0.0

        def their_value(state, our_choice, opp_choice):
            """Opponent's mean value-net score of the one-ply successor (1 = they win)."""
            from value_features import featurize
            import numpy as _np
            import torch
            branches = generate_instructions(
                state, self._engine_choice(our_choice), self._engine_choice(opp_choice))
            feats, ws = [], []
            for si in branches:
                if si.percentage <= 0:
                    continue
                feats.append(featurize(state.apply_instructions(si), self.value_case_fix))
                ws.append(si.percentage)
            if not feats:
                return None
            with torch.no_grad():
                ours = torch.sigmoid(
                    self._value_model(torch.from_numpy(_np.stack(feats)))).numpy()
            w = _np.asarray(ws)
            return float(1.0 - (ours * w).sum() / w.sum())

        dead = set()
        try:
            for cand in setups:
                votes = checked = 0
                for state, res in self._worlds[:2]:
                    opp_moves = [o.move_choice for o in res.side_two
                                 if not o.move_choice.startswith("switch")
                                 and o.move_choice != "No Move"]
                    if not opp_moves:
                        continue
                    kills = [k for k in opp_moves
                             if faint_prob(state, cand, k) >= 0.95]
                    if not kills:
                        checked += 1
                        continue
                    top_reply = max(res.side_two, key=lambda o: o.visits).move_choice
                    vt = their_value(state, alt, top_reply)
                    if vt is None:
                        continue
                    checked += 1
                    # Punishing is FREE if some kill reply is ~as good as their preferred
                    # line even against our non-setup alternative.
                    if any((vk := their_value(state, alt, k)) is not None
                           and vk >= vt - 0.02 for k in kills):
                        votes += 1
                if checked and votes * 2 >= checked and votes > 0:
                    dead.add(cand)
        except Exception:
            self.diag["gamble_err"] = self.diag.get("gamble_err", 0) + 1
            return ranked
        if not dead:
            return ranked
        survivors = [c for c, s in ranked
                     if c not in dead and (c == ranked[0][0] or not c.endswith("-tera"))]
        if not survivors:
            return ranked
        if ranked[0][0] in dead:
            self.diag["gamble_veto"] = self.diag.get("gamble_veto", 0) + 1
        self._vetoed |= dead
        return ([rc for rc in ranked if rc[0] not in dead]
                + [rc for rc in ranked if rc[0] in dead])

    def _damage_tiebreak(self, battle, ranked, eps=None):
        """Among effectively-tied top choices, prefer the one that damages the REVEALED
        opponent.

        Observed live: Close Combat and Brave Bird each won 3 worlds with identical pooled
        shares against a Ghost-type, and the coin flip landed on the immune move.
        Determinization noise can't separate an exact tie; a one-shot damage estimate
        against the known opponent can. Switches in the tie keep their relative order but
        rank below any move that deals damage."""
        if eps is None:
            # Robust mode: 0.15 share gap within the same world-win count is a tie.
            # Averaging-mode scores are pure shares where 0.15 is a real preference gap,
            # so "tied" is much tighter there.
            eps = 0.15 if self.robust_vote else 0.03
        if len(ranked) < 2 or ranked[0][1] - ranked[1][1] >= eps:
            return ranked
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        if me is None or opp is None:
            return ranked
        top = ranked[0][1]
        tied = [rc for rc in ranked if top - rc[1] < eps]

        def dmg(choice):
            if choice.startswith("switch "):
                return -1.0
            move_id = choice[:-5] if choice.endswith("-tera") else choice
            for mv in battle.available_moves:
                if mv.id == move_id:
                    try:
                        return float(estimate_damage_fraction(me, mv, opp))
                    except Exception:
                        return 0.0
            return -1.0

        argmax = ranked[0][0]
        tied.sort(key=lambda rc: dmg(rc[0]), reverse=True)
        # '-tera' only as the argmax, same as every other reranker. dmg() strips the suffix
        # so a tera twin sorts by its plain damage anyway, and promoting one on a tie spends
        # the resource on a coin flip (observed live: spikes 0.19 vs woodhammer-tera 0.18 ->
        # tera burned on a 0.01 gap; this was the last reranker with the hole).
        tied = ([rc for rc in tied if rc[0] == argmax or not rc[0].endswith("-tera")]
                + [rc for rc in tied if rc[0] != argmax and rc[0].endswith("-tera")])
        # A tied MOVE dealing nothing while a tied alternative deals real damage is exactly
        # the immune-click case this guard exists for -- keep it out of the root mix too.
        # Switches (dmg sentinel -1) are a different action class and stay mixable.
        best_dmg = dmg(tied[0][0])
        if best_dmg > 0.1:
            self._vetoed |= {c for c, _s in tied
                             if not c.startswith("switch ") and dmg(c) <= 0.0}
        return tied + ranked[len(tied):]

    def _record(self, battle, pooled, choice, fallback, mixed=False, reorders=None, ms=None):
        """Per-turn decision snapshot for post-hoc loss analysis (record=True).

        opp_boosts and ms are here so mining doesn't have to re-parse the replay HTML and
        re-join it to this file on turn number: the setup-pressure metrics (mine_losses
        boost_pressure / mechanism counters) are the ones that separate a cohort, and
        deriving them needed a protocol-log join that made the loop annoying enough to
        skip. Both are read with .get downstream, so older traces still load."""
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        top = sorted(pooled.items(), key=lambda kv: kv[1], reverse=True)[:3]
        entry = {
            "turn": battle.turn,
            "me": f"{me.species} {me.current_hp_fraction:.0%}" if me else "-",
            "opp": f"{opp.species} {opp.current_hp_fraction:.0%}" if opp else "-",
            "my_team_alive": sum(1 for m in battle.team.values() if not m.fainted),
            "opp_team_alive": 6 - sum(1 for m in battle.opponent_team.values() if m.fainted),
            "top": [(c, round(s, 2)) for c, s in top],
            "choice": choice,
            "fallback": fallback,
            "opp_boosts": {k: v for k, v in (opp.boosts or {}).items() if v} if opp else {},
        }
        if ms is not None:
            entry["ms"] = round(ms, 1)
        if mixed:
            entry["mixed"] = True
        if reorders:
            entry["reorders"] = reorders
        self.traces.setdefault(battle.battle_tag, []).append(entry)

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
