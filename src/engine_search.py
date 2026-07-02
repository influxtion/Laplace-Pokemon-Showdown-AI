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

from poke_engine import monte_carlo_tree_search, generate_instructions

from knowledge import estimate_damage_fraction, get_move, safe_priority, _estimate_stat
from poke_engine_adapter import build_state


def _spe_mult(boost):
    """Speed multiplier for a boost stage."""
    return (2 + boost) / 2 if boost >= 0 else 2 / (2 - boost)


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
                 debug=False, record=False, use_stats=True, speed_inference=True,
                 value_model_path=None, value_worlds=4, value_opp_moves=2,
                 value_margin=11.0, **kwargs):
        super().__init__(*args, **kwargs)
        if n_determinizations is not None:
            self.N_DETERMINIZATIONS = n_determinizations
        if search_time_ms is not None:
            self.SEARCH_TIME_MS = search_time_ms
        if threads is not None:
            self.THREADS = threads
        self.debug = debug
        self.use_stats = use_stats   # use the real randbats item/ability/tera feed in determinization
        self.speed_inference = speed_inference   # infer Choice Scarf from observed turn order
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
        # battle_tag -> {species_id: "scarf" | "noscarf"}: Choice Scarf verdicts inferred from
        # observed turn order. Randbats spreads are fixed (85 EVs / 31 IVs / neutral), so an
        # opponent's raw speed is essentially known; if it moved first when its raw speed says
        # it shouldn't have, it's Scarf'd (or has a speed ability -- same modeling either way).
        self._opp_speed = {}
        # battle_tag -> pending delayed effects: '<side>_wish' -> (lands_on_turn, heal_amount),
        # '<side>_fs' -> hits_at_end_of_turn. Sides are 'p1'/'p2' as on the protocol.
        self._pending = {}
        # Learned value head (Track A). The engine's MCTS leaf eval under-fears opponent
        # setup (observed: a guaranteed-fail Substitute scored 0.478 vs 0.480 for the best
        # move against a +3/+3/+3 sweeper), so near-tied root candidates are re-ranked by
        # rolling each one ply with generate_instructions and scoring the successor states
        # with a net trained on self-play outcomes (train_value.py / value_features.py).
        self._value_model = None
        self._worlds = []            # [(determinized state, MctsResult)] from the last turn
        self.value_worlds = value_worlds
        self.value_opp_moves = value_opp_moves
        self.value_margin = value_margin
        if value_model_path:
            self._load_value_model(value_model_path)

    def _load_value_model(self, path):
        import torch
        from train_value import ValueNet
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net = ValueNet(ckpt["n_in"])
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        torch.set_num_threads(1)     # tiny net; don't fight the engine threads
        self._value_model = net

    # --- speed inference from turn order --------------------------------------

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
        """Collect who-moved-first evidence (Scarf inference) and pending delayed effects
        (Wish / Future Sight) from a message payload. Speed comparisons are skipped for turns
        'tainted' by mid-turn switches (pivots) or in-turn speed-boost changes, since the
        end-of-payload battle state then doesn't reflect move-order conditions."""
        moves = []          # (side, move_id) in order within the current turn
        tainted = False
        wish_side = fs_side = None      # sides that queued a delayed effect this payload
        pend = self._pending.setdefault(battle.battle_tag, {})
        for msg in split_messages:
            if len(msg) < 2:
                continue
            tag = msg[1]
            if tag == "turn":
                # A new turn line anchors any delayed effect queued earlier in the payload:
                # Wish heals at the end of the turn we're now deciding; Future Sight one later.
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
        """Update the Scarf verdict for the opponent's active from one same-priority
        move pair. Conservative: 5% tolerance both ways for state-timing noise; a 'scarf'
        verdict (they outsped their known raw speed) overrides 'noscarf', never vice versa."""
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
        """Ranked {move_choice: score} over N determinized searches, or {} on failure.

        Aggregation is a *robust vote*, not a plain average of visit shares. Averaging has a
        known determinization pathology (observed live: Scale Shot clicked into a revealed
        Fairy): when worlds disagree on the best move, a hedge move that is every world's
        mediocre runner-up can top the pooled average despite being no world's best choice.
        So a move is only eligible while some world made it the outright winner; rank
        eligible moves by worlds won, then pooled visit share. Non-winning moves keep their
        pooled share (scaled below every winner) purely as fallback order for legality."""
        pooled = {}
        wins = {}
        runs = 0
        self._worlds = []
        opp_used = self._opp_used_since_switch(battle)
        speed_hints = self._opp_speed.get(battle.battle_tag, {})
        for _ in range(self.N_DETERMINIZATIONS):
            try:
                state = build_state(battle, use_stats=self.use_stats,
                                    opp_used_since_switch=opp_used,
                                    opp_speed_hints=speed_hints,
                                    pending=self._pending.get(battle.battle_tag))
                res = monte_carlo_tree_search(state, self.SEARCH_TIME_MS, threads=self.THREADS)
            except BaseException:
                # poke-engine can *panic* on an inconsistent state (raises PanicException, which
                # is NOT an Exception subclass); skip that sample rather than crash the turn.
                self.diag["det_fail"] += 1
                continue
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
        # Score: world-winners sort above everything by (wins, pooled share); the rest keep a
        # small share-proportional score so the legality fallback still has a sane order.
        scores = {}
        for choice, share in pooled.items():
            share /= runs
            w = wins.get(choice, 0)
            scores[choice] = (w * 10.0 + share) if w else share * 1e-3
        return scores

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
            ranked = sorted(pooled.items(), key=lambda kv: kv[1], reverse=True)
            ranked = self._damage_tiebreak(battle, ranked)
            ranked = self._value_rerank(ranked)
            for choice, _share in ranked:
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

    # --- learned value re-ranking ---------------------------------------------

    @staticmethod
    def _engine_choice(choice):
        """Root move_choice string -> the string generate_instructions accepts
        ('switch <species>' becomes the bare species name; moves/-tera pass through)."""
        return choice.split(" ", 1)[1] if choice.startswith("switch ") else choice

    def _value_rerank(self, ranked):
        """Re-rank near-tied world-winning candidates by learned state value.

        For each candidate: roll it one ply forward in up to `value_worlds` determinized
        worlds against the opponent's top MCTS replies (visit-weighted), featurize every
        chance branch of the successor states, and average the value net's win
        probabilities. The search's own ordering stands unless the value net prefers a
        different near-tied candidate. `value_margin` is in robust-vote score units where
        one world-win = 10 (+ up to 1 of pooled share), so the default of 11 only lets the
        net overturn candidates within one world-win of the leader -- the search stays in
        charge of clear decisions."""
        if self._value_model is None or len(ranked) < 2 or not self._worlds:
            return ranked
        top_score = ranked[0][1]
        if top_score < 10.0:      # top choice won no world: nothing trustworthy to rerank
            return ranked
        tied = [c for c, s in ranked if s >= 10.0 and top_score - s <= self.value_margin]
        if len(tied) < 2:
            return ranked
        tied = tied[:3]
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
                            feats.append(featurize(state.apply_instructions(si)))
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
            valued = [(c, float(num[i] / den[i])) for i, c in enumerate(tied) if den[i] > 0]
            if not valued:
                return ranked
            valued.sort(key=lambda cv: cv[1], reverse=True)
            if valued[0][0] != ranked[0][0]:
                self.diag["value_rerank"] = self.diag.get("value_rerank", 0) + 1
            order = [c for c, _v in valued]
            rest = [rc for rc in ranked if rc[0] not in set(order)]
            scores = dict(ranked)
            return [(c, scores[c]) for c in order] + rest
        except Exception:
            self.diag["value_err"] = self.diag.get("value_err", 0) + 1
            return ranked

    def _damage_tiebreak(self, battle, ranked, eps=0.15):
        """Among effectively-tied top choices, prefer the one that actually damages the
        *revealed* current opponent. Observed live: Close Combat and Brave Bird each won 3
        worlds with identical pooled shares vs a Ghost-type, and the coin flip landed on the
        immune move. Determinization noise can't distinguish exact ties; a one-shot damage
        estimate against the known opponent can. Switches in the tie keep their position
        relative to each other but rank below any move that deals damage."""
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

        tied.sort(key=lambda rc: dmg(rc[0]), reverse=True)
        return tied + ranked[len(tied):]

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
