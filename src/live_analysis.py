r"""Live terminal commentary for one battle: the play-by-play plus what the search was
actually thinking when it picked each move.

The bot is normally a black box -- ladder.py prints one line per game and the reasoning
only surfaces later in the .trace.json that mine_losses.py reads. This renders it AS THE
GAME HAPPENS, so a single ladder game can be watched and understood live instead of
reconstructed afterwards.

Two feeds, interleaved into one transcript:

  * The BATTLE, from the Showdown protocol (LiveAnalyzer.feed): moves, switches, HP swings,
    crits/misses, status, boosts, hazards, weather, tera, faints. Parsed from raw protocol
    lines rather than read off poke-env's battle object, because the lines carry their own
    HP numbers -- that's what lets the commentary print BEFORE poke-env processes the
    payload, keeping "what happened" above "what we decided next".

  * The SEARCH, via EnginePlayer's observer hook (LiveAnalyzer.__call__): worlds completing
    one at a time, pooled candidate shares with per-world win counts and the engine's eval,
    which hidden sets the sampler drew, which guard moved the front-runner, and whether the
    mixed root sampled off the argmax.

Read-only instrumentation throughout: the observer never feeds a value back into a
decision, so a game played with the analyzer attached is the same game. Nothing here may
raise into the client loop either -- EnginePlayer._emit swallows observer errors and the
protocol feed is wrapped the same way, because a broken renderer must never cost a rated
game. Renderer failures are counted and reported in summary() instead of dying quietly.

Consumer: analyze_battle.py.
"""

import asyncio
import re
import shutil
import sys
import time

from poke_env.data import GenData, to_id_str

from engine_search import EnginePlayer
from knowledge import get_move

_DEX = GenData.from_gen(9).pokedex


# --- terminal capability ------------------------------------------------------------------

def enable_ansi():
    """Turn on VT escape processing on a Windows console. No-op elsewhere, or if it fails.

    Windows 11 terminals normally have this on already; older conhost does not, and without
    it every colour code prints as literal garbage."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        for handle in (-11, -12):                       # stdout, stderr
            mode = ctypes.c_uint32()
            h = k.GetStdHandle(handle)
            if k.GetConsoleMode(h, ctypes.byref(mode)):
                k.SetConsoleMode(h, mode.value | 0x0004)   # VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


def stdout_can(text):
    """Can stdout actually encode this text? A cp1252 console cannot print block glyphs,
    and one UnicodeEncodeError inside a print would take the game down with it."""
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        text.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


_COLORS = {
    "reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
    "us": "\033[36m", "opp": "\033[35m", "good": "\033[32m", "bad": "\033[31m",
    "warn": "\033[33m", "hdr": "\033[1;97m",
}

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

# Every non-ASCII glyph this module can print, and its plain-ASCII stand-in. Applied at the
# LAST moment (line / _progress) rather than at each call site: a redirected stdout on
# Windows is cp1252, and one stray block glyph raises UnicodeEncodeError mid-game -- inside
# an observer callback, where _emit silently swallows it and the turn's output just
# vanishes. One-to-one by construction, so column widths survive the substitution.
_ASCII_FALLBACK = str.maketrans({
    "·": "|", "═": "=", "─": "-", "█": "#", "░": ".", "●": "+", "○": ".", "✕": "x",
    "▁": ".", "▂": ":", "▃": "-", "▄": "=", "▅": "+", "▆": "*", "▇": "#", "▉": "%",
})


def _vislen(text):
    """Printed width, ignoring colour escapes. Naive padding breaks every column."""
    return len(_ANSI_RE.sub("", text))


def _pad(text, width):
    return text + " " * max(0, width - _vislen(text))


# --- naming ---------------------------------------------------------------------------------

def _species_name(species):
    """'greattusk' -> 'Great Tusk' via the shipped pokedex. Ids are unreadable in a log."""
    entry = _DEX.get(to_id_str(species or ""))
    return entry["name"] if entry else str(species or "?").capitalize()


def _move_name(move_id):
    """'closecombat' -> 'Close Combat', falling back to the raw id for engine pseudo-moves."""
    try:
        mv = get_move(move_id)
        if mv is not None:
            name = mv.entry.get("name")
            if name:
                return name
    except Exception:
        pass
    return str(move_id or "?").capitalize()


def _pretty(choice):
    """A move_choice as a human would say it: 'switch greattusk' -> 'switch Great Tusk',
    'closecombat-tera' -> 'Close Combat +TERA'."""
    if not choice:
        return "-"
    if choice.startswith("switch "):
        return "switch " + _species_name(choice.split(" ", 1)[1])
    if choice == "No Move":
        return "(no move)"
    if choice.startswith("<"):          # <fallback> / <crash> sentinels
        return choice
    if choice.endswith("-tera"):
        return _move_name(choice[:-5]) + " +TERA"
    return _move_name(choice)


def _mmss(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _si(n):
    """1980000 -> '1.98M'. Visit counts get big and the exact digits never matter."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(int(n))


class LiveAnalyzer:
    """Renders the battle and the search into one terminal transcript.

    Wired up two ways (AnalyzedPlayer does both):
        agent.observer = analyzer        # search internals, via EnginePlayer._emit
        analyzer.feed(split_messages)    # protocol lines, before poke-env parses them
    """

    N_CANDIDATES = 4        # candidate moves shown per turn

    def __init__(self, username, width=None, color=True, unicode=None, show_worlds=True):
        self.username = to_id_str(username or "")
        term = shutil.get_terminal_size((100, 30)).columns
        self.width = max(64, min(width or term, 108))
        self.color = bool(color) and sys.stdout.isatty()
        # Block glyphs make the bars readable, but only where the console can encode them.
        # The ASCII fallback carries the same information.
        self.unicode = stdout_can("█░▁▇●·✕") if unicode is None else bool(unicode)
        self.show_worlds = show_worlds
        self.log = []                    # every line printed, for --save-log
        self.agent = None
        self.t_start = time.time()

        # --- per-turn search accumulators (reset by _on_turn_start) ---
        self._t_turn = 0.0
        self._worlds = []                # per-world dicts: visits / eval / winner / reply
        self._world_fails = 0
        self._budget = (0, 0, 0)         # (worlds, ms each, threads each)
        self._pooled_order = []          # candidate order before any guard/reranker
        self._stages = []                # (stage, changed the front-runner?, was, now)
        self._vetoed = {}                # choice -> the stage that vetoed it
        self._value = None               # [(choice, P(win))] from the value net
        self._mix = None                 # (eligible, pick, suppressed)
        self._progress_open = False

        # --- game-long stats ---
        self.turns_decided = 0
        self.think_ms = []
        self.visits = 0
        self.det_fails = 0
        self.evals = []                  # (turn, engine eval of the chosen move)
        self.stage_fires = {}            # stage -> times it moved the front-runner
        self.vetoes = {}                 # stage -> choices it vetoed
        self.mix_fires = 0
        self.mix_suppressed = 0
        self.fallbacks = 0
        self.feed_errors = 0             # protocol payloads this renderer failed on
        self.predictions = [0, 0]        # [correct, resolved]
        self._pending_pred = None        # (turn, predicted opponent choice)
        self.hax = {"for": 0, "against": 0, "detail": {}}
        self.dealt = 0.0                 # their HP fractions removed by our direct moves
        self.taken = 0.0
        self.players = {}                # 'p1'/'p2' -> username (NO rating: see _ev_player)
        self.tera = {}                   # 'us'/'opp' -> (turn, species, tera type)

        # --- protocol parser state ---
        self.me = None                   # 'p1' / 'p2', from the |player| lines
        self.turn = 0
        self._species = {}               # 'p1a: Nick' -> species as the protocol spelled it
        self._hp = {}                    # 'p1a: Nick' -> last known hp fraction
        self._action = None              # the move currently being narrated
        self._post_faint = set()         # sides whose next switch is a forced replacement

    # --- output -----------------------------------------------------------------

    def c(self, name, text):
        if not self.color:
            return text
        return f"{_COLORS.get(name, '')}{text}{_COLORS['reset']}"

    def line(self, text=""):
        """Print one transcript line (and keep it for --save-log)."""
        self._close_progress()
        self.log.append(text)
        if not self.unicode:
            text = text.translate(_ASCII_FALLBACK)
        try:
            print(text, flush=True)
        except UnicodeEncodeError:
            # A console that can't take a player's name (or a glyph we missed) still gets
            # the line: losing a character beats losing the commentary.
            enc = getattr(sys.stdout, "encoding", None) or "ascii"
            print(text.encode(enc, "replace").decode(enc, "replace"), flush=True)

    def rule(self, label="", ch="─"):
        if not label:
            self.line(self.c("dim", ch * self.width))
            return
        left = f"{ch}{ch} {label} "
        self.line(self.c("dim", ch * 2) + f" {label} "
                  + self.c("dim", ch * max(0, self.width - _vislen(left))))

    def _progress(self, text):
        """Overwrite-in-place status line. The search costs 1-2s per turn, so world-by-world
        progress is what makes the wait legible; the next real line erases it."""
        if not sys.stdout.isatty():
            return
        self._progress_open = True
        if not self.unicode:
            text = text.translate(_ASCII_FALLBACK)
        try:
            print("\r" + _pad(text, self.width - 1), end="", flush=True)
        except UnicodeEncodeError:
            self._progress_open = False

    def _close_progress(self):
        if self._progress_open:
            self._progress_open = False
            print("\r" + " " * (self.width - 1) + "\r", end="", flush=True)

    def bar(self, frac, cells=12):
        """A [0,1] bar, coloured so a dying mon is visible at a glance."""
        frac = max(0.0, min(1.0, float(frac or 0.0)))
        filled = int(round(frac * cells))
        body = "█" * filled + "░" * (cells - filled)
        return self.c("good" if frac > 0.5 else "warn" if frac > 0.2 else "bad", body)

    def _stamp(self):
        return self.c("dim", _pad(f"T{self.turn}", 5))

    # --- observer protocol (EnginePlayer._emit) ---------------------------------

    def __call__(self, event, **data):
        handler = getattr(self, "_on_" + event, None)
        if handler is not None:
            handler(**data)

    def attach(self, agent):
        """Remember the agent. The transcript also shows inference state it keeps across
        turns (Scarf verdicts, moves seen since switch-in) and its raw world results."""
        self.agent = agent

    # --- the decision, rendered -------------------------------------------------

    def _on_turn_start(self, battle):
        self._t_turn = time.perf_counter()
        self._worlds, self._world_fails = [], 0
        self._stages, self._vetoed, self._pooled_order = [], {}, []
        self._value = self._mix = None
        if self.me is None:
            self.me = battle.player_role
        self.line()
        forced = self.c("warn", " (forced switch)") \
            if getattr(battle, "force_switch", False) else ""
        self.rule(self.c("hdr", f"Turn {battle.turn}") + forced
                  + "  " + self.c("dim", _mmss(time.time() - self.t_start)), ch="═")
        self._side_line("us", battle.active_pokemon, battle, mine=True)
        self._side_line("opp", battle.opponent_active_pokemon, battle, mine=False)
        field = self._field_line(battle)
        if field:
            self.line("  " + self.c("dim", "field    ") + field)

    def _side_line(self, label, mon, battle, mine):
        if mon is None:
            self.line(f"  {label:4s} -")
            return
        alive = (sum(1 for m in battle.team.values() if not m.fainted) if mine
                 else 6 - sum(1 for m in battle.opponent_team.values() if m.fainted))
        frac = mon.current_hp_fraction or 0.0
        bits = []
        if mon.status is not None:
            bits.append(self.c("bad", mon.status.name.lower()))
        boosts = " ".join(f"{k}{v:+d}" for k, v in (mon.boosts or {}).items() if v)
        if boosts:
            bits.append(self.c("warn", boosts))
        if getattr(mon, "is_terastallized", False):
            bits.append(self.c("warn", "TERA"))
        if mon.ability:
            bits.append(str(mon.ability))
        # poke-env carries an unrevealed item as the string 'unknown_item', not None.
        bits.append(self.c("dim", "?item") if not mon.item
                    or mon.item == GenData.UNKNOWN_ITEM else str(mon.item))
        name = _species_name(mon.species)[:18]
        self.line(f"  {self.c('dim', _pad(label, 4))} "
                  f"{self.c('us' if mine else 'opp', _pad(name, 19))}"
                  f"{frac:4.0%} {self.bar(frac)} {alive}/6  "
                  + self.c("dim", " · ").join(bits))

    #: poke-env stores a LAYER COUNT for these and the START TURN for everything else, so
    #: only these may be rendered as 'xN'. The adapter's _side_conditions splits them the
    #: same way. Printing 'stealth_rock x5' for rocks set on turn 5 would be a lie.
    _LAYERED = ("SPIKES", "TOXIC_SPIKES")

    def _field_line(self, battle):
        parts = [w.name.lower() for w in (battle.weather or {})]
        parts += [f.name.lower() for f in (battle.fields or {})]
        for label, conds in (("ours", battle.side_conditions),
                             ("theirs", battle.opponent_side_conditions)):
            for sc, val in (conds or {}).items():
                layers = f" x{val}" if sc.name in self._LAYERED and (val or 0) > 1 else ""
                parts.append(f"{sc.name.lower()}{layers} {label}")
        return self.c("dim", " · ").join(parts)

    def _on_search_start(self, worlds, time_ms, threads, parallel):
        self._budget = (worlds, time_ms, threads)
        self._progress(f"  determinizing and searching {worlds} worlds "
                       f"x {time_ms}ms x {threads} threads ...")

    def _on_world(self, index, total, state, res, ms):
        best = max(res.side_one, key=lambda o: o.visits, default=None)
        reply = max((o for o in res.side_two if o.move_choice != "No Move"),
                    key=lambda o: o.visits, default=None)
        # total_score/visits is the engine's own mean outcome for a root move, i.e. its win
        # estimate for us. The parallel-worlds path drops it (results cross a process
        # boundary as plain tuples), hence the getattr.
        score = getattr(best, "total_score", None) if best else None
        self._worlds.append({
            "visits": res.total_visits,
            # `score is not None`, not `score`: a total_score of exactly 0.0 is a real
            # reading -- every rollout of the best root move lost -- and testing it for
            # truth dropped the eval on precisely the turns worth looking at.
            "eval": (score / best.visits) if score is not None and best.visits else None,
            "winner": best.move_choice if best else None,
            "reply": reply.move_choice if reply else None,
            "reply_share": (reply.visits / res.total_visits)
                           if reply and res.total_visits else 0.0,
            "set": self._sampled_set(state),
            "ms": ms,
        })
        done = len(self._worlds)
        ev = self._mean_eval()
        self._progress(f"  worlds [{'●' * done}{'○' * max(0, total - done)}] {done}/{total}"
                       f"  {_si(sum(w['visits'] for w in self._worlds))} visits"
                       + (f"  eval {ev:.2f}" if ev is not None else "")
                       + f"  {time.perf_counter() - self._t_turn:.1f}s")

    def _on_world_fail(self, index, total):
        self._world_fails += 1
        self.det_fails += 1

    @staticmethod
    def _sampled_set(state):
        """(item, ability, tera) this world guessed for the opponent's active.

        Determinization is how the bot copes with hidden information, so the SPREAD of
        these across worlds is usually the most informative thing in the turn."""
        try:
            side = state.side_two
            mon = side.pokemon[int(str(side.active_index))]
            return (str(mon.item or "none").lower(), str(mon.ability or "none").lower(),
                    str(mon.tera_type or "-").lower())
        except Exception:
            return None

    def _mean_eval(self):
        evs = [w["eval"] for w in self._worlds if w["eval"] is not None]
        return sum(evs) / len(evs) if evs else None

    def _on_pooled(self, battle, ranked, worlds):
        self._pooled_order = [c for c, _s in ranked]

    def _on_stage(self, name, ranked, changed, was, vetoed):
        self._stages.append((name, changed, was, ranked[0][0] if ranked else None))
        for choice in vetoed:
            self._vetoed.setdefault(choice, name)
        if changed:
            self.stage_fires[name] = self.stage_fires.get(name, 0) + 1

    def _on_value(self, valued, margin, free_switch, branches):
        self._value = valued

    def _on_mix(self, eligible, pick, suppressed):
        self._mix = (eligible, pick, suppressed)
        if suppressed:
            self.mix_suppressed += 1

    def _on_decision(self, battle, choice, ranked, mixed, reorders, fallback, ms):
        self._close_progress()
        self.turns_decided += 1
        self.think_ms.append(ms)
        self.visits += sum(w["visits"] for w in self._worlds)
        n_worlds, ms_world, threads = self._budget
        ok = len(self._worlds)
        ev = self._mean_eval()

        self.line("  " + self.c("dim",
                  f"search   {ok}/{n_worlds} worlds x {ms_world}ms x {threads}t  ·  "
                  f"{_si(sum(w['visits'] for w in self._worlds))} visits  ·  {ms / 1000:.1f}s"
                  + (f"  ·  {self._world_fails} failed" if self._world_fails else "")))
        self._candidate_table(ranked, choice, ok)
        self._pipeline_lines(mixed)
        self._sampling_line()
        self._inference_line(battle)

        if fallback:
            self.fallbacks += 1
            self.line("  " + self.c("bad", f">> plays {choice} -- no usable search result"))
            return
        reply, share = self._expected_reply()
        tail = f"[{ms / 1000:.1f}s" + (f", eval {ev:.2f}" if ev is not None else "") + "]"
        expect = ""
        if reply:
            expect = ("   " + self.c("dim", "expects ")
                      + self.c("opp", _pretty(reply)) + self.c("dim", f" {share:.0%}"))
            if reply != "No Move":
                self._pending_pred = (battle.turn, reply)
        self.line("  " + self.c("good", ">> plays ") + self.c("bold", _pretty(choice))
                  + "  " + self.c("dim", tail) + expect)
        if ev is not None:
            self.evals.append((battle.turn, ev))

    def _candidate_table(self, ranked, choice, ok):
        if not ranked:
            return
        shares, wins = self._pooled_shares()
        evals = self._choice_evals()
        pooled_rank = {c: i for i, c in enumerate(self._pooled_order)}
        self.line("  " + self.c("dim", f"   {_pad('candidate', 27)}{_pad('share', 15)}"
                                       f"{_pad('worlds', 8)}eval"))
        for i, (ch, _score) in enumerate(ranked[:self.N_CANDIDATES]):
            share = shares.get(ch, 0.0)
            ce = evals.get(ch)
            mark = self.c("good", "->") if ch == choice else "  "
            label = _pretty(ch)
            if len(label) > 26:           # 'switch Polteageist-Antique' and friends
                label = label[:25] + "."
            name = _pad(label, 27)
            moved = pooled_rank.get(ch, i) - i
            note = self.c("warn", f" ^{moved}") if moved > 0 else ""
            veto = self._vetoed.get(ch)
            if veto:
                note = self.c("bad", f" vetoed:{veto}")
            self.line(f"  {mark} {name}{self.bar(min(share * 2, 1.0), 8)} {share:5.2f}  "
                      f"{_pad(f'{wins.get(ch, 0)}/{ok}', 8)}"
                      + (f"{ce:.2f}" if ce is not None else "   -") + note)

    def _pipeline_lines(self, mixed):
        """Which guard/reranker changed its mind, and how the mixed root resolved."""
        value_fired = False
        for name, changed, was, now in self._stages:
            if not changed:
                continue
            extra = ""
            if name == "value":
                value_fired = True
                vals = dict(self._value or [])
                if now in vals and was in vals:
                    extra = self.c("dim", f"   (net {vals[now]:.2f} vs {vals[was]:.2f})")
            self.line("  " + self.c("warn", _pad(name, 9))
                      + f"{_pretty(was)} -> {self.c('bold', _pretty(now))}{extra}")
        for ch, stage in self._vetoed.items():
            self.vetoes.setdefault(stage, []).append(ch)
        if self._value and not value_fired:
            self.line("  " + self.c("dim", _pad("value", 9) + "net agrees: " + ", ".join(
                f"{_pretty(c)} {v:.2f}" for c, v in self._value[:3])))
        if self._mix:
            elig, pick, suppressed = self._mix
            if suppressed:
                self.line("  " + self.c("dim", _pad("mix", 9) +
                          f"suppressed: {len(elig)} candidates within "
                          f"{getattr(self.agent, 'mix_collapse_eps', 0.03)} of each other "
                          f"-- a flat policy is noise, not a strategy"))
            elif pick is not None and mixed:
                self.mix_fires += 1
                self.line("  " + self.c("warn", _pad("mix", 9))
                          + f"sampled {_pretty(pick)} from {len(elig)} near-tied  "
                          + self.c("dim", "(a deterministic argmax is exploitable)"))

    def _pooled_shares(self):
        """Pooled visit share and world-win count per candidate, recomputed from the raw
        world results.

        Deliberately independent of whichever score encoding the search is using (robust
        vote vs plain averaging), so the table reads the same under either."""
        worlds = list(getattr(self.agent, "_worlds", []) or [])
        shares, wins, n = {}, {}, len(worlds) or 1
        for _state, res in worlds:
            total = res.total_visits or 1
            for opt in res.side_one:
                shares[opt.move_choice] = shares.get(opt.move_choice, 0.0) + opt.visits / total
            best = max(res.side_one, key=lambda o: o.visits, default=None)
            if best is not None:
                wins[best.move_choice] = wins.get(best.move_choice, 0) + 1
        return {k: v / n for k, v in shares.items()}, wins

    def _choice_evals(self):
        """Visit-weighted mean of the engine's own score per root move, across worlds."""
        num, den = {}, {}
        for _state, res in list(getattr(self.agent, "_worlds", []) or []):
            for opt in res.side_one:
                score = getattr(opt, "total_score", None)
                if score is None or not opt.visits:
                    continue    # 0.0 is a score, not a missing one -- see _on_world
                num[opt.move_choice] = num.get(opt.move_choice, 0.0) + score
                den[opt.move_choice] = den.get(opt.move_choice, 0) + opt.visits
        return {k: num[k] / den[k] for k in num if den.get(k)}

    def _expected_reply(self):
        """The opponent move the worlds spent most of their search on -- our predicted
        reply -- and how much of the opponent's visit mass it held."""
        tally = {}
        for w in self._worlds:
            if w["reply"]:
                tally[w["reply"]] = tally.get(w["reply"], 0.0) + w["reply_share"]
        if not tally:
            return None, 0.0
        best = max(tally, key=tally.get)
        return best, tally[best] / max(len(self._worlds), 1)

    def _sampling_line(self):
        """What the determinizer guessed about the opponent's active, tallied over worlds."""
        sets = [w["set"] for w in self._worlds if w["set"]]
        if not sets or not self.show_worlds:
            return

        def tally(i):
            counts = {}
            for s in sets:
                counts[s[i]] = counts.get(s[i], 0) + 1
            return " ".join(f"{k} x{v}" for k, v in
                            sorted(counts.items(), key=lambda kv: -kv[1])[:3])

        self.line("  " + self.c("dim", _pad("guessed", 9) + f"item {tally(0)}  ·  "
                                f"ability {tally(1)}  ·  tera {tally(2)}"))

    def _inference_line(self, battle):
        """Hard inferences the adapter feeds the search: Scarf verdicts read off turn order,
        and what this opponent has shown since switching in (Choice-lock evidence)."""
        if self.agent is None:
            return
        bits = []
        opp = battle.opponent_active_pokemon
        hints = self.agent._opp_speed.get(battle.battle_tag, {})
        if opp is not None and hints.get(to_id_str(opp.species)):
            bits.append(self.c("warn", f"speed {hints[to_id_str(opp.species)]}"))
        used = (self.agent._opp_tracker.get(battle.battle_tag, {}) or {}).get("used") or set()
        if used:
            bits.append("used since switch-in: "
                        + ", ".join(_move_name(m) for m in sorted(used)))
        if bits:
            self.line("  " + self.c("dim", _pad("inferred", 9))
                      + self.c("dim", " · ").join(bits))

    # --- protocol feed (the battle itself) --------------------------------------

    def feed(self, split_messages):
        """Narrate one protocol payload. Called BEFORE poke-env processes it, so the events
        of the turn that just resolved print above the decision they lead into."""
        try:
            for msg in split_messages:
                if len(msg) > 1:
                    self._event(msg)
        except Exception:
            # Commentary must never break the client loop. But silent swallowing hides
            # renderer bugs -- it hid a Unicode crash during development -- so count it and
            # report in summary() instead.
            self.feed_errors += 1

    _HANDLERS = {
        "player": "_ev_player", "turn": "_ev_turn", "move": "_ev_move",
        "switch": "_ev_switch", "drag": "_ev_switch", "replace": "_ev_switch",
        "-damage": "_ev_hp", "-heal": "_ev_hp", "faint": "_ev_faint",
        "-crit": "_ev_flag", "-supereffective": "_ev_flag", "-resisted": "_ev_flag",
        "-immune": "_ev_flag", "-fail": "_ev_flag", "-miss": "_ev_miss",
        "-status": "_ev_status", "-boost": "_ev_boost", "-unboost": "_ev_boost",
        "-terastallize": "_ev_tera", "cant": "_ev_cant", "-weather": "_ev_field",
        "-fieldstart": "_ev_field", "-sidestart": "_ev_field", "-item": "_ev_reveal",
        "-ability": "_ev_reveal", "win": "_ev_end", "tie": "_ev_end",
    }

    def _event(self, p):
        name = self._HANDLERS.get(p[1])
        if name:
            getattr(self, name)(p)

    # -- identity helpers --

    def _side_of(self, ident):
        return ident[:2] if ident[:2] in ("p1", "p2") else None

    def _mine(self, ident):
        return self._side_of(ident) == self.me

    def _name(self, ident):
        species = self._species.get(ident) or ident.split(":", 1)[-1].strip()
        return self.c("us" if self._mine(ident) else "opp", _species_name(species))

    @staticmethod
    def _frac(token):
        """'185/246' / '63/100 tox' / '0 fnt' -> hp fraction, or None if unparseable."""
        token = (token or "").strip()
        if token.startswith("0 fnt"):
            return 0.0
        head = token.split(" ")[0]
        if "/" not in head:
            return None
        cur, _, mx = head.partition("/")
        try:
            cur, mx = int(cur), int(mx)
        except ValueError:
            return None
        return cur / mx if mx else None

    # -- events --

    def _ev_player(self, p):
        if len(p) < 4 or not p[3]:
            return
        # p[5] is the player's ladder rating. Deliberately dropped rather than stored:
        # the transcript never reports Elo -- ours or theirs -- so there is nothing to
        # keep it for, and an unstored value cannot leak into a later render.
        self.players[p[2]] = p[3]
        if to_id_str(p[3]) == self.username:
            self.me = p[2]

    def _ev_turn(self, p):
        self._flush()
        # An unresolved prediction means the opponent did nothing we could score it against
        # (e.g. we were taking a free post-faint switch); drop it rather than mis-credit it.
        self._pending_pred = None
        try:
            self.turn = int(p[2])
        except ValueError:
            pass

    def _ev_switch(self, p):
        self._flush()
        ident = p[2]
        if len(p) > 3:
            self._species[ident] = p[3].split(",")[0].strip()
        frac = self._frac(p[-1])
        if frac is None and len(p) > 4:
            frac = self._frac(p[4])
        self._hp[ident] = frac if frac is not None else 1.0
        side = self._side_of(ident)
        forced = side in self._post_faint
        self._post_faint.discard(side)
        if self._pending_pred and not self._mine(ident) and not forced:
            self._resolve_prediction("switch")
        verb = {"switch": "sends in", "drag": "is dragged in",
                "replace": "was really"}[p[1]]
        tail = self.c("warn", "   (Illusion broken)") if p[1] == "replace" else ""
        self.line(f"   {self._stamp()} {'us ' if self._mine(ident) else 'opp'} {verb} "
                  f"{self._name(ident)} {self.c('dim', f'{self._hp[ident]:.0%}')}{tail}")

    def _ev_move(self, p):
        if len(p) > 4 and any("[from]" in x for x in p[4:]):
            return          # a called move (Sleep Talk / Copycat): the parent line is enough
        self._flush()
        self._action = {"ident": p[2], "move": p[3] if len(p) > 3 else "?", "flags": [],
                        "target": None, "before": None, "after": None}
        if self._pending_pred and not self._mine(p[2]):
            self._resolve_prediction(to_id_str(self._action["move"]))

    def _ev_hp(self, p):
        ident = p[2]
        frac = self._frac(p[3] if len(p) > 3 else "")
        if frac is None:
            return
        before = self._hp.get(ident)
        self._hp[ident] = frac
        passive = any(x.startswith("[from]") for x in p[4:])
        act = self._action
        # Direct damage from the move being narrated: fold it into that one line.
        if (act is not None and not passive and p[1] == "-damage"
                and ident != act["ident"]):
            act["target"], act["before"], act["after"] = ident, before, frac
            if before is not None:
                if self._mine(act["ident"]):
                    self.dealt += max(0.0, before - frac)
                else:
                    self.taken += max(0.0, before - frac)
            return
        if before is None or abs(before - frac) < 0.005:
            return
        source = next((x.split("]", 1)[-1].strip() for x in p[4:]
                       if x.startswith("[from]")), "")
        self._flush()
        self.line(f"   {self._stamp()} {' ' * 4}{self._name(ident)} "
                  f"{'heals' if frac > before else 'loses'} {abs(frac - before):.0%} "
                  + self.c("dim", f"-> {frac:.0%}" + (f"  {source}" if source else "")))

    def _ev_flag(self, p):
        label = {"-crit": "CRIT", "-supereffective": "super effective",
                 "-resisted": "resisted", "-immune": "IMMUNE", "-fail": "failed"}[p[1]]
        if p[1] == "-crit" and len(p) > 2:
            self._hax("crit", p[2])
        if self._action is not None:
            self._action["flags"].append(label)

    def _ev_miss(self, p):
        if self._action is not None:
            self._action["flags"].append("MISSED")
        if len(p) > 2:
            self._hax("miss", p[2])          # the attacker lost the turn

    def _ev_cant(self, p):
        self._flush()
        reason = p[3] if len(p) > 3 else "?"
        self.line(f"   {self._stamp()} {' ' * 4}{self._name(p[2])} "
                  + self.c("bad", f"cannot move ({reason})"))
        if reason in ("par", "flinch", "frz"):
            self._hax(reason, p[2])

    def _ev_status(self, p):
        if len(p) < 4:
            return
        if self._action is not None:
            self._action["flags"].append(p[3])
            return
        self._flush()
        self.line(f"   {self._stamp()} {' ' * 4}{self._name(p[2])} is "
                  + self.c("bad", p[3]))

    def _ev_boost(self, p):
        if self._action is None or len(p) < 5:
            return
        sign = "+" if p[1] == "-boost" else "-"
        self._action["flags"].append(f"{p[3]} {sign}{p[4]}")

    def _ev_tera(self, p):
        self._flush()
        ttype = p[3] if len(p) > 3 else "?"
        self.tera["us" if self._mine(p[2]) else "opp"] = (
            self.turn, self._species.get(p[2], p[2]), ttype)
        self.line(f"   {self._stamp()} {' ' * 4}{self._name(p[2])} "
                  + self.c("warn", f"TERASTALLIZED {ttype}"))

    def _ev_field(self, p):
        if p[1] == "-weather":
            # Showdown re-sends the standing weather every turn as '|-weather|Sun|[upkeep]',
            # which rendered as a bare 'sunnyday' line on every single turn of a weather
            # game; and the line that ENDS weather is '|-weather|none', which rendered as
            # the word 'none'. Only real changes are worth a line.
            if any(str(x) == "[upkeep]" for x in p[3:]):
                return
            self._flush()
            what = "weather ended" if to_id_str(p[2]) == "none" else p[2]
            self.line(f"   {self._stamp()} {' ' * 4}" + self.c("dim", what))
            return
        self._flush()
        what = (p[3] if p[1] == "-sidestart" and len(p) > 3 else p[2]).replace("move: ", "")
        where = ""
        if p[1] == "-sidestart":
            # The side named on a -sidestart line is the side the hazard LANDS on.
            where = " on our side" if self._side_of(p[2]) == self.me else " on their side"
        self.line(f"   {self._stamp()} {' ' * 4}" + self.c("dim", f"{what}{where}"))

    def _ev_reveal(self, p):
        if len(p) < 4 or self._mine(p[2]):
            return          # our own item/ability is never news
        self._flush()
        label = "item" if p[1] == "-item" else "ability"
        self.line(f"   {self._stamp()} {' ' * 4}{self._name(p[2])} "
                  + self.c("warn", f"{label} revealed: {p[3]}"))

    def _ev_faint(self, p):
        self._flush()
        self._hp[p[2]] = 0.0
        self._post_faint.add(self._side_of(p[2]))
        self.line(f"   {self._stamp()} {' ' * 4}{self._name(p[2])} "
                  + self.c("bad", "✕ fainted"))

    def _ev_end(self, p):
        self._flush()

    def _flush(self):
        """Print the buffered move as one line: actor, move, target, HP swing, flags."""
        act, self._action = self._action, None
        if act is None:
            return
        text = (f"   {self._stamp()} {'us ' if self._mine(act['ident']) else 'opp'} "
                f"{self._name(act['ident'])} used {self.c('bold', act['move'])}")
        if act["target"] is not None and act["before"] is not None:
            text += (f" {self.c('dim', '->')} {self._name(act['target'])} "
                     f"{act['before']:.0%} {self.c('dim', '->')} {act['after']:.0%}")
        if act["flags"]:
            loud = any(f in ("CRIT", "MISSED", "IMMUNE", "failed") for f in act["flags"])
            text += "  " + self.c("bad" if loud else "dim", "(" + ", ".join(act["flags"]) + ")")
        self.line(text)

    def _hax(self, kind, hurt_ident):
        """Luck ledger, same convention as mine_losses.hax_events: a crit hurts its target,
        a miss / full-para / flinch hurts the mon that lost its turn -- in both cases the
        mon named on the line. Sleep excluded (Rest makes it strategy, not luck)."""
        key = "against" if self._mine(hurt_ident) else "for"
        self.hax[key] += 1
        tag = f"{kind} {key}"
        self.hax["detail"][tag] = self.hax["detail"].get(tag, 0) + 1

    def _resolve_prediction(self, actual):
        """Score the search's guess at the opponent's reply. Approximate by construction:
        one prediction per decision, resolved against the first thing that opponent does."""
        _turn, predicted = self._pending_pred
        self._pending_pred = None
        self.predictions[1] += 1
        pred = predicted[:-5] if predicted.endswith("-tera") else predicted
        hit = pred.startswith("switch") if actual == "switch" else pred == actual
        if hit:
            self.predictions[0] += 1

    # --- end-of-game report -----------------------------------------------------

    def summary(self, battle=None, result=None, replay_path=None, diag=None):
        """The post-game block: what the search cost, what the pipeline did, how well we
        read the opponent, the luck ledger, and where the game actually turned."""
        self.line()
        self.rule(self.c("hdr", "Game analysis"), ch="═")
        wall = time.time() - self.t_start
        turns = battle.turn if battle is not None else self.turn
        verdict = {"won": self.c("good", "WON"), "lost": self.c("bad", "LOST"),
                   "tie": self.c("warn", "TIE")}.get(result, result or "unfinished")
        who = "  vs  ".join(
            f"{'us' if side == self.me else 'opp'} {name}"
            for side, name in sorted(self.players.items()))
        self.line(f"  result      {verdict} in {turns} turns  ·  {who}")

        search_s = sum(self.think_ms) / 1000.0
        self.line(f"  clock       {_mmss(wall)} wall  ·  {_mmss(search_s)} thinking "
                  f"({search_s / max(wall, 1):.0%})")
        if self.think_ms:
            self.line(f"  per turn    {sum(self.think_ms) / len(self.think_ms) / 1000:.2f}s avg"
                      f"  ·  {max(self.think_ms) / 1000:.2f}s max  ·  "
                      f"{_si(self.visits / max(self.turns_decided, 1))} visits/turn")
        self.line(f"  search      {self.turns_decided} decisions  ·  {_si(self.visits)} total "
                  f"MCTS visits  ·  {self.det_fails} world failure(s)"
                  + (f"  ·  {self.c('bad', str(self.fallbacks) + ' fallback(s)')}"
                     if self.fallbacks else ""))

        fires = ", ".join(f"{k} {v}" for k, v in sorted(self.stage_fires.items())) or "none"
        self.line(f"  overrides   {fires}"
                  + (f"  ·  mix sampled {self.mix_fires}" if self.mix_fires else "")
                  + (f"  ·  mix suppressed {self.mix_suppressed}"
                     if self.mix_suppressed else ""))
        for stage, choices in sorted(self.vetoes.items()):
            uniq = sorted(set(choices))
            self.line(f"  vetoed      {stage}: "
                      + ", ".join(_pretty(c) for c in uniq[:6])
                      + (f" (+{len(uniq) - 6} more)" if len(uniq) > 6 else ""))

        correct, resolved = self.predictions
        if resolved:
            self.line(f"  opp read    called their move {correct}/{resolved} "
                      f"({correct / resolved:.0%})  "
                      + self.c("dim", "(the search's top predicted reply)"))
        f, a = self.hax["for"], self.hax["against"]
        if f or a:
            detail = ", ".join(f"{k} {v}" for k, v in sorted(self.hax["detail"].items()))
            self.line(f"  luck        {f} for / {a} against  (net {f - a:+d})  "
                      + self.c("dim", detail))
        # A SUM of per-Pokemon HP fractions, so its natural unit is "Pokemon-worth of HP"
        # and it is not bounded by the size of a team -- anything healed has to be chewed
        # through twice. Printed as a percentage under the label "of their team" it read
        # 'dealt 577% of their team', which is not a thing.
        self.line(f"  damage      dealt {self.dealt:.2f} Pokemon-worth of their HP  ·  "
                  f"taken {self.taken:.2f} of ours")
        for who_, (turn, species, ttype) in sorted(self.tera.items()):
            self.line(f"  tera        {who_}: {_species_name(species)} -> {ttype} on T{turn}")

        if self.evals:
            self.line(f"  eval curve  {self.evals[0][1]:.2f} {self._sparkline()} "
                      f"{self.evals[-1][1]:.2f}  "
                      + self.c("dim", "(the search's own win estimate, per decision)"))
            for turn, delta in self._swings():
                self.line("  " + self.c("warn", f"swing       T{turn}: {delta:+.2f}  ")
                          + self.c("dim", "-- worth replaying"))
        if battle is not None:
            revealed = [_species_name(m.species) for m in battle.opponent_team.values()]
            self.line(f"  their team  {', '.join(revealed)}  ({len(revealed)}/6 revealed)")
        if self.feed_errors:
            self.line("  " + self.c("bad", f"renderer    {self.feed_errors} protocol "
                                           f"payload(s) failed to render -- a bug in the "
                                           f"commentary, not in the bot"))
        if replay_path:
            self.line(f"  replay      {replay_path}")
        if diag:
            self.line(f"  diag        {diag}")
        self.rule(ch="═")

    def _sparkline(self):
        glyphs = "▁▂▃▄▅▆▇▉"        # ASCII-mapped in line(); see _ASCII_FALLBACK
        return "".join(glyphs[min(len(glyphs) - 1, max(0, int(v * len(glyphs))))]
                       for _t, v in self.evals)

    def _swings(self, n=2, floor=0.12):
        """The decisions where the search's own estimate moved most -- where to look first
        in the replay. Descriptive: a swing is as often their good play as our bad one.

        Ranked by MAGNITUDE. Sorting the signed delta ascending, which is what this did,
        returns the most negative two -- and in a game whose qualifying swings all happen to
        be upward it returns the two SMALLEST of them, i.e. the opposite of what the label
        promises."""
        deltas = [(self.evals[i][0], self.evals[i][1] - self.evals[i - 1][1])
                  for i in range(1, len(self.evals))]
        return sorted((d for d in deltas if abs(d[1]) >= floor),
                      key=lambda td: -abs(td[1]))[:n]

    def save(self, path):
        """Write the transcript (colour codes stripped) next to the replay."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_ANSI_RE.sub("", l) for l in self.log) + "\n")


class AnalyzedPlayer(EnginePlayer):
    """EnginePlayer that narrates the protocol as it arrives.

    The feed runs BEFORE super() handles the payload on purpose: super() parses the turn's
    messages and, on the |request|, calls choose_move -- so narrating first keeps "here is
    what happened" above "here is what I decided next". Parsing the raw lines instead of the
    updated battle object is what makes that ordering possible."""

    def __init__(self, *args, analyzer=None, min_move_s=0.0, **kwargs):
        super().__init__(*args, observer=analyzer, **kwargs)
        self.analyzer = analyzer
        self.min_move_s = max(0.0, float(min_move_s or 0.0))
        if analyzer is not None:
            analyzer.attach(self)

    def choose_move(self, battle):
        """Decide, then hold the move back until the turn has taken min_move_s.

        The search answers in ~2s, which is faster than Showdown animates the turn that
        just resolved -- so the analysis for turn N+1 lands on top of the animation for
        turn N and there is no time to read it. A floor makes the transcript watchable.

        poke-env accepts an Awaitable from choose_move (player.py: `if isinstance(choice,
        Awaitable): choice = await choice`) and awaits it on its own POKE_LOOP, so this is
        an asyncio.sleep on that loop, NOT a blocking one: the websocket keeps answering
        pings and the timer keeps ticking down normally. A time.sleep here would stall the
        connection instead.

        The pause is AFTER the decision, so the analysis prints immediately and you read it
        while the pad runs. The '[2.0s]' in the transcript and the post-game search-cost
        block stay the real think time -- padding those would make the analysis lie.

        Off (0.0) unless an entry point asks for it, so this costs nothing to the players
        that subclass AnalyzedPlayer without wanting the pacing."""
        t0 = time.perf_counter()
        order = super().choose_move(battle)
        remaining = self.min_move_s - (time.perf_counter() - t0)
        if remaining <= 0:
            return order
        if self.analyzer is not None:
            self.analyzer.line("  " + self.analyzer.c(
                "dim", f"pacing   +{remaining:.1f}s idle to the "
                       f"{self.min_move_s:g}s per-move floor"))

        async def _paced():
            await asyncio.sleep(remaining)
            return order

        return _paced()

    async def _handle_battle_message(self, split_messages):
        if self.analyzer is not None:
            self.analyzer.feed(split_messages)
        await super()._handle_battle_message(split_messages)
