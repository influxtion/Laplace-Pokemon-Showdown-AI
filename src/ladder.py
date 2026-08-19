r"""Play the bot on the real Showdown ladder against humans.

Unlike everything else here this connects to the OFFICIAL server, not localhost, so the
local server does NOT need to be running.

Setup (once):
  1. Register the bot's username at https://play.pokemonshowdown.com (gear -> Register).
  2. Credentials come from a gitignored .env at the project root, or SHOWDOWN_USERNAME /
     SHOWDOWN_PASSWORD, or --username/--password:
         SHOWDOWN_USERNAME=YourBotName
         SHOWDOWN_PASSWORD=your-password

From the project root:
    python -u src\ladder.py                       # 10 ranked games
    python -u src\ladder.py --battles 25
    python -u src\ladder.py --mode accept         # wait for a challenge
    python -u src\ladder.py --mode challenge --opponent SomeUser

Ladder games are rated, so GXE/Elo shows on the account profile -- the honest test the
offline benchmarks can't give. Follow Showdown's rules for automated play: keep the account
present, don't flood the ladder, don't farm a specific person.

Every game is archived to replays/ladder/ as <result>-<tag>.html plus a .trace.json of the
search's per-turn reasoning. Covers Ctrl-C and crashes too -- the battle room is gone once
the process exits, so anything unwritten is lost.

Those are local and always written. Hosted, shareable replays on
replay.pokemonshowdown.com are OFF by default and opt-in per run:

    python -u src\ladder.py --upload-first        # first 5 games, PUBLIC links
    python -u src\ladder.py --upload-first 20     # first 20

A dropped websocket does NOT end the run: the connection watchdog notices, the segment is
torn down (replays swept, ratings logged) and the remaining games are played on a fresh
connection. --reconnects 0 turns that off. See watch_connection / play_segment.

Every finished game's post-game Elo is checked against the standing record in
replays/ladder/elo_high.json. A new high is announced as it happens, its replay is kept a
second time under replays/ladder/records/, and the run STOPS there -- the queue is cancelled
rather than handing a fresh peak straight back. --no-stop-on-record keeps laddering.
See check_elo_record / watch_stop.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --fork: Phase-2 engine build -- value net AT THE MCTS LEAF (blend 0.5 with the stock
# eval, UCT c2=0.1) at 600ms/world. Cross-engine validated 43/60 = 71.7% vs the stock 120ms
# build. The fork wheel lives in poke-engine-fork/py and is shadowed onto sys.path; the
# venv's stock poke-engine is untouched, so omitting the flag is a complete rollback.
#
# MUST run before engine_search imports poke_engine, hence the sys.argv peek rather than
# argparse. The LAPLACE_* vars are read once by the Rust engine on first search; without
# LAPLACE_VALUE_NET the fork is bit-identical to stock. setdefault so an explicit env
# override wins.
FORK_MODE = "--fork" in sys.argv
if FORK_MODE:
    sys.path.insert(0, os.path.join(_ROOT, "poke-engine-fork", "py"))
    os.environ.setdefault("LAPLACE_VALUE_NET",
                          os.path.join(_ROOT, "poke-engine-fork", "value_net_v5.bin"))
    os.environ.setdefault("LAPLACE_VALUE_NET_BLEND", "0.5")
    os.environ.setdefault("LAPLACE_UCT_C2", "0.1")

from poke_env import AccountConfiguration, ShowdownServerConfiguration
# poke_env drives its websocket from POKE_LOOP, a loop running in its own thread; this is
# the hop every public Player method takes to get onto it. See request_upload().
from poke_env.concurrency import handle_threaded_coroutines
from poke_env.data import to_id_str

from engine_search import EnginePlayer

# Checks if fork actually loaded
if FORK_MODE:
    import poke_engine
    if not hasattr(poke_engine, "featurize_state"):
        sys.exit(f"--fork: loaded STOCK poke_engine from {poke_engine.__file__} -- "
                 f"is the fork wheel installed in poke-engine-fork/py?")
    if not os.path.exists(os.environ["LAPLACE_VALUE_NET"]):
        sys.exit(f"--fork: weight file missing: {os.environ['LAPLACE_VALUE_NET']}")

BATTLE_FORMAT = "gen9randombattle"
SPECTATE_URL = "https://play.pokemonshowdown.com/"   # + battle room id = watchable link

# Anchored to the project root, NOT cwd, so `python path\to\src\ladder.py` from anywhere
# still writes to the one replay archive.
REPLAY_DIR = os.path.join(_ROOT, "replays", "ladder")
RATING_LOG = os.path.join(REPLAY_DIR, "ratings.csv")

# Showdown-hosted (shareable) replays, uploaded via /savereplay. See request_upload().
# OFF unless --upload-first is passed: the links are public, so publishing is a deliberate
# act, not a side effect of laddering. UPLOAD_FIRST_N is what a bare --upload-first means.
REPLAY_UPLOAD_URL = "https://replay.pokemonshowdown.com/"
UPLOAD_FIRST_N = 5
SEND_TIMEOUT = 15.0         # how long a poll loop waits on one websocket send

# Auto-resume after a dropped connection (see watch_connection). MAX_RECONNECTS is the
# default for --reconnects and counts CONSECUTIVE failures: any segment that finishes a game
# resets it, so a long run can survive more than this many drops as long as it keeps making
# progress. The wait gives a restarting Showdown time to come back and keeps a hard-down
# network from spinning the reconnect loop.
MAX_RECONNECTS = 5
RECONNECT_WAIT = 15.0
CANCEL_GRACE = 5.0          # after /cancelsearch, how long a match already made can still land

# Peak-Elo tracking (see final_rating / check_elo_record). Every game that ends above the
# standing record archives a second copy of its replay here and moves the record on.
# ELO_HIGH_FILE is the running one and wins once written; ELO_HIGH_SEED is only the floor
# for the very first run -- the account's peak as of 2026-08-17, and the same number the
# README reports. replays/ is gitignored, so ELO_HIGH_FILE is machine-local: this constant
# is the only committed record of the peak, and is worth bumping when one is set.
ELO_DIR = os.path.join(REPLAY_DIR, "records")
ELO_HIGH_FILE = os.path.join(REPLAY_DIR, "elo_high.json")
ELO_HIGH_SEED = 2227

# .env lives in the project root, one level up from src/.
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def load_dotenv(path=_ENV_PATH):
    """Read KEY=value lines from .env into os.environ. Real env vars win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def build_agent(account, fork=False, cls=EnginePlayer, **extra):
    """The shipped Laplace config, pointed at the official server.

    THE single source of truth for ladder settings. `cls`/`extra` exist so another entry
    point can run the SHIPPED config with a different Player subclass or one extra kwarg
    (analyze_battle passes the live-analysis player) without forking this function -- a copy
    would silently drift."""
    # poke-engine needs no trained model; it searches the position directly. Ladder is one
    # game at a time, so we spend a generous per-turn budget: more determinizations and
    # threads than the throughput-tuned benchmark default, ~1-2s/turn, well inside the move
    # timer. record=True keeps per-turn traces, dumped next to the replay for mining.
    value_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "models", "value_net.pt")
    # Settings that were all decided the hard way:
    #
    # value_boost_margin OFF (was 26). Cohort-2 mining (11W-19L, 2018 -> 1848): with the
    # opponent boosted, the stale v3 net used the widened band to override the search toward
    # worse moves -- Freezing Glare over Hurricane x4 into a +6 Calm Mind sweep, every pick
    # inside the 0.26 margin. The net keeps its A/B-supported base tie-break authority
    # (value_margin=11). Re-widen only after a retrain earns it.
    #
    # Fork mode spends 600ms/world vs 150 stock: informed search converts time into strength
    # where the stock eval provably can't (budget falsified twice for stock; fork600 vs
    # stock120 = 71.7%/60). ~5s/turn at det8 sequential -- inside the timer, but with less
    # bank cushion in very long games.
    kwargs = dict(
        account_configuration=account,
        server_configuration=ShowdownServerConfiguration,
        battle_format=BATTLE_FORMAT,
        start_timer_on_battle_start=True,   # an AFK opponent can't stall us out
        n_determinizations=8, search_time_ms=600 if fork else 150, threads=8, record=True,
        value_model_path=value_path if os.path.exists(value_path) else None,
        value_boost_margin=0,
        robust_vote=False, mix_root=True, mix_frac=0.9,
    )
    kwargs.update(extra)
    return cls(**kwargs)


def replay_url(battle_tag):
    """The public URL /savereplay will publish this battle at.

    Derived, not guessed. Server side (rooms.ts getReplayData + replays.ts add): the replay
    id is the room id minus the leading 'battle-', and a password room's full id is
    id + '-' + password + 'pw', which reassembles to exactly the same string. So for both
    plain and '-...pw' rooms it's just the tag with 'battle-' stripped.

    The upload popup is the authoritative confirmation (see PopupWatcher); this is printed
    immediately so the link is in the log even if the popup is missed."""
    replay_id = battle_tag[len("battle-"):] if battle_tag.startswith("battle-") else battle_tag
    return REPLAY_UPLOAD_URL + replay_id


class PopupWatcher(logging.Handler):
    """Print the upload confirmation Showdown sends back after /savereplay.

    The server answers with a popup -- link or error -- and poke_env only logs it ('Popup
    message received'). Rather than patch poke_env's message loop we tap its logger. Worst
    case we miss a confirmation; the local .html archive is unaffected."""

    _URL = re.compile(r"https?://replay\.pokemonshowdown\.com/[^\s\"'<>]+")

    def __init__(self):
        super().__init__()
        self.seen = set()

    def emit(self, record):
        try:
            msg = record.getMessage()
            if "opup" not in msg:
                return
            for url in self._URL.findall(msg):
                url = url.rstrip("\\")
                if url not in self.seen:
                    self.seen.add(url)
                    print(f"    uploaded -> {url}", flush=True)
            if "could not be saved" in msg:
                print(f"    WARNING: Showdown refused the replay upload: {msg}", flush=True)
        except Exception:       # a logging handler must never break the client loop
            pass


async def send_to_showdown(agent, message, room="", timeout=SEND_TIMEOUT):
    """Send one command on poke_env's websocket from a task on the MAIN loop.

    MUST go through handle_threaded_coroutines. The websocket belongs to poke_env's
    POKE_LOOP, which runs in ITS OWN THREAD; our callers (report_progress, watch_stop,
    announce in analyze_battle) are tasks on the main asyncio.run loop. Awaiting
    ps_client.send_message directly therefore drove websocket.send from the wrong thread,
    writing into the SSL transport concurrently with POKE_LOOP's own writes. That corrupts
    sslproto's _write_backlog deque -- 'IndexError: deque index out of range' out of
    _do_write, then 'no close frame received or sent' as the connection dies and the whole
    run ends mid-battle. Hopping onto POKE_LOOP serialises this send with every other one.

    Bounded, because the callers are poll loops that must keep ticking: an await that never
    returns (a dying socket, ps_client's send lock held by a stuck write) would stop the
    monitor dead -- no start/finish lines and no per-battle replay saves for the rest of the
    run. SHIELDED, so the timeout only stops US waiting; cancelling a half-written websocket
    frame is exactly the transport corruption described above.

    room="" is how poke_env sends global commands (see its search_ladder_game)."""
    sending = asyncio.ensure_future(handle_threaded_coroutines(
        agent.ps_client.send_message(message, room=room), agent.ps_client.loop))
    await asyncio.wait_for(asyncio.shield(sending), timeout)


async def request_upload(agent, tag):
    """Ask Showdown to publish this battle as a hosted replay.

    Sent at battle START, not end, for two reasons found in the server/client source:
    poke_env sends '/leave' the moment a battle ends (player.py), so saving on the
    'finished' poll races the room teardown; and once a battle has been saved once, the
    server re-uploads the COMPLETE log automatically at game end (room-battle.ts), silently
    overwriting the partial. Saving early is both the safe order and the complete one."""
    try:
        await send_to_showdown(agent, "/savereplay", room=tag)
        print(f"    /savereplay sent -> {replay_url(tag)}", flush=True)
    except Exception as exc:
        # Never fatal: the local .html archive is written either way.
        print(f"    WARNING: /savereplay failed for {tag}: {exc!r}", flush=True)


async def cancel_search(agent):
    """Leave the matchmaking queue. Never fatal -- worst case we play one more game."""
    try:
        await send_to_showdown(agent, "/cancelsearch")
    except Exception as exc:
        print(f"    WARNING: /cancelsearch failed: {exc!r}", flush=True)


def result_label(battle):
    """won / lost / tie / unfinished, for the replay filename and the progress line.

    GOTCHA: battle.tied is a METHOD, so the obvious `getattr(battle, "tied", False)`
    returns a truthy bound method and mislabels every LOSS as a tie. Check `won is None`.
    battle.won is True / False / None."""
    if not battle.finished:
        return "unfinished"
    if battle.won:
        return "won"
    if battle.won is None:
        return "tie"
    return "lost"


def save_battle(agent, tag, battle, saved, warned):
    """Write <result>-<tag>.html (+ .trace.json) for one battle. True if it wrote.

    Saving EVERY game matters: the live battle room vanishes once the game ends, so an
    unsaved game is gone for good, and ties especially need a replay to diagnose (Endless
    Battle Clause / sim-KO).

    `saved` dedupes across the poll loop and the exit sweep. A failure is reported once per
    battle and left unsaved so a later pass retries -- the old code swallowed every
    exception, so a broken archive looked healthy."""
    if tag in saved:
        return False
    label = result_label(battle)
    try:
        os.makedirs(REPLAY_DIR, exist_ok=True)
        agent.save_replay(tag, os.path.join(REPLAY_DIR, f"{label}-{tag}.html"))
        # Per-turn decision traces (what the search considered), for mine_losses.
        traces = getattr(agent, "traces", {}).get(tag)
        if traces:
            with open(os.path.join(REPLAY_DIR, f"{label}-{tag}.trace.json"),
                      "w", encoding="utf-8") as f:
                json.dump(traces, f, indent=1)
    except Exception as exc:
        if tag not in warned:
            warned.add(tag)
            print(f"  WARNING: could not save replay for {tag}: {exc!r}", flush=True)
        return False
    saved.add(tag)
    return True


def sweep_replays(agent, saved, warned):
    """Save every battle the agent still holds that the poll loop didn't get to.

    Two gaps the loop can't cover: the last game of a run (agent.ladder() returns as soon as
    it ends and the monitor is cancelled before its next tick), and Ctrl-C / crash
    mid-session. In-progress battles are saved too -- a partial log is still worth reading,
    and the 'unfinished-' prefix keeps them out of the won/lost stats."""
    n = sum(save_battle(agent, tag, battle, saved, warned)
            for tag, battle in list(agent.battles.items()))
    if n:
        print(f"  Saved {n} replay(s) on exit -> {REPLAY_DIR}", flush=True)
    return n


def log_rating(tag, battle, logged):
    """Append this battle's rating row. True once written, False to retry next tick.

    GOTCHA 1 -- it is the rating BEFORE this game, whatever poke_env's docstring claims.
    Showdown sends "<user>'s rating: 2007 &rarr; <strong>2023</strong>" as a |raw| message
    and poke_env keeps rating_info[:4], the number LEFT of the arrow. So a row reads "went
    into this game at 2007 and got <result>", and the final game's update is not in this
    file -- read that one off the profile.

    GOTCHA 2 -- those |raw| lines are part of the end-of-battle burst but don't reliably
    land on the same poll tick as |win|, so rating is often still None the first time a
    finished battle is seen. Returning False keeps tag out of `logged` so the next tick
    retries -- which is why the caller must NOT gate this on `tag not in finished`."""
    if tag in logged or battle.rating is None:
        return False
    try:
        os.makedirs(REPLAY_DIR, exist_ok=True)
        new = not os.path.exists(RATING_LOG)
        with open(RATING_LOG, "a", encoding="utf-8", newline="") as f:
            if new:
                f.write("tag,engine,result,turns,rating_before,opp_rating_before\n")
            f.write(f"{tag},{'fork' if FORK_MODE else 'stock'},{result_label(battle)},"
                    f"{battle.turn},{battle.rating},{battle.opponent_rating or ''}\n")
    except Exception as exc:
        # Unlike save_battle this gives up on the row instead of leaving it for a retry: a
        # disk error will just recur, and one lost row in 60 costs less than a warning
        # printed every 3s for the rest of a 5-hour cohort.
        print(f"  WARNING: could not log rating for {tag}: {exc!r}", flush=True)
        logged.add(tag)
        return False
    logged.add(tag)
    return True


# |raw|influxobot's rating: 1844 &rarr; <strong>1863</strong><br />(+19 for winning)
# Greedy on the name: a username can contain "'s ", but the line can only carry one
# "'s rating: ", so the LAST match is always the real split point.
_RATING_LINE = re.compile(r"(.+)'s rating: (\d+) &rarr; <strong>(\d+)</strong>")


def final_rating(battle):
    """Our Elo AFTER this battle, or None until Showdown has reported it.

    NOT battle.rating -- that is the rating we went IN at. poke_env keeps rating_info[:4],
    the number to the LEFT of the arrow (abstract_battle.parse_message; see log_rating
    GOTCHA 1), so the post-game number exists nowhere in its public API. It is in the raw
    line itself though, and parse_message archives every message verbatim in _replay_data
    before interpreting it, so this re-reads it from there.

    Scanned newest-first and matched on username: Showdown sends one of these lines per
    player and the opponent's is identical in shape."""
    me = to_id_str(battle.player_username)
    for msg in reversed(getattr(battle, "_replay_data", None) or []):
        if len(msg) > 2 and msg[1] == "raw":
            hit = _RATING_LINE.search(msg[2])
            if hit and to_id_str(hit.group(1)) == me:
                return int(hit.group(3))
    return None


def load_elo_high():
    """(record Elo, the battle that set it). ELO_HIGH_SEED until the file exists.

    An unreadable file falls back to the seed but SAYS SO: silently starting from the seed
    when the real record is higher would let a worse game overwrite it."""
    if not os.path.exists(ELO_HIGH_FILE):
        return ELO_HIGH_SEED, None
    try:
        with open(ELO_HIGH_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return int(data["elo"]), data.get("tag")
    except Exception as exc:
        print(f"  WARNING: could not read {ELO_HIGH_FILE} ({exc!r}); "
              f"falling back to {ELO_HIGH_SEED}", flush=True)
        return ELO_HIGH_SEED, None


def check_elo_record(agent, tag, battle, run):
    """Handle one finished battle's post-game Elo. True once it has been checked.

    False while Showdown hasn't sent the rating yet, so the poll loop keeps retrying -- the
    |raw| lines trail |win| by a tick or two, exactly like log_rating. Unrated games never
    produce one and are simply never checked.

    A new record archives a SECOND copy of the replay under ELO_DIR (the normal
    won-/lost- copy is still written by save_battle) and moves the record on disk, so the
    peak survives across runs."""
    if not run.track_elo or tag in run.elo_checked:
        return False
    try:
        elo = final_rating(battle)
    except Exception as exc:
        # This runs inside the progress monitor. A surprise in the raw line must not take
        # the monitor down with it, and must not be retried every poll tick either.
        run.elo_checked.add(tag)
        print(f"    WARNING: could not read the post-game Elo for {tag}: {exc!r}", flush=True)
        return True
    if elo is None:
        return False
    run.elo_checked.add(tag)
    if elo <= run.elo_high:
        return True
    was = run.elo_high
    run.elo_high, run.elo_high_tag = elo, tag
    run.elo_records.append((elo, tag))
    print(f"    *** NEW ELO HIGH: {elo}  (previous best {was}) ***", flush=True)
    # End the run here rather than queue again and hand the new peak straight back.
    # watch_stop does the actual stopping -- we are already in the matchmaking queue by now.
    if run.stop_on_record and run.stop_reason is None:
        run.stop_reason = f"new Elo record {elo}"
    # Both writes are guarded separately: losing the archived replay must not cost us the
    # record, and vice versa.
    try:
        os.makedirs(ELO_DIR, exist_ok=True)
        path = os.path.join(ELO_DIR, f"elo{elo}-{tag}.html")
        agent.save_replay(tag, path)
        traces = getattr(agent, "traces", {}).get(tag)
        if traces:
            with open(os.path.join(ELO_DIR, f"elo{elo}-{tag}.trace.json"),
                      "w", encoding="utf-8") as f:
                json.dump(traces, f, indent=1)
        print(f"    archived -> {path}", flush=True)
    except Exception as exc:
        print(f"    WARNING: could not archive the record replay: {exc!r}", flush=True)
    try:
        with open(ELO_HIGH_FILE, "w", encoding="utf-8") as f:
            json.dump({"elo": elo, "tag": tag, "engine": "fork" if FORK_MODE else "stock",
                       "when": datetime.now().isoformat(timespec="seconds")}, f, indent=1)
    except Exception as exc:
        print(f"    WARNING: could not update {ELO_HIGH_FILE}: {exc!r}", flush=True)
    return True


class Run:
    """Everything about a run that has to outlive the agent playing it.

    A dropped websocket can't be repaired in place (watch_connection explains why), so the
    remaining games are played by a BRAND NEW agent -- new battles dict, counters back at
    zero. Game numbering, the W/L tally and the save/rating dedupe sets live here so a
    resumed run reads as one run instead of restarting at 'Battle 1/20'.

    `total` counts games played to a result. A game abandoned mid-battle to a disconnect is
    forfeited on Showdown's timer and its result never reaches us, so it is recorded in
    `abandoned` and replaced by another game rather than counted.

    track_elo=False for callers that must not touch the peak record -- ladder_old plays
    deliberately weakened builds, and a record set by one of those would be a lie.

    stop_reason, once set, ends the run at the next opportunity; watch_stop acts on it."""

    def __init__(self, total, upload_first=0, track_elo=True, stop_on_record=True):
        self.total = total
        self.upload_first = upload_first
        self.saved, self.warned, self.logged = set(), set(), set()
        self.done = self.won = self.lost = 0
        self.abandoned = []
        self.track_elo = track_elo
        self.elo_high, self.elo_high_tag = load_elo_high() if track_elo else (0, None)
        self.elo_checked = set()    # battles whose post-game Elo has been seen
        self.elo_records = []       # (elo, tag) per new high set this run, in order
        self.stop_on_record = stop_on_record
        self.stop_reason = None

    def absorb(self, agent):
        """Fold a retired agent's results into the totals. Once per agent, after it stops."""
        self.done += agent.n_finished_battles
        self.won += agent.n_won_battles
        self.lost += agent.n_lost_battles
        self.abandoned += [tag for tag, b in agent.battles.items() if not b.finished]


async def watch_connection(agent, poll=2.0):
    """Return once poke_env has stopped listening to the websocket -- i.e. the run is dead.

    poke_env's PSClient.listen() catches everything, logs it and RETURNS (ps_client.py); it
    never reconnects. So after a drop -- 'no close frame received or sent', a ConnectionReset,
    a Showdown restart -- nothing is reading the socket, no |request| ever arrives, and
    Player._ladder blocks forever on _battle_count_queue.join(). The process just sits there:
    the current game is forfeited on the timer, its replay is never written and the rest of
    the run never happens. This is what turns that silent hang into a clean segment end.

    The listener is an asyncio task on POKE_LOOP wrapped in a concurrent.futures.Future, so
    'has it stopped' is one .done() check; polling it keeps the watchdog off POKE_LOOP.
    listen() swallows the cause, so the reason is only ever in the logged traceback."""
    fut = getattr(agent.ps_client, "_listening_coroutine", None)
    if fut is None:
        # Unrecognised poke_env layout. Watch nothing rather than risk a false positive
        # tearing down a healthy run.
        print("  NOTE: connection watchdog off (no ps_client._listening_coroutine)", flush=True)
    while fut is None or not fut.done():
        await asyncio.sleep(poll)


async def watch_stop(agent, run, poll=1.0):
    """Complete once the run should end early, having left the ladder queue cleanly.

    Set by check_elo_record: a game that breaks the Elo record ends the run rather than
    putting the new peak straight back at risk.

    Why this can't just cancel the play coroutine -- poke_env's Player._ladder sends
    '/search' for the NEXT game the instant the previous one ends: it is woken by
    _battle_end_condition inside the |win| handler, which runs before the |raw| rating lines
    that tell us we set a record. So by the time we know, we are already in the matchmaking
    queue, and cancelling _ladder would leave us sitting in it -- Showdown would match us and
    start a game with nothing driving the queue bookkeeping. The queue has to be left
    explicitly."""
    while not run.stop_reason:
        await asyncio.sleep(poll)
    print(f"  STOPPING EARLY: {run.stop_reason} -- not queuing for another game.", flush=True)
    while True:
        await cancel_search(agent)
        # A match made just before /cancelsearch reached the server still arrives. Give its
        # |init| time to land before calling the queue clean: returning early here would
        # tear the run down on top of a game that had just started and forfeit it.
        await asyncio.sleep(CANCEL_GRACE)
        if not any(not b.finished for b in agent.battles.values()):
            return
        # Matchmaking beat us to it. That game is rated and already under way, so it gets
        # played out: dropping it would forfeit, which is the opposite of the point. Once it
        # ends, _ladder queues another search, hence the loop back to /cancelsearch.
        print("  ... a game had already started; playing it out, then stopping.", flush=True)
        while any(not b.finished for b in agent.battles.values()):
            await asyncio.sleep(poll)


async def report_progress(agent, run, poll=3.0):
    """Print queue/start/finish updates while the play coroutine runs alongside.

    Polls agent.battles rather than hooking events: it grows as games start, and each
    battle flips .finished when it ends. Spectate link on start, running W/L on finish,
    and the pre-game rating appended to RATING_LOG once Showdown reports it.

    Counts are offset by `run` so numbering and the W/L tally continue across a reconnect --
    this agent only knows about its own segment."""
    started, finished, announced_search = set(), set(), False
    while True:
        for tag, battle in list(agent.battles.items()):
            if tag not in started:
                started.add(tag)
                announced_search = False
                n = run.done + len(started)
                print(f"  Battle {n}/{run.total} started  ->  watch: {SPECTATE_URL}{tag}",
                      flush=True)
                # Opt-in only (--upload-first): the first N games of the run also get a
                # hosted, shareable replay.
                if n <= run.upload_first:
                    await request_upload(agent, tag)
            if battle.finished and tag not in finished:
                finished.add(tag)
                save_battle(agent, tag, battle, run.saved, run.warned)
                result = {"won": "WON ", "lost": "LOST", "tie": "TIE "}[result_label(battle)]
                print(f"  Battle {run.done + len(finished)}/{run.total} {result} in "
                      f"{battle.turn} turns  (running {run.won + agent.n_won_battles}W / "
                      f"{run.lost + agent.n_lost_battles}L)", flush=True)
            # Deliberately OUTSIDE the `not in finished` gate above: the |raw| rating lines
            # can arrive a tick or two after |win|, so these have to keep looking.
            if battle.finished:
                if log_rating(tag, battle, run.logged):
                    vs = f" vs {battle.opponent_rating}" if battle.opponent_rating else ""
                    print(f"    rated: went in at {battle.rating}{vs}", flush=True)
                check_elo_record(agent, tag, battle, run)
        # No unfinished battle and not done yet -> sitting in the matchmaking queue.
        active = any(not b.finished for b in agent.battles.values())
        done = run.done + len(finished)
        if not active and done < run.total and not announced_search:
            announced_search = True
            print(f"  Searching for the next opponent... ({done}/{run.total} done)", flush=True)
        await asyncio.sleep(poll)


async def play_segment(agent, run, play):
    """Play one connection's worth of games. True when the run is over, don't resume.

    True covers both `play` finishing normally and watch_stop deciding to end the run early
    (a new Elo record). False means the watchdog won the race: the websocket is gone and
    `play` would have hung forever, so it is cancelled and the caller decides whether to
    resume on a new agent.

    The teardown runs on ALL paths -- normal finish, early stop, disconnect, Ctrl-C, crash --
    because it is the only chance to archive this agent's games: the battle rooms are gone
    once the process exits, and after a drop the agent itself is discarded."""
    monitor = asyncio.ensure_future(report_progress(agent, run))
    watchdog = asyncio.ensure_future(watch_connection(agent))
    stopper = asyncio.ensure_future(watch_stop(agent, run))
    playing = asyncio.ensure_future(play)
    try:
        await asyncio.wait({playing, watchdog, stopper},
                           return_when=asyncio.FIRST_COMPLETED)
        if playing.done():
            playing.result()        # re-raise a genuine failure instead of reporting success
            return True
        if stopper.done():
            stopper.result()        # a bug in the stopper should surface, not stop us quietly
            return True
        return False
    finally:
        for task in (monitor, watchdog, stopper, playing):
            task.cancel()
        await asyncio.gather(monitor, watchdog, stopper, playing, return_exceptions=True)
        # Catches the final game and anything the last poll tick missed. In-progress battles
        # are saved too, under the 'unfinished-' prefix.
        sweep_replays(agent, run.saved, run.warned)
        # Same gap sweep_replays exists for: the monitor is cancelled the instant the last
        # game ends, so that game's rating row -- and the Elo record it might have set --
        # would otherwise never be written.
        for tag, battle in list(agent.battles.items()):
            if battle.finished:
                log_rating(tag, battle, run.logged)
                check_elo_record(agent, tag, battle, run)
        run.absorb(agent)
        # parallel_worlds only: don't leak a pool of engine processes per reconnect.
        pool = getattr(agent, "_executor", None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


async def main():
    load_dotenv()  # populate SHOWDOWN_* before argparse reads them as defaults
    parser = argparse.ArgumentParser(description="Play the bot on the real Showdown ladder.")
    parser.add_argument("--mode", choices=("ladder", "accept", "challenge"), default="ladder",
                        help="ladder = ranked games; accept = wait for a challenge; "
                             "challenge = challenge --opponent")
    parser.add_argument("--battles", type=int, default=10, help="how many games to play")
    parser.add_argument("--opponent", default=None, help="username to challenge (challenge mode)")
    parser.add_argument("--username", default=os.environ.get("SHOWDOWN_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("SHOWDOWN_PASSWORD"))
    parser.add_argument("--upload-first", type=int, nargs="?", const=UPLOAD_FIRST_N,
                        default=0, metavar="N",
                        help=f"publish the first N games as hosted Showdown replays via "
                             f"/savereplay -- PUBLIC links. Off unless passed; bare "
                             f"--upload-first means {UPLOAD_FIRST_N}. Every game is archived "
                             f"locally to replays/ladder either way.")
    parser.add_argument("--fork", action="store_true",
                        help="run the Phase-2 fork engine (net-at-leaf, 600ms/world); "
                             "handled at import time, this flag just documents it")
    parser.add_argument("--no-stop-on-record", action="store_true",
                        help="keep laddering after a new Elo record. By default a record "
                             "ends the run immediately -- the queue is cancelled rather "
                             "than putting the fresh peak straight back at risk.")
    parser.add_argument("--reconnects", type=int, default=MAX_RECONNECTS, metavar="N",
                        help=f"after a dropped websocket, resume the remaining games on a "
                             f"fresh connection, up to N consecutive failures "
                             f"(default {MAX_RECONNECTS}; 0 = stop at the first drop)")
    args = parser.parse_args()

    if not args.username:
        parser.error("no username: set SHOWDOWN_USERNAME or pass --username "
                     "(register it first at play.pokemonshowdown.com).")
    if args.mode == "challenge" and not args.opponent:
        parser.error("challenge mode needs --opponent USERNAME.")

    account = AccountConfiguration(args.username, args.password)
    if FORK_MODE:
        import poke_engine
        print(f"Playing as '{args.username}' (Laplace FORK: net-at-leaf blend 0.5, "
              f"c2={os.environ['LAPLACE_UCT_C2']}, 600ms/world).\n"
              f"  engine: {poke_engine.__file__}", flush=True)
    else:
        print(f"Playing as '{args.username}' (Laplace: poke-engine MCTS).", flush=True)

    print(f"Replays -> {REPLAY_DIR}", flush=True)
    if args.upload_first > 0:
        print(f"  first {args.upload_first} game(s) ALSO published as public replays on "
              f"{REPLAY_UPLOAD_URL}", flush=True)

    # One agent per connection. A drop retires the agent (poke_env can't reconnect one) and
    # the loop builds another for whatever is left to play, so a 20-game run isn't ended by
    # a 2-second network hiccup at game 6.
    run = Run(args.battles, args.upload_first,
              stop_on_record=not args.no_stop_on_record)
    start_high = run.elo_high
    print(f"  Elo record to beat: {start_high}"
          f"{f' (set in {run.elo_high_tag})' if run.elo_high_tag else ''}  "
          f"-- new highs archived to {ELO_DIR}", flush=True)
    print("    breaking it ENDS the run" if run.stop_on_record
          else "    --no-stop-on-record: the run continues past a new record", flush=True)
    failures = 0
    while run.done < run.total:
        remaining = run.total - run.done
        agent = build_agent(account, fork=FORK_MODE)
        if args.upload_first > 0:
            # The popup is the only confirmation Showdown sends, and poke_env logs it at
            # WARNING. If something raised the client's level above that we'd never see it,
            # so lower it back. No-op at poke_env's default, which leaves the level unset.
            if agent.ps_client.logger.getEffectiveLevel() > logging.WARNING:
                agent.ps_client.logger.setLevel(logging.WARNING)
            agent.ps_client.logger.addHandler(PopupWatcher())

        if args.mode == "ladder":
            print(f"Queuing for {remaining} ranked ladder game(s)...", flush=True)
            play = agent.ladder(remaining)
        elif args.mode == "accept":
            who = args.opponent or "anyone"
            print(f"Waiting for {remaining} challenge(s) from {who} "
                  f"(challenge '{args.username}' in {BATTLE_FORMAT})...", flush=True)
            play = agent.accept_challenges(args.opponent, remaining)
        else:
            print(f"Challenging {args.opponent} to {remaining} game(s)...", flush=True)
            play = agent.send_challenges(args.opponent, remaining)

        before, before_abandoned = run.done, len(run.abandoned)
        if await play_segment(agent, run, play):
            break
        # Lost the websocket. Anything still in progress is already archived as
        # 'unfinished-'; on Showdown it runs down the clock and is forfeited, and since we
        # never see its |win| it is replaced rather than counted.
        for tag in run.abandoned[before_abandoned:]:
            print(f"  CONNECTION LOST -- abandoned {tag} (forfeits on the timer)", flush=True)
        if run.stop_reason:
            # The record already ended this run; losing the socket on the way out is not a
            # reason to reconnect and start playing again.
            break
        # Consecutive drops with nothing to show for them. A segment that finished a game
        # starts the streak over, so a long run can ride out more than --reconnects hiccups;
        # what the budget stops is a loop that reconnects and immediately drops again.
        failures = 1 if run.done > before else failures + 1
        if run.done >= run.total:
            break
        if failures > args.reconnects:
            print(f"\nGiving up: {failures} connection loss(es) in a row with no game "
                  f"finished. {run.done}/{run.total} games played.", flush=True)
            break
        print(f"  Connection lost at {run.done}/{run.total}. Reconnecting in "
              f"{RECONNECT_WAIT:.0f}s (attempt {failures}/{args.reconnects})...", flush=True)
        await asyncio.sleep(RECONNECT_WAIT)

    if run.done:
        print(f"\nDone: won {run.won}/{run.done} ({run.won / run.done:.0%}).", flush=True)
    else:
        print("\nNo games played.", flush=True)
    if run.elo_records:
        print(f"  NEW ELO RECORD: {run.elo_high} (was {start_high}), "
              f"{len(run.elo_records)} new high(s) this run -> {ELO_DIR}", flush=True)
    elif run.track_elo:
        print(f"  Elo record unchanged at {run.elo_high}.", flush=True)
    # Guarded on done < total: the LAST game setting a record is not an early stop, and
    # check_elo_record still sets stop_reason there (play_segment's teardown sweep).
    if run.stop_reason and run.done < run.total:
        print(f"  Stopped early at {run.done}/{run.total} games: {run.stop_reason}.",
              flush=True)
    if run.abandoned:
        print(f"  {len(run.abandoned)} game(s) abandoned to a dropped connection and "
              f"forfeited -- not in the count above:", flush=True)
        for tag in run.abandoned:
            print(f"    {tag}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # main()'s finally already swept the replays out; no traceback needed.
        print("\nInterrupted.", flush=True)
