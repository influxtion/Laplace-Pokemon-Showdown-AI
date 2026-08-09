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

Those are local. The first N games of each run are ALSO published to
replay.pokemonshowdown.com via /savereplay (--upload-first N, 0 to disable). Those links
are PUBLIC.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys

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

from engine_search import EnginePlayer

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

# Showdown-hosted (shareable) replays, uploaded via /savereplay. See request_upload().
REPLAY_UPLOAD_URL = "https://replay.pokemonshowdown.com/"
UPLOAD_FIRST_N = 5          # per run; --upload-first overrides

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
    # robust_vote=False + mix_frac 0.9 is the FP recipe: plain share averaging plus a mixed
    # strategy in a wide window. vs Foul Play 57% (17-13) against 28.3% (17-43) for robust
    # argmax; mirror gate 47.5%/120. The small mirror tax buys unexploitability the mirror
    # structurally can't price -- humans switch immunity absorbers into deterministic clicks.
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


async def request_upload(agent, tag):
    """Ask Showdown to publish this battle as a hosted replay.

    Sent at battle START, not end, for two reasons found in the server/client source:
    poke_env sends '/leave' the moment a battle ends (player.py), so saving on the
    'finished' poll races the room teardown; and once a battle has been saved once, the
    server re-uploads the COMPLETE log automatically at game end (room-battle.ts), silently
    overwriting the partial. Saving early is both the safe order and the complete one."""
    try:
        await agent.ps_client.send_message("/savereplay", room=tag)
        print(f"    /savereplay sent -> {replay_url(tag)}", flush=True)
    except Exception as exc:
        print(f"    WARNING: /savereplay failed for {tag}: {exc!r}", flush=True)


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


async def report_progress(agent, total, saved, warned, upload_first=UPLOAD_FIRST_N, poll=3.0):
    """Print queue/start/finish updates while the play coroutine runs alongside.

    Polls agent.battles rather than hooking events: it grows as games start, and each
    battle flips .finished when it ends. Spectate link on start, running W/L on finish."""
    started, finished, announced_search = set(), set(), False
    while True:
        for tag, battle in list(agent.battles.items()):
            if tag not in started:
                started.add(tag)
                announced_search = False
                print(f"  Battle {len(started)}/{total} started  ->  watch: {SPECTATE_URL}{tag}",
                      flush=True)
                # First N games of the run also get a hosted, shareable replay.
                if len(started) <= upload_first:
                    await request_upload(agent, tag)
            if battle.finished and tag not in finished:
                finished.add(tag)
                save_battle(agent, tag, battle, saved, warned)
                result = {"won": "WON ", "lost": "LOST", "tie": "TIE "}[result_label(battle)]
                print(f"  Battle {len(finished)}/{total} {result} in {battle.turn} turns  "
                      f"(running {agent.n_won_battles}W / {agent.n_lost_battles}L)", flush=True)
        # No unfinished battle and not done yet -> sitting in the matchmaking queue.
        active = any(not b.finished for b in agent.battles.values())
        if not active and len(finished) < total and not announced_search:
            announced_search = True
            print(f"  Searching for the next opponent... ({len(finished)}/{total} done)", flush=True)
        await asyncio.sleep(poll)


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
    parser.add_argument("--upload-first", type=int, default=UPLOAD_FIRST_N, metavar="N",
                        help=f"publish the first N games as hosted Showdown replays via "
                             f"/savereplay (default {UPLOAD_FIRST_N}, 0 = none). Every game "
                             f"is archived locally regardless.")
    parser.add_argument("--fork", action="store_true",
                        help="run the Phase-2 fork engine (net-at-leaf, 600ms/world); "
                             "handled at import time, this flag just documents it")
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

    agent = build_agent(account, fork=FORK_MODE)

    if args.mode == "ladder":
        print(f"Queuing for {args.battles} ranked ladder game(s)...", flush=True)
        play = agent.ladder(args.battles)
    elif args.mode == "accept":
        who = args.opponent or "anyone"
        print(f"Waiting for {args.battles} challenge(s) from {who} "
              f"(challenge '{args.username}' in {BATTLE_FORMAT})...", flush=True)
        play = agent.accept_challenges(args.opponent, args.battles)
    else:
        print(f"Challenging {args.opponent} to {args.battles} game(s)...", flush=True)
        play = agent.send_challenges(args.opponent, args.battles)

    print(f"Replays -> {REPLAY_DIR}", flush=True)
    if args.upload_first > 0:
        print(f"  first {args.upload_first} game(s) also published to "
              f"{REPLAY_UPLOAD_URL}", flush=True)
        # The popup is the only confirmation Showdown sends, and poke_env logs it at
        # WARNING. If something raised the client's level above that we'd never see it, so
        # lower it back. No-op at poke_env's default, which leaves the level unset.
        if agent.ps_client.logger.getEffectiveLevel() > logging.WARNING:
            agent.ps_client.logger.setLevel(logging.WARNING)
        agent.ps_client.logger.addHandler(PopupWatcher())

    # Run the games alongside a monitor that prints start/finish/queue updates and saves
    # each replay as it lands. saved/warned are shared with the exit sweep below so nothing
    # is written twice and nothing is dropped.
    saved, warned = set(), set()
    monitor = asyncio.ensure_future(
        report_progress(agent, args.battles, saved, warned, args.upload_first))
    try:
        await play
    finally:
        monitor.cancel()
        try:
            await monitor
        except asyncio.CancelledError:
            pass
        # Normal path AND Ctrl-C / exception: catches the final game and anything the last
        # poll tick missed.
        sweep_replays(agent, saved, warned)

    won = agent.n_won_battles
    total = len(agent.battles)
    print(f"\nDone: won {won}/{total} ({won / total:.0%})." if total else "\nNo games played.",
          flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # main()'s finally already swept the replays out; no traceback needed.
        print("\nInterrupted.", flush=True)
