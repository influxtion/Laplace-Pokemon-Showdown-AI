r"""Play ONE ladder game and narrate it live in the terminal.

Same bot, same shipped ladder config as `ladder.py` (this imports build_agent from it, so
the settings cannot drift) -- the difference is that the terminal becomes a live analysis
board instead of a progress log. Per turn you see:

  * the position: both actives with HP bars, boosts, status, hazards, weather, team counts;
  * the search working: determinized worlds completing one by one, visits accumulating, the
    engine's own win estimate firming up;
  * the candidates: pooled visit share, how many worlds each move won, the engine's eval,
    and which ones a guard vetoed;
  * the hidden-information guesses: which item / ability / Tera type the worlds drew for the
    opponent's active, plus Choice-Scarf verdicts inferred from turn order;
  * the pipeline: which guard or reranker moved the front-runner, what the value net scored,
    whether the mixed root sampled off the argmax, and the move finally sent;
  * the resolution: what both sides actually did, with damage, crits, misses and faints.

Then a post-game block: time spent thinking, MCTS volume, guard/reranker activity, how often
the search called the opponent's move correctly, the luck ledger, the eval curve with its
biggest swings, and the replay path.

One game only, on purpose -- this is for watching and understanding, not for laddering. Use
`ladder.py` for a cohort.

    python -u src\analyze_battle.py                     # one rated ladder game, narrated
    python -u src\analyze_battle.py --upload            # ... and publish a shareable replay
    python -u src\analyze_battle.py --ascii --no-color  # dumb-terminal / redirect friendly
    python -u src\analyze_battle.py --mode challenge --opponent SomeUser

Credentials come from .env / SHOWDOWN_USERNAME / --username, exactly as in ladder.py. The
game is rated and archived to replays/ladder/ like any other ladder game; the transcript is
saved next to the replay as <result>-<tag>.analysis.txt unless --no-save-log.
"""

import argparse
import asyncio
import logging
import os
import sys

# ladder first: its module-level --fork block has to run before poke_engine is imported
# anywhere, and importing it here is also what keeps the shipped ladder config in one place.
import ladder
from live_analysis import AnalyzedPlayer, LiveAnalyzer, enable_ansi

from poke_env import AccountConfiguration


async def announce(agent, analyzer, waiting, upload, local=False, poll=0.4):
    """Print the waiting/matched banner while the game coroutine runs.

    Polls agent.battles rather than hooking events (same approach as
    ladder.report_progress): the battle appears in it the moment the server pairs us. The
    poll is short so the watch link lands above the first turn rather than inside it."""
    said = False
    base = "http://localhost:8000/" if local else ladder.SPECTATE_URL
    while True:
        for tag in list(agent.battles):
            analyzer.line(f"  matched -> watch live at {base}{tag}")
            if upload:
                await ladder.request_upload(agent, tag)
            return
        if not said:
            said = True
            analyzer.line("  " + waiting)
        await asyncio.sleep(poll)


def print_header(agent, analyzer, args, fork):
    """One block describing exactly what is about to play, read off the live agent so it
    can't drift from build_agent()."""
    a = analyzer
    a.rule(a.c("hdr", "Laplace -- live analysis of one ladder game"), ch="═")
    engine = "poke-engine MCTS" + (" [FORK: value net at the leaf]" if fork else "")
    where = "localhost:8000" if args.local else "official Showdown server"
    a.line(f"  account    {args.username} @ {where}")
    a.line(f"  engine     {engine}")
    a.line(f"  search     {agent.N_DETERMINIZATIONS} determinized worlds x "
           f"{agent.SEARCH_TIME_MS}ms x {agent.THREADS} threads"
           f"   value net {'on' if agent._value_model is not None else 'off'}")
    root = "plain share averaging" if not agent.robust_vote else "robust world vote"
    mix = f"mixed strategy in a {agent.mix_frac:.2f} window" if agent.mix_root \
        else "deterministic argmax"
    a.line(f"  root       {root}, {mix}")
    guards = [n for n, on in (("absorb", agent.absorb_guard),
                              ("futility", agent.futility_guard),
                              ("gamble", agent.gamble_guard)) if on]
    a.line(f"  guards     {', '.join(guards + ['no-op', 'dead-lock', 'damage tiebreak'])}")
    a.line(f"  replays    {ladder.REPLAY_DIR}")
    a.line()
    a.line("  " + a.c("dim", "reading a turn:  share = pooled MCTS visit share across "
                             "worlds  ·  worlds = worlds that move won"))
    a.line("  " + a.c("dim", "                 eval  = the engine's own win estimate for "
                             "us (0-1)  ·  ^n = promoted n places"))
    a.line("  " + a.c("dim", "                 stage lines (absorb/futility/tiebreak/value/"
                             "noop/deadlock/mix) = who changed the pick"))
    a.rule(ch="═")


async def main():
    ladder.load_dotenv()      # SHOWDOWN_* from .env before argparse reads them as defaults
    ap = argparse.ArgumentParser(
        description="Play one ladder game with live search analysis in the terminal.")
    ap.add_argument("--mode", choices=("ladder", "accept", "challenge"), default="ladder",
                    help="ladder = one rated game (default); accept = wait for a challenge; "
                         "challenge = challenge --opponent")
    ap.add_argument("--opponent", default=None, help="username (accept/challenge mode)")
    ap.add_argument("--username", default=os.environ.get("SHOWDOWN_USERNAME"))
    ap.add_argument("--password", default=os.environ.get("SHOWDOWN_PASSWORD"))
    ap.add_argument("--local", action="store_true",
                    help="play on the local server (localhost:8000) instead of the official "
                         "one -- for trying the analysis out without spending a rated game; "
                         "pair it with --mode challenge/accept, the local server has no "
                         "ladder pool")
    ap.add_argument("--upload", action="store_true",
                    help="also publish this game as a hosted Showdown replay "
                         "(a PUBLIC link; off by default)")
    ap.add_argument("--no-color", action="store_true", help="no ANSI colour")
    ap.add_argument("--ascii", action="store_true",
                    help="ASCII bars instead of block glyphs (auto-detected otherwise)")
    ap.add_argument("--no-worlds", action="store_true",
                    help="hide the per-turn hidden-set guesses (item/ability/tera tally)")
    ap.add_argument("--no-save-log", action="store_true",
                    help="don't write the transcript next to the replay")
    ap.add_argument("--width", type=int, default=None, help="transcript width in columns")
    ap.add_argument("--fork", action="store_true",
                    help="run the Phase-2 fork engine (net-at-leaf, 600ms/world); handled at "
                         "import time by ladder.py, this flag just documents it")
    args = ap.parse_args()

    if not args.username:
        ap.error("no username: set SHOWDOWN_USERNAME in .env or pass --username "
                 "(register it first at play.pokemonshowdown.com).")
    if args.mode == "challenge" and not args.opponent:
        ap.error("challenge mode needs --opponent USERNAME.")

    enable_ansi()
    # An exotic opponent username must not be able to kill the run on a narrow console.
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    analyzer = LiveAnalyzer(args.username, width=args.width, color=not args.no_color,
                            unicode=False if args.ascii else None,
                            show_worlds=not args.no_worlds)
    extra = {}
    if args.local:
        from poke_env import LocalhostServerConfiguration
        extra["server_configuration"] = LocalhostServerConfiguration
    agent = ladder.build_agent(AccountConfiguration(args.username, args.password),
                               fork=ladder.FORK_MODE, cls=AnalyzedPlayer, analyzer=analyzer,
                               **extra)
    print_header(agent, analyzer, args, ladder.FORK_MODE)

    if args.mode == "ladder":
        play, waiting = agent.ladder(1), "queued on the ladder, waiting for an opponent..."
    elif args.mode == "accept":
        play = agent.accept_challenges(args.opponent, 1)
        waiting = (f"waiting for a {ladder.BATTLE_FORMAT} challenge from "
                   f"{args.opponent or 'anyone'}...")
    else:
        play = agent.send_challenges(args.opponent, 1)
        waiting = f"challenging {args.opponent}..."

    if args.upload:
        # The upload confirmation only ever arrives as a popup poke_env logs at WARNING.
        if agent.ps_client.logger.getEffectiveLevel() > logging.WARNING:
            agent.ps_client.logger.setLevel(logging.WARNING)
        agent.ps_client.logger.addHandler(ladder.PopupWatcher())

    banner = asyncio.ensure_future(
        announce(agent, analyzer, waiting, args.upload, local=args.local))
    saved, warned = set(), set()
    try:
        await play
    finally:
        banner.cancel()
        try:
            await banner
        except asyncio.CancelledError:
            pass
        # Runs on the normal path AND on Ctrl-C: the battle room is gone once we exit, so an
        # unsaved game (or an unwritten transcript) is lost for good.
        tag, battle = next(iter(agent.battles.items()), (None, None))
        if battle is None:
            print("\nNo game was played.", flush=True)
        else:
            result = ladder.result_label(battle)
            path = None
            if ladder.save_battle(agent, tag, battle, saved, warned):
                path = os.path.join(ladder.REPLAY_DIR, f"{result}-{tag}.html")
            analyzer.summary(battle=battle, result=result, replay_path=path,
                             diag=dict(agent.diag))
            if not args.no_save_log and path:
                log_path = path[: -len(".html")] + ".analysis.txt"
                try:
                    analyzer.save(log_path)
                    print(f"  transcript  {log_path}", flush=True)
                except OSError as exc:
                    print(f"  WARNING: could not write the transcript: {exc!r}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # main()'s finally has already saved the replay and printed the summary.
        print("\nInterrupted.", flush=True)
