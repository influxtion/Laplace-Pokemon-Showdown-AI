r"""Play the trained bot on the real Pokemon Showdown ladder against humans.

Unlike every other script here, this connects to the OFFICIAL server (sim3.psim.us), not
the local one -- so the local server does NOT need to be running. 

Setup (once):
  1. Go to https://play.pokemonshowdown.com, click the gear (Options) -> Register, and
     register the username you want the bot to play as, with a password.
  2. Credentials come from a .env file in the project root (gitignored), or from the
     SHOWDOWN_USERNAME / SHOWDOWN_PASSWORD environment variables, or --username/--password.
     The .env format is two lines:
         SHOWDOWN_USERNAME=Influxobot
         SHOWDOWN_PASSWORD=your-password

Then, from the project root:
    python -u src\ladder.py                       # play 10 ranked ladder games
    python -u src\ladder.py --battles 25          # play 25
    python -u src\ladder.py --mode accept         # wait for someone to challenge you
    python -u src\ladder.py --mode challenge --opponent SomeUser

Ladder games are rated, so your GXE/ELO shows on the account's profile -- the honest test
the training scripts couldn't give us. Follow Showdown's rules for automated play: keep the
account present, don't flood the ladder, and don't farm a specific person.
"""

import argparse
import asyncio
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --fork: run the Phase-2 engine build -- value net at the MCTS leaf (blend 0.5 with the
# stock eval, UCT c2=0.1) at 600ms/world, cross-engine validated 43/60 = 71.7% vs the
# stock 120ms build (2026-07-06). The fork wheel lives in poke-engine-fork/py and is
# shadowed onto sys.path; the venv's stock poke-engine is untouched, so omitting the
# flag is a complete rollback. This block must run BEFORE engine_search imports
# poke_engine, hence the sys.argv peek instead of argparse. The LAPLACE_* env vars are
# read once by the Rust engine on first search; without LAPLACE_VALUE_NET the fork
# behaves bit-identically to stock. setdefault = an explicit env override wins.
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
SPECTATE_URL = "https://play.pokemonshowdown.com/"   # + battle room id = a watchable link

# .env lives in the project root, one level up from this src/ file.
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def load_dotenv(path=_ENV_PATH):
    """Read KEY=value lines from .env into the environment (real env vars win)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def build_agent(account, fork=False):
    """The Laplace bot (poke-engine MCTS over determinized opponent teams), pointed at the
    official server under the given account."""
    # poke-engine needs no trained model; it searches the position directly. Ladder is one
    # game at a time on the official server, so we spend a generous per-turn budget (more
    # determinizations + threads than the throughput-tuned benchmark default) -- ~1-2s/turn,
    # well within Showdown's move timer. record=True keeps per-turn decision traces, which
    # get dumped next to the replay for post-hoc loss analysis.
    # The learned value head re-ranks near-tied root candidates (fixes the engine eval's
    # blindness to opponent setup / wasted turns); it ships whenever the trained net exists.
    value_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "models", "value_net.pt")
    # value_boost_margin: OFF (was 26). Cohort-2 mining 2026-07-03 (11W-19L, 2018->1848):
    # with the opponent boosted, the stale v3 net used the widened band to override the
    # search toward worse moves (Freezing Glare over Hurricane x4 into a +6 Calm Mind
    # sweep, every pick inside the 0.26 margin). The net keeps its A/B-supported base
    # tie-break authority (value_margin=11); re-widen only after a v4 retrain earns it.
    # FP-recipe root (2026-07-03): plain share averaging + mixed strategy in a 0.9
    # window. vs Foul Play 57% (17-13) at mix_frac=0.9 vs 28.3% (17-43) for robust
    # argmax; mirror gate 47.5%/120 (the small tax buys unexploitability the mirror
    # can't price -- humans switch immunity absorbers into deterministic clicks).
    # Fork mode spends 600ms/world (vs 150ms stock): informed search converts time into
    # strength where the stock eval provably can't (budget falsified twice for stock;
    # fork600-vs-stock120 = 71.7%/60). ~5s/turn at det8 sequential -- inside the move
    # timer, but with less bank cushion in very long games.
    return EnginePlayer(
        account_configuration=account,
        server_configuration=ShowdownServerConfiguration,
        battle_format=BATTLE_FORMAT,
        start_timer_on_battle_start=True,   # start the clock so an AFK opponent can't stall the bot
        n_determinizations=8, search_time_ms=600 if fork else 150, threads=8, record=True,
        value_model_path=value_path if os.path.exists(value_path) else None,
        value_boost_margin=0,
        robust_vote=False, mix_root=True, mix_frac=0.9,
    )


async def report_progress(agent, total, poll=3.0):
    """Print queue/start/finish updates while the play coroutine runs in parallel.

    Polls the agent's battle state rather than hooking events: agent.battles grows as games
    start, and each battle flips .finished when it ends. Prints a spectate link on start and a
    running W/L tally on finish."""
    started, finished, announced_search = set(), set(), False
    while True:
        for tag, battle in list(agent.battles.items()):
            if tag not in started:
                started.add(tag)
                announced_search = False
                print(f"  Battle {len(started)}/{total} started  ->  watch: {SPECTATE_URL}{tag}",
                      flush=True)
            if battle.finished and tag not in finished:
                finished.add(tag)
                # battle.won is True (win) / False (loss) / None (genuine tie). NOTE: `battle.tied`
                # is a *method*, so the old `getattr(battle, "tied", False)` returned the truthy
                # bound method and mislabeled every LOSS as a tie. Check `won is None` instead.
                if battle.won:
                    result = "WON "
                elif battle.won is None:
                    result = "TIE "
                else:
                    result = "LOST"
                # Save every finished game as a replay so post-hoc analysis is possible (the
                # live battle room vanishes once it ends). Ties especially -- they're frequent
                # vs humans and we need a replay to see why (Endless Battle Clause / sim-KO).
                try:
                    os.makedirs(os.path.join("replays", "ladder"), exist_ok=True)
                    rpath = os.path.join("replays", "ladder", f"{result.strip().lower()}-{tag}.html")
                    agent.save_replay(tag, rpath)
                    # Per-turn decision traces (what the search considered), for loss analysis.
                    traces = getattr(agent, "traces", {}).get(tag)
                    if traces:
                        import json
                        tpath = os.path.join("replays", "ladder",
                                             f"{result.strip().lower()}-{tag}.trace.json")
                        with open(tpath, "w", encoding="utf-8") as f:
                            json.dump(traces, f, indent=1)
                except Exception:
                    pass
                print(f"  Battle {len(finished)}/{total} {result} in {battle.turn} turns  "
                      f"(running {agent.n_won_battles}W / {agent.n_lost_battles}L)", flush=True)
        # No unfinished battle and not done yet -> sitting in the matchmaking queue.
        active = any(not b.finished for b in agent.battles.values())
        if not active and len(finished) < total and not announced_search:
            announced_search = True
            print(f"  Searching for the next opponent... ({len(finished)}/{total} done)", flush=True)
        await asyncio.sleep(poll)


async def main():
    load_dotenv()  # populate SHOWDOWN_* from .env before reading them as argparse defaults
    parser = argparse.ArgumentParser(description="Play the bot on the real Showdown ladder.")
    parser.add_argument("--mode", choices=("ladder", "accept", "challenge"), default="ladder",
                        help="ladder = ranked games; accept = wait for a challenge; "
                             "challenge = challenge --opponent")
    parser.add_argument("--battles", type=int, default=10, help="how many games to play")
    parser.add_argument("--opponent", default=None, help="username to challenge (challenge mode)")
    parser.add_argument("--username", default=os.environ.get("SHOWDOWN_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("SHOWDOWN_PASSWORD"))
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

    # Run the games alongside a monitor that prints start/finish/queue updates.
    monitor = asyncio.ensure_future(report_progress(agent, args.battles))
    try:
        await play
    finally:
        monitor.cancel()
        try:
            await monitor
        except asyncio.CancelledError:
            pass

    won = agent.n_won_battles
    total = len(agent.battles)
    print(f"\nDone: won {won}/{total} ({won / total:.0%})." if total else "\nNo games played.",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
