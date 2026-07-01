r"""Play the trained bot on the real Pokemon Showdown ladder against humans.

Unlike every other script here, this connects to the OFFICIAL server (sim3.psim.us), not
the local one -- so the local server does NOT need to be running. By default it plays the
agent we ship (the policy + 1-ply search) as a registered account; pass --raw for the bare
policy.

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
    python -u src\ladder.py --raw                 # bare policy, no search
    python -u src\ladder.py --mode accept         # wait for someone to challenge you
    python -u src\ladder.py --mode challenge --opponent SomeUser

Ladder games are rated, so your GXE/ELO shows on the account's profile -- the honest test
the training scripts couldn't give us. Follow Showdown's rules for automated play: keep the
account present, don't flood the ladder, and don't farm a specific person.
"""

import argparse
import asyncio
import os

from sb3_contrib import MaskablePPO

from poke_env import AccountConfiguration, ShowdownServerConfiguration

from eval_search import pick_model_path
from search import SearchPlayer
from engine_search import EnginePlayer
from opponent import ModelPlayer

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


def build_agent(model, account, kind):
    """The bot to ladder with, pointed at the official server under the given account.
    kind: "engine" (poke-engine MCTS, strongest), "search" (1-ply), or "raw" (bare policy)."""
    kwargs = dict(
        account_configuration=account,
        server_configuration=ShowdownServerConfiguration,
        battle_format=BATTLE_FORMAT,
        start_timer_on_battle_start=True,   # start the clock each game so an AFK opponent can't stall the bot
    )
    if kind == "engine":
        # poke-engine needs no trained model; it searches the position directly. Ladder is one
        # game at a time on the official server, so we spend a generous per-turn budget (more
        # determinizations + threads than the throughput-tuned benchmark default) -- ~1-2s/turn,
        # well within Showdown's move timer. record=True keeps per-turn decision traces, which
        # get dumped next to the replay for post-hoc loss analysis.
        return EnginePlayer(n_determinizations=8, search_time_ms=150, threads=8, record=True,
                            **kwargs)
    if kind == "search":
        return SearchPlayer(model=model, **kwargs)
    return ModelPlayer(model=model, deterministic=True, **kwargs)


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
    parser.add_argument("--raw", action="store_true",
                        help="use the bare policy instead of search")
    parser.add_argument("--search", action="store_true",
                        help="use the 1-ply policy+damage search instead of the poke-engine MCTS")
    parser.add_argument("--model", default=None,
                        help="model .zip to load (default: best available, see eval_search)")
    parser.add_argument("--username", default=os.environ.get("SHOWDOWN_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("SHOWDOWN_PASSWORD"))
    args = parser.parse_args()

    if not args.username:
        parser.error("no username: set SHOWDOWN_USERNAME or pass --username "
                     "(register it first at play.pokemonshowdown.com).")
    if args.mode == "challenge" and not args.opponent:
        parser.error("challenge mode needs --opponent USERNAME.")

    account = AccountConfiguration(args.username, args.password)
    kind = "raw" if args.raw else "search" if args.search else "engine"
    labels = {"engine": "poke-engine MCTS", "search": "1-ply search", "raw": "raw policy"}
    # The engine agent is model-free; the search/raw agents need a trained policy.
    model = None
    if kind != "engine":
        model_path = args.model or pick_model_path()
        print(f"Loading {model_path}.", flush=True)
        model = MaskablePPO.load(model_path)
    print(f"Playing as '{args.username}' ({labels[kind]}).", flush=True)

    agent = build_agent(model, account, kind)

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
