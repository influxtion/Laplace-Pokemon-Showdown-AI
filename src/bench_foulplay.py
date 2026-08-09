r"""Head-to-head against Foul Play on the local server.

Foul Play (pmariglia/foul-play) is the strongest open-source Showdown bot and the project
ours descends from, so beating it answers "is this the strongest open bot?". Both sides get
matched compute (states x time x threads) -- the result is about the ideas, not the box.

Setup (once): clone foul-play/ with its own fp-venv; its pinned deps conflict with ours.
Start the pieces in this order:
  1. local server:  cd server && node pokemon-showdown start --no-security
  2. Foul Play (accepts our challenges):
     foul-play\fp-venv\Scripts\python.exe foul-play\run.py ^
       --websocket-uri ws://localhost:8000/showdown/websocket ^
       --ps-username FoulPlayFP --bot-mode accept_challenge ^
       --pokemon-format gen9randombattle --run-count 999 ^
       --search-time-ms 120 --search-parallelism 6 --search-threads 2
  3. this script:   python -u src\bench_foulplay.py --battles 20

Replays land in replays/foulplay/ for post-hoc mining of its wins.
"""

import argparse
import asyncio
import json
import os

from poke_env import AccountConfiguration, LocalhostServerConfiguration

from engine_search import EnginePlayer

BATTLE_FORMAT = "gen9randombattle"
OUT_DIR = os.path.join("replays", "foulplay")


async def main():
    ap = argparse.ArgumentParser(description="Challenge Foul Play on the local server.")
    ap.add_argument("--battles", type=int, default=20)
    ap.add_argument("--opponent", default="FoulPlayFP")
    ap.add_argument("--det", type=int, default=6)
    ap.add_argument("--time-ms", type=int, default=120)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--mix", action="store_true",
                    help="mixed root strategy (sample among near-tied world-winners)")
    ap.add_argument("--avg", action="store_true",
                    help="FP-style share averaging instead of the robust vote")
    ap.add_argument("--mix-frac", type=float, default=0.75,
                    help="mix window as a fraction of the top score")
    ap.add_argument("--username", default="influxbench",
                    help="account name; use a distinct one per config so mining can split runs")
    ap.add_argument("--net", default=None,
                    help="value net path override (default models/value_net.pt)")
    ap.add_argument("--boost-margin", type=float, default=0,
                    help="value_boost_margin (0 = ladder config; 26 = the pre-cohort-2 arm)")
    args = ap.parse_args()

    value_path = args.net or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "value_net.pt")
    agent = EnginePlayer(
        account_configuration=AccountConfiguration(args.username, None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=BATTLE_FORMAT,
        n_determinizations=args.det, search_time_ms=args.time_ms, threads=args.threads,
        value_model_path=value_path if os.path.exists(value_path) else None,
        value_boost_margin=args.boost_margin, record=True, mix_root=args.mix,
        mix_frac=args.mix_frac, robust_vote=not args.avg,
        start_timer_on_battle_start=True,
    )

    print(f"Challenging {args.opponent} to {args.battles} game(s) "
          f"(det={args.det}, {args.time_ms}ms, {args.threads}t, mix={args.mix}, "
          f"avg={args.avg})...", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    saved = set()
    # One challenge at a time with a pause: Foul Play only accepts while it's back in its
    # 'waiting for challenge' state, so back-to-back challenges get dropped and both sides
    # deadlock (observed live). Saving replays per game also survives interruption.
    for i in range(args.battles):
        if i:
            await asyncio.sleep(8)
        await agent.send_challenges(args.opponent, n_challenges=1)
        for tag, battle in agent.battles.items():
            if tag in saved or not battle.finished:
                continue
            saved.add(tag)
            result = "won" if battle.won else ("tie" if battle.won is None else "lost")
            try:
                agent.save_replay(tag, os.path.join(OUT_DIR, f"{result}-{tag}.html"))
                traces = agent.traces.get(tag)
                if traces:
                    with open(os.path.join(OUT_DIR, f"{result}-{tag}.trace.json"), "w",
                              encoding="utf-8") as f:
                        json.dump(traces, f, indent=1)
            except Exception:
                pass
        print(f"  game {i + 1}/{args.battles} done "
              f"(running {agent.n_won_battles}W / {agent.n_lost_battles}L)", flush=True)

    w, l = agent.n_won_battles, agent.n_lost_battles
    print(f"RESULT: {w}W / {l}L vs {args.opponent} "
          f"({w / max(w + l, 1):.0%})  diag={dict(agent.diag)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
