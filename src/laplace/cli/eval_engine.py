r"""Fast benchmark: EnginePlayer vs SimpleHeuristicsPlayer.

poke-engine holds the GIL during a search, so battles inside one process serialize even
with max_concurrent_battles>1. To use all cores we shard battles across worker PROCESSES
(Foul Play uses a ProcessPoolExecutor for the same reason). Wall clock with W workers is
~(battles / W) x per-battle-time.

Two knobs, easy to confuse: `threads` is search threads PER TURN (stronger search in the
same budget, not faster eval); `workers` is cross-battle parallelism. Keep
workers x threads near the core count.

This yardstick is SATURATED at ~92% -- it's a regression check, not a strength test. The
real one is bench_foulplay.py.

From the project root, with the local server running:
    python -m laplace.cli.eval_engine
    python -m laplace.cli.eval_engine --battles 100 --workers 10
    python -m laplace.cli.eval_engine --determinizations 8 --time-ms 150 --threads 2
"""

import argparse
import asyncio
import os
import time
from concurrent.futures import ProcessPoolExecutor

BATTLE_FORMAT = "gen9randombattle"


def _play_share(n, det, time_ms, threads, idx):
    """Worker process: play `n` battles, return wins. Top-level so it pickles on Windows."""
    from poke_env import AccountConfiguration
    from poke_env.player import SimpleHeuristicsPlayer
    from laplace.agent.engine_search import EnginePlayer

    async def run():
        engine = EnginePlayer(
            account_configuration=AccountConfiguration.generate(f"eng{idx}", rand=True),
            battle_format=BATTLE_FORMAT,
            n_determinizations=det, search_time_ms=time_ms, threads=threads)
        opponent = SimpleHeuristicsPlayer(
            account_configuration=AccountConfiguration.generate(f"heur{idx}", rand=True),
            battle_format=BATTLE_FORMAT)
        await engine.battle_against(opponent, n_battles=n)
        return engine.n_won_battles

    return asyncio.new_event_loop().run_until_complete(run())


def _split(total, workers):
    """Divide `total` battles as evenly as possible into `workers` shares, dropping empties."""
    base, extra = divmod(total, workers)
    return [base + (1 if i < extra else 0) for i in range(workers) if base + (1 if i < extra else 0) > 0]


def main():
    parser = argparse.ArgumentParser(description="Benchmark the poke-engine search bot.")
    parser.add_argument("--battles", type=int, default=60)
    parser.add_argument("--determinizations", type=int, default=6)
    parser.add_argument("--time-ms", type=int, default=120)
    parser.add_argument("--threads", type=int, default=2, help="search threads per turn")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                        help="worker processes (cross-battle parallelism)")
    args = parser.parse_args()

    shares = _split(args.battles, args.workers)
    print(f"EnginePlayer (det={args.determinizations}, {args.time_ms}ms, {args.threads} threads) "
          f"vs heuristic; {args.battles} battles over {len(shares)} workers.", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(shares)) as ex:
        futures = [ex.submit(_play_share, n, args.determinizations, args.time_ms,
                             args.threads, i) for i, n in enumerate(shares)]
        wins = sum(f.result() for f in futures)
    dt = time.time() - t0

    wr = wins / args.battles
    print(f"\nEnginePlayer vs heuristic: {wr:.0%}  ({wins}/{args.battles})   "
          f"[{dt:.0f}s wall, {dt/args.battles:.1f}s/battle effective]", flush=True)


if __name__ == "__main__":
    main()
