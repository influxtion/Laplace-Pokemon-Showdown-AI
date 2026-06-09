r"""Benchmark the test-time search agent against the raw policy.

Runs the trained model two ways against SimpleHeuristicsPlayer and prints both win rates, to
see whether the 1-ply lookahead helps:
  - raw policy     : the network's move, no search (opponent.ModelPlayer)
  - search (1-ply) : every action scored by the damage model (search.SearchPlayer)

Uses poke-env's battle_against. Loads the best available model (see pick_model_path).

Run from the project root, with the local Showdown server running:
    python -u src\eval_search.py
"""

import asyncio
import os

from sb3_contrib import MaskablePPO

from poke_env import AccountConfiguration
from poke_env.player import SimpleHeuristicsPlayer

from rl_env import N_FEATURES
from search import SearchPlayer
from opponent import ModelPlayer

BATTLE_FORMAT = "gen9randombattle"
N_BATTLES = 400


def pick_model_path():
    # Best first: the self-play agent, then plain v3, then the heuristic-trained fallback.
    # Layering search on the best base measures the full stack.
    candidates = (
        f"ppo_v3_anneal_selfplay_best_obs{N_FEATURES}.zip",   # Influxobot (final): anneal -> self-play
        f"ppo_v3_anneal_selfplay_obs{N_FEATURES}.zip",
        f"ppo_v3_anneal_best_obs{N_FEATURES}.zip",
        f"ppo_v3_anneal_obs{N_FEATURES}.zip",
        f"ppo_v3_selfplay_best_obs{N_FEATURES}.zip",
        f"ppo_v3_selfplay_obs{N_FEATURES}.zip",
        f"ppo_v3_obs{N_FEATURES}.zip",
        f"ppo_vs_heuristic_obs{N_FEATURES}.zip",
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No trained model found (ppo_v3_selfplay / ppo_v3 / ppo_vs_heuristic).")


async def bench(player, tag, n):
    opponent = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration.generate(f"heur{tag}", rand=True),
        battle_format=BATTLE_FORMAT,
    )
    await player.battle_against(opponent, n_battles=n)
    return player.n_won_battles / n


async def main():
    path = pick_model_path()
    print(f"Loading {path}; {N_BATTLES} battles each.", flush=True)
    model = MaskablePPO.load(path)

    raw = ModelPlayer(
        model=model, deterministic=True,
        account_configuration=AccountConfiguration.generate("rawpolicy", rand=True),
        battle_format=BATTLE_FORMAT,
    )
    raw_wr = await bench(raw, "raw", N_BATTLES)
    print(f"raw policy     vs heuristic: {raw_wr:.0%}", flush=True)

    searcher = SearchPlayer(
        model=model,
        account_configuration=AccountConfiguration.generate("searchbot", rand=True),
        battle_format=BATTLE_FORMAT,
    )
    search_wr = await bench(searcher, "search", N_BATTLES)
    print(f"search (1-ply) vs heuristic: {search_wr:.0%}", flush=True)

    print(f"\nlift from search: {(search_wr - raw_wr) * 100:+.0f} points", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
