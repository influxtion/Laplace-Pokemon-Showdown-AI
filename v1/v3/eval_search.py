r"""Benchmark the test-time search agent, and compare it to the raw policy.

Runs the trained model two ways against SimpleHeuristicsPlayer and prints both win rates so
we can see whether the 1-ply lookahead actually helps:
  - raw policy        : the network's move, no search (selfplay.ModelPlayer)
  - search (1-ply)    : policy proposes top-k, damage model picks (search.SearchPlayer)

Uses poke-env's native player-vs-player (battle_against). Loads the v3 model if present,
otherwise the heuristic-trained 215 model.

Run from the project root, with the local Showdown server running:
    python -u v1\v3\eval_search.py
"""

import asyncio
import os
import sys

# This file lives in v1/v3/. Put v1/ on the path (for rl_env) and v1/selfplay/ on the path
# (for the ModelPlayer used as the raw-policy baseline).
_V1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _V1)
sys.path.insert(0, os.path.join(_V1, "selfplay"))

from sb3_contrib import MaskablePPO

from poke_env import AccountConfiguration
from poke_env.player import SimpleHeuristicsPlayer

from rl_env import N_FEATURES
from search import SearchPlayer
from opponent import ModelPlayer

BATTLE_FORMAT = "gen9randombattle"
N_BATTLES = 200


def pick_model_path():
    for path in (f"ppo_v3_obs{N_FEATURES}.zip", f"ppo_vs_heuristic_obs{N_FEATURES}.zip"):
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No trained model found (ppo_v3 or ppo_vs_heuristic).")


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
