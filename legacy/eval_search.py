r"""How much does searching actually help?

Plays four versions of the bot against the same fixed opponent for the same number of
games, and prints their win rates side by side:

  - the raw network, just picking a move
  - the one-turn searcher, scoring every option with the damage model
  - that, plus credit for status, setup, hazards and healing
  - the two-turn searcher, playing the game out on a small simulator

All four use the same trained network, so the differences are down to the searching.

Run from the project root, with the local server going:
    python -u src\eval_search.py
"""

import asyncio
import os

from sb3_contrib import MaskablePPO

from poke_env import AccountConfiguration
from poke_env.player import SimpleHeuristicsPlayer

from rl_env import N_FEATURES
from search import SearchPlayer
from heuristic_search import HeuristicSearchPlayer
from deep_search import DeepSearchPlayer
from opponent import ModelPlayer

BATTLE_FORMAT = "gen9randombattle"
N_BATTLES = 300


def pick_model_path():
    # Best available first. Putting the search on top of the strongest network means we're
    # measuring the whole stack rather than an artificially weak one.
    candidates = (
        f"ppo_v3_anneal_selfplay_best_obs{N_FEATURES}.zip",   # the final one
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

    # Same network and same prior as the plain searcher, so any difference here is down to
    # the extra scoring and nothing else.
    heur = HeuristicSearchPlayer(
        model=model,
        account_configuration=AccountConfiguration.generate("heurbot", rand=True),
        battle_format=BATTLE_FORMAT,
    )
    heur_wr = await bench(heur, "heur", N_BATTLES)
    print(f"heuristic+     vs heuristic: {heur_wr:.0%}", flush=True)

    # And the two-turn version, which is the test of whether depth is worth anything.
    deep = DeepSearchPlayer(
        model=model,
        account_configuration=AccountConfiguration.generate("deepbot", rand=True),
        battle_format=BATTLE_FORMAT,
    )
    deep_wr = await bench(deep, "deep", N_BATTLES)
    print(f"deep (2-ply)   vs heuristic: {deep_wr:.0%}", flush=True)

    print(f"\nlift, search    vs raw:       {(search_wr - raw_wr) * 100:+.0f} points", flush=True)
    print(f"lift, heuristic+ vs raw:      {(heur_wr - raw_wr) * 100:+.0f} points", flush=True)
    print(f"lift, deep 2-ply vs raw:      {(deep_wr - raw_wr) * 100:+.0f} points", flush=True)
    print(f"deep 2-ply vs search:         {(deep_wr - search_wr) * 100:+.0f} points", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
