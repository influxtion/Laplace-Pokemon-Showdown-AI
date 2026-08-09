r"""Watch a game instead of just counting wins.

eval_search.py plays hundreds of games to get a number. This plays a couple and saves each
one as a replay you can open in a browser and watch turn by turn. By default it uses the
best agent of the era: the network wrapped in the one-turn searcher.

Each replay is a single .html file with the whole battle log inside it. Opening it loads
Showdown's own replay viewer and plays the game back with sprites and animations. The
viewer needs internet, but the game itself is in the file.

Run from the project root, with the local server going:
    python -u src\play.py                       # one game against the heuristic
    python -u src\play.py --battles 3
    python -u src\play.py --raw                 # just the network, no searching
    python -u src\play.py --opponent random
    python -u src\play.py --model ppo_v3_obs215.zip
"""

import argparse
import asyncio
import os

from sb3_contrib import MaskablePPO

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

from eval_search import pick_model_path
from search import SearchPlayer
from heuristic_search import HeuristicSearchPlayer
from deep_search import DeepSearchPlayer
from engine_search import EnginePlayer
from opponent import ModelPlayer

BATTLE_FORMAT = "gen9randombattle"
REPLAY_DIR = "replays"          # where the .html files land

OPPONENTS = {"heuristic": SimpleHeuristicsPlayer, "random": RandomPlayer}


def build_agent(model, agent, debug=False):
    """Build whichever generation of bot we've been asked to watch."""
    if agent == "engine":
        return EnginePlayer(
            debug=debug,
            account_configuration=AccountConfiguration.generate("enginebot", rand=True),
            battle_format=BATTLE_FORMAT,
        )
    if agent == "deep":
        return DeepSearchPlayer(
            model=model, debug=debug,
            account_configuration=AccountConfiguration.generate("deepbot", rand=True),
            battle_format=BATTLE_FORMAT,
        )
    if agent == "heuristic":
        return HeuristicSearchPlayer(
            model=model, debug=debug,
            account_configuration=AccountConfiguration.generate("heurbot", rand=True),
            battle_format=BATTLE_FORMAT,
        )
    if agent == "search":
        return SearchPlayer(
            model=model,
            account_configuration=AccountConfiguration.generate("searchbot", rand=True),
            battle_format=BATTLE_FORMAT,
        )
    return ModelPlayer(
        model=model, deterministic=True,
        account_configuration=AccountConfiguration.generate("rawpolicy", rand=True),
        battle_format=BATTLE_FORMAT,
    )


def save_replays(agent, opp_label):
    """Save every game just played as a replay file, and say how they went."""
    os.makedirs(REPLAY_DIR, exist_ok=True)
    wins = 0
    for i, (tag, battle) in enumerate(agent.battles.items(), start=1):
        result = "WON " if battle.won else "lost"
        wins += 1 if battle.won else 0
        # The battle tag is already a perfectly good filename.
        path = os.path.join(REPLAY_DIR, f"{tag}.html")
        agent.save_replay(tag, path)
        print(f"  Battle {i}: {result} in {battle.turn} turns  ->  {path}", flush=True)
    print(f"\nWon {wins}/{len(agent.battles)} vs {opp_label}. "
          f"Open the .html file(s) in a browser to watch.", flush=True)


async def main():
    parser = argparse.ArgumentParser(description="Watch one of the bots play a game.")
    parser.add_argument("--battles", type=int, default=1, help="how many games to play")
    parser.add_argument("--opponent", choices=OPPONENTS, default="heuristic",
                        help="who to play against")
    parser.add_argument("--raw", action="store_true",
                        help="just the trained network, with no searching on top")
    parser.add_argument("--heuristic", action="store_true",
                        help="the one-turn searcher plus status, setup, hazards and healing")
    parser.add_argument("--deep", action="store_true",
                        help="the two-turn searcher")
    parser.add_argument("--engine", action="store_true",
                        help="the real engine bot")
    parser.add_argument("--debug", action="store_true",
                        help="print the top-scoring options each turn")
    parser.add_argument("--model", default=None,
                        help="a specific saved model to load")
    args = parser.parse_args()

    which = ("engine" if args.engine else "deep" if args.deep else "heuristic" if args.heuristic
             else "raw" if args.raw else "search")
    labels = {"engine": "poke-engine MCTS bot", "deep": "2-ply deep searcher",
              "heuristic": "heuristic+ agent", "raw": "raw policy", "search": "search agent"}
    # The engine bot needs no trained network; everything else does.
    model = None
    if which != "engine":
        model_path = args.model or pick_model_path()
        print(f"Loading {model_path} -> playing {args.battles} battle(s) as the {labels[which]} "
              f"vs {args.opponent}.", flush=True)
        model = MaskablePPO.load(model_path)
    else:
        print(f"Playing {args.battles} battle(s) as the {labels[which]} vs {args.opponent}.",
              flush=True)

    agent = build_agent(model, which, debug=args.debug)
    opponent = OPPONENTS[args.opponent](
        account_configuration=AccountConfiguration.generate("opp", rand=True),
        battle_format=BATTLE_FORMAT,
    )

    await agent.battle_against(opponent, n_battles=args.battles)
    save_replays(agent, args.opponent)


if __name__ == "__main__":
    asyncio.run(main())
