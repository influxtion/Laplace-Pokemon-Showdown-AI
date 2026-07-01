r"""Watch the trained agent play a battle.

Where eval_search.py runs hundreds of battles for a win-rate number, this plays a few and
saves each as a Showdown replay you can open in a browser and watch move by move. By default
it plays the agent we'd ship: the policy wrapped in the 1-ply search (search.py). Pass --raw
for the bare policy.

A replay is a self-contained .html with the full battle log embedded; opening it loads
Showdown's replay viewer (needs internet for the viewer script, but the battle is in the file)
and plays back with sprites, HP bars, and animations.

Run from the project root, with the local Showdown server running:
    python -u src\play.py                       # 1 battle, search agent vs the heuristic
    python -u src\play.py --battles 3           # 3 battles
    python -u src\play.py --raw                 # bare policy, no search
    python -u src\play.py --opponent random     # vs RandomPlayer instead of the heuristic
    python -u src\play.py --model ppo_v3_obs215.zip   # a specific model file
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
REPLAY_DIR = "replays"          # where the .html replays are written (project root)

OPPONENTS = {"heuristic": SimpleHeuristicsPlayer, "random": RandomPlayer}


def build_agent(model, agent, debug=False):
    """The agent to watch: the poke-engine MCTS bot (--engine), the 1-ply searcher (default),
    heuristic+ (--heuristic), the 2-ply deep searcher (--deep), or the bare policy (--raw)."""
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
    """Write each battle the agent just played to an .html replay and report the result."""
    os.makedirs(REPLAY_DIR, exist_ok=True)
    wins = 0
    for i, (tag, battle) in enumerate(agent.battles.items(), start=1):
        result = "WON " if battle.won else "lost"
        wins += 1 if battle.won else 0
        # tag is like "battle-gen9randombattle-NN", already a safe filename.
        path = os.path.join(REPLAY_DIR, f"{tag}.html")
        agent.save_replay(tag, path)
        print(f"  Battle {i}: {result} in {battle.turn} turns  ->  {path}", flush=True)
    print(f"\nWon {wins}/{len(agent.battles)} vs {opp_label}. "
          f"Open the .html file(s) in a browser to watch.", flush=True)


async def main():
    parser = argparse.ArgumentParser(description="Watch the trained agent play a battle.")
    parser.add_argument("--battles", type=int, default=1, help="how many battles to play")
    parser.add_argument("--opponent", choices=OPPONENTS, default="heuristic",
                        help="who to play against")
    parser.add_argument("--raw", action="store_true",
                        help="use the bare policy instead of the 1-ply search wrapper")
    parser.add_argument("--heuristic", action="store_true",
                        help="use heuristic+ (SearchPlayer + status/setup/hazard/recovery)")
    parser.add_argument("--deep", action="store_true",
                        help="use the 2-ply deep searcher (deep_search.py)")
    parser.add_argument("--engine", action="store_true",
                        help="use the poke-engine MCTS bot (engine_search.py)")
    parser.add_argument("--debug", action="store_true",
                        help="(engine/heuristic/deep only) print the top scored actions each turn")
    parser.add_argument("--model", default=None,
                        help="model .zip to load (default: best available, see eval_search)")
    args = parser.parse_args()

    which = ("engine" if args.engine else "deep" if args.deep else "heuristic" if args.heuristic
             else "raw" if args.raw else "search")
    labels = {"engine": "poke-engine MCTS bot", "deep": "2-ply deep searcher",
              "heuristic": "heuristic+ agent", "raw": "raw policy", "search": "search agent"}
    # The engine agent is model-free; the others load a trained policy.
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
