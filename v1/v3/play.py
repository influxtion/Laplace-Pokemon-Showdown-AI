r"""Watch the trained agent actually play a battle.

eval_search.py runs hundreds of battles and prints a win-rate number; this is the opposite --
it plays a SMALL number of battles and saves each as a Showdown replay you can open in a
browser and watch move by move. By default it plays the agent we'd actually ship: the trained
policy wrapped in the 1-ply test-time search (search.py). Pass --raw to watch the bare policy
instead.

How "watching" works: a Showdown replay is a self-contained .html file with the full battle
log embedded; opening it loads Showdown's replay viewer (needs internet for the viewer's
script, but the battle itself is in the file) and plays the battle back with sprites, HP bars,
and move animations -- exactly what you'd see on the site.

Run from the project root, with the local Showdown server running:
    python -u v1\v3\play.py                       # 1 battle, search agent vs the heuristic
    python -u v1\v3\play.py --battles 3           # 3 battles
    python -u v1\v3\play.py --raw                 # bare policy, no search
    python -u v1\v3\play.py --opponent random     # vs RandomPlayer instead of the heuristic
    python -u v1\v3\play.py --model ppo_v3_obs215.zip   # a specific model file
"""

import argparse
import asyncio
import os
import sys

# This file lives in v1/v3/. Put v1/ (rl_env, knowledge) and v1/selfplay/ (ModelPlayer) on the
# path so the bare imports below resolve when run as a script.
_V1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _V1)
sys.path.insert(0, os.path.join(_V1, "selfplay"))

from sb3_contrib import MaskablePPO

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

from eval_search import pick_model_path
from search import SearchPlayer
from opponent import ModelPlayer

BATTLE_FORMAT = "gen9randombattle"
REPLAY_DIR = "replays"          # where the watchable .html files are written (project root)

OPPONENTS = {"heuristic": SimpleHeuristicsPlayer, "random": RandomPlayer}


def build_agent(model, use_search):
    """The agent to watch: the search wrapper (what we ship) or the bare policy (--raw)."""
    if use_search:
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
    """Write every battle the agent just played to a watchable .html and report each result."""
    os.makedirs(REPLAY_DIR, exist_ok=True)
    wins = 0
    for i, (tag, battle) in enumerate(agent.battles.items(), start=1):
        result = "WON " if battle.won else "lost"
        wins += 1 if battle.won else 0
        # tag looks like "battle-gen9randombattle-NN"; it's already a safe filename.
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
    parser.add_argument("--model", default=None,
                        help="model .zip to load (default: best available, see eval_search)")
    args = parser.parse_args()

    model_path = args.model or pick_model_path()
    label = "raw policy" if args.raw else "search agent"
    print(f"Loading {model_path} -> playing {args.battles} battle(s) as the {label} "
          f"vs {args.opponent}.", flush=True)
    model = MaskablePPO.load(model_path)

    agent = build_agent(model, use_search=not args.raw)
    opponent = OPPONENTS[args.opponent](
        account_configuration=AccountConfiguration.generate("opp", rand=True),
        battle_format=BATTLE_FORMAT,
    )

    await agent.battle_against(opponent, n_battles=args.battles)
    save_replays(agent, args.opponent)


if __name__ == "__main__":
    asyncio.run(main())
