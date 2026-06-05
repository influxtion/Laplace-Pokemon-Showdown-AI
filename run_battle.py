"""Run the heuristic bot against a baseline and report its win rate.

Prerequisite: the local Showdown server must be running. In a separate
terminal, from the `server/` folder, run:

    node pokemon-showdown start --no-security

Then run this script:

    python run_battle.py
"""

import asyncio

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import RandomPlayer

from heuristic_bot import MaxDamagePlayer

# How many battles to play. More = more reliable win-rate estimate.
N_BATTLES = 100

# "Random battles" hand each side a random legal team, so we can test the bot
# without building teams yet. gen9 = the current generation.
BATTLE_FORMAT = "gen9randombattle"


async def main():
    # Our rule-based bot.
    heuristic = MaxDamagePlayer(
        account_configuration=AccountConfiguration("HeuristicBot", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=BATTLE_FORMAT,
    )
    # The baseline: a bot that picks legal moves completely at random.
    # If our heuristic is doing anything useful, it should crush this.
    baseline = RandomPlayer(
        account_configuration=AccountConfiguration("RandomBot", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=BATTLE_FORMAT,
    )

    await heuristic.battle_against(baseline, n_battles=N_BATTLES)

    won = heuristic.n_won_battles
    print(f"\nHeuristic bot won {won} / {N_BATTLES} battles ({won / N_BATTLES:.0%}) vs RandomPlayer")


if __name__ == "__main__":
    asyncio.run(main())
