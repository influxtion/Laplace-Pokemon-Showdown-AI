"""Put the rule-based bot up against a random player and see how it does.

You'll need the local server running first, in another terminal, from `server/`:
    node pokemon-showdown start --no-security

Then:
    python run_battle.py
"""

import asyncio

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import RandomPlayer

from heuristic_bot import MaxDamagePlayer

# The more games, the more you can trust the number.
N_BATTLES = 100

# Random Battle hands both sides a team, so we don't have to build any.
BATTLE_FORMAT = "gen9randombattle"


async def main():
    heuristic = MaxDamagePlayer(
        account_configuration=AccountConfiguration("HeuristicBot", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=BATTLE_FORMAT,
    )
    # Picks moves at random. If our rules are worth anything, this should be a slaughter.
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
