r"""Play the bot on the Gen 9 OU ladder.

This is `laplace.cli.ladder` with the OU defaults filled in -- same harness, same
reconnect/archive/Elo machinery, same search. Every flag that file documents works here
too; the only difference is that --format defaults to gen9ou and --team is required.

    python -m laplace.cli.ladder_ou --team balance                 # 10 ranked OU games
    python -m laplace.cli.ladder_ou --team balance --battles 40
    python -m laplace.cli.ladder_ou --team balance --lead 1        # fixed lead
    python -m laplace.cli.ladder_ou --team balance --mode challenge --opponent SomeUser

Before the first run:

  1. Build the set priors (once, and again whenever the tier shifts):
         python -m laplace.cli.fetch_ou_data
  2. Put a team in `teams/` as a Showdown export, e.g. teams/balance.txt. Copy it straight
     out of the teambuilder with Import/Export. --team takes the file stem.

Games are archived to replays/ladder_ou/ -- a different directory and a different Elo
record from the Random Battle runs, so the two never contaminate each other's history.

What is DIFFERENT about OU, and what to expect:

  * Team preview hands us the opponent's six before turn 1, which removes the largest
    source of noise in the Random Battle determinizer -- it never has to invent a bench.
  * In exchange, the SET behind each of those six is a human's choice rather than a
    generator's, so the prior is weaker: usage statistics say what is popular, not what
    this opponent brought. Expect worse hidden-information inference and better endgame
    reasoning than the Random Battle bot.
  * The value net is OFF here. It was trained on Random Battle self-play states and has no
    standing to re-rank OU positions; see FormatProfile.use_value_net in ladder.py.
  * The team is now part of the bot. A result is a statement about the search AND the team
    together, and swapping teams invalidates the comparison as thoroughly as swapping
    engines does.
"""

import asyncio

from laplace.cli import ladder


def run():
    """Sync entry point: the `laplace-ou` console script, and `python -m`."""
    try:
        asyncio.run(ladder.main(default_format="gen9ou"))
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)


if __name__ == "__main__":
    run()
