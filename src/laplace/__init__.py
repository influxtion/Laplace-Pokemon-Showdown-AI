"""Laplace -- a Gen 9 Pokemon Showdown bot: Random Battle and OU.

The search is one algorithm and knows nothing about either format. What a format has to
supply -- how to determinize a position, what the set prior says, what to do at team
preview -- is a `laplace.agent.metagame.Metagame`, and there are two.

Layout:
    laplace.agent     the search, plus the Gen 9 Random Battle metagame
    laplace.ou        the Gen 9 OU metagame: set prior, determinizer, teams, team preview
    laplace.value     the learned value head: features and the network
    laplace.analysis  live commentary and post-hoc replay mining
    laplace.cli       entry points (python -m laplace.cli.<name>)
"""

__version__ = "1.0.0"
