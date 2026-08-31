"""Gen 9 OU: everything the search needs that Random Battle gets for free.

Random Battle hands the bot two things OU does not. The server ships the exact generator
sheet, so the opponent's set is drawn from a KNOWN distribution; and the team is generated,
so there is nothing to bring and nothing to choose at preview. OU replaces both:

    usage.py        the set prior, from public Smogon data (curated sets + usage marginals)
    knowledge.py    OUKnowledge -- ability prediction and stat estimation over that prior
    adapter.py      observed battle -> determinized poke-engine State, OU rules
    teams.py        our own team: load a Showdown export, hand it to poke-env
    teampreview.py  the one decision Random Battle never has to make -- who leads
    metagame.py     the four above, bundled as the object EnginePlayer plugs in

The search itself (laplace.agent.engine_search), the value features and every guard are
format-agnostic and are reused unchanged; the seam they plug into is
laplace.agent.metagame.Metagame.
"""
