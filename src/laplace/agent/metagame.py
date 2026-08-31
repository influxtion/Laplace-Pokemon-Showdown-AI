r"""The seam between the search and the format it is playing.

`engine_search.EnginePlayer` is a general algorithm: determinize the hidden state, MCTS each
world, pool, guard, mix. Nothing in that depends on Random Battle -- but until this module
existed the format was wired in by direct imports, so the bot could only ever play one.

A Metagame supplies the three things the search cannot derive from the protocol:

    build_state(battle, **kwargs)   one determinized poke-engine State. The kwargs are the
                                    observations the search accumulates across turns
                                    (Choice-lock history, Scarf and Boots verdicts, Wish and
                                    Future Sight timers); every format gets the same ones and
                                    is free to use or ignore them.

    knowledge                       the set prior the deterministic guards consult. Three
                                    methods, no more (see the Knowledge note below).

    teampreview(battle, player)     the '/team ...' order, or None to let poke-env pick.
                                    Random Battle never sees a preview; OU always does.

Two instances exist: RANDBATS here, which is the shipped Gen 9 Random Battle bot and is what
`EnginePlayer` still uses when nobody says otherwise, and `laplace.ou.metagame.OU`.

--- the Knowledge protocol -------------------------------------------------------------

`knowledge` is duck-typed rather than an ABC, because both implementations predate this file
and an ABC would only add a place for them to disagree. It must provide:

    predicted_abilities(mon) -> {ability_id: probability}
        Consulted by the absorb guard, which demotes a move outright when an absorbing
        ability totals >= 0.99, and by the priority-ability check that decides whether a
        turn-order observation is interpretable at all. So the contract is about the
        MEANING of 1.0: it must mean "certain", not "unobserved".

    estimate_stat(mon, key) -> int
        A point estimate of one stat. Exact for our own side, since the request gives us the
        real numbers.

    speed_bounds(mon) -> (slowest, fastest)
        The unboosted Speed range this Pokemon could legally have. Random Battle collapses
        it to a point (spreads are fixed); OU cannot, and the Choice Scarf verdict is only
        safe because it is told so.
"""

from laplace.agent import knowledge as _knowledge
from laplace.agent.poke_engine_adapter import build_state as _randbats_build_state


class Metagame:
    """One format's worth of prior knowledge. Subclass to add a format.

    The base class IS the Random Battle implementation rather than an abstract shell: it is
    the tested one, and making it the default means adding OU could not change what the
    shipped bot does even by accident."""

    name = "gen9randombattle"

    # Whether the format requires us to bring a team. Random Battle generates one; OU does
    # not, and a Player started without one is refused by the server at match time rather
    # than at startup, which is a miserable way to find out.
    needs_team = False

    def __init__(self, knowledge=None):
        self.knowledge = knowledge or _knowledge.KNOWLEDGE

    def build_state(self, battle, **kwargs):
        return _randbats_build_state(battle, **kwargs)

    def teampreview(self, battle, player):
        """The team-preview order, or None to fall back to poke-env's default.

        Takes the player because a preview decision may want the agent's own configuration
        (and because the caller must be able to mark the chosen Pokemon on the battle)."""
        return None

    def __repr__(self):
        return f"<Metagame {self.name}>"


RANDBATS = Metagame()
