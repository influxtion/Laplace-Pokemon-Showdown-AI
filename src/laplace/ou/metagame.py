r"""Gen 9 OU, as the object `EnginePlayer` plugs into.

Everything specific to the tier is assembled here and nowhere else: the determinizer, the
set prior the guards consult, and the team-preview decision. The search, the guards, the
value features and the whole ladder harness are shared with the Random Battle bot and see
only this object.

    from laplace.ou.metagame import OU
    EnginePlayer(..., battle_format=OU.name, team=team_text, metagame=OU)
"""

from laplace.agent.metagame import Metagame
from laplace.ou import adapter, teampreview
from laplace.ou.knowledge import KNOWLEDGE


class OUMetagame(Metagame):
    """Gen 9 OU. Team preview, chosen sets, chosen spreads, and a team we bring ourselves."""

    name = "gen9ou"
    needs_team = True

    def __init__(self, fixed_lead=None, knowledge=None, curated_frac=None):
        super().__init__(knowledge=knowledge or KNOWLEDGE)
        # 1-based slot to always lead with, or None to score the matchup every game. Some
        # teams have exactly one lead and saying so beats rediscovering it.
        self.fixed_lead = fixed_lead
        # Share of determinized worlds drawn from the curated sets rather than the usage
        # marginals; None takes adapter.CURATED_FRACTION. The one knob worth turning here,
        # and the reason it is a knob is in that constant's comment.
        self.curated_frac = curated_frac

    def build_state(self, battle, **kwargs):
        if self.curated_frac is not None and kwargs.get("use_joint", True):
            kwargs["use_joint"] = self.curated_frac
        return adapter.build_state(battle, **kwargs)

    def teampreview(self, battle, player):
        return teampreview.choose_order(battle, fixed_lead=self.fixed_lead)

    def ready(self):
        """(ok, message). Whether the priors are actually on disk.

        The bot runs without them -- every loader degrades to 'no prior' rather than raising
        -- but it runs BADLY, and a silent 60% drop in strength is the kind of thing that
        gets diagnosed as a bug in the search. Checked once at startup instead."""
        usage = self.knowledge.usage
        curated = self.knowledge.curated
        if usage.loaded and curated.loaded:
            meta = usage.meta
            return True, (f"OU priors: {len(usage.by_species)} species from "
                          f"{meta.get('_month')} @ {meta.get('_cutoff')}, "
                          f"{len(curated.by_species)} with curated sets")
        missing = [name for name, feed in (("usage statistics", usage),
                                           ("curated sets", curated)) if not feed.loaded]
        return False, (f"OU priors MISSING ({', '.join(missing)}) -- the opponent's sets "
                       f"will be guessed from revealed moves only. "
                       f"Run: python -m laplace.cli.fetch_ou_data")


OU = OUMetagame()
