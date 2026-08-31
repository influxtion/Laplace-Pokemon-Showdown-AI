r"""OU ability prediction and stat estimation -- the OU half of `agent.knowledge`.

Same three questions the guards ask in Random Battle, answered from a different prior:

  predicted_abilities   what ability is the opponent holding? The absorb guard vetoes a
                        move on this, so it has to be CERTAIN before it says so.
  estimate_stat         how hard does it hit, how hard is it to hit? Feeds the damage
                        tiebreak and the Wish estimate.
  speed_bounds          how fast could it possibly be? Feeds the Choice Scarf verdict.

The interesting difference is that Random Battle spreads are fixed and OU spreads are
chosen. `agent.knowledge._estimate_stat` returns one number because in randbats there is
only one number; here the same Great Tusk is a 0-EV wall or a 252-EV lead depending on the
set, so a single estimate is wrong for both. estimate_stat answers with the usage-weighted
mean (the right point estimate for damage arithmetic) and speed_bounds answers with the
legal range (the right answer for an inference that must not fire on a guess).

Certainty is the other difference. In randbats an ability is certain once every surviving
generated set agrees, because the generator is the ground truth. Usage statistics are not
ground truth -- they say what the ladder PLAYS, not what is legal -- so an ability only
reads as certain here when the species has exactly one, or when the server has shown it.
See predicted_abilities.
"""

from functools import lru_cache

from poke_env.data import GenData, to_id_str

from laplace.agent.knowledge import _estimate_stat
from laplace.ou.usage import CURATED, USAGE, LEVEL

GEN = 9
_GEN_DATA = GenData.from_gen(GEN)
_POKEDEX = _GEN_DATA.pokedex
_LEARNSET = _GEN_DATA.learnset

# Floor on any dex-legal ability, whatever the usage feed says about it. Big enough to hold
# a multi-ability species below the guards' 0.99 certainty bar, small enough that it does
# not distort the ranking of the abilities people actually run.
#
# A FLOOR, not a default for missing keys. That distinction is the whole fix: the feed
# records Jolteon as 99.57% Volt Absorb and Vaporeon as 99.98% Water Absorb, so "only fill
# in abilities the feed never mentions" left both of them above the absorb guard's 0.99 bar
# -- and the guard would then refuse to click an Electric move at a Jolteon that could
# perfectly legally be Quick Feet. Four species reached an unearned veto that way. An
# ability recorded at 0.0002 and an ability recorded at nothing are the same claim about
# legality, and neither is evidence of impossibility.
_UNPLAYED_FLOOR = 0.02


@lru_cache(maxsize=None)
def full_learnset(species_id):
    """Every move a species can legally know in gen 9, its pre-evolutions included.

    The dex files list moves per species and let the simulator walk the evolution chain, so
    reading one species' entry alone reports Clefable as unable to learn Soft-Boiled. The
    only consumer is the Illusion tell, where a false 'this move is impossible' is exactly
    the bug that would matter."""
    moves = set()
    seen = set()
    sid = species_id
    while sid and sid not in seen:
        seen.add(sid)
        entry = (_LEARNSET.get(sid) or {}).get("learnset") or {}
        for move, sources in entry.items():
            if any(str(s).startswith("9") for s in sources):
                moves.add(move)
        sid = to_id_str((_POKEDEX.get(sid) or {}).get("prevo") or "")
    return moves


def _dex_abilities(mon):
    """Ability ids the dex allows for this Pokemon, via poke-env's own lookup."""
    try:
        return [to_id_str(a) for a in (mon.possible_abilities or []) if a]
    except Exception:
        return []


class OUKnowledge:
    """The prior the OU search reasons with. One instance, built at import."""

    def __init__(self, usage=USAGE, curated=CURATED):
        self.usage = usage
        self.curated = curated

    @property
    def loaded(self):
        return self.usage.loaded or self.curated.loaded

    # --- species resolution ---------------------------------------------------

    def key(self, mon):
        """The usage-table key for a Pokemon, or None. Mid-battle formes fall back to the
        base species, exactly as the randbats adapter does for Mimikyu-Busted."""
        return self.usage.resolve(mon)

    @staticmethod
    def dex_key(mon):
        """The pokedex key to read base stats from: the live forme if it exists (a
        Terastallized Ogerpon has different stats to the base one), else the base species."""
        sid = to_id_str(mon.species)
        if sid in _POKEDEX:
            return sid
        base = to_id_str(getattr(mon, "base_species", "") or "")
        return base if base in _POKEDEX else sid

    # --- the Knowledge protocol (see laplace.agent.metagame) -------------------

    def predicted_abilities(self, mon):
        """ability_id -> P(this Pokemon has it).

        1.0 once the server has revealed it, and 1.0 for a species with a single legal
        ability -- those two are the only sources of certainty in OU. Otherwise the usage
        marginals, with every unrecorded but legal ability given _UNPLAYED_FLOOR. That floor
        is the whole point: the absorb guard demotes a move outright at >= 0.99, and reading
        'nobody on the ladder runs Flash Fire on this' as 'it cannot have Flash Fire' would
        hand a confident veto to a statistic that never claimed to be exhaustive."""
        if mon is None:
            return {}
        if mon.ability:
            return {to_id_str(mon.ability): 1.0}
        legal = _dex_abilities(mon)
        if len(legal) == 1:
            return {legal[0]: 1.0}
        sid = self.key(mon)
        probs = {a: p for a, p in self.usage.abilities(sid).items()
                 if not legal or a in legal} if sid else {}
        if not probs and not legal:
            return {}
        for ability in legal:
            probs[ability] = max(probs.get(ability, 0.0), _UNPLAYED_FLOOR)
        total = sum(probs.values()) or 1.0
        return {a: p / total for a, p in probs.items()}

    def estimate_stat(self, mon, key):
        """Real value when we know it (our own side), else the usage-weighted mean.

        Falls through to agent.knowledge._estimate_stat -- the neutral 85-EV convention --
        for a species the usage feed has never seen. That estimate is wrong for OU in
        general, but it is wrong by less than any other constant available, and the
        alternative is no number at all."""
        known = (getattr(mon, "stats", None) or {}).get(key)
        if known:
            return int(known)
        sid = self.key(mon)
        if sid:
            stats = self.usage.expected_stats(sid, self.dex_key(mon))
            if stats and stats.get(key):
                return int(stats[key])
        return int(_estimate_stat(mon, key))

    def speed_bounds(self, mon):
        """(slowest, fastest) unboosted Speed this Pokemon could legally have.

        Collapses to a single point once the real value is known, which is what our own
        side always is. For the opponent it is the format bound rather than anything
        observed -- see UsageStats.speed_bounds for why the observed range would be the
        wrong one to hand a veto."""
        known = (getattr(mon, "stats", None) or {}).get("spe")
        if known:
            return int(known), int(known)
        bounds = self.usage.speed_bounds(self.dex_key(mon))
        if bounds:
            return bounds
        est = int(_estimate_stat(mon, "spe"))
        return est, est

    # --- OU-only extras -------------------------------------------------------

    def movepool(self, mon):
        """Every move this Pokemon could legally know, for the Illusion tell."""
        return full_learnset(self.dex_key(mon))

    def level(self, mon):
        return getattr(mon, "level", None) or LEVEL


KNOWLEDGE = OUKnowledge()
