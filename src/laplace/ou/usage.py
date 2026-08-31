r"""The Gen 9 OU set prior: what the opponent is likely to be running.

This is the OU answer to `poke_engine_adapter._JointSets` / `_SetSheet` / `_StatsFeed`, and
it keeps the same two-tier shape for the same reason. Complete sets first, marginals only as
a fallback:

  CuratedSets   Smogon analysis sets. One draw yields moves, item, ability, nature, EVs and
                Tera that were written to go together. This is the OU stand-in for the
                counted joint sets, and it exists for the lesson that produced those: when
                the randbats bot composed a set from independent marginals it drew
                incoherent Pokemon (the Scale-Shot-into-Fairy pathology) and lost its
                compute-matched benchmark. ~108 species.

  UsageStats    Smogon chaos marginals -- items, abilities, moves, spreads, Tera, teammates,
                each weighted by how often the ladder plays it. Covers ~208 species, i.e.
                everything the curated file misses, and takes over whenever the evidence has
                ruled out every curated set. It also owns the STAT estimator, because in OU
                a Pokemon's stats are a choice: Great Tusk's Speed is 0 EVs or 252 EVs
                depending on the set, and no single number is right for both.

Both files are built by `python -m laplace.cli.fetch_ou_data` and committed, so nothing here
touches the network. A missing or unreadable file degrades to "no prior" rather than
raising: the bot still plays, it just guesses worse, which is the same failure mode the
randbats loaders were written for.
"""

import json
import random

from poke_env.data import GenData, to_id_str

from laplace import paths

GEN = 9
_GEN_DATA = GenData.from_gen(GEN)
_POKEDEX = _GEN_DATA.pokedex
_NATURES = _GEN_DATA.natures

LEVEL = 100          # OU is level 100, always. No per-species level table to consult.
DEFAULT_IV = 31

# Item ids the chaos feed uses for "the slot is empty", translated to the engine's spelling
# at sample time rather than in the data file, so the committed file stays a faithful copy
# of the source.
_NO_ITEM = {"nothing", "none", ""}

_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")


def _weighted(dist):
    """Sample a key from {key: probability}. None if the distribution is empty.

    Same contract as the randbats _StatsFeed helper, deliberately: the two feeds are read
    by parallel code paths and a difference here would be a difference nobody would look
    for."""
    if not dist:
        return None
    total = sum(dist.values())
    if total <= 0:
        return None
    r = random.random() * total
    acc = 0.0
    for key, p in dist.items():
        acc += p
        if r <= acc:
            return key
    return next(iter(dist))


def _load(path, key="data"):
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}, {}
    meta = {k: v for k, v in raw.items() if k.startswith("_")}
    return raw.get(key) or {}, meta


# --- marginals ----------------------------------------------------------------------------

class UsageStats:
    """Per-species marginals from the Smogon chaos feed."""

    def __init__(self, path=paths.OU_USAGE):
        self.by_species, self.meta = _load(path)
        self._stat_cache = {}
        self._speed_cache = {}
        # Species ordered by usage, for the one case where we must invent a bench slot with
        # nothing to go on (a format or a game with no team preview).
        self._ranked = sorted(self.by_species, key=lambda s: -self.by_species[s]["usage"])

    @property
    def loaded(self):
        return bool(self.by_species)

    def entry(self, species_id):
        return self.by_species.get(species_id)

    def resolve(self, mon_or_id, base_id=None):
        """The key this species is filed under, following the same fallback the randbats
        adapter uses for mid-battle formes: exact species first, then base species.

        Ogerpon-Wellspring-Tera, Palafin-Hero and Mimikyu-Busted are all species that only
        exist once a battle is under way and appear in no usage table."""
        sid = mon_or_id if isinstance(mon_or_id, str) else to_id_str(mon_or_id.species)
        if sid in self.by_species:
            return sid
        if base_id is None and not isinstance(mon_or_id, str):
            base_id = to_id_str(getattr(mon_or_id, "base_species", "") or "")
        return base_id if base_id in self.by_species else None

    # --- single-field draws ---------------------------------------------------

    def item(self, species_id, exclude=()):
        e = self.entry(species_id)
        if not e:
            return None
        dist = {k: v for k, v in e["items"].items() if k not in exclude}
        return _weighted(dist)

    def ability(self, species_id):
        e = self.entry(species_id)
        return _weighted(e["abilities"]) if e else None

    def abilities(self, species_id):
        """{ability_id: probability}, for the guards' certainty test."""
        e = self.entry(species_id)
        return dict(e["abilities"]) if e else {}

    def tera(self, species_id):
        e = self.entry(species_id)
        return _weighted(e["tera"]) if e else None

    def spread(self, species_id):
        """(nature, [ev x6]) sampled by observed weight, or None."""
        e = self.entry(species_id)
        if not e or not e["spreads"]:
            return None
        spreads = e["spreads"]
        pick = random.choices(spreads, weights=[s[2] for s in spreads])[0]
        return pick[0], list(pick[1])

    def moves(self, species_id, revealed=(), k=4):
        """`revealed` topped up to k with a weighted draw from the rest of the movepool.

        Without replacement and weighted: OU move marginals are steep (Great Tusk's top four
        carry ~78% of its mass), so sampling with replacement would collapse most draws onto
        the same two moves and hide the fifth-slot tech that actually decides games."""
        chosen = list(dict.fromkeys(revealed))[:k]
        e = self.entry(species_id)
        if not e:
            return chosen
        pool = {m: p for m, p in e["moves"].items() if m not in chosen and p > 0}
        while len(chosen) < k and pool:
            pick = _weighted(pool)
            if pick is None:
                break
            pool.pop(pick, None)
            chosen.append(pick)
        return chosen

    def teammate(self, species_id, exclude=()):
        """A likely teammate of `species_id`, for filling a bench slot with no preview."""
        e = self.entry(species_id)
        if not e:
            return None
        dist = {k: v for k, v in e["teammates"].items()
                if k not in exclude and k in self.by_species}
        return _weighted(dist)

    def popular(self, exclude=()):
        """The most-used species we have not placed yet. Last-resort bench filler."""
        for sid in self._ranked:
            if sid not in exclude:
                return sid
        return None

    # --- stats ----------------------------------------------------------------

    def expected_stats(self, species_id, dex_id=None):
        """{stat: value} averaged over the species' observed spreads, cached.

        The MEAN, not the mode, and computed on the finished stat rather than on the EVs:
        it is the number that minimises squared error in a damage estimate, which is the
        only thing this feeds. A species with no usage entry gets None and the caller falls
        back to the neutral 85-EV convention, which is what the randbats estimator has
        always used."""
        key = (species_id, dex_id)
        if key in self._stat_cache:
            return self._stat_cache[key]
        e = self.entry(species_id)
        dex = _POKEDEX.get(dex_id or species_id) or {}
        base = dex.get("baseStats")
        if not e or not e["spreads"] or not base:
            self._stat_cache[key] = None
            return None
        acc = {k: 0.0 for k in _STAT_KEYS}
        total = 0.0
        for nature, evs, w in e["spreads"]:
            total += w
            for stat in _STAT_KEYS:
                acc[stat] += w * raw_stat(base, stat, evs[_STAT_KEYS.index(stat)],
                                          DEFAULT_IV, LEVEL, nature)
        out = {k: int(round(v / total)) for k, v in acc.items()} if total > 0 else None
        self._stat_cache[key] = out
        return out

    def speed_bounds(self, species_id, dex_id=None):
        """(slowest, fastest) plausible unboosted Speed for this species, cached.

        This is what makes the Choice Scarf verdict safe in OU. The randbats version can
        compare against a single number because randbats spreads are fixed; here the same
        Pokemon legitimately ranges from 0 EVs with a minus-Speed nature to 252 EVs with a
        plus one -- a ~1.6x spread -- so a single estimate would read half the metagame as
        Scarfed. Bounding it means the verdict only fires when the observation is
        impossible at ANY legal spread, which is the standard engine_search._infer_scarf
        was written to.

        The bounds are the format's, not the species' observed sets: a set nobody has
        registered on the ladder is still legal, and a false Scarf verdict pins choicescarf
        into every determinized world for the rest of the game."""
        key = (species_id, dex_id)
        if key in self._speed_cache:
            return self._speed_cache[key]
        dex = _POKEDEX.get(dex_id or species_id) or {}
        base = dex.get("baseStats")
        if not base:
            self._speed_cache[key] = None
            return None
        lo = raw_stat(base, "spe", 0, DEFAULT_IV, LEVEL, "brave")     # 0 EVs, -Speed
        hi = raw_stat(base, "spe", 252, DEFAULT_IV, LEVEL, "jolly")   # 252 EVs, +Speed
        self._speed_cache[key] = (lo, hi)
        return lo, hi


def raw_stat(base_stats, stat, ev, iv, level, nature):
    """One stat from base / EV / IV / level / nature, by the gen-3+ formula.

    poke_env.stats.compute_raw_stats does all six at once from a dex lookup; this computes
    one from base stats we already hold, which is what both callers here actually want and
    what lets `expected_stats` average over a dozen spreads without a dex hit each time."""
    b = base_stats.get(stat, 80)
    if stat == "hp":
        if b == 1:                       # Shedinja
            return 1
        return (2 * b + iv + ev // 4) * level // 100 + level + 10
    mult = (_NATURES.get(nature) or {}).get(stat, 1.0)
    return int(((2 * b + iv + ev // 4) * level // 100 + 5) * mult)


# --- curated sets -------------------------------------------------------------------------

class CuratedSets:
    """Smogon analysis sets: complete, correlated, and small enough to filter exhaustively.

    Two things this is NOT, and both cost something the randbats joint sets do not:

    It is not exhaustive. 108 species, 1.9 sets each, written by analysts to describe a tier
    rather than to enumerate it. Measured against the same month's usage feed, the sets can
    express only 84.5% of the item mass their species actually run (median 87%), and for
    some heavily-played ones far less -- Rocky Helmet is 34% of real Zapdos and appears in
    no Zapdos set at all. Whatever a set cannot express, this sampler can never draw. That
    is why the adapter blends it with the marginals rather than preferring it outright.

    It is not counted. The randbats joint file carries an observed count per set, so
    sampling by weight reproduces the generator. These carry nothing, and drawing uniformly
    among them is its own distortion: Gholdengo's four sets made Choice Scarf 25% of worlds
    when the ladder runs it 39%. `_weights` fixes that from the usage feed -- see there."""

    def __init__(self, path=paths.OU_SETS, usage=None):
        self.by_species, self.meta = _load(path)
        self._usage = usage
        self._weight_cache = {}

    @property
    def loaded(self):
        return bool(self.by_species)

    def has(self, species_id):
        return species_id in self.by_species

    def movepool(self, species_id):
        """Every move any curated set for this species can run. None if it has no sets."""
        sets = self.by_species.get(species_id)
        if not sets:
            return None
        pool = set()
        for s in sets:
            for slot in s["moves"]:
                pool |= set(slot)
        return pool

    def abilities(self, species_id):
        """{ability_id: probability} across the species' sets, uniform over sets."""
        sets = self.by_species.get(species_id)
        if not sets:
            return {}
        probs = {}
        for s in sets:
            abilities = s["abilities"]
            if not abilities:
                continue
            w = 1.0 / (len(sets) * len(abilities))
            for a in abilities:
                probs[a] = probs.get(a, 0.0) + w
        return probs

    def _weights(self, species_id):
        """How likely each of a species' sets is to be the one in front of us.

        The sets carry no counts, so the weight is borrowed from the usage feed by
        APPORTIONING each item's observed mass across the sets that can run it. A set's
        weight is the mass apportioned to it; an item only one set runs hands that set all
        of its mass, an item three sets share is split three ways.

        Two properties make this the right shape rather than just a heuristic. If the sets
        run disjoint items it reproduces the item marginal exactly. And mass belonging to
        items NO set can express is simply dropped rather than redistributed -- which is
        honest, because it is mass this sampler was never able to reach, and the adapter's
        blend with the raw marginals is what covers it.

        Uniform weights whenever the usage feed has nothing to say, which is the old
        behaviour and no worse than it."""
        cached = self._weight_cache.get(species_id)
        if cached is not None:
            return cached
        sets = self.by_species.get(species_id) or []
        entry = self._usage.entry(species_id) if self._usage else None
        weights = None
        if entry and sets:
            items = entry.get("items") or {}
            owners = {}
            for i, s in enumerate(sets):
                for item in s["items"]:
                    owners.setdefault(item, []).append(i)
            acc = [0.0] * len(sets)
            for item, holders in owners.items():
                mass = items.get(item, 0.0)
                if mass <= 0:
                    continue
                for i in holders:
                    acc[i] += mass / len(holders)
            if sum(acc) > 0:
                weights = acc
        self._weight_cache[species_id] = weights
        return weights

    def weights_for(self, species_id, subset):
        """The weights of `subset` (a filtered list of this species' sets), or None."""
        weights = self._weights(species_id)
        if not weights:
            return None
        index = {id(s): i for i, s in enumerate(self.by_species.get(species_id) or [])}
        picked = [weights[index[id(s)]] for s in subset if id(s) in index]
        if len(picked) != len(subset) or sum(picked) <= 0:
            return None
        return picked

    def sample(self, species_id, revealed_moves=(), item=None, ability=None,
               item_exclude=(), force_scarf=False):
        """One complete set consistent with the evidence, or None if there are no sets.

        The filter is the same ladder that never empties as _JointSets.sample: each
        constraint narrows the pool only if something survives it. The curated file is a
        snapshot of a metagame that moves, so a constraint it cannot satisfy has to degrade
        to 'less informed', never to 'no set' -- an empty return here drops the search to
        the marginal path for that world, which is strictly worse information."""
        sets = self.by_species.get(species_id)
        if not sets:
            return None
        revealed = set(revealed_moves)

        def narrow(pool, pred):
            kept = [s for s in pool if pred(s)]
            return kept or pool

        sets = narrow(sets, lambda s: revealed <= _all_moves(s))
        if item:
            sets = narrow(sets, lambda s: item in s["items"])
        elif force_scarf:
            sets = narrow(sets, lambda s: "choicescarf" in s["items"])
        elif item_exclude:
            sets = narrow(sets, lambda s: any(i not in item_exclude for i in s["items"]))
        if ability:
            sets = narrow(sets, lambda s: ability in s["abilities"])

        weights = self.weights_for(species_id, sets)
        s = random.choices(sets, weights=weights)[0] if weights else random.choice(sets)
        evs = random.choice(s["evs"]) if s["evs"] else {}
        ivs = random.choice(s["ivs"]) if s["ivs"] else {}
        return {
            "name": s["name"],
            "moves": _resolve_moves(s["moves"], revealed),
            "item": _resolve_item(s["items"], item, item_exclude, force_scarf),
            "ability": ability or (random.choice(s["abilities"]) if s["abilities"] else None),
            "nature": random.choice(s["natures"]),
            "evs": {k: int(v) for k, v in evs.items()},
            "ivs": {k: int(v) for k, v in ivs.items()},
            "tera": random.choice(s["tera"]) if s["tera"] else None,
        }


def _all_moves(s):
    pool = set()
    for slot in s["moves"]:
        pool |= set(slot)
    return pool


def _resolve_item(options, known, exclude, force_scarf):
    """Pick one of a set's item options, honouring the evidence where it can."""
    if known:
        return known
    pool = options
    if force_scarf and "choicescarf" in pool:
        return "choicescarf"
    if exclude:
        pool = [i for i in pool if i not in exclude] or options
    pick = random.choice(pool)
    return "none" if pick in _NO_ITEM else pick


def _resolve_moves(slots, revealed):
    """One move per slot, with the revealed moves placed first.

    Placing a revealed move consumes the SCARCEST slot that offers it. A set writes its
    fixed moves as one-option slots and its choices as multi-option ones, so consuming the
    narrow slot first is what keeps 'Clefable showed Moonblast' from eating the flex slot
    and leaving Stealth Rock to be drawn at random. A revealed move no slot offers is still
    kept -- it is an observation, not a guess -- and simply costs a slot."""
    remaining = [list(slot) for slot in slots]
    chosen = []
    for move in dict.fromkeys(revealed):
        if move in chosen:
            continue
        holders = [i for i, slot in enumerate(remaining) if move in slot]
        if holders:
            remaining.pop(min(holders, key=lambda i: len(remaining[i])))
        chosen.append(move)
    for slot in remaining:
        if len(chosen) >= 4:
            break
        options = [m for m in slot if m not in chosen]
        if options:
            chosen.append(random.choice(options))
    return chosen[:4]


# Loaded once at import, like KNOWLEDGE and SETS on the randbats side. CURATED is given the
# usage feed so it can weight its sets by what the ladder actually plays (CuratedSets._weights).
USAGE = UsageStats()
CURATED = CuratedSets(usage=USAGE)
