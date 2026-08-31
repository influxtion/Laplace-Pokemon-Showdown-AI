r"""Build the two OU data files the determinizer runs on, from public Smogon sources.

Random Battle needs no download: the server checkout ships the exact generator sheet, so
the bot knows the true set distribution. OU has no such sheet -- players build their own
teams -- so the prior has to come from what the metagame actually plays. Two sources, and
the split mirrors the Random Battle one exactly (see poke_engine_adapter's _JointSets vs
_SetSheet/_StatsFeed):

  sets_gen9ou.json   COMPLETE, hand-curated Smogon analysis sets (pkmn/smogon mirror).
                     Moves, item, ability, nature, EVs and Tera all come from the same set,
                     so the correlations survive -- a Choice Scarf set is Jolly and carries
                     a pivot move, which independent marginals can never reproduce. This is
                     the preferred sampler. ~110 species.

  usage_gen9ou.json  Smogon chaos usage statistics: per-species MARGINALS for items,
                     abilities, moves, spreads, Tera types and teammates, weighted by how
                     often the ladder actually plays them. Covers everything (~235 species)
                     and is the fallback when a species has no curated set, or when the
                     evidence has eliminated every curated set. Also supplies the stat
                     estimator: OU spreads are chosen, not fixed, so "base + level + 85 EVs"
                     (the Random Battle convention) is simply wrong here.

Both are condensed to the head of each distribution before being written, which is what
keeps a ~30 MB chaos dump down to a committable file without changing any decision the
sampler would make: the tail is individually sub-0.5% and collectively noise.

    python -m laplace.cli.fetch_ou_data                  # latest month, 1825 cutoff
    python -m laplace.cli.fetch_ou_data --month 2026-06
    python -m laplace.cli.fetch_ou_data --cutoff 1695    # mid-ladder instead of top-cut

Cutoff is the ladder rating the sample is drawn from. 1825 is the top cut: it is the
metagame the bot faces once it climbs, and it is the cleanest sample. Drop to 1695 if the
account is going to spend its life below ~1700, where the sets on the other side are
visibly different.

Re-run it when the tier shifts (a suspect test, a home-page metagame change). The files are
committed, so a fresh clone plays without network access.
"""

import argparse
import gzip
import json
import os
import urllib.request
from datetime import date

from poke_env.data import to_id_str

from laplace import paths

CHAOS_URL = "https://www.smogon.com/stats/{month}/chaos/{fmt}-{cutoff}.json.gz"
SETS_URL = "https://pkmn.github.io/smogon/data/sets/{fmt}.json"

FORMAT = "gen9ou"
DEFAULT_CUTOFF = 1825

# How much of each distribution to keep. Measured against the 2026-07 @ 1825 dump, weighted
# by species usage, rather than guessed:
#
#   items    top 10 -> 99.4% of observed mass (worst species 96.0%)
#   moves    top 24 -> 100%  (worst 97.5%)
#   tera     top 10 -> 99.7% (worst 97.2%)
#
# SPREADS ARE DIFFERENT and were the one place a plausible-sounding cutoff was wrong. EVs
# are free-form, so the tail is enormous -- Great Tusk alone has 2647 distinct spreads, and
# no single one holds more than 26% -- and a top-12 cut keeps only 83% of the raw mass.
#
# Raw mass is the wrong measure, though: most of that tail is 252/252/4 vs 248/252/8, which
# no damage roll can tell apart. The measure that matters is whether the ARCHETYPE survives
# -- the nature, and which stats carry real investment. By that measure top-12 keeps 94.4%
# (worst species 82.9%) and top-30 keeps 98.7% (worst 94.1%), for ~126 KB. A species whose
# archetype we cannot draw is one we systematically mis-model in every world, so the 30 is
# bought cheaply and worth buying.
#
# `moves` is wide for a different reason: a 4-slot draw without replacement reaches further
# down the list than any single-slot draw does.
KEEP_ITEMS = 10
KEEP_MOVES = 24
KEEP_SPREADS = 30
KEEP_TERA = 10
KEEP_TEAMMATES = 12

# Species below this usage are dropped entirely. At 1825 that is a handful of one-off
# novelties; a species we have no data for still plays (the adapter falls back to revealed
# moves and a neutral spread), it just gets no prior.
MIN_USAGE = 0.0002


def _get(url, timeout=180):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _norm(pairs, keep):
    """Top `keep` of {key: weight} as {key: probability}, renormalised over what's kept.

    Renormalising rather than keeping the raw shares is deliberate: the sampler draws from
    this dict, so the mass of the discarded tail has to land somewhere, and spreading it
    over the head in proportion is the only choice that leaves the head's ORDER and relative
    odds untouched."""
    items = [(k, float(v)) for k, v in pairs.items() if k and float(v) > 0]
    items.sort(key=lambda kv: -kv[1])
    items = items[:keep]
    total = sum(w for _k, w in items)
    if total <= 0:
        return {}
    return {k: round(w / total, 6) for k, w in items}


def _parse_spread(key):
    """'Jolly:0/252/4/0/0/252' -> ('jolly', [0, 252, 4, 0, 0, 252]), or None if malformed."""
    if ":" not in key:
        return None
    nature, evs = key.split(":", 1)
    parts = evs.split("/")
    if len(parts) != 6:
        return None
    try:
        return nature.strip().lower(), [int(p) for p in parts]
    except ValueError:
        return None


def condense_usage(chaos):
    """Chaos JSON -> the committed per-species marginals."""
    data = chaos["data"]
    out = {}
    for name, info in data.items():
        if float(info.get("usage", 0.0)) < MIN_USAGE:
            continue
        spreads = {}
        for key, w in info.get("Spreads", {}).items():
            parsed = _parse_spread(key)
            if parsed is None or float(w) <= 0:
                continue
            spreads[key] = float(w)
        top_spreads = sorted(spreads.items(), key=lambda kv: -kv[1])[:KEEP_SPREADS]
        total = sum(w for _k, w in top_spreads) or 1.0
        entry = {
            "usage": round(float(info.get("usage", 0.0)), 6),
            "abilities": _norm(info.get("Abilities", {}), 8),
            # 'nothing' is a real observation (an empty item slot), so it is kept as-is and
            # translated to the engine's 'none' at sample time, not dropped here.
            "items": _norm(info.get("Items", {}), KEEP_ITEMS),
            "moves": _norm(info.get("Moves", {}), KEEP_MOVES),
            "tera": _norm(info.get("Tera Types", {}), KEEP_TERA),
            "spreads": [[_parse_spread(k)[0], _parse_spread(k)[1], round(w / total, 6)]
                        for k, w in top_spreads],
            "teammates": {to_id_str(k): round(v, 6) for k, v in
                          _norm(info.get("Teammates", {}), KEEP_TEAMMATES).items()},
        }
        out[to_id_str(name)] = entry
    return out


def _opts(value):
    """A curated-set field -> a list of options. Scalars become one-element lists."""
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if v]
    return [value]


def _stat_opts(value):
    """An EV / IV field -> a list of {stat: value} options.

    Alternative SPREADS are written as a list of dicts (Ninetales runs a physically bulky
    and a special-attacking EV split under one set), so this is not the same shape as
    _opts's list-of-scalars and has to keep each dict whole."""
    if not value:
        return [{}]
    options = value if isinstance(value, list) else [value]
    return [{k: int(v) for k, v in opt.items()} for opt in options if isinstance(opt, dict)] \
        or [{}]


def condense_sets(raw):
    """pkmn/smogon analysis sets -> {species_id: [set, ...]} with ids normalised.

    The source encodes "pick one of these" as a nested list -- both for a move SLOT and for
    the item / ability / nature / Tera fields. That structure is preserved rather than
    flattened: it is the only thing that says which choices are alternatives to each other,
    and the sampler resolves one option per slot so the result is a set someone would
    actually build."""
    out = {}
    for species, sets in raw.items():
        parsed = []
        for set_name, s in sets.items():
            moves = [[to_id_str(m) for m in _opts(slot)] for slot in s.get("moves", [])]
            moves = [slot for slot in moves if slot]
            if not moves:
                continue
            parsed.append({
                "name": set_name,
                "moves": moves,
                "items": [to_id_str(i) for i in _opts(s.get("item"))] or ["none"],
                "abilities": [to_id_str(a) for a in _opts(s.get("ability"))],
                "natures": [str(n).lower() for n in _opts(s.get("nature"))] or ["serious"],
                "evs": _stat_opts(s.get("evs")),
                "ivs": _stat_opts(s.get("ivs")),
                "tera": [str(t).lower() for t in _opts(s.get("teratypes"))],
            })
        if parsed:
            out[to_id_str(species)] = parsed
    return out


def latest_month(fmt, cutoff, back=4):
    """The most recent month Smogon has published this format for.

    Stats land partway through the following month, so 'this month' is normally absent and
    'last month' is the usual answer; walking back a few keeps the tool working during the
    gap instead of failing on a URL that does not exist yet."""
    y, m = date.today().year, date.today().month
    for _ in range(back):
        month = f"{y:04d}-{m:02d}"
        try:
            req = urllib.request.Request(
                CHAOS_URL.format(month=month, fmt=fmt, cutoff=cutoff), method="HEAD")
            with urllib.request.urlopen(req, timeout=30):
                return month
        except Exception:
            pass
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    raise SystemExit(f"no published {fmt}-{cutoff} stats in the last {back} months")


def main():
    ap = argparse.ArgumentParser(description="Download and condense the Gen 9 OU priors.")
    ap.add_argument("--month", default=None, metavar="YYYY-MM",
                    help="Smogon stats month (default: the latest published)")
    ap.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF,
                    help=f"ladder rating cutoff of the usage sample (default {DEFAULT_CUTOFF})")
    ap.add_argument("--format", default=FORMAT, help=f"tier (default {FORMAT})")
    ap.add_argument("--usage-only", action="store_true", help="skip the curated sets")
    ap.add_argument("--sets-only", action="store_true", help="skip the usage statistics")
    args = ap.parse_args()

    os.makedirs(paths.OU_DATA_DIR, exist_ok=True)

    if not args.sets_only:
        month = args.month or latest_month(args.format, args.cutoff)
        url = CHAOS_URL.format(month=month, fmt=args.format, cutoff=args.cutoff)
        print(f"usage stats  {url}", flush=True)
        chaos = json.loads(gzip.decompress(_get(url)))
        info = chaos.get("info", {})
        usage = condense_usage(chaos)
        payload = {
            "_source": url,
            "_month": month,
            "_cutoff": args.cutoff,
            "_battles": info.get("number of battles"),
            "data": usage,
        }
        with open(paths.OU_USAGE, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        print(f"  {len(usage)} species  ->  {paths.OU_USAGE} "
              f"({os.path.getsize(paths.OU_USAGE) / 1e6:.1f} MB, "
              f"{info.get('number of battles')} battles)", flush=True)

    if not args.usage_only:
        url = SETS_URL.format(fmt=args.format)
        print(f"curated sets {url}", flush=True)
        sets = condense_sets(json.loads(_get(url)))
        with open(paths.OU_SETS, "w", encoding="utf-8") as f:
            json.dump({"_source": url, "data": sets}, f, separators=(",", ":"))
        n_sets = sum(len(v) for v in sets.values())
        print(f"  {len(sets)} species / {n_sets} sets  ->  {paths.OU_SETS} "
              f"({os.path.getsize(paths.OU_SETS) / 1e3:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
