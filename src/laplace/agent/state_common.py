r"""The format-agnostic half of the poke-env -> poke-engine translation.

`poke_engine_adapter` does two separable jobs. Most of it just re-encodes what the server
has already told us -- HP, status, boosts, hazards, screens with their real turn counters,
Wish and Future Sight timers, Taunt/Encore durations, our own fully-known side -- and none
of that knows or cares which format is being played. The rest is the determinizer, which is
entirely a statement about the format's set distribution and is the only part OU has to
replace.

This module names the first half, so `laplace.ou.adapter` has a documented import site
instead of reaching into another module's underscore names, and so the boundary between
"encoding the board" and "guessing the hidden set" is written down somewhere.

It re-exports rather than re-implements on purpose. There is exactly one encoder in this
repo and there should stay exactly one: the randbats bot is the tested artifact, and a
second copy of `_side_conditions` would be a second place for the Reflect clock to drift.
If the code is ever physically moved, this is where it moves TO, and every caller keeps
working.
"""

from laplace.agent.poke_engine_adapter import (   # noqa: F401  (re-export)
    GEN,
    # --- names, enums, small conversions ---
    _STATUS, _WEATHER, _TERRAIN,
    _status_str, _types_tuple, _base_types_tuple, _stat, _maxhp, _hp,
    # --- moves and PP ---
    _max_pp, _moves_from_objs, _moves_from_ids, _observed_pp, _last_used_str,
    _active_own_moves,
    # --- counters the protocol only exposes indirectly ---
    _sleep_turns, _toxic_count, _remaining,
    # --- item slot state ---
    _UNKNOWN_ITEM, _item_evidence, _resolve_item, _CHOICE_ITEMS,
    # --- side / field assembly ---
    _side_conditions, _boosts, _volatile_set, _duration_volatiles, _sub_health, _delayed,
    _our_side, _dummy,
    _weather_str, _terrain_str, _trick_room,
    # --- our own Pokemon (never sampled: the request tells us everything) ---
    _own_pokemon,
)
