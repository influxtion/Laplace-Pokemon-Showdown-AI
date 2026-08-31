r"""Who leads. The one OU decision the search cannot take.

Every other decision in a game is made by rolling the position forward. At team preview
there is no position: nothing is on the field, no move has been made, and poke-engine has
nothing to search. What there IS, uniquely, is complete knowledge of both rosters -- our six
are ours and their six are on the screen -- so the decision is a matchup table, and a matchup
table is exactly what a heuristic does well.

Bring all six (singles OU has no bring-4), so the order only decides the lead. The lead is
scored on three things, in the order they decide games:

  1. What it threatens and what threatens it. For each of their six, the best damage we
     would deal and the worst we would take, from the moves we actually have and the moves
     the usage data says they actually run. Averaged, not maximised: a lead is chosen
     against an unknown one of six, not against a chosen one.
  2. Speed. Winning the first turn is worth a lot more at preview than at any later point,
     because neither side has committed anything yet.
  3. Job. A hazard setter or a pivot leads better than a late-game cleaner, and burning the
     sweeper on turn 1 is a real and common way to lose.

This is a heuristic and it is meant to be one. It is deliberately NOT a place to spend
search: the honest ceiling on lead choice is low, the variance is high, and every one of the
project's compute experiments has said the same thing about spending time where the
evaluation is already flat.
"""

from poke_env.data import to_id_str

from laplace.agent.knowledge import estimate_damage_fraction, get_move
from laplace.ou.knowledge import KNOWLEDGE
from laplace.ou.usage import USAGE

# Moves whose value is highest on turn 1 -- what you actually want to be doing with the
# opening slot. Hazards on turn 1 pay for the whole game; a pivot move keeps the lead from
# being a commitment at all.
_LEAD_MOVES = frozenset((
    "stealthrock", "spikes", "toxicspikes", "stickyweb",
    "uturn", "voltswitch", "flipturn", "partingshot", "teleport", "chillyreception",
    "taunt", "defog", "rapidspin", "courtchange",
))

# How much each term is worth, in units of "fraction of a health bar". Damage is the scale
# everything else is quoted against, so these read as: winning the speed tie is worth ~8% of
# a bar, and having a turn-1 job is worth ~12%.
W_SPEED = 0.08
W_LEAD_MOVE = 0.12
# Setup moves are the counterweight: a Pokemon whose whole plan is to boost is the wrong
# lead even when its raw matchup numbers are good, because it has to survive a turn it has
# not earned yet.
W_SETUP_ONLY = -0.10

# How many of their moves to price. The usage marginals are steep, so the top four are the
# set the overwhelming majority of the time, and going deeper mostly prices tech moves that
# a lead matchup should not be decided on.
OPP_MOVES = 4


def _opp_moves(mon):
    """The moves this previewed opponent most likely carries, as Move objects."""
    sid = USAGE.resolve(mon)
    if not sid:
        return []
    entry = USAGE.entry(sid) or {}
    ranked = sorted(entry.get("moves", {}).items(), key=lambda kv: -kv[1])[:OPP_MOVES]
    return [mv for move_id, _p in ranked if (mv := get_move(move_id)) is not None]


def _our_moves(mon):
    return [mv for mv in (mon.moves or {}).values() if mv is not None]


def _best_damage(attacker, moves, defender):
    best = 0.0
    for mv in moves:
        try:
            best = max(best, float(estimate_damage_fraction(attacker, mv, defender,
                                                            KNOWLEDGE)))
        except Exception:
            continue
    return best


def _role_bonus(mon):
    """Turn-1 usefulness of our own Pokemon's moveset."""
    move_ids = set(mon.moves or {})
    if move_ids & _LEAD_MOVES:
        return W_LEAD_MOVE
    setup = 0
    attacking = 0
    for mid in move_ids:
        mv = get_move(mid)
        if mv is None:
            continue
        if mv.base_power > 0:
            attacking += 1
        boosts = dict(mv.boosts or {})
        boosts.update(mv.self_boost or {})
        if any(v > 0 for v in boosts.values()):
            setup += 1
    # "Setup only" means it has a boosting move and barely anything else to do turn 1.
    return W_SETUP_ONLY if setup and attacking <= 2 else 0.0


def score_lead(mon, opponents):
    """How good `mon` is as a lead against a previewed enemy team. Higher is better."""
    ours = _our_moves(mon)
    my_speed = KNOWLEDGE.estimate_stat(mon, "spe")
    deal = take = speed = 0.0
    for opp in opponents:
        theirs = _opp_moves(opp)
        deal += min(_best_damage(mon, ours, opp), 1.0)
        take += min(_best_damage(opp, theirs, mon), 1.0)
        speed += 1.0 if my_speed > KNOWLEDGE.speed_bounds(opp)[1] else \
            0.5 if my_speed > sum(KNOWLEDGE.speed_bounds(opp)) / 2 else 0.0
    n = max(len(opponents), 1)
    return (deal - take) / n + W_SPEED * (speed / n) + _role_bonus(mon)


def choose_order(battle, fixed_lead=None):
    """The '/team ...' string for this battle, and mark what we bring.

    `fixed_lead` (1-based) skips the heuristic entirely -- some teams have exactly one lead
    and the right thing is to say so rather than let a scorer rediscover it every game.

    Marking `_selected_in_teampreview` is not optional bookkeeping: poke-env reads it to
    know which Pokemon are in play, and every one of ours is."""
    team = list(battle.team.values())
    if not team:
        return None
    order = list(range(1, len(team) + 1))
    if fixed_lead and 1 <= fixed_lead <= len(team):
        lead = fixed_lead
    else:
        opponents = list(battle.teampreview_opponent_team or [])
        if opponents:
            scores = [score_lead(mon, opponents) for mon in team]
            lead = max(order, key=lambda i: scores[i - 1])
        else:
            lead = 1
    order.remove(lead)
    order.insert(0, lead)
    for mon in team:
        mon._selected_in_teampreview = True
    return "/team " + "".join(str(i) for i in order)
