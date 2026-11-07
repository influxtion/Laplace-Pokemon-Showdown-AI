r"""Mine saved replays (+ decision traces) for recurring blunder signatures.

The improvement loop that actually moved Elo: play ladder games (ladder.py saves every
replay plus a .trace.json of what the search considered each turn), then run this to
cluster losses into failure classes.

  immune_click   we used a move the active opponent was immune to (the Scale Shot class)
  fail_click     our move failed outright, e.g. Rest at full HP
  died_no_act    our active fainted on a turn it neither moved nor was switched in --
                 outsped + KO'd, i.e. the "stayed in on lethal" signature
  opp_boosted    the opponent's active reached net +3 or more: we let something set up
  slept_hit      turns spent asleep while taking damage (Rest-loop cost)
  tera_wasted    our tera'd mon fainted within 2 turns of tera with no opponent faint in
                 between. 48% of tera-losses vs 12% of tera-wins when first measured.

Signatures are heuristics for WHERE TO LOOK, not verdicts -- open the replay/trace for any
flagged turn before calling it a bug. Wins are mined too, as a control: a signature that
shows up equally in wins is background noise, not a loss cause. See classify_loss.

    python -u src\mine_losses.py                    # mine replays/ladder
    python -u src\mine_losses.py --dir some\dir --include-file extra-replay.html
"""

import argparse
import glob
import json
import os
import re

from laplace import paths

LOG_RE = re.compile(r'<script type="text/plain" class="battle-log-data">(.*?)</script>', re.S)


BOT_USERNAME = "influxobot"


def read_log(path, username=None):
    """The protocol log lines of a saved replay, plus which side (p1/p2) is the bot."""
    username = (username or BOT_USERNAME).lower()
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    m = LOG_RE.search(text)
    lines = (m.group(1) if m else text).replace("\\/", "/").split("\n")
    lines = [l.rstrip() for l in lines if l.startswith("|")]
    me = None
    for l in lines:
        p = l.split("|")
        if len(p) > 3 and p[1] == "player" and p[3].lower() == username:
            me = p[2]
    return lines, me


_BOOST_TAGS = ("-boost", "-unboost", "-setboost", "-clearboost", "-clearallboost",
               "-invertboost")


def _fold_boost(boosts, p, tag, prefix):
    """Fold one boost-protocol message into {stat: stage} for the side `prefix`.

    Centralised because three call sites need identical semantics, and writing it three
    times is how they diverge. What each message means:

      -boost / -unboost   relative change, the common case
      -setboost           ABSOLUTE set. This is the only way Belly Drum's +6 Attack is
                          reported, so handling just -boost/-unboost scores a Belly Drum
                          sweep as peak boost ZERO -- which is exactly what happened to
                          Azumarill and Eiscue in the mined cohort, and it is the most
                          explosive setup move in the format.
      -clearboost         clears ONE target, so it must be gated on the prefix
      -clearallboost      Haze: clears both sides unconditionally
      -invertboost        Topsy-Turvy; approximated as a clear

    -clearnegativeboost is deliberately ignored: it removes only negative stages, and every
    metric built on this dict counts net POSITIVE stages, so it cannot change the answer.

    Returns the (possibly new) dict."""
    if tag == "-clearallboost":
        return {}
    if len(p) < 3 or not p[2].startswith(prefix):
        return boosts
    if tag in ("-clearboost", "-invertboost"):
        return {}
    if len(p) < 5:
        return boosts
    try:
        amount = int(p[4])
    except ValueError:
        return boosts
    if tag == "-setboost":
        boosts[p[3]] = amount
    else:
        boosts[p[3]] = boosts.get(p[3], 0) + amount * (1 if tag == "-boost" else -1)
    return boosts


def _net_positive(boosts):
    return sum(v for v in boosts.values() if v > 0)


def mine_game(lines, me, fracs=None):
    """Signature -> turns it fired on, for one game, from side `me`'s perspective.

    `fracs` is {turn: (our team hp fraction, theirs)} from hp_curve, used to gate
    died_no_act_even. Optional: without it that one signature is simply not emitted.

    Two non-list keys ride along under leading underscores because every caller already
    special-cases '_faints': '_turns' is the game length, needed because signature rates
    MUST be per-turn (losses run ~30% longer than wins here, and per-game rates read that
    length difference as a signal -- died_no_act was 4.2x loss-correlated per game, 3.5x
    per turn, and 1.25x per turn once you also control for being behind). '_boost_peak' is
    the highest net positive boost the opponent's active ever held, because the old
    opp_boosted flag fired ONCE at +3 and then latched, which made a Dunsparce that Coiled
    to +18 the same datum as a single Calm Mind."""
    opp = "p2" if me == "p1" else "p1"
    sig = {"immune_click": [], "fail_click": [], "died_no_act": [],
           "opp_boosted": [], "slept_hit": [], "tera_wasted": []}
    if fracs is not None:
        # died_no_act, but only while the game is still level. The raw signature is
        # dominated by the endgame of a game already lost -- once you are down to two mons
        # against six, everything dies without acting, and that is the SYMPTOM of losing,
        # not a decision worth reviewing. Per-turn normalisation alone does NOT fix it
        # (0.105 vs 0.033 on the cohort this was written for); gating on parity does
        # (0.030 vs 0.024, i.e. nothing). This is the version worth flagging on.
        sig["died_no_act_even"] = []
    turn = 0
    opp_boosts = {}
    opp_boost_flagged = False
    boost_peak = 0
    acted_this_turn = False      # our active moved or was switched this turn
    our_last_move_line = -1
    slept_this_turn = False
    tera_turn = tera_mon = None  # our first Terastallization
    opp_faints_since_tera = 0

    def mine_prefix(s):
        return s.startswith(f"{me}a:")

    def opp_prefix(s):
        return s.startswith(f"{opp}a:")

    for i, l in enumerate(lines):
        p = l.split("|")
        tag = p[1] if len(p) > 1 else ""
        if tag == "turn":
            turn = int(p[2])
            acted_this_turn = False
            slept_this_turn = False
        elif tag in ("switch", "drag") and len(p) > 2:
            if mine_prefix(p[2]):
                acted_this_turn = True
            if opp_prefix(p[2]):
                opp_boosts = {}
                opp_boost_flagged = False
        elif tag == "move" and len(p) > 2:
            if mine_prefix(p[2]):
                acted_this_turn = True
                our_last_move_line = i
        elif tag == "cant" and len(p) > 3 and mine_prefix(p[2]) and p[3] == "slp":
            acted_this_turn = True     # it "acted" (slept); don't double-count as died_no_act
            slept_this_turn = True
        elif tag == "-damage" and len(p) > 2 and mine_prefix(p[2]) and slept_this_turn:
            sig["slept_hit"].append(turn)
            slept_this_turn = False    # one per turn
        elif tag == "-immune" and len(p) > 2 and opp_prefix(p[2]):
            # ours was the immune'd attack iff our move was the most recent move line
            if our_last_move_line >= 0 and i - our_last_move_line <= 3:
                sig["immune_click"].append(turn)
        elif tag == "-fail" and len(p) > 2 and mine_prefix(p[2]):
            if our_last_move_line >= 0 and i - our_last_move_line <= 3:
                sig["fail_click"].append(turn)
        elif tag in _BOOST_TAGS:
            opp_boosts = _fold_boost(opp_boosts, p, tag, f"{opp}a")
            net = _net_positive(opp_boosts)
            boost_peak = max(boost_peak, net)
            if net >= 3 and not opp_boost_flagged:
                sig["opp_boosted"].append(turn)
                opp_boost_flagged = True
        elif tag == "-terastallize" and len(p) > 2 and mine_prefix(p[2]) and tera_turn is None:
            tera_turn = turn
            tera_mon = p[2].split(":")[-1].strip()
        elif tag == "faint" and len(p) > 2:
            if opp_prefix(p[2]) and tera_turn is not None:
                opp_faints_since_tera += 1
            if mine_prefix(p[2]):
                sig.setdefault("_faints", []).append(turn)
                if not acted_this_turn and turn > 0:
                    sig["died_no_act"].append(turn)
                    if fracs is not None:
                        mine_f, opp_f = fracs.get(turn, (1.0, 1.0))
                        if mine_f >= opp_f - 0.05:
                            sig["died_no_act_even"].append(turn)
                if (tera_mon is not None and p[2].split(":")[-1].strip() == tera_mon
                        and turn - tera_turn <= 2 and opp_faints_since_tera == 0):
                    sig["tera_wasted"].append(turn)
                    tera_mon = None      # count once
    sig["_turns"] = turn
    sig["_boost_peak"] = boost_peak
    return sig


# --- deeper per-game analyses -------------------------------------------------------------
# These fill the signature miner's known blind spots. Plenty of losses contain no blunder at
# all -- we just traded worse and ran out of team, which is invisible to everything above
# (cf. the Foul Play mining). So: HP-curve divergence, luck accounting, switch quality.
# Same caveat as the signatures: pointers for where to look, not verdicts.

_HP_RE = re.compile(r"^(\d+)/(\d+)")


def _frac(token):
    """'185/246' / '63/100 tox' / '0 fnt' -> hp fraction, or None if unparseable."""
    if token.startswith("0 fnt"):
        return 0.0
    m = _HP_RE.match(token)
    if not m:
        return None
    cur, mx = int(m.group(1)), int(m.group(2))
    return cur / mx if mx else None


def _hp_frac(parts):
    """First parseable HP token in a protocol line's fields, or None.

    Positional indexing does not survive contact with the real protocol. A plain switch is
    |switch|IDENT|DETAILS|HP, so the token is both p[4] and p[-1] -- but a pivot switch-in
    is |switch|IDENT|DETAILS|HP|[from] Volt Switch, where p[-1] is the [from] tag and p[3]
    is the species details, so reading either yields None and the mon silently keeps its
    stale (or full) HP. U-turn / Volt Switch / Flip Turn / Parting Shot entries are a large
    share of all switch-ins, so that quietly skewed every HP-derived number here."""
    for token in parts[3:]:
        f = _frac(token)
        if f is not None:
            return f
    return None


def hp_curve(lines, me):
    """Per-turn (our team fraction, opp team fraction). Unrevealed mons count as full.

    Both sides' HP tokens are cur/max IN THEIR OWN UNITS (ours raw stats, theirs /100), so
    per-mon FRACTIONS are exact and comparable while raw HP differencing would be garbage.

    Returns (curve, first_behind_turn, recovered), where 'behind' means diff < -0.10."""
    opp = "p2" if me == "p1" else "p1"
    mons = {"me": {}, "opp": {}}     # nickname -> last known frac
    curve = []
    turn = 0
    first_behind = None
    recovered = False

    def team(side):
        d = mons[side]
        return (sum(d.values()) + (6 - len(d))) / 6.0

    for l in lines:
        p = l.split("|")
        tag = p[1] if len(p) > 1 else ""
        if tag == "turn":
            turn = int(p[2])
            diff = team("me") - team("opp")
            curve.append((turn, round(team("me"), 3), round(team("opp"), 3)))
            if diff < -0.10 and first_behind is None:
                first_behind = turn
            if first_behind is not None and diff >= 0:
                recovered = True
        elif tag in ("switch", "drag", "-damage", "-heal", "replace") and len(p) > 3:
            side = "me" if p[2].startswith(f"{me}a") else \
                   "opp" if p[2].startswith(f"{opp}a") else None
            if side:
                f = _frac(p[3]) if tag in ("-damage", "-heal") else _hp_frac(p)
                if f is not None:
                    mons[side][p[2]] = f
        elif tag == "faint" and len(p) > 2:
            side = "me" if p[2].startswith(f"{me}a") else \
                   "opp" if p[2].startswith(f"{opp}a") else None
            if side:
                mons[side][p[2]] = 0.0
    return curve, first_behind, recovered


def hax_events(lines, me):
    """Luck ledger: crits, misses, full-para, flinch/frz stops, confusion self-hits.

    Sleep is EXCLUDED -- Rest makes it strategy, not luck. Reported symmetrically: 'for' =
    events that helped us, 'against' = events that hurt us."""
    opp_pre = ("p2" if me == "p1" else "p1") + "a"
    me_pre = f"{me}a"
    ev = {"for": 0, "against": 0}

    def bump(hurt_side_prefix):
        ev["against" if hurt_side_prefix == me_pre else "for"] += 1

    for l in lines:
        p = l.split("|")
        tag = p[1] if len(p) > 1 else ""
        if tag == "-crit" and len(p) > 2:            # target got hurt
            bump(me_pre if p[2].startswith(me_pre) else opp_pre)
        elif tag == "-miss" and len(p) > 2:          # attacker lost the turn
            bump(me_pre if p[2].startswith(me_pre) else opp_pre)
        elif tag == "cant" and len(p) > 3 and p[3] in ("par", "flinch", "frz"):
            bump(me_pre if p[2].startswith(me_pre) else opp_pre)
        elif tag == "-damage" and "[from] confusion" in l and len(p) > 2:
            bump(me_pre if p[2].startswith(me_pre) else opp_pre)
    return ev


def switch_punish(lines, me):
    """(voluntary healthy switch-ins, how many were punished).

    Punished = the switched-in mon lost >=40% max HP to DIRECT opponent move damage, or
    fainted, within its entry turn block.

    Deliberately excludes: forced post-faint replacements (a different decision class),
    hazard/passive chip ([from]-tagged -- already priced by the search), and entries below
    60% HP (endgame sacks are intentional)."""
    opp = "p2" if me == "p1" else "p1"
    vol = punished = 0
    cur_mon = None
    entry_frac = None
    our_faint_this_block = False
    lost = 0.0
    for l in lines:
        p = l.split("|")
        tag = p[1] if len(p) > 1 else ""
        if tag == "turn":
            if cur_mon is not None and lost >= 0.40:
                punished += 1
            cur_mon, entry_frac, lost = None, None, 0.0
            our_faint_this_block = False
        elif tag == "faint" and len(p) > 2 and p[2].startswith(f"{me}a"):
            if cur_mon is not None and p[2] == cur_mon:
                punished += 1
                cur_mon = None
            our_faint_this_block = True
        elif tag == "switch" and len(p) > 3 and p[2].startswith(f"{me}a"):
            f = _hp_frac(p)
            if not our_faint_this_block and f is not None and f > 0.60:
                vol += 1
                cur_mon, entry_frac, lost = p[2], f, 0.0
        elif (tag == "-damage" and cur_mon is not None and len(p) > 3
              and p[2] == cur_mon and "[from]" not in l):
            f = _frac(p[3])
            if f is not None and entry_frac is not None:
                lost = entry_frac - f
    return vol, punished


def tera_roi(lines, me):
    """Descriptive tera stats: (tera turn, opp faints while the tera mon lived, turns it
    survived).

    OUTCOME-CONFOUNDED by construction -- winners' teras always look good. Trend-watching
    only, never a gate."""
    opp = "p2" if me == "p1" else "p1"
    turn = 0
    tera_turn = tera_mon = None
    kills = 0
    faint_turn = None
    for l in lines:
        p = l.split("|")
        tag = p[1] if len(p) > 1 else ""
        if tag == "turn":
            turn = int(p[2])
        elif tag == "-terastallize" and len(p) > 2 and p[2].startswith(f"{me}a") \
                and tera_turn is None:
            tera_turn, tera_mon = turn, p[2]
        elif tag == "faint" and len(p) > 2 and tera_turn is not None:
            if p[2].startswith(f"{opp}a") and faint_turn is None:
                kills += 1
            elif p[2] == tera_mon and faint_turn is None:
                faint_turn = turn
    if tera_turn is None:
        return None
    return tera_turn, kills, (faint_turn or turn) - tera_turn


def opp_killers(lines, me):
    """Opponent species credited with our faints. APPROXIMATE -- passive damage credits
    whatever happens to be active. Aggregated across losses as a 'who beats us' pointer."""
    opp = "p2" if me == "p1" else "p1"
    active = None
    credit = {}
    for l in lines:
        p = l.split("|")
        tag = p[1] if len(p) > 1 else ""
        if tag in ("switch", "drag", "replace") and len(p) > 3 and p[2].startswith(f"{opp}a"):
            active = p[3].split(",")[0].strip()
        elif tag == "faint" and len(p) > 2 and p[2].startswith(f"{me}a") and active:
            credit[active] = credit.get(active, 0) + 1
    return credit


def boost_pressure(lines, me):
    """(turns spent facing a net +2-or-better opposing active, total turns).

    The single cleanest separator found in the 50-game cohort this was written for: 33.0%
    of turns in losses vs 15.6% in wins, while OUR OWN peak boost was identical across the
    two (+2.71 vs +2.59). So it is not 'we set up less than they do', it is 'we do not stop
    them'. Counted per turn rather than per game on purpose -- see mine_game's docstring."""
    opp = "p2" if me == "p1" else "p1"
    boosts = {}
    turns = pressured = 0
    for l in lines:
        p = l.split("|")
        tag = p[1] if len(p) > 1 else ""
        if tag == "turn":
            turns += 1
            if _net_positive(boosts) >= 2:
                pressured += 1
        elif tag in ("switch", "drag") and len(p) > 2 and p[2].startswith(f"{opp}a"):
            boosts = {}
        elif tag in _BOOST_TAGS:
            boosts = _fold_boost(boosts, p, tag, f"{opp}a")
    return pressured, turns


def sweep_profile(lines, me):
    """How concentrated our faints were, and what the mon that caused them was doing.

    Returns (species, kills, our_faints, peak_net_boost, terastallized) for the opposing
    Pokemon credited with the most of our faints, or None if we never fainted.

    Raw 'one mon got 3+ of our faints' is nearly useless on its own -- it fires on 20/21
    losses, and a class that fires on 95% of losses hides exactly as much as one that fires
    on none. The peak-boost and tera fields are what make it actionable, because they split
    that one bucket into three causes pointing at three different fixes (see classify_loss).

    Credit is APPROXIMATE, same caveat as opp_killers: passive damage credits whichever
    opposing mon happens to be active. Normalise by our faint count before comparing across
    wins and losses -- a loss has ~6 faints to concentrate and a win ~2.7."""
    opp = "p2" if me == "p1" else "p1"
    active = None
    credit, peaks, teras = {}, {}, set()
    boosts = {}
    our_faints = 0
    for l in lines:
        p = l.split("|")
        tag = p[1] if len(p) > 1 else ""
        if tag in ("switch", "drag", "replace") and len(p) > 3 and p[2].startswith(f"{opp}a"):
            active = p[3].split(",")[0].strip()
            boosts = {}
        elif tag in _BOOST_TAGS:
            boosts = _fold_boost(boosts, p, tag, f"{opp}a")
            if active:
                peaks[active] = max(peaks.get(active, 0), _net_positive(boosts))
        elif tag == "-terastallize" and len(p) > 2 and p[2].startswith(f"{opp}a") and active:
            teras.add(active)
        elif tag == "faint" and len(p) > 2 and p[2].startswith(f"{me}a"):
            our_faints += 1
            if active:
                credit[active] = credit.get(active, 0) + 1
    if not credit:
        return None
    species, kills = max(credit.items(), key=lambda kv: kv[1])
    return species, kills, our_faints, peaks.get(species, 0), species in teras


_RECOVERY = {"recover", "roost", "softboiled", "slackoff", "synthesis", "moonlight",
             "morningsun", "shoreup", "milkdrink", "strengthsap", "rest", "protect",
             "detect", "substitute", "wish"}

# Slack on the opponent's HP that still counts as "no progress" -- leftovers-scale jitter
# should not excuse a stall. Same 0.02 scale as engine_search.FUTILE_HP_EPS, deliberately,
# so the miner and the shipped guard agree on what counts as progress.
WASTED_HP_EPS = 0.02


def wasted_recovery(lines, me):
    """Turns where we clicked the same recovery/protect move >=3 times running and the
    matchup got no better for it.

    The mirror image of the chip futility the search already guards: engine_search's
    futile() only classifies damaging moves and status moves WITH A POSITIVE BOOST TABLE,
    so Roost / Recover / Protect / Substitute fall straight through it, and _noop_demote
    tests exact equality against the pass baseline so a five-HP heal escapes that too.

    Observed live (lost-...-2665357974 T20-24): Noctowl Roosted five turns running at ~full
    HP while a Lokix healed 16% -> 74% off Leech Life and then set up; the fifth Roost
    failed outright. Confidence was 0.42-0.59 throughout, so no tie-window guard could
    engage.

    'Got no better' is measured on the OPPONENT's HP, not ours. Keying it on our own HP
    looks right and is wrong: across the Noctowl run our HP genuinely climbed 65% -> 100%,
    because Roost was doing exactly what Roost does. What made those five turns worthless is
    that the opponent climbed 26% -> 74% over the same window. So the test is: a run of
    >= 3 consecutive identical no-damage clicks in one matchup, ending with the opponent at
    or above the HP it started the run on.

    That clears the honest cases by construction: Roost out-pacing chip damage shows the
    opponent's HP falling, and stalling a Toxic'd or Leech-Seeded target shows the same.
    It will still flag a deliberate PP-stall or a wait-out-the-boost hold, which is a known
    false positive -- like every signature in this file it is a pointer to a turn worth
    reading, not a verdict. Returns the turns each qualifying run ended on."""
    opp = "p2" if me == "p1" else "p1"
    turn = 0
    run_move = None
    run_len = 0
    my_hp = opp_hp = None
    start_my = start_opp = None
    flagged = []

    def close_run():
        nonlocal run_move, run_len, start_my, start_opp
        if (run_len >= 3 and opp_hp is not None and start_opp is not None
                and opp_hp >= start_opp - WASTED_HP_EPS):
            flagged.append(turn)
        run_move, run_len, start_my, start_opp = None, 0, None, None

    for l in lines:
        p = l.split("|")
        tag = p[1] if len(p) > 1 else ""
        if tag == "turn":
            turn = int(p[2])
        elif tag in ("switch", "drag", "replace") and len(p) > 3:
            f = _hp_frac(p)
            if p[2].startswith(f"{me}a"):
                close_run()
                my_hp = f
            elif p[2].startswith(f"{opp}a"):
                close_run()          # new matchup, fresh evidence
                opp_hp = f
        elif tag in ("-damage", "-heal") and len(p) > 3:
            f = _frac(p[3])
            if f is not None:
                if p[2].startswith(f"{me}a"):
                    my_hp = f
                elif p[2].startswith(f"{opp}a"):
                    opp_hp = f
        elif tag == "move" and len(p) > 3 and p[2].startswith(f"{me}a"):
            mid = re.sub(r"[^a-z0-9]", "", p[3].lower())
            if mid not in _RECOVERY:
                close_run()
                continue
            if mid != run_move:
                close_run()
                run_move, run_len = mid, 1
                start_my, start_opp = my_hp, opp_hp
            else:
                run_len += 1
    close_run()
    return flagged


def saw_illusion(lines, me):
    """Did the opponent reveal a Zoroark forme this game (via replace or naming)?"""
    opp = "p2" if me == "p1" else "p1"
    return any(l.split("|")[1:2] == ["replace"] and l.split("|")[2].startswith(f"{opp}a")
               or (f"|{opp}a: Zoroark" in l) for l in lines)


TAXONOMY = ("blundered", "swept_setup", "swept_tera", "swept_raw",
            "hax", "out_traded", "close")

# A single opposing mon taking this share of our faints, and at least this many, is a
# sweep rather than an even trade. The share gate is what makes the class survive
# normalisation: a loss has ~6 faints to concentrate and a win ~2.7, so a raw '3+ kills'
# test fires on 20/21 losses AND on any short win.
SWEEP_MIN_KILLS = 3
SWEEP_MIN_SHARE = 0.5
SWEEP_SETUP_BOOST = 2

# Minimum loss-side events before a signature may be called loss-correlated. The old rule
# had no count floor at all, only a rate one, so on a 50-game cohort fail_click qualified
# off SIX occurrences against two -- a coin flip, promoted to a verdict, which then decided
# whether five games got labelled 'blundered'. Correlation drives classification here, so
# the floor has to be high enough that the label means something.
CORRELATE_MIN_EVENTS = 10


def classify_loss(sig, hax_net, first_behind, recovered, game_len, correlated, sweep=None):
    """Coarse loss taxonomy, precedence-ordered (see TAXONOMY). A game may qualify for
    several. Pointer for where to spend reading time.

    'blundered' only counts signatures that are LOSS-CORRELATED in this cohort (rate in
    losses > 2x wins). An immune_click that appears equally in wins is hidden-set background
    noise, and labelling those losses 'blundered' buries the attrition story -- caught on
    the FP cohort validation.

    The three swept_* classes replace what used to fall through to 'close'. On the cohort
    that motivated them, 18 of 21 losses landed in 'close' -- the fallthrough bucket -- and
    the real pattern (one opposing Pokemon accounting for most of our team) sat inside it
    unnamed for a whole cohort. Adding a single 'swept' class would not have helped: it
    fires on 20/21, and a class that catches 95% of losses tells you as little as one that
    catches none. So it is split by CAUSE, because the three causes have three different
    fixes:

      swept_setup  the sweeper reached net +2 or better  -> boost-removal, leaf eval,
                                                            value_boost_margin
      swept_tera   it terastallized and never boosted    -> tera-type priors, determinizer
      swept_raw    neither; no answer to its base stats  -> matchup eval, or a genuinely
                                                            unwinnable draw

    'close' now means 'nothing fired at all' and should be RARE. See report()'s collapse
    check: any class over COLLAPSE_FRAC of losses is a sign the taxonomy has gone blind
    again and needs splitting, which is the failure this rewrite exists to prevent."""
    blunder_sigs = {"immune_click", "fail_click", "tera_wasted"} & correlated
    if any(sig.get(k) for k in blunder_sigs):
        return "blundered"
    if sweep:
        _species, kills, faints, peak, tera = sweep
        if kills >= SWEEP_MIN_KILLS and faints and kills / faints >= SWEEP_MIN_SHARE:
            if peak >= SWEEP_SETUP_BOOST:
                return "swept_setup"
            return "swept_tera" if tera else "swept_raw"
    if hax_net <= -3:
        return "hax"
    if first_behind is not None and game_len and first_behind <= game_len * 0.5 \
            and not recovered:
        return "out_traded"
    return "close"


# --- mechanism metrics --------------------------------------------------------------------
# Everything else in this file is conditioned on the result, which means it is confounded by
# the very thing it is trying to explain -- a losing bot gives away free turns, so its
# opponent's boosts go up BECAUSE it is losing as well as causing it. Those numbers are fine
# for reading a cohort and useless for judging a change.
#
# These are different: counted over every game regardless of outcome, they describe what the
# BOT DID, not what happened to it. That makes them the readout an A/B can actually use.
# "Anti-setup clicks went 4 -> 40 per 1000 decisions and the win rate did not drop" is
# decidable on one cohort; "win rate went 58% -> 61%" needs several hundred games (the
# Wilson interval at n=60 is about +/-12 points, which is wider than any effect being
# chased here).

_ANTI_SETUP = {"haze", "clearsmog", "roar", "whirlwind", "dragontail", "circlethrow",
               "taunt", "encore", "spectralthief", "coreenforcer"}
_HAZARD_SET = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
_HAZARD_CLEAR = {"rapidspin", "defog", "mortalspin", "tidyup"}

# A pooled top1-top2 gap under this is a policy that did not separate its candidates: the
# vote is determinization noise and whatever ranks first is close to arbitrary. Same scale
# as engine_search._damage_tiebreak's averaging-mode eps, deliberately.
FLAT_POLICY_EPS = 0.03


def _mech_from_log(mech, lines, me):
    """Accumulate the protocol-log half of the mechanism counters."""
    for l in lines:
        p = l.split("|")
        tag = p[1] if len(p) > 1 else ""
        if tag == "turn":
            mech["turns"] = mech.get("turns", 0) + 1
        elif tag == "move" and len(p) > 3 and p[2].startswith(f"{me}a"):
            mid = re.sub(r"[^a-z0-9]", "", p[3].lower())
            mech["our_moves"] = mech.get("our_moves", 0) + 1
            for key, group in (("antisetup", _ANTI_SETUP), ("hazard_set", _HAZARD_SET),
                               ("hazard_clear", _HAZARD_CLEAR)):
                if mid in group:
                    mech[key] = mech.get(key, 0) + 1
        elif tag == "switch" and len(p) > 2 and p[2].startswith(f"{me}a"):
            mech["our_switches"] = mech.get("our_switches", 0) + 1
    mech["wasted_recovery"] = mech.get("wasted_recovery", 0) + len(wasted_recovery(lines, me))


def _mech_from_trace(mech, path):
    """Accumulate the decision-trace half. Silently no-ops on a missing or older trace --
    every field is read with .get so a trace written before a field existed still loads."""
    if not os.path.exists(path):
        return
    try:
        entries = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return
    for e in entries:
        mech["decisions"] = mech.get("decisions", 0) + 1
        top = e.get("top") or []
        if len(top) > 1 and top[0][1] - top[1][1] < FLAT_POLICY_EPS:
            mech["flat_policy"] = mech.get("flat_policy", 0) + 1
        if top and e.get("choice") != top[0][0]:
            mech["off_argmax"] = mech.get("off_argmax", 0) + 1
            if "value" in (e.get("reorders") or []):
                mech["value_override"] = mech.get("value_override", 0) + 1
        if e.get("mixed"):
            mech["mixed"] = mech.get("mixed", 0) + 1
        if e.get("fallback"):
            mech["fallback"] = mech.get("fallback", 0) + 1
        ms = e.get("ms")
        if ms:
            mech["ms_total"] = mech.get("ms_total", 0.0) + float(ms)
            mech["ms_n"] = mech.get("ms_n", 0) + 1
        ob = e.get("opp_boosts")
        if ob is not None:
            mech["boost_turns"] = mech.get("boost_turns", 0) + 1
            if sum(v for v in ob.values() if v > 0) >= 2:
                mech["boost_pressured"] = mech.get("boost_pressured", 0) + 1


def analyze_dir(directory, username, include=(), since=None, last=None):
    """Aggregate every analysis over a replay dir. Returns the totals dict.

    `last` keeps only the N most recently modified replays -- the archive is cumulative
    (900+ games and growing), so mining it whole averages the current build together with
    every build before it. A cohort is what you actually want to look at, and needing a
    hand-copied temp directory to get one is why this loop stops being run."""
    paths = sorted(glob.glob(os.path.join(directory, "*.html")))
    if last:
        paths = sorted(paths, key=os.path.getmtime)[-last:]
    games = []
    for path in paths:
        if since is not None and os.path.getmtime(path) < since:
            continue
        name = os.path.basename(path)
        result = "lost" if name.startswith("lost") else \
                 "won" if name.startswith("won") else "tie"
        games.append((path, result))
    games += [(f, "lost") for f in include]

    agg = {
        "counts": {"won": 0, "lost": 0, "tie": 0},
        "turns": {"won": 0, "lost": 0, "tie": 0},         # denominator for every sig rate
        "sig": {"won": {}, "lost": {}, "tie": {}},
        "faints": {"won": 0, "lost": 0, "tie": 0},
        "hax": {"won": [0, 0], "lost": [0, 0]},          # [for, against]
        "switch": {"won": [0, 0], "lost": [0, 0]},        # [voluntary, punished]
        "tera": {"won": [], "lost": []},                  # (turn, kills, survived)
        "classes": {},                                    # loss label -> count
        "killers": {},
        "illusion": {"won": 0, "lost": 0},
        "reorders": {"won": {}, "lost": {}},
        "boost": {"won": [0, 0], "lost": [0, 0]},         # [pressured turns, total turns]
        "boost_peak": {"won": [], "lost": []},            # per-game opponent peak
        "sweep_share": {"won": [], "lost": []},           # top killer's share of our faints
        "mech": {},                                       # outcome-INDEPENDENT counters
        "loss_lines": [],
        "_pending_losses": [],
    }
    for path, result in games:
        lines, me = read_log(path, username)
        if me is None:
            print(f"  ?? could not find {username} in {path}")
            continue
        agg["counts"][result] += 1
        curve, first_behind, recovered = hp_curve(lines, me)
        sig = mine_game(lines, me, fracs={t: (a, b) for t, a, b in curve})
        faints = len(sig.pop("_faints", []))
        agg["turns"][result] += sig.pop("_turns", 0)
        boost_peak = sig.pop("_boost_peak", 0)
        agg["faints"][result] += faints
        flagged = {k: v for k, v in sig.items() if v}
        for k, v in flagged.items():
            agg["sig"][result][k] = agg["sig"][result].get(k, 0) + len(v)
        tag = os.path.basename(path).replace(".html", "")
        # Mechanism counters are pooled across EVERY game, not split by result. That is the
        # point of them: every other number in this file is outcome-conditioned and so
        # confounded by the thing it is trying to explain, which makes it useless as an A/B
        # readout. These move when the bot changes and can be read off a single cohort.
        _mech_from_log(agg["mech"], lines, me)
        _mech_from_trace(agg["mech"], os.path.join(os.path.dirname(path),
                                                   tag + ".trace.json"))
        if result == "tie":
            continue
        agg["boost_peak"][result].append(boost_peak)
        pressured, tturns = boost_pressure(lines, me)
        agg["boost"][result][0] += pressured
        agg["boost"][result][1] += tturns
        sweep = sweep_profile(lines, me)
        # Only games where we actually lost most of a team. Unrestricted, this share is not
        # a signal at all -- over the full archive it reads 58% in losses against 61% in
        # wins, because a win where two mons died and one opponent killed both scores 100%.
        # Gated at 4+ faints it separates properly (71% vs 48% on the mined cohort).
        if sweep and sweep[2] >= 4:
            agg["sweep_share"][result].append(sweep[1] / sweep[2])
        hx = hax_events(lines, me)
        agg["hax"][result][0] += hx["for"]
        agg["hax"][result][1] += hx["against"]
        vol, pun = switch_punish(lines, me)
        agg["switch"][result][0] += vol
        agg["switch"][result][1] += pun
        tr = tera_roi(lines, me)
        if tr:
            agg["tera"][result].append(tr)
        if saw_illusion(lines, me):
            agg["illusion"][result] += 1
        trace = os.path.join(os.path.dirname(path), tag + ".trace.json")
        if os.path.exists(trace):
            try:
                for e in json.load(open(trace, encoding="utf-8")):
                    for r in e.get("reorders", []):
                        agg["reorders"][result][r] = agg["reorders"][result].get(r, 0) + 1
            except (OSError, ValueError):
                pass
        if result == "lost":
            game_len = curve[-1][0] if curve else 0
            agg["_pending_losses"].append(
                (tag, flagged, first_behind, recovered, game_len,
                 hx["for"] - hx["against"], sweep))
            for sp, n in opp_killers(lines, me).items():
                agg["killers"][sp] = agg["killers"].get(sp, 0) + n

    # Second pass: classification needs the whole cohort's win-control first, i.e. which
    # signatures are actually loss-correlated here. Same rule as the table marker below.
    #
    # PER TURN, not per game. Losses in a normal cohort run ~30% longer than wins, so a
    # per-game rate reads that length difference as evidence: died_no_act scored 4.2x on
    # this cohort per game and 3.5x per turn, and the residual is mostly just 'the endgame
    # of a loss has more faints in it'. The old per-game version marked it loss-correlated
    # and it is not one.
    tl, tw = max(agg["turns"]["lost"], 1), max(agg["turns"]["won"], 1)
    correlated = {k for k in set(agg["sig"]["lost"]) | set(agg["sig"]["won"])
                  if agg["sig"]["lost"].get(k, 0) >= CORRELATE_MIN_EVENTS
                  and agg["sig"]["lost"].get(k, 0) / tl
                  > 2 * agg["sig"]["won"].get(k, 0) / tw}
    for tag, flagged, fb, rec, game_len, hax_net, sweep in agg.pop("_pending_losses"):
        label = classify_loss(flagged, hax_net, fb, rec, game_len, correlated, sweep)
        agg["classes"][label] = agg["classes"].get(label, 0) + 1
        agg["loss_lines"].append((tag, label, flagged, fb, hax_net, sweep))
    return agg


# Any single loss class above this share means the taxonomy has stopped discriminating and
# needs splitting. This exists because 'close' silently swallowed 86% of one cohort's losses
# and the dominant failure lived inside it, unnamed, for the whole cohort.
COLLAPSE_FRAC = 0.6


def wilson(k, n, z=1.96):
    """Wilson score interval for k/n. Printed next to every headline rate because at the
    cohort sizes this project runs (50-60 games) the interval is about +/-12 points, which
    is wider than most of the effects being tested -- that fact should be on screen, not
    left for the reader to remember."""
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def report(agg, baseline=None):
    c = agg["counts"]
    played = c["lost"] + c["won"]
    lo, hi = wilson(c["won"], played)
    print(f"\n=== {c['lost']} losses / {c['won']} wins"
          + (f" / {c['tie']} ties" if c["tie"] else "")
          + (f"  --  {c['won']/played:.0%} (95% CI {lo:.0%}-{hi:.0%})" if played else "")
          + " ===")
    for tag, label, flagged, fb, hax_net, sweep in agg["loss_lines"]:
        extras = []
        if fb is not None:
            extras.append(f"behind@t{fb}")
        if hax_net:
            extras.append(f"hax{hax_net:+d}")
        if sweep:
            sp, kills, faints, peak, tera = sweep
            extras.append(f"{sp}x{kills}/{faints}"
                          + (f"+{peak}" if peak else "") + ("T" if tera else ""))
        print(f"LOSS [{label:11s}] {tag}  {' '.join(extras)}")
        for k, turns in flagged.items():
            print(f"    {k}: turns {turns}")

    tl, tw = max(agg["turns"]["lost"], 1), max(agg["turns"]["won"], 1)
    small = c["lost"] < 15
    print(f"\n--- signature rates PER TURN "
          f"(L {agg['turns']['lost']}t / W {agg['turns']['won']}t)"
          f"{'  (n<15: directional only)' if small else ''} ---")
    keys = sorted(set(agg["sig"]["lost"]) | set(agg["sig"]["won"]))
    for k in keys:
        l = agg["sig"]["lost"].get(k, 0) / tl
        w = agg["sig"]["won"].get(k, 0) / tw
        marker = ("  <-- loss-correlated" if l > 2 * w
                  and agg["sig"]["lost"].get(k, 0) >= CORRELATE_MIN_EVENTS
                  and not small else "")
        base = ""
        if baseline:
            bt = max(baseline["turns"]["lost"], 1)
            base = f"   (baseline L {baseline['sig']['lost'].get(k, 0) / bt:5.3f})"
        print(f"  {k:14s} L {l:5.3f}   W {w:5.3f}{base}{marker}")
    lf, wf = agg["faints"]["lost"], agg["faints"]["won"]
    if lf and wf:
        l = agg["sig"]["lost"].get("died_no_act", 0) / lf
        w = agg["sig"]["won"].get("died_no_act", 0) / wf
        print(f"  died_no_act per faint: L {l:.2f}  W {w:.2f}"
              f"{'  <-- loss-correlated' if l > 1.5 * w else '  (faint-count artifact)'}")

    print("\n--- setup pressure (the cohort's cleanest separator) ---")
    for r in ("lost", "won"):
        pressured, tt = agg["boost"][r]
        peaks = agg["boost_peak"][r]
        mean_peak = sum(peaks) / len(peaks) if peaks else 0.0
        shares = agg["sweep_share"][r]
        print(f"  {r}: {pressured}/{tt} turns facing a +2-or-better active "
              f"= {pressured/max(tt,1):5.1%}   mean opponent peak +{mean_peak:.2f}"
              + (f"   top killer took {sum(shares)/len(shares):.0%} of our faints "
                 f"(n={len(shares)} games with 4+ faints)" if shares else ""))

    print("\n--- loss taxonomy (precedence: " + " > ".join(TAXONOMY) + ") ---")
    for k in TAXONOMY:
        n = agg["classes"].get(k, 0)
        if n:
            print(f"  {k:11s} {n:3d}  ({n/max(c['lost'],1):4.0%})")
    worst = max(agg["classes"].items(), key=lambda kv: kv[1], default=(None, 0))
    if c["lost"] and worst[1] / c["lost"] > COLLAPSE_FRAC:
        print(f"  !! '{worst[0]}' holds {worst[1]/c['lost']:.0%} of losses -- the taxonomy "
              f"has stopped discriminating.\n"
              f"     Split it before trusting this table; a class this broad is what hid "
              f"the last dominant failure.")

    print("--- luck ledger (crits/misses/full-para/flinch/confusion; sleep excluded) ---")
    for r in ("lost", "won"):
        f, a = agg["hax"][r]
        n = max(c[r], 1)
        print(f"  {r}: for us {f/n:.2f}/game, against us {a/n:.2f}/game, net {(f-a)/n:+.2f}")

    print("--- voluntary healthy switch-ins punished (>=40% direct dmg or KO on entry) ---")
    for r in ("lost", "won"):
        v, pu = agg["switch"][r]
        print(f"  {r}: {pu}/{v} = {pu/max(v,1):.0%}")

    print("--- tera ROI (DESCRIPTIVE ONLY -- outcome-confounded) ---")
    for r in ("lost", "won"):
        ts = agg["tera"][r]
        if ts:
            import statistics as st
            print(f"  {r}: tera'd {len(ts)}/{c[r]} games, median turn "
                  f"{st.median(t for t, _, _ in ts):.0f}, kills/tera "
                  f"{sum(k for _, k, _ in ts)/len(ts):.2f}, survived "
                  f"{sum(s for _, _, s in ts)/len(ts):.1f} turns")

    if agg["illusion"]["won"] or agg["illusion"]["lost"]:
        print(f"--- opponent Zoroark games: {agg['illusion']['won']}W / "
              f"{agg['illusion']['lost']}L ---")
    if any(agg["reorders"].values()):
        print("--- reranker overrides PER TURN (needs traces with 'reorders') ---")
        keys = sorted(set(agg["reorders"]["lost"]) | set(agg["reorders"]["won"]))
        for k in keys:
            print(f"  {k:12s} L {agg['reorders']['lost'].get(k,0)/tl:5.3f}"
                  f"   W {agg['reorders']['won'].get(k,0)/tw:5.3f}")
    if agg["killers"]:
        top = sorted(agg["killers"].items(), key=lambda kv: -kv[1])[:8]
        print("--- top killers in losses (approximate credit) ---")
        print("  " + ", ".join(f"{sp} x{n}" for sp, n in top if n >= 2))

    m = agg["mech"]
    if m:
        # The A/B readout. Outcome-INDEPENDENT by construction, so unlike everything above
        # these can be compared between two builds on one cohort each.
        dec = max(m.get("decisions", 0), 1)
        mv = max(m.get("our_moves", 0), 1)
        turns = max(m.get("turns", 0), 1)
        print(f"\n=== MECHANISM METRICS (all {played} games pooled; judge A/Bs on these) ===")
        rows = [
            ("anti-setup clicks", f"{m.get('antisetup', 0)}", f"{m.get('antisetup',0)/mv*1000:6.1f} per 1k of our moves"),
            ("hazards set", f"{m.get('hazard_set', 0)}", f"{m.get('hazard_set',0)/mv*1000:6.1f} per 1k"),
            ("hazard removal", f"{m.get('hazard_clear', 0)}", f"{m.get('hazard_clear',0)/mv*1000:6.1f} per 1k"),
            ("wasted recovery runs", f"{m.get('wasted_recovery', 0)}", f"{m.get('wasted_recovery',0)/played:6.2f} per game" if played else ""),
            ("our switch rate", f"{m.get('our_switches', 0)}", f"{m.get('our_switches',0)/turns:6.3f} per turn"),
            ("flat pooled policy", f"{m.get('flat_policy', 0)}", f"{m.get('flat_policy',0)/dec:6.1%} of decisions (gap < {FLAT_POLICY_EPS})"),
            ("value-net overrides", f"{m.get('value_override', 0)}", f"{m.get('value_override',0)/dec:6.1%} of decisions"),
            ("mixed-strategy picks", f"{m.get('mixed', 0)}", f"{m.get('mixed',0)/dec:6.1%} of decisions"),
            ("fallback / crash picks", f"{m.get('fallback', 0)}", f"{m.get('fallback',0)/dec:6.2%} of decisions"),
        ]
        if m.get("boost_turns"):
            rows.append(("turns facing +2 (trace)", f"{m.get('boost_pressured', 0)}",
                         f"{m.get('boost_pressured',0)/m['boost_turns']:6.1%} of decisions"))
        if m.get("ms_n"):
            rows.append(("mean think time", f"{m['ms_total']/m['ms_n']:.0f} ms",
                         "of a 150 s/turn allowance"))
        for name, raw, rate in rows:
            base = ""
            if baseline and baseline.get("mech"):
                bm = baseline["mech"]
                bdec = max(bm.get("decisions", 0), 1)
                bmv = max(bm.get("our_moves", 0), 1)
                key = {"anti-setup clicks": ("antisetup", bmv, 1000),
                       "flat pooled policy": ("flat_policy", bdec, 1),
                       "value-net overrides": ("value_override", bdec, 1)}.get(name)
                if key:
                    fld, den, scale = key
                    base = f"   (baseline {bm.get(fld,0)/den*scale:.3f})"
            print(f"  {name:24s} {raw:>8s}   {rate}{base}")


def main():
    ap = argparse.ArgumentParser(description="Mine replays for blunder signatures.")
    ap.add_argument("--dir", default=paths.LADDER_REPLAY_DIR)
    ap.add_argument("--include-file", action="append", default=[],
                    help="extra replay html(s) to mine (counted as losses)")
    ap.add_argument("--username", default=BOT_USERNAME,
                    help="the bot's username in the replays")
    ap.add_argument("--since", default=None,
                    help="only mine replays modified after 'YYYY-MM-DD HH:MM'")
    ap.add_argument("--last", type=int, default=None, metavar="N",
                    help="only mine the N most recent replays -- the usual way to look at "
                         "one cohort instead of the whole cumulative archive")
    ap.add_argument("--baseline", default=None,
                    help="second replay dir to diff signature rates against")
    ap.add_argument("--baseline-username", default=None,
                    help="bot username in the baseline dir (default: same as --username)")
    ap.add_argument("--baseline-last", type=int, default=None, metavar="N",
                    help="--last, applied to the baseline dir")
    args = ap.parse_args()

    since = None
    if args.since:
        import datetime as _dt
        since = _dt.datetime.strptime(args.since, "%Y-%m-%d %H:%M").timestamp()

    agg = analyze_dir(args.dir, args.username, include=args.include_file, since=since,
                      last=args.last)
    if not any(agg["counts"].values()):
        print(f"No replays found in {args.dir}.")
        return
    baseline = None
    if args.baseline:
        baseline = analyze_dir(args.baseline, args.baseline_username or args.username,
                               last=args.baseline_last)
        b = baseline["counts"]
        if not (b["lost"] or b["won"]):
            print(f"!! baseline {args.baseline} matched 0 games for username "
                  f"'{args.baseline_username or args.username}' -- check --baseline-username")
            baseline = None
        else:
            print(f"[baseline: {args.baseline} -- {b['lost']}L/{b['won']}W]")
    report(agg, baseline=baseline)


if __name__ == "__main__":
    main()
