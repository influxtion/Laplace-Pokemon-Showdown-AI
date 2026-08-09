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


def mine_game(lines, me):
    """Signature -> turns it fired on, for one game, from side `me`'s perspective."""
    opp = "p2" if me == "p1" else "p1"
    sig = {"immune_click": [], "fail_click": [], "died_no_act": [],
           "opp_boosted": [], "slept_hit": [], "tera_wasted": []}
    turn = 0
    opp_boosts = {}
    opp_boost_flagged = False
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
        elif tag in ("-boost", "-unboost") and len(p) > 4 and opp_prefix(p[2]):
            d = int(p[4]) * (1 if tag == "-boost" else -1)
            opp_boosts[p[3]] = opp_boosts.get(p[3], 0) + d
            net = sum(v for v in opp_boosts.values() if v > 0)
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
                if (tera_mon is not None and p[2].split(":")[-1].strip() == tera_mon
                        and turn - tera_turn <= 2 and opp_faints_since_tera == 0):
                    sig["tera_wasted"].append(turn)
                    tera_mon = None      # count once
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
                f = _frac(p[-1] if tag in ("switch", "drag", "replace") else p[3])
                if f is None and tag in ("switch", "drag", "replace"):
                    f = _frac(p[3])
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
            f = _frac(p[-1]) or _frac(p[3])
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


def saw_illusion(lines, me):
    """Did the opponent reveal a Zoroark forme this game (via replace or naming)?"""
    opp = "p2" if me == "p1" else "p1"
    return any(l.split("|")[1:2] == ["replace"] and l.split("|")[2].startswith(f"{opp}a")
               or (f"|{opp}a: Zoroark" in l) for l in lines)


def classify_loss(sig, hax_net, first_behind, recovered, game_len, correlated):
    """Coarse loss taxonomy, precedence-ordered: blundered > hax > out_traded > close.
    A game may qualify for several. Pointer for where to spend reading time.

    'blundered' only counts signatures that are LOSS-CORRELATED in this cohort (rate in
    losses > 2x wins). An immune_click that appears equally in wins is hidden-set background
    noise, and labelling those losses 'blundered' buries the attrition story -- caught on
    the FP cohort validation."""
    blunder_sigs = {"immune_click", "fail_click", "tera_wasted"} & correlated
    if any(sig.get(k) for k in blunder_sigs):
        return "blundered"
    if hax_net <= -3:
        return "hax"
    if first_behind is not None and game_len and first_behind <= game_len * 0.5 \
            and not recovered:
        return "out_traded"
    return "close"


def analyze_dir(directory, username, include=(), since=None):
    """Aggregate every analysis over a replay dir. Returns the totals dict."""
    games = []
    for path in sorted(glob.glob(os.path.join(directory, "*.html"))):
        if since is not None and os.path.getmtime(path) < since:
            continue
        name = os.path.basename(path)
        result = "lost" if name.startswith("lost") else \
                 "won" if name.startswith("won") else "tie"
        games.append((path, result))
    games += [(f, "lost") for f in include]

    agg = {
        "counts": {"won": 0, "lost": 0, "tie": 0},
        "sig": {"won": {}, "lost": {}, "tie": {}},
        "faints": {"won": 0, "lost": 0, "tie": 0},
        "hax": {"won": [0, 0], "lost": [0, 0]},          # [for, against]
        "switch": {"won": [0, 0], "lost": [0, 0]},        # [voluntary, punished]
        "tera": {"won": [], "lost": []},                  # (turn, kills, survived)
        "classes": {},                                    # loss label -> count
        "killers": {},
        "illusion": {"won": 0, "lost": 0},
        "reorders": {"won": {}, "lost": {}},
        "loss_lines": [],
        "_pending_losses": [],
    }
    for path, result in games:
        lines, me = read_log(path, username)
        if me is None:
            print(f"  ?? could not find {username} in {path}")
            continue
        agg["counts"][result] += 1
        sig = mine_game(lines, me)
        faints = len(sig.pop("_faints", []))
        agg["faints"][result] += faints
        flagged = {k: v for k, v in sig.items() if v}
        for k, v in flagged.items():
            agg["sig"][result][k] = agg["sig"][result].get(k, 0) + len(v)
        if result == "tie":
            continue
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
        tag = os.path.basename(path).replace(".html", "")
        trace = os.path.join(os.path.dirname(path), tag + ".trace.json")
        if os.path.exists(trace):
            try:
                for e in json.load(open(trace, encoding="utf-8")):
                    for r in e.get("reorders", []):
                        agg["reorders"][result][r] = agg["reorders"][result].get(r, 0) + 1
            except (OSError, ValueError):
                pass
        if result == "lost":
            curve, fb, rec = hp_curve(lines, me)
            game_len = curve[-1][0] if curve else 0
            agg["_pending_losses"].append(
                (tag, flagged, fb, rec, game_len, hx["for"] - hx["against"]))
            for sp, n in opp_killers(lines, me).items():
                agg["killers"][sp] = agg["killers"].get(sp, 0) + n

    # Second pass: classification needs the whole cohort's win-control first, i.e. which
    # signatures are actually loss-correlated here. Same rule as the table marker below.
    nl, nw = max(agg["counts"]["lost"], 1), max(agg["counts"]["won"], 1)
    correlated = {k for k in set(agg["sig"]["lost"]) | set(agg["sig"]["won"])
                  if agg["sig"]["lost"].get(k, 0) / nl > 0.3
                  and agg["sig"]["lost"].get(k, 0) / nl
                  > 2 * agg["sig"]["won"].get(k, 0) / nw}
    for tag, flagged, fb, rec, game_len, hax_net in agg.pop("_pending_losses"):
        label = classify_loss(flagged, hax_net, fb, rec, game_len, correlated)
        agg["classes"][label] = agg["classes"].get(label, 0) + 1
        agg["loss_lines"].append((tag, label, flagged, fb, hax_net))
    return agg


def report(agg, baseline=None):
    c = agg["counts"]
    print(f"\n=== {c['lost']} losses / {c['won']} wins"
          + (f" / {c['tie']} ties" if c["tie"] else "") + " ===")
    for tag, label, flagged, fb, hax_net in agg["loss_lines"]:
        extras = []
        if fb is not None:
            extras.append(f"behind@t{fb}")
        if hax_net:
            extras.append(f"hax{hax_net:+d}")
        print(f"LOSS [{label:10s}] {tag}  {' '.join(extras)}")
        for k, turns in flagged.items():
            print(f"    {k}: turns {turns}")

    nl, nw = max(c["lost"], 1), max(c["won"], 1)
    small = c["lost"] < 15
    print(f"\n--- signature rates per game{'  (n<15: directional only)' if small else ''} ---")
    keys = sorted(set(agg["sig"]["lost"]) | set(agg["sig"]["won"]))
    for k in keys:
        l = agg["sig"]["lost"].get(k, 0) / nl
        w = agg["sig"]["won"].get(k, 0) / nw
        marker = "  <-- loss-correlated" if l > 2 * w and l > 0.3 and not small else ""
        base = ""
        if baseline:
            bl = baseline["sig"]["lost"].get(k, 0) / max(baseline["counts"]["lost"], 1)
            base = f"   (baseline L {bl:4.2f})"
        print(f"  {k:14s} L {l:4.2f}   W {w:4.2f}{base}{marker}")
    lf, wf = agg["faints"]["lost"], agg["faints"]["won"]
    if lf and wf:
        l = agg["sig"]["lost"].get("died_no_act", 0) / lf
        w = agg["sig"]["won"].get("died_no_act", 0) / wf
        print(f"  died_no_act per faint: L {l:.2f}  W {w:.2f}"
              f"{'  <-- loss-correlated' if l > 1.5 * w else '  (faint-count artifact)'}")

    print("\n--- loss taxonomy (precedence: blundered > hax > out_traded > close) ---")
    for k in ("blundered", "hax", "out_traded", "close"):
        if agg["classes"].get(k):
            print(f"  {k:10s} {agg['classes'][k]}")

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
        print("--- reranker overrides per game (needs traces with 'reorders') ---")
        keys = sorted(set(agg["reorders"]["lost"]) | set(agg["reorders"]["won"]))
        for k in keys:
            print(f"  {k:12s} L {agg['reorders']['lost'].get(k,0)/nl:.2f}"
                  f"   W {agg['reorders']['won'].get(k,0)/nw:.2f}")
    if agg["killers"]:
        top = sorted(agg["killers"].items(), key=lambda kv: -kv[1])[:8]
        print("--- top killers in losses (approximate credit) ---")
        print("  " + ", ".join(f"{sp} x{n}" for sp, n in top if n >= 2))


def main():
    ap = argparse.ArgumentParser(description="Mine replays for blunder signatures.")
    ap.add_argument("--dir", default=os.path.join("replays", "ladder"))
    ap.add_argument("--include-file", action="append", default=[],
                    help="extra replay html(s) to mine (counted as losses)")
    ap.add_argument("--username", default=BOT_USERNAME,
                    help="the bot's username in the replays")
    ap.add_argument("--since", default=None,
                    help="only mine replays modified after 'YYYY-MM-DD HH:MM'")
    ap.add_argument("--baseline", default=None,
                    help="second replay dir to diff signature rates against")
    ap.add_argument("--baseline-username", default=None,
                    help="bot username in the baseline dir (default: same as --username)")
    args = ap.parse_args()

    since = None
    if args.since:
        import datetime as _dt
        since = _dt.datetime.strptime(args.since, "%Y-%m-%d %H:%M").timestamp()

    agg = analyze_dir(args.dir, args.username, include=args.include_file, since=since)
    if not any(agg["counts"].values()):
        print(f"No replays found in {args.dir}.")
        return
    baseline = None
    if args.baseline:
        baseline = analyze_dir(args.baseline, args.baseline_username or args.username)
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
