r"""Mine saved ladder replays (+ decision traces) for recurring blunder signatures.

The improvement loop that actually moves ELO: play ladder games (ladder.py saves every
replay to replays/ladder/ and, since the robust-vote commit, a .trace.json of what the
search considered each turn), then run this to cluster losses into failure classes:

  * immune_click   -- we used a move the active opponent was immune to (the Scale Shot class)
  * fail_click     -- our move failed outright (e.g. Rest at full HP)
  * died_no_act    -- our active fainted on a turn it neither moved nor was switched in
                      (outsped + KO'd: the "stayed in on lethal" signature)
  * opp_boosted    -- the opponent's active reached a net +3 or more boost stages
                      (the setup-snowball signature: we let something set up)
  * slept_hit      -- turns we spent asleep while taking damage (Rest-loop cost)
  * tera_wasted    -- our Terastallized mon fainted within 2 turns of tera without a
                      single opponent faint in between (burned the once-per-game
                      resource for nothing; 48% of tera-losses vs 12% of tera-wins
                      when first measured, 2026-07-02)

Signatures are heuristics for *where to look*, not verdicts -- open the replay/trace for
any flagged turn before calling it a bug. Wins are mined too, as a control: a signature
that shows up equally in wins is background noise, not a loss cause.

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
    """The protocol log lines of a saved replay, and which side (p1/p2) is the bot."""
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
    """Signature counts for one game, from the bot's (side `me`) perspective."""
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
            acted_this_turn = True     # it "acted" (slept); died_no_act shouldn't double-count
            slept_this_turn = True
        elif tag == "-damage" and len(p) > 2 and mine_prefix(p[2]) and slept_this_turn:
            sig["slept_hit"].append(turn)
            slept_this_turn = False    # one per turn
        elif tag == "-immune" and len(p) > 2 and opp_prefix(p[2]):
            # our attack was immune-d if our move was the most recent move line
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


def main():
    ap = argparse.ArgumentParser(description="Mine ladder replays for blunder signatures.")
    ap.add_argument("--dir", default=os.path.join("replays", "ladder"))
    ap.add_argument("--include-file", action="append", default=[],
                    help="extra replay html(s) to mine (counted as losses)")
    ap.add_argument("--username", default=BOT_USERNAME,
                    help="the bot's username in the replays (default influxobot)")
    args = ap.parse_args()

    games = []
    for path in sorted(glob.glob(os.path.join(args.dir, "*.html"))):
        name = os.path.basename(path)
        result = "lost" if name.startswith("lost") else \
                 "won" if name.startswith("won") else "tie"
        games.append((path, result))
    games += [(f, "lost") for f in args.include_file]
    if not games:
        print(f"No replays found in {args.dir}.")
        return

    totals = {"won": {}, "lost": {}, "tie": {}}
    counts = {"won": 0, "lost": 0, "tie": 0}
    for path, result in games:
        lines, me = read_log(path, args.username)
        if me is None:
            print(f"  ?? could not find influxobot in {path}")
            continue
        counts[result] += 1
        sig = mine_game(lines, me)
        faints = len(sig.pop("_faints", []))
        totals[result]["_faints"] = totals[result].get("_faints", 0) + faints
        flagged = {k: v for k, v in sig.items() if v}
        for k, v in flagged.items():
            totals[result][k] = totals[result].get(k, 0) + len(v)
        if result == "lost" and flagged:
            tag = os.path.basename(path).replace(".html", "")
            trace = os.path.join(os.path.dirname(path), tag + ".trace.json")
            has_trace = os.path.exists(trace)
            print(f"LOSS {tag}{' [trace]' if has_trace else ''}")
            for k, turns in flagged.items():
                print(f"    {k}: turns {turns}")

    print(f"\n=== signature rate per game (losses n={counts['lost']} "
          f"vs wins n={counts['won']}) ===")
    lf = totals["lost"].pop("_faints", 0)
    wf = totals["won"].pop("_faints", 0)
    keys = sorted(set(totals["lost"]) | set(totals["won"]))
    for k in keys:
        l = totals["lost"].get(k, 0) / max(counts["lost"], 1)
        w = totals["won"].get(k, 0) / max(counts["won"], 1)
        marker = "  <-- loss-correlated" if l > 2 * w and l > 0.3 else ""
        print(f"  {k:14s} losses {l:4.2f}/game   wins {w:4.2f}/game{marker}")
    # died_no_act is confounded by faint count (losses have ~6 by definition); the per-faint
    # rate is the honest comparison for it.
    if lf and wf:
        l = totals["lost"].get("died_no_act", 0) / lf
        w = totals["won"].get("died_no_act", 0) / wf
        print(f"  died_no_act per faint: losses {l:.2f}   wins {w:.2f}"
              f"{'   <-- loss-correlated' if l > 1.5 * w else '   (mostly faint-count artifact)'}")


if __name__ == "__main__":
    main()
