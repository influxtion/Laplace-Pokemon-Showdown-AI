# Teams

One Showdown export per file, `<name>.txt`. The file stem is what `--team` takes:

```powershell
.\.venv\Scripts\python.exe -u -m laplace.cli.ladder_ou --team balance
```

reads `teams/balance.txt`.

## Getting a team in here

Open the Showdown teambuilder, pick the team, **Import/Export**, copy the whole thing, and
paste it into a new file. That text is the format — nothing to convert:

```
Great Tusk @ Booster Energy
Ability: Protosynthesis
Tera Type: Steel
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Headlong Rush
- Ice Spinner
- Rapid Spin
- Close Combat

Gholdengo @ Air Balloon
...
```

Six Pokemon, separated by a blank line. `laplace.ou.teams.load` parses and counts the file
before a single game is queued, so a malformed export fails at startup with a readable
message rather than at match time, several minutes into a run.

## Legality

Parsing is not validation. Showdown checks tier legality — banned Pokemon, banned moves,
illegal learnsets — when the **match is made**, and an illegal team is rejected with a popup
that poke-env has no handler for: the game simply never starts, and the run sits in the
queue looking like an empty ladder. `PopupWatcher` in `laplace/cli/ladder.py` reprints that
popup so it reads as the one-line error it should be, but it is still cheaper to check
first. With the local server checked out:

```powershell
# packed format -- laplace.ou.teams.load gives you it, or export from the teambuilder
cd server; node pokemon-showdown validate-team gen9ou < ..\packed-team.txt
```

Exit 0 means legal; anything else prints the reason (`Dragapult's move Tera Blast is
banned.`).

## Why teams live in the repo

In Random Battle the team is generated and the bot brings nothing. In OU the team is half
the bot: the same search plays a completely different game behind hyper-offence than behind
balance, and a win rate is a statement about the search **and** the team together. Keeping
teams committed and named is what makes "the bot went 12-8" mean something reproducible —
swapping teams invalidates a comparison as thoroughly as swapping engines does.

## Lead choice

By default `laplace.ou.teampreview` scores every one of your six against the opponent's
previewed six and leads with the best. If a team has exactly one sensible lead, say so
instead and skip the scoring:

```powershell
... --team balance --lead 1        # always lead with slot 1
```
