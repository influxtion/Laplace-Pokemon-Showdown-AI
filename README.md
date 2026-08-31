# Laplace, a Pokémon Showdown battle AI

Laplace plays [Pokémon Showdown](https://pokemonshowdown.com/) Gen 9 Random Battle. It peaked at
**2231 Elo / ~83.3 GXE** on the live ladder, top 1% of ranked players and #135 in the world, well
past the point where the opponents are strong humans rather than other bots. It garnered over 200k views on social media.

The bot utilizes MCTS and a trained value net to rank actions. Every turn, it guesses the opponents' hidden stats (items, abilities, stats, etc) and guesses forward with a real Gen 9 engine, picking from the pooled result. 

It also plays **Gen 9 OU**, bringing a team of your own — same search, same guards, a
different prior over what the opponent is holding. That ladder is new and unproven; the
numbers above are Random Battle's. See [Playing OU](#playing-ou).

The name comes from Lapras, whose Japanese name is *Laplace* (ラプラス), and from
[Laplace's demon](https://en.wikipedia.org/wiki/Laplace%27s_demon), the idea of an intelligence
that could predict the future if it knew the complete hidden state of the world. Filling in the
opponent's unseen set and rolling the game forward is a rough approximation of that.

## Why it's hard

Random Battle is imperfect-information, simultaneous-move and stochastic all at once. The
opponent's moves, item, ability and Tera type stay hidden until revealed, both players commit at
the same time, and damage rolls and crits add chance on top. You can't just read the board and
pick the best reply, because half the board is unknown and "best" depends on what the opponent
does on the very same turn. The one thing that makes it tractable is that teams aren't arbitrary:
Random Battle builds them from a public recipe, so the space of hidden sets is large but
structured, and you can weight your guesses by how the generator actually works.

## How it works

1. **Read the state.** [`poke-env`](https://github.com/hsahovic/poke-env) tracks what the server
   has revealed: moves, items, HP, boosts, hazards, weather, field conditions.

2. **Fill in the unknowns.** The bot samples eight complete opponent teams with real items,
   abilities, moves and Tera types, drawn from Random Battle usage counts and filtered against
   everything seen so far. The filter is the interesting part: it folds in revealed moves, Choice
   locks inferred from move history, and Choice Scarf verdicts read off turn order (Random Battle
   speeds are deterministic, so an opponent that outsped when it shouldn't have is holding a
   Scarf). This is `agent/poke_engine_adapter.py`, the core of the project.

3. **Search each world.** Each sampled team goes to
   [`poke-engine`](https://github.com/pmariglia/poke-engine), a fast Rust Gen 9 engine running
   MCTS at 150 ms per world, roughly 2M visits a turn in total. Damage, priority, hazards,
   residuals, abilities, turn order and Terastallization are modelled exactly, and MCTS handles
   the simultaneous-move turn properly where plain minimax would let the opponent peek at our move.

4. **Pool and guard.** The per-world rankings get combined, then deterministic guards veto known
   blunders: spending a turn on a move that does nothing, locking the last Pokémon into a status
   move, clicking into an immunity. The value net re-ranks the handful of turns where the engine's
   own evaluation is weak, mainly when it under-fears an opponent setting up.

5. **Play a mixed strategy.** Always taking the top move is exploitable, since humans learn to
   switch an immunity absorber into a predictable click. So among near-tied candidates the final
   pick is sampled. The exception is Terastallization, which happens once a game and never on a
   coin flip.

## How it was built

Every ladder game is saved as a replay plus a per-turn trace of what the search considered.
`analysis/mine_losses.py` clusters losses into recurring signatures (a wasted turn, a move clicked
into an immunity, a Pokémon that died without acting, a Tera burned for nothing) using wins as a
control, so a signature has to actually separate losses from wins. Each signature then gets
reproduced, traced to a specific flaw, and fixed with the smallest change that addresses it. The
no-op guard is a good example: the engine correctly predicted that a full-HP recovery move would
do nothing, but its flat evaluation kept clicking it anyway.

Nothing ships without a head-to-head test: new code vs old code, self-play on a local server, 60+
games, judged on a Wilson interval. This is the part that keeps the project honest, and plausible
ideas failed it repeatedly. Feeding the search real item statistics without Choice-lock modelling
*lost* at 39%. Adding search time did nothing, because the engine had already converged. Falsified
ideas are logged next to the ones that worked so a dead lever doesn't get retried later.

## Setup

Needs Python 3.12, Node.js 18+, git, and a Rust toolchain ([rustup](https://rustup.rs)) to build
poke-engine. From the project root:

```powershell
# 1. Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # prompt should show "(.venv)"
pip install -r requirements.txt

# 2. Local Showdown server (supplies the Random Battle data, and hosts local games)
git clone --depth 1 https://github.com/smogon/pokemon-showdown.git server
cd server; npm install; cd ..

# 3. poke-engine, built with the Gen 9 Terastallization feature
pip install -v --force-reinstall --no-cache-dir "poke-engine==0.0.47" `
  --config-settings="build-args=--features poke-engine/terastallization --no-default-features"

# 4. The bot itself, editable so `laplace` imports from anywhere
pip install -e . --no-deps
```

`--no-deps` keeps pip from touching the poke-engine build you just did by hand. Editable means the
install points at `src/`, so edits take effect with no reinstall. A `ModuleNotFoundError` almost
always means the venv isn't active.

To play the real ladder, register a username at play.pokemonshowdown.com and put the credentials
in a gitignored `.env` at the project root:

```
SHOWDOWN_USERNAME=YourBotName
SHOWDOWN_PASSWORD=your-password
```

## Running it

The ladder connects to the official server, so it doesn't need the local one:

```powershell
.\.venv\Scripts\python.exe -u -m laplace.cli.ladder                # 10 ranked games
.\.venv\Scripts\python.exe -u -m laplace.cli.ladder --battles 40   # a full cohort
```

Every game lands in `replays/ladder/` as `<result>-<battle-tag>.html` plus a `.trace.json` of the
per-turn decisions, which is the raw material for the loss-mining loop. Open the `.html` in a
browser to watch it back. Anything in flight when you Ctrl-C is flushed on exit.

A cohort is meant to run unattended for hours, so two things happen on their own. A dropped
websocket doesn't end the run: a watchdog spots the dead socket, archives what that connection
played, and resumes on a fresh one with continuous game numbering. And each finished game's rating
is checked against the record in `replays/ladder/elo_high.json`; beating it archives a second copy
under `records/` and stops the run rather than handing a fresh peak straight back to the next
opponent. `--reconnects 0` and `--no-stop-on-record` turn these off.

Games can also be published to `replay.pokemonshowdown.com`. Those links are public, so it's
opt-in per run: `--upload-first` does the first 5, `--upload-first N` the first N.

### Watching a game

`analyze_battle` plays one ladder game and turns the terminal into a live analysis board. Same
bot, same shipped config (it imports `ladder`'s `build_agent`, so the two can't drift apart):

```powershell
.\.venv\Scripts\python.exe -u -m laplace.cli.analyze_battle           # one rated game, narrated
.\.venv\Scripts\python.exe -u -m laplace.cli.analyze_battle --upload  # + a public replay link
.\.venv\Scripts\python.exe -u -m laplace.cli.analyze_battle --local   # local server, no rated game
```

## Playing OU

The same bot also plays **Gen 9 OU**. The search, the guards, the mixed root and the whole
ladder harness are shared; what changes is everything Random Battle gets for free.

```powershell
# 1. Build the OU set priors (once, and again whenever the tier shifts)
.\.venv\Scripts\python.exe -u -m laplace.cli.fetch_ou_data

# 2. Put a team in teams/ as a Showdown export -- see teams/README.md
#    teams/balance.txt  ->  --team balance

# 3. Ladder
.\.venv\Scripts\python.exe -u -m laplace.cli.ladder_ou --team balance
.\.venv\Scripts\python.exe -u -m laplace.cli.ladder_ou --team balance --battles 40
.\.venv\Scripts\python.exe -u -m laplace.cli.analyze_battle --format gen9ou --team balance
```

Games land in `replays/ladder_ou/`, with their own rating log and their own Elo record, so
the two formats never contaminate each other's history. Every `laplace.cli.ladder` flag
works: `ladder_ou` is that file with `--format gen9ou` filled in.

### What actually differs

**Team preview removes the biggest guess.** Random Battle hides the opponent's whole team,
so the determinizer has to invent unseen bench slots out of the dex -- the single largest
source of noise in its worlds. OU shows all six before turn 1, so what stays hidden is each
Pokemon's *set*, not its identity.

**Sets are chosen, not generated.** There is no sheet to sample from, so the prior is the
metagame itself, from two public sources that fail in opposite directions. Hand-written
Smogon analysis sets are *coherent* -- moves, item, nature and Tera came from one set
somebody would really build -- but narrow: 108 species, 1.9 sets each, and they can express
only ~85% of the item mass those species actually run (Rocky Helmet is 34% of real Zapdos
and appears in no Zapdos set at all). Smogon usage *marginals* cover everything with the
real frequencies, but draw each field independently, so they also build Pokemon nobody
would bring. Measured over the top 25 species:

| prior | item fidelity | incoherent worlds |
|---|---|---|
| curated only | 78.0% | 0.94% |
| marginal only | 98.7% | 7.67% |
| **50/50 mix** | **88.8%** | **4.31%** |

So the worlds are drawn from a *mixture*, not a preference — determinization is exactly the
mechanism for hedging between two priors, since the worlds get pooled anyway. Within the
curated sets, which set you get is weighted by what the ladder actually plays rather than
uniformly (`CuratedSets._weights`; worth +5.9pp of fidelity). Both files are condensed and
committed by `fetch_ou_data`; nothing hits the network at play time.

**Stats are part of the set.** Random Battle fixes every spread, which is why its adapter
can estimate any stat from base + level. An OU player picks the EVs and the nature, and the
same Great Tusk is a 0-EV wall or a 252-EV lead -- a ~60% swing on a stat. So the spread is
sampled *with* the set and the stats computed from it. That also forces the Choice Scarf
inference to change shape: it now tests against the *fastest legal* spread rather than a
single estimate, because "it outsped me" is only evidence of a Scarf if no legal spread
could have done it.

**Illusion gets easier, not harder.** In Random Battle any Pokemon might be a disguised
Zoroark and the only tell is an off-movepool click. With team preview the disguise is
impossible unless a Zoroark was previewed, and it is over once one has been seen.

**One decision has no position to search.** Team preview happens before anything is on the
field, so poke-engine has nothing to roll and the lead is chosen by a matchup heuristic
(`laplace.ou.teampreview`): what each of ours threatens and is threatened by across their
six, who wins the speed race, and whether it has a turn-1 job. `--lead N` overrides it.

**The value net is off.** It was trained on Random Battle self-play states and has no
standing to re-rank OU positions; an out-of-distribution reranker with authority over
near-ties is worse than none. Re-enable it only after training one on OU data.

**The team is now part of the bot.** A result is a statement about the search *and* the
team. Swapping teams invalidates a comparison as thoroughly as swapping engines does, which
is why teams are committed and named rather than pasted into a flag.

### Other entry points

These play games, so they need the local server running in another terminal
(`cd server; node pokemon-showdown start --no-security`):

```powershell
.\.venv\Scripts\python.exe -u -m laplace.cli.eval_engine        # quick check vs the built-in heuristic
.\.venv\Scripts\python.exe -u -m laplace.cli.bench_foulplay     # benchmark vs Foul Play
.\.venv\Scripts\python.exe -u -m laplace.cli.gen_value_data     # self-play value-net training data
```

These two work offline, reading the saved replays and data chunks:

```powershell
.\.venv\Scripts\python.exe -u -m laplace.analysis.mine_losses   # cluster replays into blunder signatures
.\.venv\Scripts\python.exe -u -m laplace.cli.train_value        # retrain -> models/value_net.pt
```

Retrain the value net only after a real strength jump. It should model games at the bot's current
level, not an older one.

## Repo layout

The bot is one installable package, `laplace`, under `src/`. The split that matters is
between the **search**, which is format-agnostic, and a **metagame**, which is everything
the search cannot derive from the protocol. `laplace.agent.metagame.Metagame` is that seam:
it supplies a determinizer, a set prior, and a team-preview decision, and it is the only
thing `laplace.ou` had to provide.

```
src/laplace/
├── paths.py                     every on-disk location, resolved from the repo root
├── agent/                       the search, and the Gen 9 Random Battle metagame
│   ├── engine_search.py         determinized MCTS, vote pooling, guards, mixed root
│   ├── metagame.py              the seam: what a format has to supply
│   ├── poke_engine_adapter.py   observed battle -> concrete engine states (randbats)
│   ├── state_common.py          the format-agnostic half of that translation
│   └── knowledge.py             set/ability prediction and damage estimation
├── ou/                          the Gen 9 OU metagame
│   ├── usage.py                 the set prior: curated Smogon sets + usage marginals
│   ├── knowledge.py             ability prediction and stat estimation over that prior
│   ├── adapter.py               observed battle -> concrete engine states (OU)
│   ├── teams.py                 our own team: load, validate, hand to poke-env
│   ├── teampreview.py           who leads -- the one decision with nothing to search
│   └── metagame.py              the five above, bundled
├── value/
│   ├── value_features.py        368 hand-built features over an engine state
│   └── net.py                   the network itself
├── analysis/
│   ├── live_analysis.py         the live terminal board
│   └── mine_losses.py           replay clustering
└── cli/                         entry points, run with python -m
```

`data/` holds the Random Battle usage data and, under `data/ou/`, the two committed OU
priors; `models/value_net.pt` is the shipped net; `teams/` holds the OU teams (see
[`teams/README.md`](teams/README.md)); and `legacy/` archives the pre-search era, the
heuristic bot and the PPO experiments (see [`legacy/README.md`](legacy/README.md)).
`server/` and `replays/` are generated, not committed. The install also puts
`laplace-ladder`, `laplace-ou`, `laplace-fetch-ou`, `laplace-analyze`, `laplace-mine` and
`laplace-train-value` on PATH.
