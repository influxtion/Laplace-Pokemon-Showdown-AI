# Laplace, a Pokémon Showdown battle AI

Laplace plays [Pokémon Showdown](https://pokemonshowdown.com/) Gen 9 Random Battle and
reached a peak of **2137 Elo** on the live ladder, top 1% of active, ranked players, and well into
the range where the opponents are strong humans rather than other bots. The bot achieved top 500 (#447) on the ladder.
This is almost at the level of professional players and at the top of the best human pokemon players. 

The bot is a combination of a search policy backed by a trained value net portion with 368 hand-built
features, trained on self-play games to predict the winner (~70% accuracy) and used
only as a tie-breaker on maybe 5–10% of turns. Everything else is search, which is why the bot is
judged on win rates with confidence intervals, not loss curves. This acts as an "intuition" kind of layer to
assist against sweeps and other common threats hard to search for. 

The name comes from Lapras, whose Japanese name is *Laplace* (ラプラス), and from
[Laplace's demon](https://en.wikipedia.org/wiki/Laplace%27s_demon), the idea of an intelligence
that could predict the future if it knew the complete hidden state of the world. Filling in the
opponent's unseen set and rolling the game forward is a rough approximation of exactly that.

## Why it's hard

Random Battle is an imperfect-information, simultaneous-move, stochastic game. The opponent's
moves, item, ability, and Tera type stay hidden until revealed; both players commit at the same
time; and damage rolls, crits, and secondary effects add chance on top. You can't just read the
board and pick the best reply; the board is half-unknown, and the "best" reply depends on what
the opponent does on the same turn.

The one thing that makes it tractable: teams aren't arbitrary. Random Battle generates them from
a public recipe, so the space of hidden sets is large but structured, and you can weight your
guesses by how the game actually builds teams.

## How it works

Each turn runs through five steps.

1. **Read the state.** [`poke-env`](https://github.com/hsahovic/poke-env) tracks everything the
   server has revealed: moves, items, HP, boosts, hazards, weather, field conditions.

2. **Fill in the unknowns.** The opponent's exact set is hidden, so the bot samples several
   complete, concrete opponent teams — full item, ability, move, and Tera sets drawn from real
   Random Battle usage counts and filtered to be consistent with everything seen so far. That
   filter matters: it folds in revealed moves, Choice locks inferred from move history, and
   Choice Scarf verdicts read off of turn order (Random Battle speeds are deterministic, so an
   opponent that outsped when it shouldn't have is holding a Scarf). This is `poke_engine_adapter.py`,
   and it is the core of the project.

3. **Search each world.** Every sampled team is handed to
   [`poke-engine`](https://github.com/pmariglia/poke-engine), a fast Rust Gen 9 engine that runs
   Monte-Carlo Tree Search (~250k iterations in ~120 ms). Damage, priority, hazards, residuals,
   abilities, turn order, and Terastallization are all modelled exactly, and MCTS handles the
   simultaneous-move turn correctly where plain minimax would let the opponent peek at our move.

4. **Pool and guard.** The per-world rankings are combined, and a few deterministic guards veto
   known blunders: wasting a turn on a move that does nothing, locking the last Pokémon into a
   status move, clicking a move the revealed opponent is immune to. A small learned value net
   re-ranks the handful of turns where the engine's own evaluation is weak (it under-fears an
   opponent setting up).

5. **Play a mixed strategy.** Always taking the single top move is exploitable — humans learn to
   switch an immunity absorber into a predictable click. So among near-tied candidates the final
   choice is sampled rather than fixed, with one exception: the once-per-game Terastallization is
   never spent on a coin flip.

## How it was built and tested

The interesting engineering here is the loop, not any single component. It goes:

**Mine the losses.** Every ladder game is saved as a replay plus a per-turn trace of what the
search considered. `mine_losses.py` clusters losses into recurring signatures: a wasted turn, a
move clicked into an immunity, a Pokémon that died without acting, a Tera burned for nothing,
with wins used as a control so a "signature" has to actually separate losses from wins.

**Find the root cause, fix it narrowly.** Each signature gets reproduced, traced to a specific
flaw, and fixed with the smallest change that addresses it. Most of the shipped features started
this way. For example: the engine correctly predicted that a full-HP recovery move would do
nothing, but its flat evaluation kept clicking it anyway, so a deterministic no-op guard demotes
any move whose outcome equals doing nothing.

**A/B before it ships.** No strength change is committed without a head-to-head test — new code
vs old code, self-play on a local server, 60+ games, judged on a Wilson confidence interval.
This is the part that keeps the project honest. Plausible ideas failed the test repeatedly:
feeding the search real item statistics without Choice-lock modelling *lost* at 39%, and adding
more search time did nothing because the engine is already converged. Falsified ideas are logged
next to the ones that worked, so a dead lever doesn't get retried later.

One thing this discipline made explicit: some fixes don't show up in a self-play mirror because
both sides share the same blind spot. Those get judged on whether they fix the reproduced
behavior plus how they do against real humans, not on a mirror win rate near 50%. The strongest
open-source bot, [Foul Play](https://github.com/pmariglia/foul-play), was used the same way — as
a benchmark opponent that exposed structural gaps (its distribution-correct team sampling and its
mixed root strategy both came out of losing to it and reading its source).

## Results

| Opponent | Result |
|---|---|
| Live Showdown ladder (humans) | peak **2137 Elo**, ~81 GXE — top 1% and top 500 in the world |

## Setup

Requires Python 3.12, Node.js 18+ (for the local server), git, and a Rust toolchain
([rustup](https://rustup.rs)) to build poke-engine. From the project root:

```powershell
# 1. Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # prompt should show "(.venv)"
pip install -r requirements.txt

# 2. Local Showdown server (supplies the Random Battle data the bot reads, and hosts local games)
git clone --depth 1 https://github.com/smogon/pokemon-showdown.git server
cd server; npm install; cd ..

# 3. poke-engine, built with the Gen 9 Terastallization feature
pip install -v --force-reinstall --no-cache-dir "poke-engine==0.0.47" `
  --config-settings="build-args=--features poke-engine/terastallization --no-default-features"
```

Everything lives in `.venv` — activate it per terminal or call `.\.venv\Scripts\python.exe`
directly. A `ModuleNotFoundError` almost always means the venv isn't active.

To play the real ladder, register a username at play.pokemonshowdown.com and put the credentials
in a gitignored `.env` at the project root:

```
SHOWDOWN_USERNAME=YourBotName
SHOWDOWN_PASSWORD=your-password
```

## Running it

Most commands need the local server running in a separate terminal:

```powershell
cd server; node pokemon-showdown start --no-security      # ready when it prints: listening on 0.0.0.0:8000
```

Play the real ladder (this one connects to the official server, so the local server is not
needed):

```powershell
.\.venv\Scripts\python.exe -u src\ladder.py                  # 10 ranked games
.\.venv\Scripts\python.exe -u src\ladder.py --battles 40     # a full cohort
```

Every game is saved to `replays/ladder/` as `<result>-<battle-tag>.html` plus a
`.trace.json` of the per-turn decisions — the raw material for the loss-mining loop. The
archive is complete: the last game of a run and anything in flight when you Ctrl-C are
flushed on exit (an interrupted game is saved as `unfinished-…`), and a replay that can't
be written prints a warning instead of failing silently. Open the `.html` in a browser to
watch the game back.

The first 5 games of each run are additionally published as hosted replays on
`replay.pokemonshowdown.com` (`/savereplay`), and the shareable link is printed as each one
starts — handy for sharing a game without shipping an HTML file. Use `--upload-first N` to
change the count, `--upload-first 0` to publish nothing. Note these links are **public**.

### Watching a game being played

`analyze_battle.py` plays **one** ladder game and turns the terminal into a live analysis
board for it — same bot, same shipped config (it imports `ladder.py`'s `build_agent`, so the
settings can't drift), but narrated:

```powershell
.\.venv\Scripts\python.exe -u src\analyze_battle.py            # one rated game, narrated
.\.venv\Scripts\python.exe -u src\analyze_battle.py --upload    # + a public shareable replay
.\.venv\Scripts\python.exe -u src\analyze_battle.py --local --mode challenge --opponent SomeBot
```

Each turn prints the position (both actives with HP bars, boosts, status, hazards, weather),
then the search working — determinized worlds completing one at a time with visits and the
engine's own win estimate firming up — then the candidate table:

```
== Turn 2 ==========================================================================
  us   Iron Hands        82% ##########.. 6/6  quarkdrive · assaultvest
  opp  Scrafty          100% ############ 6/6  intimidate · ?item
  search   8/8 worlds x 150ms x 8t  ·  2.43M visits  ·  1.8s
     candidate                  share          worlds  eval
  -> Heavy Slam                 ####....  0.27  4/8     0.57
     Close Combat               ###.....  0.19  2/8     0.54
     Volt Switch                ###.....  0.19  1/8     0.53
  value    net agrees: Heavy Slam 0.53, Close Combat 0.50, Volt Switch 0.50
  guessed  item unknownitem x8  ·  ability intimidate x8  ·  tera poison x8
  inferred used since switch-in: Drain Punch
  >> plays Heavy Slam  [1.8s, eval 0.55]   expects switch Misdreavus 10%
   T2    opp Scrafty used Drain Punch -> Iron Hands 82% -> 64%
   T2    us  Iron Hands used Heavy Slam -> Scrafty 100% -> 69%
```

`share` is the pooled MCTS visit share across worlds, `worlds` how many worlds made that move
their outright winner, `eval` the engine's own win estimate. `guessed` is the determinizer:
which item / ability / Tera type the sampled hidden sets drew this turn. Any guard or
reranker that moves the front-runner prints its own line (`absorb`, `futility`, `tiebreak`,
`value`, `noop`, `deadlock`, `mix`), as do Choice-lock and Scarf inferences.

The game ends with a post-game block: time spent thinking, MCTS volume, which guards fired
and what they vetoed, how often the search called the opponent's move, the luck ledger, the
eval curve with its biggest swings, and the revealed opponent team. The transcript is saved
next to the replay as `<result>-<battle-tag>.analysis.txt`.

Useful flags: `--local` plays on the local server instead (for trying it out without
spending a rated game), `--ascii` / `--no-color` for dumb terminals and redirects,
`--no-worlds` to drop the hidden-set tally, `--fork` for the Phase-2 engine.

Other entry points:

```powershell
.\.venv\Scripts\python.exe -u src\eval_engine.py             # fast check vs the built-in heuristic
.\.venv\Scripts\python.exe -u src\mine_losses.py             # cluster saved replays into blunder signatures
.\.venv\Scripts\python.exe -u src\gen_value_data.py          # generate value-net training data via self-play
.\.venv\Scripts\python.exe -u src\train_value.py             # retrain the value net -> models/value_net.pt
```

Retrain the value net after a real strength jump — the net should model games at the bot's
current level, not an older one.

## Repo layout

| Path | Purpose |
|---|---|
| `src/engine_search.py` | The bot: determinized MCTS search, vote pooling, blunder guards, value-net re-ranking, mixed-strategy root. |
| `src/poke_engine_adapter.py` | Builds concrete engine states from the observed battle: set determinization, Choice-lock and Scarf inference, condition durations. |
| `src/knowledge.py` | Opponent set/ability prediction from Random Battle data, plus damage estimation. |
| `src/value_features.py`, `src/train_value.py` | The learned value head: feature extraction and its trainer. |
| `src/ladder.py` | Plays ranked games on the official ladder (the shipped config lives here). |
| `src/analyze_battle.py`, `src/live_analysis.py` | One ladder game, narrated live: the play-by-play plus the search's candidates, guards, hidden-set guesses and eval, then a post-game report. |
| `src/eval_engine.py` | Fast benchmark vs the built-in heuristic. |
| `src/gen_value_data.py`, `src/mine_losses.py` | Self-play data generation and replay-based loss analysis. |
| `data/` | Random Battle usage data: counted joint sets and per-role item/ability/Tera stats. |
| `models/value_net.pt` | The shipped value net. |
| `server/` | Local Showdown server (cloned during setup, not committed). |
