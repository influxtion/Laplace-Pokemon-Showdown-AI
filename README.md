# Laplace — a Pokémon Showdown battle AI

**Laplace** plays [Pokémon Showdown](https://pokemonshowdown.com/) Gen 9 Random Battle at a
strong human level (peak **~1920 Elo / ~75 GXE** on the real ladder). It is a *search* bot:
every turn it guesses the opponent's hidden team, simulates the position thousands of times
with a real battle engine, and plays a strategy that is deliberately hard to exploit.

The name is a double reference: **Lapras** is called *Laplace* (ラプラス) in Japanese, and
[Laplace's demon](https://en.wikipedia.org/wiki/Laplace%27s_demon) is the thought experiment
about an intelligence that, knowing the complete hidden state of the world, could predict its
future — which is exactly what this bot approximates when it fills in the opponent's unseen
set and rolls the game forward.

> The bot started as a reinforcement-learning project and became a search engine. The RL era
> (a rule-based baseline, then PPO agents, then a 1-ply searcher) is preserved in
> [`legacy/`](legacy/) — it's the story of how the project got here. This README is about the
> bot that shipped.

## Results

| Opponent | Result | Notes |
|---|---|---|
| **Real Showdown ladder** (humans) | **peak ~1920 Elo / 75.6 GXE** | gen9randombattle; beats 1700–1800-rated players |
| **[Foul Play](https://github.com/pmariglia/foul-play)** (strongest open-source bot) | **~45% head-to-head** | up from ~28% before the mixed-strategy work, while searching *less* compute than it does |
| `SimpleHeuristicsPlayer` (poke-env's built-in) | **~92%** | a beginner-level yardstick — saturated, no longer informative |

The Foul Play number is the headline: it is the reference open-source Showdown AI and the
project this architecture descends from. Getting from 28% to ~45% against it (at its own
higher compute budget) is the difference between "a weaker copy" and "a peer."

## How it works

Laplace never trains on the decision it has to make — it *computes* the answer each turn:

1. **Perceive.** [`poke-env`](https://github.com/hsahovic/poke-env) maintains the battle state
   from the server protocol: revealed moves, items, HP, boosts, hazards, field conditions.
2. **Determinize** (`poke_engine_adapter.py`). The opponent's exact set is hidden, so the bot
   samples several *complete, concrete* opponent teams — full item / ability / move / Tera
   sets drawn from real Random Battle usage counts (`data/joint_sets_gen9.json`), filtered to
   be consistent with everything revealed (shown moves, inferred Choice locks, Choice-Scarf
   verdicts from turn order, etc.). This is how it copes with imperfect information.
3. **Search** (`engine_search.py` → [`poke-engine`](https://github.com/pmariglia/poke-engine)).
   Each sampled world is handed to poke-engine, a fast Rust Gen 9 engine that runs Monte-Carlo
   Tree Search (~250k iterations in ~120 ms). Damage, priority, hazards, residuals, abilities,
   turn order and Terastallization are all modelled *exactly*, and MCTS correctly handles the
   simultaneous-move nature of a turn that plain minimax cannot.
4. **Aggregate + guard.** The per-world move rankings are pooled, a few deterministic guards
   veto known blunder patterns (wasted no-op turns, locking the last Pokémon into a status
   move, clicking a move the revealed opponent is immune to), and a small learned **value
   net** re-ranks the handful of turns where the search's own leaf evaluation is known to be
   weak (it under-fears opponent setup).
5. **Play a mixed strategy.** Rather than always taking the single top-scoring move
   (exploitable — humans learn to switch immunity absorbers into a predictable click), the
   final choice is *sampled* among near-tied candidates. This is the Foul Play insight that
   closed most of the gap to it, tuned so the once-per-game Terastallization is never spent on
   a coin flip.

The only learned component is the value net (`value_features.py` / `train_value.py`): a small
MLP over ~368 hand-built features, trained on self-play outcomes to predict the winner (~70%
accuracy), used only as a tie-breaker on ~5–10% of turns. Everything else is search — which is
why the bot is validated in **win rates with confidence intervals**, not loss curves.

## Setup

Requirements: **Python 3.12**, **Node.js 18+** (local server), **git**, and a **Rust
toolchain** ([rustup](https://rustup.rs)) to build poke-engine. From the project root:

```powershell
# 1. Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # prompt should show "(.venv)"
pip install -r requirements.txt

# 2. Local Showdown server (provides the randbats data the bot reads, and hosts local games)
git clone --depth 1 https://github.com/smogon/pokemon-showdown.git server
cd server; npm install; cd ..

# 3. poke-engine, built with Gen 9 Terastallization features
pip install -v --force-reinstall --no-cache-dir "poke-engine==0.0.47" \
  --config-settings="build-args=--features poke-engine/terastallization --no-default-features"
```

All packages live in `.venv` — activate it per terminal or call
`.\.venv\Scripts\python.exe` directly. A `ModuleNotFoundError` almost always means the venv
isn't active.

To play on the **real** ladder, register a username at play.pokemonshowdown.com and put the
credentials in a `.env` file at the project root (gitignored):

```
SHOWDOWN_USERNAME=YourBotName
SHOWDOWN_PASSWORD=your-password
```

## Running it

Most commands need the local server running (a separate terminal):

```powershell
cd server; node pokemon-showdown start --no-security      # ready at: listening on 0.0.0.0:8000
```

**Play the real ladder** (connects to the official server; the local one is *not* needed):

```powershell
.\.venv\Scripts\python.exe -u src\ladder.py                     # 10 ranked games
.\.venv\Scripts\python.exe -u src\ladder.py --battles 40        # a full cohort
.\.venv\Scripts\python.exe -u src\ladder.py --mode accept       # wait to be challenged
```

Every game is saved to `replays/ladder/` as a `.html` replay plus a `.trace.json` of what the
search considered each turn — the raw material for loss analysis.

**Benchmark vs Foul Play** (the real strength test — needs a Foul Play clone; see the header
of `src/bench_foulplay.py` for its one-time setup):

```powershell
.\.venv\Scripts\python.exe -u src\bench_foulplay.py --battles 30 --mix   # the shipped config
```

**Quick sanity check vs the built-in heuristic** (saturated, but fast):

```powershell
.\.venv\Scripts\python.exe -u src\eval_engine.py
```

**Analyse losses** — cluster saved replays into recurring blunder signatures (wins mined as a
control):

```powershell
.\.venv\Scripts\python.exe -u src\mine_losses.py
```

**Retrain the value net** — generate self-play data, then train (run after a big strength jump,
since the net should model games at the bot's current level):

```powershell
.\.venv\Scripts\python.exe -u src\gen_value_data.py             # → data_value*/ (gitignored)
.\.venv\Scripts\python.exe -u src\train_value.py               # → models/value_net.pt
```

## Repo layout

| Path | Purpose |
|---|---|
| `src/engine_search.py` | **The bot.** Determinized MCTS search, robust/averaged vote pooling, blunder guards, mixed-strategy root, value-net re-ranking. |
| `src/poke_engine_adapter.py` | Builds concrete poke-engine states from the poke-env battle: opponent-set determinization, Choice-lock / Scarf inference, condition durations. |
| `src/knowledge.py` | Opponent set/ability prediction from Random Battle data, and damage estimation. |
| `src/value_features.py`, `src/train_value.py` | The learned value head: feature extraction and its trainer. |
| `src/ladder.py` | Play ranked games on the official ladder (the shipped config lives here). |
| `src/bench_foulplay.py` | Head-to-head benchmark against Foul Play on the local server. |
| `src/eval_engine.py` | Fast benchmark vs the built-in heuristic. |
| `src/gen_value_data.py` | Generate value-net training data via self-play. |
| `src/mine_losses.py` | Cluster saved replays into blunder signatures for the improvement loop. |
| `data/` | Random Battle usage data: counted joint sets and per-role item/ability/Tera stats. |
| `models/value_net.pt` | The shipped value net. |
| `server/` | Local Showdown server (cloned during setup, not committed). |
| `legacy/` | The pre-poke-engine era: rule-based bot, PPO reinforcement-learning agents, 1-ply searcher, and the attention experiment. See [`legacy/README.md`](legacy/README.md). |

## How Laplace got here (the short version)

The full journey, models and lessons, is in [`legacy/README.md`](legacy/README.md). In brief:

- **Rule-based baseline → PPO reinforcement learning.** A neural net trained from scratch on
  game rewards. It climbed to ~45% vs the heuristic and plateaued around 1213 Elo on the
  ladder — a competent but unremarkable player. The lesson that stuck: *a plateau is usually a
  representation problem, and small evaluations invent improvements that aren't real.*
- **The pivot to search.** Pokémon's rules are perfectly known and simulable hundreds of
  thousands of times per second. So instead of *learning* to approximate the consequences of
  moves, *compute* them — and spend the learning budget only on the genuinely hard parts:
  guessing hidden information (usage statistics) and long-horizon judgment (the small value
  net). That pivot took the bot from ~1213 to the ~1900 band.
- **Everything since** has been better world-modelling (joint-set determinization, condition
  and lock inference) and better decision-making (blunder guards, the value head, and the
  mixed-strategy root that closed the gap to Foul Play) — each change gated by a head-to-head
  A/B test before it shipped.
