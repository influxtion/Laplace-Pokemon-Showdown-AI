# Pokemon Showdown Battle AI

An AI that plays [Pokemon Showdown](https://pokemonshowdown.com/) singles battles
(Gen 9 Random Battle). It is built in stages, from a hand-written rule-based bot up to a
neural network trained with reinforcement learning.

## Rule-based vs. learned

The contrast between the two kinds of bot here is the whole point of the project.

A **rule-based bot** does exactly what a person tells it. The logic is written by hand --
"play the highest-damage move, switch out on a bad matchup" -- and it follows those rules
the same way every game. The only way to make it better is to write more rules. The
heuristic bot here is that kind; it serves as the fixed opponent and the baseline.

The **reinforcement-learning agent** is given no rules about how to play. It is a neural
network that starts out choosing at random and learns by playing. After each battle it gets
a reward (a little for dealing damage or scoring a knockout, a lot for winning), and the
training algorithm (PPO) adjusts the network's internal weights so the choices that led
toward wins become more likely next time. Over thousands of games it works out its own
strategy -- nobody writes "switch here." That is what makes it real machine learning rather
than a script: the part choosing each move is a set of learned parameters tuned by game
outcomes. It is *reinforcement* learning specifically because there is no answer key of
correct moves to copy; the agent only ever sees the reward of how a game turned out and has
to work out for itself which decisions earned it.

## How it works

The code talks to a local copy of the Showdown server through
[`poke-env`](https://github.com/hsahovic/poke-env), which handles the connection and exposes
the battle state as Python objects. Everything runs locally, so the agent can play thousands
of fast battles for free.

Each turn, the environment (`src/rl_env.py`) turns the battle into a list of numbers (the
*observation*) for the network, and turns the result into a *reward*. Training uses
**MaskablePPO**: at every turn only some actions are legal, and poke-env supplies a mask of
the legal ones so the agent only ever picks valid moves, which makes learning far faster.
Training runs across several environments in parallel for speed.

## The models, in order

All win rates are against `SimpleHeuristicsPlayer` unless they say *random*. That heuristic
is only a beginner-level bot, so these are a fixed yardstick for comparing versions, not
ladder ratings. Evaluations are noisy (a 50-battle measurement swings ~8 points), so later
runs measure over 200-400 battles.

| # | Model | Win rate | What it added / what went wrong |
|---|---|---|---|
| 1 | **Heuristic bot** `src/heuristic_bot.py` *(no saved model)* | ~96% vs random | Hand-written rules: score moves by power, STAB, and type effectiveness; switch on a bad matchup. The baseline and sparring partner, not a learner. |
| 2 | **First RL agent, 12 features** `ppo_vs_random.zip`, `ppo_vs_heuristic.zip` | ~100% vs random; ~30% vs heuristic | Proved the pipeline learns. But with only 12 numbers it was blind to stats, status, hazards, and the bench, so it stalled against a real opponent. |
| 3 | **Richer observation, 141 features** `ppo_vs_heuristic_obs141.zip` | ~41% | Added boosts, status, the bench, move details, weather, and abilities. Broke the 30% plateau; held as the baseline far longer than expected. |
| 4 | **Attention net, 854 features** `ppo_v2_attn_obs854.zip` (+ MLP control `ppo_v2_mlp_obs854.zip`) | ~4% | The one dead end. Laid the battle out as 12 Pokemon "tokens" for a self-attention net, but most of those numbers are zero early (the opponent's team is hidden) and drowned the useful signal. Dropped. |
| 5 | **Extended observation, 215 features** `ppo_vs_heuristic_obs215.zip` | ~34-40% | Added move priority, held item, Terastallization, and a knowledge layer that predicts the opponent's set and estimates damage. No gain: the *turning point* that proved observation was no longer the bottleneck. |
| 6 | **Scaled v3 agent** `ppo_v3_obs215.zip` | ~45% peak, then ~30% | Bigger network, higher discount, win-focused reward, parallel training. Hit a new high but had no training hygiene, so it overtrained downhill and saved the latest (not best) weights. |
| 7 | **Rescaled-reward experiment** `ppo_v3_rescaled_obs215.zip` | no change | Divided every reward by 10 to test if reward scale capped the value network's fit. It didn't (the metric is scale-invariant). Ruled out reward scale; pointed at partial observability. |
| 8 | **Self-play** `ppo_selfplay_obs215.zip`, `ppo_v3_selfplay_obs215.zip` (+ `_best_`) | ~43% | Trains against refreshed snapshots of itself instead of a fixed opponent. Hygiene held this time, but it didn't transfer to the benchmark: the ceiling wasn't "the opponent is too easy." |
| 9 | **Reward-anneal + anti-panic-switch** `ppo_v3_anneal_obs215.zip` (+ `_best_`) | ~40% (no gain) | Warm-started v3 and annealed the dense shaping toward zero while holding the win bonus, plus a penalty for switching a different Pokemon two turns in a row. Ran to completion cleanly, but the raw win rate stayed flat -- confirming reward is not the lever either; the ceiling is partial observability + a reactive 1-turn policy. |
| 10 | **Test-time search** `src/search.py` *(wraps a trained model)* | **~55-62% (about +20 over the raw policy)** | Not a new model: a 1-ply evaluator that scores *every* legal action with the damage model -- it understands type matchups, switching for an offensive advantage, Terastallization, and ability immunities -- using the trained policy only as a prior. The biggest lever among the from-scratch agents, and the agent we shipped before the engine bot. |
| 11 | **poke-engine MCTS** `src/engine_search.py` (+ `poke_engine_adapter.py`) | **~92% (37/40 vs heuristic)** | The strongest agent, following the Foul Play recipe. Instead of a hand-rolled simulator (the dead end in `deep_search.py`), it feeds the position into [`poke-engine`](https://github.com/pmariglia/poke-engine) -- a fast Rust battle engine with real MCTS -- and copes with the opponent's hidden set by *determinization*: sample several full opponent teams consistent with what's revealed (from the randbats sheet), search each, and pool the visit-weighted policies. Models KO probability, hazards, items, abilities, residuals, turn order and Tera correctly, and MCTS handles the simultaneous-move nature of a turn that plain minimax can't. ~0.9s/turn. Needs no trained model -- it searches the position directly. |
| 12 | **Determinization overhaul** (same files) | **82% head-to-head vs #11** (79/96; still ~92% vs heuristic -- that yardstick is saturated) | Fixed a blind spot: 56% of randbats movepools exceed 4 moves, but the old adapter always gave the opponent the same first-4 truncation, so the search could never anticipate unrevealed moves late in the sheet order. Now each determinization *samples* a legal 4-move set around what's revealed. Added Choice-lock inference (a mon that committed to a move while holding a Choice item is locked to it; one that used 2+ moves since switching in can't hold Choice at all -- poke-engine doesn't enforce this itself, verified), correct `last_used_move` indices, and turned the real randbats item/ability/tera probabilities (`data_randbats_stats_gen9.json`) back on: with lock modeling they A/B at parity (was 39% without it), and only they let the search exploit locked opponents. |

## Lessons learned

- **A plateau is usually a representation problem first.** The agent stalled at ~-15
  `ep_rew_mean` until the observation was enriched (12 -> 141 features); more game state was
  worth more than more training.
- **More is not always better.** Going 141 -> 854 features *hurt* because most of the extra
  numbers were empty, and 141 -> 215 didn't help at all. Past a point, observation stopped
  being the bottleneck.
- **Measure honestly.** Small evals are noisy enough to invent improvements that aren't real
  (an apparent self-play "peak" was sampling noise), so headline numbers use 200-400 battles.
- **Reward shape drives behaviour.** Reward climbing while win rate stays flat means the
  agent learned to *trade*, not to *win* -- which is what the reward anneal targets.

## Setup

Requirements: **Python 3.12** (any recent 3.x), **Node.js 18+** (for the local server), and
**git**. Run once from the project root:

```powershell
# 1. Python virtual environment + dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # prompt should show "(.venv)"
pip install -r requirements.txt

# 2. Clone and build the local Showdown server (large, a few minutes)
git clone --depth 1 https://github.com/smogon/pokemon-showdown.git server
cd server; npm install; cd ..

# 3. Build poke-engine for Gen 9 (the engine bot's search). Needs a Rust toolchain
#    (https://rustup.rs); on Windows without Visual Studio, install the GNU host with
#    `rustup-init.exe --default-host x86_64-pc-windows-gnu` and have MinGW gcc on PATH.
pip install -v --force-reinstall --no-cache-dir "poke-engine==0.0.47" \
  --config-settings="build-args=--features poke-engine/terastallization --no-default-features"
```

All packages live in `.venv`, not system Python. Either activate it once per terminal
(`.\.venv\Scripts\Activate.ps1`, then `python ...`) or call it directly
(`.\.venv\Scripts\python.exe ...`). A `ModuleNotFoundError` almost always means the venv is
not active.

## Running it

The server must be running first; use a separate terminal for each long-running piece.

**1. Start the server** (leave it open; ready at `listening on 0.0.0.0:8000`):

```powershell
cd server
node pokemon-showdown start --no-security
```

**2. Train an agent** (from the project root):

```powershell
.\.venv\Scripts\python.exe -u src\train_rl.py                          # main agent vs random/heuristic
.\.venv\Scripts\python.exe -u src\train_v3.py                          # scaled agent (bigger net, win reward)
.\.venv\Scripts\python.exe -u src\train_v3_anneal.py                   # reward-anneal + anti-panic-switch finetune
.\.venv\Scripts\python.exe -u src\train_v3_anneal_selfplay.py          # "Influxobot": self-play on the annealed agent
.\.venv\Scripts\python.exe -u experiments\v2_attention\train_v2.py     # the attention experiment
```

A run loads its starting weights, prints the win rate before training, trains for
`TRAIN_STEPS` while logging to `tb_logs`, checkpoints periodically, saves, and prints the
win rate after. `-u` makes output appear live. Stopping with Ctrl+C keeps the last
checkpoint, and the next run resumes from it.

**3. Watch the agent play.** Plays a few battles (as the searching agent we'd ship) and saves
each as a Showdown replay `.html` in `replays/` that you open in a browser to watch move by
move:

```powershell
.\.venv\Scripts\python.exe -u src\play.py                 # 1 battle, search agent vs heuristic
.\.venv\Scripts\python.exe -u src\play.py --battles 3     # more battles
.\.venv\Scripts\python.exe -u src\play.py --raw           # bare policy, no search
.\.venv\Scripts\python.exe -u src\play.py --opponent random
.\.venv\Scripts\python.exe -u src\play.py --model ppo_v3_obs215.zip   # a specific model
```

It prints each result and the replay path, then you open the file to watch:

```
  Battle 1: WON  in 24 turns  ->  replays/battle-gen9randombattle-12.html
Won 1/1 vs heuristic. Open the .html file(s) in a browser to watch.
```

The replay file has the full battle embedded; opening it loads Showdown's replay viewer
(needs internet for that viewer script, but the battle itself is in the file). This is more
reliable than trying to catch the bot-vs-bot game live, since it finishes in seconds.

**4. Benchmark the test-time search** against the raw policy:

```powershell
.\.venv\Scripts\python.exe -u src\eval_search.py                  # raw / 1-ply / heuristic+ / deep
.\.venv\Scripts\python.exe -u src\eval_engine.py                  # the poke-engine MCTS bot vs heuristic
.\.venv\Scripts\python.exe -u src\eval_engine.py --vs-search      # engine head-to-head vs the 1-ply searcher
.\.venv\Scripts\python.exe -u src\play.py --engine                # watch the engine bot (saves a replay)
```

**5. Watch training (optional)** in a browser via TensorBoard:

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir tb_logs   # then open http://localhost:6006
```

Graphs worth watching on the SCALARS tab: `eval/win_rate` (the real score, ~100 battles per
point), `rollout/ep_rew_mean` (smoother progress signal), and `train/explained_variance`
(whether the value network is learning; near zero means it isn't).

**6. Benchmark the heuristic bot (optional):**

```powershell
.\.venv\Scripts\python.exe src\run_battle.py
```

## Settings and model files

The main knobs are constants at the top of `src/train_rl.py`:

| Setting | What it does |
|---|---|
| `OPPONENT` | Who the agent trains and is scored against: `"random"` or `"heuristic"`. |
| `TRAIN_STEPS` | Steps of experience to add this run (a step is one move/switch; ~300/sec). |
| `EVAL_BATTLES` / `EVAL_FREQ` | Battles per before/after measurement, and how often to measure for the live graph. |
| `N_ENVS` | Parallel training environments (lower it if the machine or server can't keep up). |

Agents are saved as `ppo_..._obs<N>.zip`, where `<N>` is the observation size; putting it in
the name stops an incompatible older model from silently loading or being overwritten. On
start, `train_rl.py` continues a matching saved agent if one exists, else warm-starts from a
same-size random-trained agent, else starts fresh. Delete a `.zip` to start that setup over.
The `.zip` files, `tb_logs/`, and `server/` are not committed (see `.gitignore`).

## Repo layout

| Path | Purpose |
|---|---|
| `src/heuristic_bot.py` | The rule-based bot. |
| `src/run_battle.py` | Runs the heuristic bot vs random and reports win rate. |
| `src/rl_env.py` | The RL environment: the 215-number observation (`embed_battle`) and reward (`calc_reward`). |
| `src/knowledge.py` | Opponent set/ability prediction (from Random Battle data) and damage estimation. |
| `src/train_rl.py` | Trains the main agent (parallel envs, win-focused reward). |
| `src/opponent.py`, `src/train_selfplay.py` | Model-driven self-play opponent, and the earlier self-play trainer. |
| `src/train_v3.py` | Scaled agent (bigger net, higher gamma, win reward); `RESCALED` toggle for the reward-scale experiment. |
| `src/train_v3_anneal.py` | Reward-anneal + anti-panic-switch finetune (KL / LR-decay / best-checkpoint hygiene). |
| `src/train_v3_selfplay.py`, `src/train_v3_anneal_selfplay.py` | Self-play on the v3 / annealed agent ("Influxobot", the ladder bot). |
| `src/search.py`, `src/eval_search.py` | Test-time 1-ply search player, and its benchmark vs the raw policy. |
| `src/engine_search.py`, `src/poke_engine_adapter.py`, `src/eval_engine.py` | The strongest agent: a poke-engine MCTS bot, the poke-env→poke-engine state adapter + opponent determinization, and its benchmark. |
| `src/play.py` | Plays a few watchable battles and saves them as browser replays (`replays/`). |
| `src/smoke_test.py`, `src/smoke_parallel.py`, `src/smoke_selfplay.py` | Quick checks that the env / parallel training / self-play start cleanly. |
| `experiments/v2_attention/` | The attention experiment (dead end): `rl_env_v2.py` (854-number obs), `team_net.py`, `train_v2.py`, `smoke_v2.py`. |
| `requirements.txt` | Python dependencies. |
| `server/` | Local Showdown server (cloned during setup, not committed). |

## Troubleshooting

- **`ModuleNotFoundError`** -- the venv is not active. Activate it or use the full
  `.\.venv\Scripts\python.exe` path.
- **A script hangs while connecting** -- the server isn't running. Start it and confirm the
  `listening on 0.0.0.0:8000` line.
- **`EADDRINUSE` / port 8000 in use** -- a server is already running. Reuse it, or stop the
  old one (Ctrl+C, or `Stop-Process -Name node`).
- **`TimeoutError: Agent is not challenging`** -- a previous crashed run left stale
  connections. Restart the server; if it persists, lower `N_ENVS`.
