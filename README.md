# Pokemon Showdown Battle AI

An AI that plays [Pokemon Showdown](https://pokemonshowdown.com/) singles battles. It
starts as a simple rule-based bot and works up to a neural network trained with
reinforcement learning.

Everything runs against a local copy of the Showdown server, which the code talks to
through [poke-env](https://github.com/hsahovic/poke-env). poke-env handles the connection
and hands back the battle state as Python objects. Because it all runs locally, the agent
can play through thousands of battles quickly and for free.

## How it's built

The project grew in stages, and the code keeps each stage around so they can be compared.

**Heuristic bot** (`v1/heuristic_bot.py`). A hand-written player with no learning. It
scores each move by base power, STAB, and type effectiveness, plays the best one, and
switches out when the matchup is bad. It wins about 96% of games against a random opponent
and also serves as the benchmark the learning agents are measured against.

**Reinforcement-learning agent** (`v1/`). This is the main line of work. The battle state
is turned into a list of numbers (the observation), fed to a network, and the network
learns which moves and switches win games. Training uses MaskablePPO: at every turn only
some of the actions are legal, and poke-env supplies a mask of the legal ones, so the
agent only ever picks valid moves. That makes learning much faster than letting it flail
at illegal actions.

Training runs across several environments in parallel (each its own process and server
connection) for a large speedup, with a reward that emphasizes winning over even trades.

The observation has grown over time. It began as a flat 141 numbers, and the agent trained
on that reached around 41% against the heuristic. The current version is larger (215
numbers) and adds two things worth calling out:

- Per-Pokemon detail the early version missed: held item, Terastallization state, and move
  priority.
- A knowledge layer (`v1/knowledge.py`) that predicts the opponent's set. Random Battle
  sets are drawn from a fixed, public pool, so when the agent sees a Haxorus it can look up
  the moves that Haxorus is likely carrying and estimate how much damage they would do.
  The prediction is probabilistic (a move in one of two possible sets reads as roughly
  50%) and sharpens as the opponent reveals moves.

**Self-play** (`v1/selfplay/`). Against a single fixed opponent the agent tops out, because
once it has solved that opponent there is nothing left to learn. Self-play replaces the
fixed opponent with a snapshot of the agent itself, refreshed periodically, so the target
keeps getting stronger as the agent does. It warm-starts from the heuristic-trained model
and is still benchmarked against the heuristic so progress stays comparable.

**Scaled agent and test-time search** (`v1/v3/`). A fresh agent that bundles the changes
needing a clean start: a bigger network, a higher discount (so a win late in a game is
still credited to the moves that set it up), the win-focused reward, and parallel training
at scale. On top of the trained policy it adds a shallow one-turn lookahead: the policy
proposes its few most likely moves, then the damage calculator and opponent set-prediction
pick the one that actually KOs or avoids being KO'd. This is honest "test-time search" --
not full game-tree MCTS, which would need a battle simulator the project doesn't have, but
the same idea of planning one step ahead on top of a learned policy. `eval_search.py`
benchmarks the searching agent against the raw policy to measure whether it helps.

**Attention experiment** (`v2/`). A larger structured observation (854 numbers laid out as
twelve Pokemon "tokens" plus global field state) fed to a custom team-attention network,
the idea being to let the network reason about each Pokemon on its own. In testing it
learned worse than the simpler flat MLP: most of that observation is empty early in a game
(the opponent's unrevealed team is all zeros), which buried the useful signal. It is kept
here as a documented experiment rather than the path forward.

## Requirements

- Python 3.12 (any recent 3.x should be fine)
- Node.js 18+ (to run the local Showdown server)
- git

## One-time setup

Run these once from the project root.

```powershell
# 1. Create the Python virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # the prompt should now show "(.venv)"
pip install -r requirements.txt

# 2. Clone and build the local Showdown server (large, takes a few minutes)
git clone --depth 1 https://github.com/smogon/pokemon-showdown.git server
cd server
npm install
cd ..
```

The Python packages live in `.venv`, not in the system Python, so every command has to use
that environment. Either activate it once per terminal with `.\.venv\Scripts\Activate.ps1`
and then run `python ...`, or call the venv's Python directly with
`.\.venv\Scripts\python.exe ...`. If you see `ModuleNotFoundError: No module named
'sb3_contrib'` (or similar), the venv is not active.

## Running it

The local server has to be running first, since every script connects to it. Use a
separate terminal for each long-running piece.

### 1. Start the server

```powershell
cd server
node pokemon-showdown start --no-security
```

Leave it open. It is ready once it prints `Worker 1 now listening on 0.0.0.0:8000`. The
`--no-security` flag turns off login and rate limits so local bots can connect freely.
Ctrl+C stops it.

### 2. Train an agent

From the project root:

```powershell
.\.venv\Scripts\python.exe -u v1\train_rl.py                  # main agent vs random/heuristic
.\.venv\Scripts\python.exe -u v1\v3\train_v3.py               # scaled "done-right" agent (bigger net, win reward)
.\.venv\Scripts\python.exe -u v1\selfplay\train_selfplay.py   # self-play (warm-starts from the heuristic agent)
.\.venv\Scripts\python.exe -u v2\train_v2.py                  # the attention experiment
```

A run does everything in order: load the starting weights (see Model files below), measure
and print the win rate before training, train for `TRAIN_STEPS` steps while logging to
`tb_logs`, save the agent, and measure the win rate again. It checkpoints periodically, so
stopping early with Ctrl+C keeps recent progress and the next run picks up from there.

The `-u` flag makes Python print output as it happens instead of buffering it, so progress
is visible live. The trainers run several environments in parallel (see `N_ENVS` in
`train_rl.py`); if a run times out connecting, restart the server to clear stale
connections, and lower `N_ENVS` if your machine or the server can't handle that many.

Once an agent is trained, benchmark the test-time search against the raw policy:

```powershell
.\.venv\Scripts\python.exe -u v1\v3\eval_search.py            # raw policy vs 1-ply search, vs the heuristic
```

### 3. Watch training (optional)

TensorBoard shows live graphs in the browser. In another terminal:

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir tb_logs
```

Open http://localhost:6006 and use the SCALARS tab. The graphs worth watching:

- `eval/win_rate` is the real score, measured against the opponent during training. Each
  point is 100 battles, so it is fairly stable (set by `LIVE_EVAL_BATTLES`).
- `rollout/ep_rew_mean` is the average reward per battle, a smoother signal that should
  trend up as the agent improves.
- `train/explained_variance` says whether the value network is learning. If it sits near
  zero the agent is not really learning, even if the reward looks busy.

### 4. Benchmark the heuristic bot (optional)

Plays the rule-based bot against a random opponent and prints its win rate. No learning
involved.

```powershell
.\.venv\Scripts\python.exe v1\run_battle.py
```

## Settings

The knobs are constants at the top of `v1/train_rl.py`:

| Setting | What it does |
|---|---|
| `OPPONENT` | Who the agent trains against and is scored against: `"random"` (random legal moves, the easy baseline) or `"heuristic"` (a real strategy bot). |
| `TRAIN_STEPS` | How many steps of experience to add this run. Larger is stronger but slower (roughly 300 steps per second). |
| `EVAL_BATTLES` | Battles used for the before/after win-rate measurement. |
| `EVAL_FREQ` | How often, in steps, to measure win rate for the live graph. |

A step is one decision (a move or a switch). A battle is roughly 40 to 60 steps, so 50,000
steps is about a thousand battles.

## Model files

Trained agents are saved as `ppo_vs_<opponent>_obs<N>.zip`, where `<opponent>` is who it
trained against and `<N>` is the observation size. Putting the observation size in the name
means that changing what the network sees will not silently load or overwrite an
incompatible older model.

When `train_rl.py` starts, it decides where to begin:

1. If a saved file already exists for the current opponent and observation size, it loads
   it and keeps training the same agent.
2. If not, but a same-size agent trained against the random opponent exists, it warm-starts
   from that (transfer learning) on a fresh training curve.
3. Otherwise it starts from a brand-new random-weights network.

To start an opponent over from scratch, delete its `.zip`.

The `.zip` files, `tb_logs/`, and the `server/` clone are not committed (see
`.gitignore`); the models and logs are regenerated by training.

## Repo layout

| Path | Purpose |
|---|---|
| `v1/heuristic_bot.py` | The rule-based bot. |
| `v1/run_battle.py` | Runs the heuristic bot vs a random opponent and reports win rate. |
| `v1/rl_env.py` | The RL environment: the 215-number observation (`embed_battle`) and the reward (`calc_reward`). |
| `v1/knowledge.py` | Opponent set prediction (from the Random Battle set data) and damage estimation. |
| `v1/train_rl.py` | Trains the main agent (parallel envs, win-focused reward) vs random/heuristic. |
| `v1/smoke_test.py` | Quick check that the v1 environment builds, resets, and steps. |
| `v1/smoke_parallel.py` | Quick check that parallel (multi-process) training starts cleanly. |
| `v1/selfplay/opponent.py` | A poke-env opponent that picks moves with a trained model. |
| `v1/selfplay/train_selfplay.py` | Trains the agent by self-play (vs refreshed snapshots of itself). |
| `v1/selfplay/smoke_selfplay.py` | Quick live check of the self-play loop. |
| `v1/v3/train_v3.py` | Trains the scaled "done-right" agent (bigger net, higher gamma, win reward). |
| `v1/v3/search.py` | Test-time 1-ply lookahead player (policy + damage-model re-ranking). |
| `v1/v3/eval_search.py` | Benchmarks the search agent vs the raw policy, both vs the heuristic. |
| `v2/rl_env_v2.py` | The experiment's structured 854-number observation. |
| `v2/team_net.py` | The team-attention network. |
| `v2/train_v2.py` | Trains the v2 agent. |
| `v2/smoke_v2.py` | End-to-end check of the v2 stack against a live battle. |
| `requirements.txt` | Python dependencies. |
| `server/` | Local Showdown server, cloned separately during setup. |

## Troubleshooting

- `ModuleNotFoundError` means the virtual environment is not active. Activate it or call
  `.\.venv\Scripts\python.exe` directly.
- A script that hangs while connecting means the server is not running. Start it and
  confirm the `listening on 0.0.0.0:8000` line.
- `EADDRINUSE` or "port 8000 already in use" means a server is already running. Reuse it,
  or stop the old one (Ctrl+C in its terminal, or `Stop-Process -Name node`).
- `TimeoutError: Agent is not challenging` means an env could not start a battle, usually
  because a previous crashed run left stale connections on the server. Restart the server
  to clear them. If it persists, lower `N_ENVS` in `train_rl.py` (too many parallel
  connections at once).
- TensorBoard showing no data usually means `--logdir` is not pointing at `tb_logs`, or no
  run has started writing yet. Use the refresh icon in the browser.
