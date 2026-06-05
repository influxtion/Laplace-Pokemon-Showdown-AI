# Pokemon Showdown Battle AI

A project that builds an AI to play [Pokemon Showdown](https://pokemonshowdown.com/)
singles battles. It is built in stages, from a simple rule-based bot up to a neural
network trained with reinforcement learning.

The code talks to a local copy of the Pokemon Showdown server through
[`poke-env`](https://github.com/hsahovic/poke-env), which handles the connection and
exposes the battle state as Python objects. Training runs locally so the agent can play
thousands of fast battles for free.

## Stages

1. Heuristic bot (`heuristic_bot.py`) - a rule-based agent. Each turn it scores the
   available moves by base power, STAB, and type effectiveness, plays the best one, and
   switches to the best matchup when its Pokemon faints. Wins about 96% against an
   opponent that plays random moves. This is also the benchmark the learning agent is
   measured against.
2. Reinforcement-learning agent (`rl_env.py` + `train_rl.py`) - a neural network that
   learns by playing, using MaskablePPO. There are no hand-written move rules: the battle
   is described to the network as numbers, the network is rewarded for winning, and it is
   adjusted over many battles so winning behaviour is reinforced.
3. In progress - training against stronger opponents (the heuristic, then self-play),
   richer observations, and eventually laddering on the live server for a real rating.

## Requirements

- Python 3.12 (any recent 3.x should work)
- Node.js 18+ (needed to run the local Showdown server)
- git

## One-time setup

Run these once, from the project folder (`C:\Users\jayde\Desktop\Coding Projects\Project`).

```powershell
# 1. Create the Python virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # activates the venv (prompt shows "(.venv)")
pip install -r requirements.txt

# 2. Clone and build the local Showdown server (large, takes a few minutes)
git clone --depth 1 https://github.com/smogon/pokemon-showdown.git server
cd server
npm install
cd ..
```

## Important: the virtual environment

All Python packages for this project (poke-env, stable-baselines3, sb3-contrib,
tensorboard) are installed inside the `.venv` folder, not in the system-wide Python.
Every Python command for this project must use that environment, or you will get errors
like `ModuleNotFoundError: No module named 'sb3_contrib'`.

Two ways to make sure you are using it:

- Activate it once per terminal, then use `python` normally:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  python -u train_rl.py
  ```
- Or call the venv's Python directly, without activating:
  ```powershell
  .\.venv\Scripts\python.exe -u train_rl.py
  ```

If you see a `ModuleNotFoundError`, the venv is almost certainly not active. Activate it
(or use the full `.\.venv\Scripts\python.exe` path) and try again.

## Running things

The local Showdown server must be running before any bot or training script, because the
scripts connect to it. Use separate terminals.

### Step 1 - start the local server

```powershell
cd server
node pokemon-showdown start --no-security
```

Leave this terminal open. It is ready when it prints
`Worker 1 now listening on 0.0.0.0:8000`. The `--no-security` flag disables login and
rate limits so local bots can connect freely. To stop the server later, press Ctrl+C in
this terminal.

### Step 2 - (optional) start TensorBoard

TensorBoard shows live graphs of training progress in a browser. In another terminal:

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir tb_logs
```

Then open http://localhost:6006 and use the SCALARS tab. Useful graphs:

- `rollout/ep_rew_mean` - average reward per battle. Trending up means the agent is
  winning more. This is the smooth, continuous signal.
- `eval/win_rate` - win rate measured every 10,000 steps against the current opponent.
  This is the "real score", but it bounces a bit because each point is only 30 battles.
- `rollout/ep_len_mean` - average battle length. Trending down means it is winning faster.

### Step 3 - run the heuristic bot benchmark (optional)

This plays the rule-based bot against a random opponent and prints its win rate. No
machine learning is involved.

```powershell
.\.venv\Scripts\python.exe run_battle.py
```

### Step 4 - train the reinforcement-learning agent

```powershell
.\.venv\Scripts\python.exe -u train_rl.py
```

One run does everything automatically, in order:

1. Loads the agent's starting weights (see "Model files" below).
2. Measures and prints the win rate before training ("Before:").
3. Trains for `TRAIN_STEPS` steps. Progress prints every few seconds and is logged to
   `tb_logs` for TensorBoard.
4. Saves the trained agent to disk automatically. You will see
   `Saved model to ppo_vs_<opponent>.zip` when this happens.
5. Measures and prints the win rate after training ("After:").

The model is only saved once, at the end of training. If you stop the run early with
Ctrl+C, that run's progress is not saved and it will resume from the last saved file next
time. Let it finish (you will see the "Saved model" line).

The `-u` flag makes Python print output immediately instead of buffering it, so you can
watch progress live.

## Choosing the opponent and other settings

The settings are constants at the top of `train_rl.py`:

| Setting | What it does |
|---|---|
| `OPPONENT` | Who the agent trains against and is measured against. `"random"` (random legal moves, the easy baseline) or `"heuristic"` (`SimpleHeuristicsPlayer`, a real strategy bot). |
| `TRAIN_STEPS` | How many steps (turns of experience) to train this run. Larger is stronger but slower. Roughly 300 steps per second. |
| `EVAL_BATTLES` | Battles used for the before/after win-rate measurement. |
| `EVAL_FREQ` | How often (in steps) to measure win rate for the live TensorBoard graph. |

A "step" is one decision (one move or switch). A full battle is roughly 40-60 steps, so
50,000 steps is about 1,000 battles.

## Model files

Each opponent gets its own saved agent so experiments do not overwrite each other:

- `ppo_vs_random.zip` - agent trained against the random opponent.
- `ppo_vs_heuristic.zip` - agent trained against the heuristic opponent.

When you run `train_rl.py`, it decides where to start from:

1. If a saved file already exists for the current `OPPONENT`, it loads that and continues
   training it (so each run keeps improving the same agent).
2. If not, but `ppo_vs_random.zip` exists, it warm-starts from the random-trained agent
   (transfer learning) and starts a fresh training curve.
3. If neither exists, it starts from a brand-new random-weights network.

To start an opponent over from scratch, delete its `.zip` file.

The `.zip` files and `tb_logs` are excluded from git (see `.gitignore`); they are
regenerated by training.

## Typical full session

```powershell
# Terminal 1: server
cd server
node pokemon-showdown start --no-security

# Terminal 2: TensorBoard (optional)
cd "C:\Users\jayde\Desktop\Coding Projects\Project"
.\.venv\Scripts\python.exe -m tensorboard.main --logdir tb_logs

# Terminal 3: training (re-run any time to keep improving the agent)
cd "C:\Users\jayde\Desktop\Coding Projects\Project"
.\.venv\Scripts\Activate.ps1
python -u train_rl.py
```

## Troubleshooting

- `ModuleNotFoundError: No module named '...'` - the virtual environment is not active.
  Activate it with `.\.venv\Scripts\Activate.ps1` or call `.\.venv\Scripts\python.exe`
  directly.
- The training script hangs or times out connecting - the local server is not running.
  Start it (Step 1) and confirm it prints the "listening on 0.0.0.0:8000" line.
- `EADDRINUSE` / port 8000 already in use - a server is already running. Either reuse it
  or stop the old one (Ctrl+C in its terminal, or `Stop-Process -Name node`).
- TensorBoard shows no data - make sure you point `--logdir` at `tb_logs` and that a
  training run has started writing to it. Use the refresh icon in the browser.

## Project layout

| Path | Purpose |
|---|---|
| `heuristic_bot.py` | The rule-based bot (`MaxDamagePlayer`). |
| `run_battle.py` | Plays the heuristic bot vs a random opponent and reports win rate. |
| `rl_env.py` | The reinforcement-learning environment: turns a battle into numbers (`embed_battle`) and into a reward (`calc_reward`). |
| `train_rl.py` | Trains the RL agent, logs to TensorBoard, evaluates, and saves the model. |
| `smoke_test.py` | Quick check that the RL environment builds, resets, and steps. |
| `requirements.txt` | Python dependencies. |
| `server/` | Local Pokemon Showdown server (cloned separately, not committed). |
| `tb_logs/` | TensorBoard logs (generated, not committed). |
| `ppo_vs_*.zip` | Saved trained agents (generated, not committed). |
