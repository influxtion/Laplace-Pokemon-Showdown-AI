# Pokémon Showdown Battle AI

Building an AI that plays [Pokémon Showdown](https://pokemonshowdown.com/) battles, in stages:

1. **Heuristic bot** (✅ done) — a rule-based agent that picks the best move by base power, STAB, and type effectiveness, and switches intelligently when forced. **Wins ~96% vs. a random-move baseline.** Serves as the foundation, benchmark, and training partner for the learning agent.
2. **Reinforcement-learning agent** (✅ working) — a neural network trained from win/loss rewards with **MaskablePPO** (Stable-Baselines3 + sb3-contrib) on poke-env's Gymnasium interface. No hand-written move rules: it learns by playing. Average reward and win rate climb visibly during training.
3. **Stretch** — train against the heuristic / self-play, learning curves, and laddering on the live server for a real Elo rating.

## Reinforcement learning (Stage 2)

Instead of rules, we describe each battle to a neural network as a vector of numbers
(`embed_battle`) and give it a reward each turn (`calc_reward`); MaskablePPO adjusts the
network over thousands of battles so reward-earning (winning) behaviour is reinforced.
poke-env supplies an **action mask** of the legal moves each turn, so the agent only ever
picks legal actions. See `rl_env.py` (the environment) and `train_rl.py` (the training loop).

```bash
# with the server running (see below):
python -u train_rl.py        # evaluates untrained -> trains -> evaluates trained, saves ppo_showdown.zip
```

## How it works

[`poke-env`](https://github.com/hsahovic/poke-env) connects Python to a local Pokémon Showdown server over WebSocket and exposes the battle state as clean Python objects. The bot subclasses `Player` and implements one method, `choose_move(battle)`, which returns a legal action each turn. See `heuristic_bot.py`.

## Setup

```bash
# 1. Python environment
python -m venv .venv
.venv\Scripts\activate            # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# 2. Local Showdown server (one-time clone)
git clone --depth 1 https://github.com/smogon/pokemon-showdown.git server
cd server && npm install && cd ..
```

## Run

```bash
# Terminal 1 — start the local battle server
cd server
node pokemon-showdown start --no-security

# Terminal 2 — run the bot against the random baseline
python run_battle.py
```

## Project layout

| File | Purpose |
|------|---------|
| `heuristic_bot.py` | The rule-based bot (`MaxDamagePlayer`) — the swappable "brain". |
| `run_battle.py` | Plays N battles vs. a baseline and reports the win rate. |
| `requirements.txt` | Python dependencies. |
| `server/` | Local Pokémon Showdown server (cloned, not committed). |
