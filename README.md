# Pokémon Showdown Battle AI

Building an AI that plays [Pokémon Showdown](https://pokemonshowdown.com/) battles, in stages:

1. **Heuristic bot** (✅ done) — a rule-based agent that picks the best move by base power, STAB, and type effectiveness, and switches intelligently when forced. **Wins ~96% vs. a random-move baseline.** Serves as the foundation, benchmark, and training partner for the learning agent.
2. **Reinforcement-learning agent** (next) — a model trained from win/loss rewards via self-play, using poke-env's Gymnasium interface + Stable-Baselines3.
3. **Stretch** — learning curves, win-rate tracking, and laddering on the live server for a real Elo rating.

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
