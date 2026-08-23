# Legacy — the pre-poke-engine era

This folder is an **archive** of everything the project built *before* it became a search bot.
None of it is on the live path (the shipped bot is in [`../src/laplace/`](../src/laplace/)); it's kept because
it's the story of how [Laplace](../README.md) got here, and because the lessons are the
interesting part of the project.

The trained PPO weight files (`ppo_*.zip`) are **not committed** — they're large binary
artifacts, regenerable from these scripts, and covered by `.gitignore`. The scripts and the
results table below are what preserve this chapter.

> These scripts originally lived in `src/` as flat modules and assumed that working directory
> (sibling imports, `models/` paths). They are archived as-is for reference, not maintained
> to run from here.

## Rule-based vs. learned

The contrast between the two kinds of bot here was the original point of the project.

A **rule-based bot** does exactly what a person tells it — "play the highest-damage move,
switch out on a bad matchup" — and follows those rules identically every game. The only way to
improve it is to write more rules. `heuristic_bot.py` is that bot; it served as the fixed
sparring partner and baseline.

The **reinforcement-learning agent** is given no rules. It's a neural network that starts by
choosing at random and learns by playing: after each battle it gets a reward (a little for
damage or a KO, a lot for winning), and PPO adjusts the network's weights so choices that led
toward wins become more likely. Over thousands of games it works out its own strategy — nobody
writes "switch here." It's *reinforcement* learning specifically because there's no answer key
of correct moves; the agent only ever sees how a game turned out and has to infer which
decisions earned the result.

## The models, in order

Win rates are vs `SimpleHeuristicsPlayer` unless noted — a beginner-level yardstick for
comparing versions, not ladder ratings. Small evals are noisy (~8 points over 50 battles), so
later runs measure over 200–400.

| # | Model | Win rate | What it added / what went wrong |
|---|---|---|---|
| 1 | **Heuristic bot** `heuristic_bot.py` | ~96% vs random | Hand-written rules. The baseline and sparring partner, not a learner. |
| 2 | **First RL agent, 12 features** `ppo_vs_random/heuristic.zip` | ~100% vs random; ~30% vs heuristic | Proved the pipeline learns, but too blind to stats/status/hazards to beat a real opponent. |
| 3 | **Richer obs, 141 features** `ppo_vs_heuristic_obs141.zip` | ~41% | Added boosts, status, bench, move details, weather, abilities. Broke the 30% plateau. |
| 4 | **Attention net, 854 features** `ppo_v2_attn_obs854.zip` | ~4% | The one dead end (`v2_attention/`): laid the battle out as 12 Pokémon "tokens" for self-attention, but most are zero early (hidden team) and drowned the signal. Dropped. |
| 5 | **Extended obs, 215 features** `ppo_vs_heuristic_obs215.zip` | ~34–40% | Added priority, item, Tera, and a knowledge layer predicting the opponent's set. No gain — the turning point that proved observation was no longer the bottleneck. |
| 6 | **Scaled v3** `ppo_v3_obs215.zip` | ~45% peak → ~30% | Bigger net, win-focused reward, parallel training. New high, but overtrained downhill with no checkpoint hygiene. |
| 7 | **Rescaled reward** `ppo_v3_rescaled_obs215.zip` | no change | Ruled out reward scale as the cap; pointed at partial observability. |
| 8 | **Self-play** `ppo_selfplay*.zip` | ~43% | Trained against snapshots of itself. Didn't transfer — the ceiling wasn't "opponent too easy." |
| 9 | **Reward-anneal + anti-panic-switch** `ppo_v3_anneal*.zip` | ~40% | Annealed dense shaping toward zero while keeping the win bonus. Flat — confirming reward wasn't the lever; the ceiling was partial observability + a reactive 1-turn policy. |
| 10 | **Test-time search** `search.py` | **~55–62%** | Not a new model: a 1-ply evaluator scoring every legal action with the damage model, using the trained policy only as a prior. The biggest lever among the from-scratch agents. |

The plateau at #10 — a 1-ply searcher over a learned policy — is what motivated the pivot to a
real battle engine and full MCTS, which became the shipped bot.

## Lessons learned

- **A plateau is usually a representation problem first.** The agent stalled until the
  observation was enriched (12 → 141 features); more game state beat more training.
- **More is not always better.** 141 → 854 features *hurt* (most were empty); 141 → 215 didn't
  help. Past a point, observation stopped being the bottleneck.
- **Measure honestly.** Small evals invent improvements that aren't real (an apparent self-play
  "peak" was sampling noise) — headline numbers used 200–400 battles.
- **Reward shape drives behaviour.** Reward climbing while win rate stays flat means the agent
  learned to *trade*, not to *win*.

These carried directly into the search era, where "measure honestly" became the rule that no
change ships without a head-to-head A/B test.

## What's here

| Path | Purpose |
|---|---|
| `heuristic_bot.py`, `run_battle.py` | The rule-based bot and its benchmark. |
| `rl_env.py` | The RL environment: the 215-number observation and reward. |
| `train_rl.py`, `train_selfplay.py`, `train_v3*.py` | The PPO trainers (scaled net, reward anneal, self-play). |
| `opponent.py` | Model-driven self-play opponent. |
| `search.py`, `deep_search.py`, `heuristic_search.py` | Test-time 1-ply / experimental 2-ply / heuristic+ searchers. |
| `eval_search.py` | Benchmark across the raw / 1-ply / heuristic+ / deep agents. |
| `play.py` | Plays watchable battles across all the old bot kinds, saved as browser replays. |
| `smoke_*.py` | Quick checks that the env / parallel training / self-play start cleanly. |
| `v2_attention/` | The attention experiment (dead end): 854-number obs, token net, trainer. |
