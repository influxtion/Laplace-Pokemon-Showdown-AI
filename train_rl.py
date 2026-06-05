"""Train a reinforcement-learning agent to battle, and measure if it improved.

Prerequisite: the local Showdown server must be running (see README):
    cd server && node pokemon-showdown start --no-security

Then:
    python -u train_rl.py

What this does:
  1. Builds the RL environment with an opponent to learn against.
  2. Measures the *untrained* agent's win rate (a baseline).
  3. Trains it with MaskablePPO for TRAIN_STEPS turns of experience.
  4. Measures the *trained* agent's win rate.
  5. Saves the trained model to ppo_showdown.zip.

If training worked, the win rate should go UP between steps 2 and 4.

Why MaskablePPO (and not plain PPO)? At every turn only some of the 26 possible
actions are legal (you can't use a move a Pokemon doesn't have, etc.). poke-env
gives us an "action mask" of the legal choices each turn. MaskablePPO uses that
mask so the agent ONLY ever picks legal actions -- which makes learning much
faster and avoids the agent flailing at illegal moves.
"""

import logging

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.environment.singles_env import SinglesEnv
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from rl_env import ShowdownSinglesEnv

BATTLE_FORMAT = "gen9randombattle"
TRAIN_STEPS = 20_000     # turns of experience to learn from (bump up for a stronger agent)
EVAL_BATTLES = 50        # battles used to estimate win rate
MODEL_PATH = "ppo_showdown.zip"


def mask_fn(env):
    """Tell MaskablePPO which of the 26 actions are legal in the current state.

    `env` is the SingleAgentWrapper; `env.env` is our ShowdownSinglesEnv, whose
    `battle1` is the agent's current battle.
    """
    return np.array(SinglesEnv.get_action_mask(env.env.battle1), dtype=bool)


def build_env():
    """Create the masked learning environment and return it plus the inner env."""
    showdown = ShowdownSinglesEnv(
        battle_format=BATTLE_FORMAT,
        strict=False,                 # safety net: illegal action -> random legal move
        log_level=logging.ERROR,      # quiet the per-turn warning spam
    )
    # The opponent the agent learns against. A random opponent gives the clearest
    # "is it learning?" signal first. Later, swap in our Stage 1 heuristic
    # (MaxDamagePlayer) or poke-env's SimpleHeuristicsPlayer for a tougher teacher.
    opponent = RandomPlayer(
        account_configuration=AccountConfiguration("RandomOpp", None),
        battle_format=BATTLE_FORMAT,
        start_listening=False,        # used only as a "brain"; needs no own connection
    )
    wrapped = SingleAgentWrapper(showdown, opponent)
    masked = ActionMasker(wrapped, mask_fn)
    return masked, showdown


def win_rate(model, env, inner, n_battles):
    """Play n_battles with the model and return the fraction it won."""
    wins = 0
    for _ in range(n_battles):
        obs, _ = env.reset()
        done = False
        while not done:
            masks = np.asarray(obs["action_mask"], dtype=bool)
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(np.int64(action))
            done = terminated or truncated
        if inner.battle1 is not None and inner.battle1.won:
            wins += 1
    return wins / n_battles


def main():
    env, inner = build_env()

    # MultiInputPolicy because our observation is a dict (state numbers + action mask).
    model = MaskablePPO("MultiInputPolicy", env, verbose=1)

    print("\n=== Evaluating UNTRAINED agent ===", flush=True)
    before = win_rate(model, env, inner, EVAL_BATTLES)
    print(f"Untrained win rate vs RandomPlayer: {before:.0%}", flush=True)

    print(f"\n=== Training for {TRAIN_STEPS:,} steps ===", flush=True)
    model.learn(total_timesteps=TRAIN_STEPS)
    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}", flush=True)

    print("\n=== Evaluating TRAINED agent ===", flush=True)
    after = win_rate(model, env, inner, EVAL_BATTLES)
    print(f"Trained win rate vs RandomPlayer:   {after:.0%}", flush=True)

    print(f"\nResult: {before:.0%} -> {after:.0%} win rate after {TRAIN_STEPS:,} steps.", flush=True)
    env.close()


if __name__ == "__main__":
    main()
