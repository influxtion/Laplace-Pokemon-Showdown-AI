r"""Fresh "done-right" agent for the 70% push (trained from scratch vs the heuristic).

This bundles the improvements that REQUIRE a clean start (they change the architecture or
the learning dynamics, so they can't be bolted onto an existing model):

  - Bigger network: net_arch [256, 256] (the SB3 default is small ~[64, 64]) for more
    capacity on the 215-feature observation and the game's complexity.
  - Higher gamma: 0.997 (default 0.99) so the terminal WIN is discounted less over a
    ~30-turn game -> better credit assignment for actually closing games.
  - Win-focused reward + parallel envs + 200-battle evals: reused from train_rl (so the
    reward is hp 0.25 / victory 150, training runs across N_ENVS processes).

Built to be trained for millions of steps; re-run to keep adding steps (it resumes).

Run from the project root, with the local Showdown server running:
    python -u v1\v3\train_v3.py
"""

import os
import sys

# This file lives in v1/v3/ but reuses v1/ modules (train_rl, rl_env). Put the parent v1/
# directory on the import path so those bare imports resolve when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Reuse the env factories, eval, callbacks, and constants from the main trainer. That keeps
# the win-focused reward and parallel-env machinery in one place.
import train_rl as base
from rl_env import N_FEATURES

NET_ARCH = [256, 256]      # bigger policy/value network
GAMMA = 0.997              # value the win more over a long game
ENT_COEF = 0.01
TRAIN_STEPS = 100_000      

MODEL_PATH = f"ppo_v3_obs{N_FEATURES}.zip"
TB_LOG_NAME = f"ppo_v3_obs{N_FEATURES}"


def main():
    train_env = (
        SubprocVecEnv([base.make_env(i) for i in range(base.N_ENVS)])
        if base.N_ENVS > 1 else DummyVecEnv([base.make_env(0)])
    )
    eval_env, eval_inner = base.build_env()

    resuming = os.path.exists(MODEL_PATH)
    if resuming:
        print(f"Found {MODEL_PATH} -> CONTINUING the v3 agent.", flush=True)
        model = MaskablePPO.load(MODEL_PATH, env=train_env, tensorboard_log=base.TB_DIR)
    else:
        print(f"Starting a FRESH v3 agent (net {NET_ARCH}, gamma {GAMMA}, {N_FEATURES} features).", flush=True)
        model = MaskablePPO(
            "MultiInputPolicy", train_env, verbose=1,
            ent_coef=ENT_COEF, gamma=GAMMA,
            policy_kwargs=dict(net_arch=NET_ARCH),
            tensorboard_log=base.TB_DIR,
        )

    print(f"\n=== Win rate vs {base.OPPONENT_LABEL} BEFORE ({base.EVAL_BATTLES} battles) ===", flush=True)
    before = base.win_rate(model, eval_env, eval_inner, base.EVAL_BATTLES)
    print(f"Before: {before:.0%}", flush=True)

    callbacks = CallbackList([
        base.WinRateCallback(eval_env, eval_inner, base.LIVE_EVAL_BATTLES, base.EVAL_FREQ),
        base.SaveCallback(MODEL_PATH, base.SAVE_FREQ),
    ])
    print(f"\n=== Training {TRAIN_STEPS:,} steps across {base.N_ENVS} envs vs {base.OPPONENT_LABEL} ===", flush=True)
    model.learn(
        total_timesteps=TRAIN_STEPS,
        reset_num_timesteps=not resuming,
        tb_log_name=TB_LOG_NAME,
        callback=callbacks,
    )
    model.save(MODEL_PATH)
    print(f"Saved {MODEL_PATH}", flush=True)

    print(f"\n=== Win rate vs {base.OPPONENT_LABEL} AFTER ({base.EVAL_BATTLES} battles) ===", flush=True)
    after = base.win_rate(model, eval_env, eval_inner, base.EVAL_BATTLES)
    print(f"After: {after:.0%}", flush=True)
    print(f"\nResult: {before:.0%} -> {after:.0%}", flush=True)
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
