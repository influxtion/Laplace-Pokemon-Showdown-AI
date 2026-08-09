r"""A bigger agent, trained from scratch, in an attempt to break the plateau.

Everything here needs a clean start, because it changes the shape of the network or how it
learns, and you can't bolt that onto an existing model:

  - A much bigger network, since the default is tiny next to a 215-number observation.
  - A higher discount, so a win 30 turns away still counts for something and the agent
    learns which early decisions led to it.
  - The win-focused rewards, parallel games and large evaluations from the main trainer.

There's also a rescaled-reward experiment behind a switch below. The theory was that reward
values running into the hundreds were making the value estimates hard to fit. Dividing
everything by ten keeps the same balance between winning and trading while shrinking the
numbers. It didn't help. The ceiling turned out to be that the agent can't see the
opponent's hidden information, not the reward scale.

Built for millions of steps. Re-run it to add more; it picks up where it stopped.

Run from the project root, with the local server going:
    python -u src\train_v3.py
"""

import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Borrow the game setup, evaluation and callbacks from the main trainer, so the rewards and
# the parallel-game machinery only exist in one place.
import train_rl as base
from rl_env import N_FEATURES

NET_ARCH = [256, 256]      # a much bigger network than the default
GAMMA = 0.997              # care more about the eventual win
ENT_COEF = 0.01
TRAIN_STEPS = 2_000_000

# The rescaled-reward experiment. Every value is the normal one divided by ten. All four
# have to be listed: leave one out and it gets filled in at full size, which wrecks the
# balance the experiment was testing.
RESCALED = False    # it made no difference. Left here as a documented dead end.
RESCALED_REWARD = {"hp_value": 0.025, "fainted_value": 0.1,
                   "status_value": 0.01, "victory_value": 15.0}
REWARD_OVERRIDE = RESCALED_REWARD if RESCALED else None

# Tagging the filename keeps the two variants' models and logs apart, so they can be trained
# and compared side by side.
TAG = "_rescaled" if RESCALED else ""
MODEL_PATH = f"ppo_v3{TAG}_obs{N_FEATURES}.zip"
TB_LOG_NAME = f"ppo_v3{TAG}_obs{N_FEATURES}"


def main():
    train_env = (
        SubprocVecEnv([base.make_env(i, REWARD_OVERRIDE) for i in range(base.N_ENVS)])
        if base.N_ENVS > 1 else DummyVecEnv([base.make_env(0, REWARD_OVERRIDE)])
    )
    eval_env, eval_inner = base.build_env(REWARD_OVERRIDE)

    resuming = os.path.exists(MODEL_PATH)
    if resuming:
        print(f"Found {MODEL_PATH} -> CONTINUING the v3 agent.", flush=True)
        model = MaskablePPO.load(MODEL_PATH, env=train_env, tensorboard_log=base.TB_DIR)
    else:
        reward_desc = f"rescaled {RESCALED_REWARD}" if RESCALED else "train_rl default (hp 0.25 / victory 150)"
        print(f"Starting a FRESH v3 agent (net {NET_ARCH}, gamma {GAMMA}, {N_FEATURES} features, reward: {reward_desc}).", flush=True)
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
