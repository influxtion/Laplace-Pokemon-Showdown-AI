r"""Self-play, this time starting from the bigger agent.

The bigger agent stalls around 40% against the fixed opponent, because once you've solved
one opponent there's nothing left to learn from it. Self-play swaps that for a copy of the
agent itself, refreshed as it improves, so the target keeps rising.

Two things differ from the earlier self-play attempt:
  - It starts from the big network rather than the small one, keeping its shape, discount
    and reward scale intact so its judgement stays calibrated.
  - It adds the safeguards the earlier big run was missing, and whose absence let that run
    quietly train itself into a worse policy: a cap on how far the policy can move at once,
    a learning rate winding down to zero, and keeping the best version rather than the last.

It still measures itself against the fixed heuristic throughout, so the numbers remain
comparable to everything before.

Run from the project root, with the local server going:
    python -u src\train_v3_selfplay.py
"""

import logging
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CallbackList

from poke_env import AccountConfiguration
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from rl_env import ShowdownSinglesEnv, N_FEATURES
import train_rl as base
from opponent import ModelPlayer
from train_selfplay import SnapshotCallback

BATTLE_FORMAT = "gen9randombattle"

# --- the safeguards the previous run lacked -----------------------------------
TARGET_KL = 0.03           # stop a batch early if it's moving the policy too far. Last time
                           # this crept up and the policy drifted downhill for hours.
LR_START = 3e-4            # winds down to zero over the run

TRAIN_STEPS = 3_000_000
REFRESH_FREQ = 50_000      # how often the opponent catches up with the learner
EVAL_FREQ = 25_000
EVAL_BATTLES = 200         # for the before-and-after comparison
LIVE_EVAL_BATTLES = 100    # per point on the live graph
SAVE_FREQ = 100_000

WARM_START_PATH = f"ppo_v3_obs{N_FEATURES}.zip"
MODEL_PATH = f"ppo_v3_selfplay_obs{N_FEATURES}.zip"          # the latest, to resume from
BEST_PATH = f"ppo_v3_selfplay_best_obs{N_FEATURES}.zip"      # the best, to actually use
SNAPSHOT_PATH = "ppo_v3_selfplay_snapshot.zip"              # scratch file for the opponent
TB_LOG_NAME = f"ppo_v3_selfplay_obs{N_FEATURES}"


def linear_schedule(start):
    """Wind the learning rate down to zero as the run progresses."""
    return lambda progress_remaining: progress_remaining * start


def make_self_opponent():
    return ModelPlayer(
        model=None, deterministic=False,
        account_configuration=AccountConfiguration.generate("v3sp-opp", rand=True),
        battle_format=BATTLE_FORMAT, start_listening=False,
    )


def build_train_env(opponent):
    """The agent against a copy of itself, using the same rewards it was trained on, so its
    sense of what a position is worth doesn't suddenly change scale."""
    showdown = ShowdownSinglesEnv(
        reward_weights=base.REWARD_WEIGHTS,
        account_configuration1=AccountConfiguration.generate("v3spA", rand=True),
        account_configuration2=AccountConfiguration.generate("v3spB", rand=True),
        battle_format=BATTLE_FORMAT, strict=False, log_level=logging.ERROR,
    )
    wrapped = SingleAgentWrapper(showdown, opponent)
    return ActionMasker(wrapped, base.mask_fn), showdown


def build_eval_env():
    """A separate game against the fixed heuristic, so the numbers stay comparable."""
    showdown = ShowdownSinglesEnv(
        account_configuration1=AccountConfiguration.generate("v3spEvA", rand=True),
        account_configuration2=AccountConfiguration.generate("v3spEvB", rand=True),
        battle_format=BATTLE_FORMAT, strict=False, log_level=logging.ERROR,
    )
    opp = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration.generate("v3sp-eval-opp", rand=True),
        battle_format=BATTLE_FORMAT, start_listening=False,
    )
    wrapped = SingleAgentWrapper(showdown, opp)
    return ActionMasker(wrapped, base.mask_fn), showdown


def main():
    opponent = make_self_opponent()
    env, _ = build_train_env(opponent)
    eval_env, eval_inner = build_eval_env()

    # Carry on from a saved self-play agent if there is one, otherwise start from the big one.
    resuming = os.path.exists(MODEL_PATH)
    if resuming:
        print(f"CONTINUING v3 self-play from {MODEL_PATH}.", flush=True)
        start_path = MODEL_PATH
    elif os.path.exists(WARM_START_PATH):
        print(f"WARM-STARTING v3 self-play from {WARM_START_PATH}.", flush=True)
        start_path = WARM_START_PATH
    else:
        raise FileNotFoundError(
            f"Need {MODEL_PATH} (to resume) or {WARM_START_PATH} (to warm-start). "
            f"Train the v3 agent first: python -u src\\train_v3.py")
    model = MaskablePPO.load(start_path, env=env, tensorboard_log=base.TB_DIR)

    # Loading keeps the network, discount and reward scale; we only change the optimiser.
    model.target_kl = TARGET_KL
    model.lr_schedule = linear_schedule(LR_START)

    # Give the opponent a copy of the learner, so it starts out as an exact mirror.
    model.save(SNAPSHOT_PATH)
    opponent.model = MaskablePPO.load(SNAPSHOT_PATH)

    print(f"\n=== Win rate vs SimpleHeuristicsPlayer BEFORE ({EVAL_BATTLES} battles) ===", flush=True)
    before = base.win_rate(model, eval_env, eval_inner, EVAL_BATTLES)
    print(f"Before: {before:.0%}", flush=True)

    callbacks = CallbackList([
        SnapshotCallback(opponent, REFRESH_FREQ),
        base.WinRateCallback(eval_env, eval_inner, LIVE_EVAL_BATTLES, EVAL_FREQ, best_path=BEST_PATH),
        base.SaveCallback(MODEL_PATH, SAVE_FREQ),
    ])
    print(f"\n=== v3 self-play: {TRAIN_STEPS:,} steps (target_kl={TARGET_KL}, lr {LR_START:g}->0) ===", flush=True)
    model.learn(
        total_timesteps=TRAIN_STEPS,
        reset_num_timesteps=not resuming,
        tb_log_name=TB_LOG_NAME,
        callback=callbacks,
    )
    model.save(MODEL_PATH)
    print(f"Saved {MODEL_PATH} (latest) and {BEST_PATH} (peak win rate).", flush=True)

    print(f"\n=== Win rate vs SimpleHeuristicsPlayer AFTER ({EVAL_BATTLES} battles) ===", flush=True)
    after = base.win_rate(model, eval_env, eval_inner, EVAL_BATTLES)
    print(f"After: {after:.0%}", flush=True)
    print(f"\nvs heuristic: {before:.0%} -> {after:.0%}  (best checkpoint saved to {BEST_PATH})", flush=True)
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
