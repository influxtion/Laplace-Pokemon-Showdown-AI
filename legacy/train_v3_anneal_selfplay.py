r"""Self-play on top of the annealed agent. This is the last of the pre-search bots.

The one experiment left to try before taking it to the ladder. The best agent so far had
only ever played one opponent, ever. Self-play at least gives it a varied, moving target.

Self-play hadn't moved the heuristic benchmark before, but that's the wrong thing to watch
here. The point isn't beating the heuristic better; it's seeing more kinds of opponent,
which might generalise to real humans in a way that a policy raised on a single bot won't.
The number that matters is the ladder rating, not the benchmark, which may well stay flat.

It picks up where the fade left off, and keeps the fade's final rewards rather than
reverting to the larger earlier ones. Snapping back to rewards ten times the size would
throw off the agent's sense of what a position is worth.

Run from the project root, with the local server going:
    python -u src\train_v3_anneal_selfplay.py
"""

import logging
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CallbackList

from poke_env import AccountConfiguration
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from rl_env import ShowdownSinglesEnv, N_FEATURES
import train_rl as base
from train_selfplay import SnapshotCallback
# The self-play machinery, borrowed wholesale from the previous run.
from train_v3_selfplay import make_self_opponent, build_eval_env, linear_schedule
# The rewards the fade ended on, so picking up from it is seamless.
from train_v3_anneal import ANNEAL_END

BATTLE_FORMAT = "gen9randombattle"

# --- the same safeguards as the other finetunes -------------------------------
TARGET_KL = 0.03
LR_START = 3e-4            # winds down to zero over the run

TRAIN_STEPS = 500_000
REFRESH_FREQ = 50_000      # how often the opponent catches up with the learner
EVAL_FREQ = 15_000
EVAL_BATTLES = 100
LIVE_EVAL_BATTLES = 100
SAVE_FREQ = 100_000

# Start from the best annealed agent, or the latest one if there's no best saved.
WARM_START_PATHS = [
    f"ppo_v3_anneal_best_obs{N_FEATURES}.zip",
    f"ppo_v3_anneal_obs{N_FEATURES}.zip",
]
MODEL_PATH = f"ppo_v3_anneal_selfplay_obs{N_FEATURES}.zip"          # the latest, to resume from
BEST_PATH = f"ppo_v3_anneal_selfplay_best_obs{N_FEATURES}.zip"      # the best, to actually use
SNAPSHOT_PATH = "ppo_v3_anneal_selfplay_snapshot.zip"              # scratch file for the opponent
TB_LOG_NAME = f"ppo_v3_anneal_selfplay_obs{N_FEATURES}"


def build_train_env(opponent):
    """The agent against a copy of itself, on the rewards the fade finished with."""
    showdown = ShowdownSinglesEnv(
        reward_weights=ANNEAL_END,
        account_configuration1=AccountConfiguration.generate("v3aspA", rand=True),
        account_configuration2=AccountConfiguration.generate("v3aspB", rand=True),
        battle_format=BATTLE_FORMAT, strict=False, log_level=logging.ERROR,
    )
    wrapped = SingleAgentWrapper(showdown, opponent)
    return ActionMasker(wrapped, base.mask_fn), showdown


def main():
    opponent = make_self_opponent()
    env, _ = build_train_env(opponent)
    eval_env, eval_inner = build_eval_env()

    # Carry on if we've already started, otherwise begin from the best annealed agent.
    resuming = os.path.exists(MODEL_PATH)
    if resuming:
        print(f"CONTINUING anneal self-play from {MODEL_PATH}.", flush=True)
        start_path = MODEL_PATH
    else:
        start_path = next((p for p in WARM_START_PATHS if os.path.exists(p)), None)
        if start_path is None:
            raise FileNotFoundError(
                f"Need {MODEL_PATH} (to resume) or one of {WARM_START_PATHS} (to warm-start). "
                f"Run the anneal finetune first: python -u src\\train_v3_anneal.py")
        print(f"WARM-STARTING anneal self-play from {start_path}.", flush=True)
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
    print(f"\n=== Anneal self-play: {TRAIN_STEPS:,} steps "
          f"(rewards hp{ANNEAL_END['hp_value']}/win{ANNEAL_END['victory_value']:g}; "
          f"target_kl={TARGET_KL}, lr {LR_START:g}->0) ===", flush=True)
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
    print(f"\nvs heuristic: {before:.0%} -> {after:.0%}  (best saved to {BEST_PATH}). "
          f"Remember the real test is the ladder, not this number.", flush=True)
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
