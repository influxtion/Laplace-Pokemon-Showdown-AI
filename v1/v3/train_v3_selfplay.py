r"""Self-play on top of the v3 agent -- the "stack #1 + #2" push toward a higher win rate.

The v3 agent plateaus around 40% vs SimpleHeuristicsPlayer because, once it has solved that
one opponent, there's nothing left to learn. Self-play replaces the fixed heuristic with a
periodically-refreshed snapshot of the agent itself, so the target keeps rising as the agent
does -- the way to push past a single opponent's ceiling.

This differs from v1/selfplay/train_selfplay.py in two ways:
  - It WARM-STARTS from the v3 model (bigger net [256,256], gamma 0.997, win-focused reward),
    not the small heuristic-trained net. Loading preserves the architecture, gamma, and reward
    scale automatically, so the value function stays calibrated.
  - It adds training hygiene that the plain v3 run lacked (and which let it overtrain into a
    worse policy): target_kl early-stopping, a learning rate that decays to zero, and BEST-model
    checkpointing so we keep the peak rather than the drifted-downhill tail.

The benchmark is still SimpleHeuristicsPlayer every EVAL_FREQ steps, so the number stays
comparable to every earlier run. At test time, layer search.py on top of the saved model
(eval_search.py) to stack the lookahead gain on the stronger self-play policy.

Run from the project root, with the local Showdown server running:
    python -u v1\v3\train_v3_selfplay.py
"""

import logging
import os
import sys

# This file lives in v1/v3/ but reuses modules in v1/ (rl_env, train_rl) and v1/selfplay/
# (the ModelPlayer opponent + SnapshotCallback). Put both on the import path.
_V1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _V1)
sys.path.insert(0, os.path.join(_V1, "selfplay"))

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

# --- hygiene (the part plain train_v3 was missing) ----------------------------
TARGET_KL = 0.03           # early-stop an update batch if it moves the policy too far (v3's
                           # approx_kl crept to 0.06 and the policy drifted downhill)
LR_START = 3e-4            # SB3 default; decays LINEARLY to 0 across this run for a stable finish

TRAIN_STEPS = 3_000_000
REFRESH_FREQ = 50_000      # how often to copy the learner's weights into the opponent
EVAL_FREQ = 25_000         # how often to benchmark vs the heuristic (live graph)
EVAL_BATTLES = 200         # battles for the before/after headline benchmark
LIVE_EVAL_BATTLES = 100    # battles per live benchmark point
SAVE_FREQ = 100_000

# Warm-start from the v3 agent; save self-play progress + the peak under their own names.
WARM_START_PATH = f"ppo_v3_obs{N_FEATURES}.zip"
MODEL_PATH = f"ppo_v3_selfplay_obs{N_FEATURES}.zip"          # latest (for resuming)
BEST_PATH = f"ppo_v3_selfplay_best_obs{N_FEATURES}.zip"      # highest win rate seen (for eval)
SNAPSHOT_PATH = "ppo_v3_selfplay_snapshot.zip"              # scratch file to seed the opponent
TB_LOG_NAME = f"ppo_v3_selfplay_obs{N_FEATURES}"


def linear_schedule(start):
    """progress_remaining goes 1 -> 0 over the run, so this decays the LR from `start` to 0."""
    return lambda progress_remaining: progress_remaining * start


def make_self_opponent():
    return ModelPlayer(
        model=None, deterministic=False,
        account_configuration=AccountConfiguration.generate("v3sp-opp", rand=True),
        battle_format=BATTLE_FORMAT, start_listening=False,
    )


def build_train_env(opponent):
    """The training env: the agent vs a snapshot of itself. Uses the v3 win-focused reward
    (train_rl.REWARD_WEIGHTS) so the value function stays on the same scale it warm-starts on."""
    showdown = ShowdownSinglesEnv(
        reward_weights=base.REWARD_WEIGHTS,
        account_configuration1=AccountConfiguration.generate("v3spA", rand=True),
        account_configuration2=AccountConfiguration.generate("v3spB", rand=True),
        battle_format=BATTLE_FORMAT, strict=False, log_level=logging.ERROR,
    )
    wrapped = SingleAgentWrapper(showdown, opponent)
    return ActionMasker(wrapped, base.mask_fn), showdown


def build_eval_env():
    """A separate env that plays the heuristic, for the honest (comparable) benchmark."""
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

    # Start from a saved self-play agent (continue), else warm-start from the v3 model.
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
            f"Train the v3 agent first: python -u v1\\v3\\train_v3.py")
    model = MaskablePPO.load(start_path, env=env, tensorboard_log=base.TB_DIR)

    # Apply the hygiene the base v3 run lacked. Loading preserves net_arch, gamma, and the
    # reward scale; we only override the optimizer behaviour here.
    model.target_kl = TARGET_KL
    model.lr_schedule = linear_schedule(LR_START)

    # Seed the opponent with a copy of the learner so it starts as an equal mirror.
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
