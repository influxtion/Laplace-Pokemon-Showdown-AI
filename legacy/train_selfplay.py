r"""Train the agent against copies of itself.

Against one fixed opponent the agent stalled around 40%. Once it has that opponent figured
out there's nothing left to learn. Self-play gives it a target that gets better as it does,
which in theory pushes past that ceiling.

The opponent runs a snapshot of the agent's own network, refreshed every so often. Keeping
it slightly behind the live version is deliberate: training against your exact current self
tends to oscillate rather than improve.

Meant to be run in two passes. First with the normal win bonus, to learn the mirror match.
Then again with it raised, so the agent starts valuing closing games out over trading
evenly. Only the one constant below changes between the two.

It still measures itself against the fixed heuristic opponent throughout, so the numbers
stay comparable to the earlier runs.

Run from the project root, with the local server going:
    python -u src\train_selfplay.py
"""

import logging
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from poke_env import AccountConfiguration
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from rl_env import ShowdownSinglesEnv, N_FEATURES
from opponent import ModelPlayer
# Borrow the benchmarking and saving machinery from the main trainer.
from train_rl import win_rate, WinRateCallback, SaveCallback, mask_fn

BATTLE_FORMAT = "gen9randombattle"

# The one thing that changes between the two passes. Start at 100, then raise it.
VICTORY_VALUE = 100.0

TRAIN_STEPS = 100_000
REFRESH_FREQ = 50_000     # how often the opponent catches up with the learner
EVAL_FREQ = 10_000        # how often to check ourselves against the heuristic
EVAL_BATTLES = 200        # games in the before-and-after comparison
LIVE_EVAL_BATTLES = 50    # games per point on the live graph
SAVE_FREQ = 100_000
TB_DIR = "tb_logs"

MODEL_PATH = f"ppo_selfplay_obs{N_FEATURES}.zip"
WARM_START_PATH = f"ppo_vs_heuristic_obs{N_FEATURES}.zip"  # what to start from
SNAPSHOT_PATH = "ppo_selfplay_snapshot.zip"               # scratch file for the opponent


def make_self_opponent():
    return ModelPlayer(
        model=None, deterministic=False,
        account_configuration=AccountConfiguration("selfplay-opp", None),
        battle_format=BATTLE_FORMAT, start_listening=False,
    )


def build_train_env(opponent):
    showdown = ShowdownSinglesEnv(
        reward_weights={"victory_value": VICTORY_VALUE},
        battle_format=BATTLE_FORMAT, strict=False, log_level=logging.ERROR,
    )
    wrapped = SingleAgentWrapper(showdown, opponent)
    return ActionMasker(wrapped, mask_fn), showdown


def build_eval_env():
    """A separate game against the fixed heuristic, which is the honest measurement."""
    showdown = ShowdownSinglesEnv(
        battle_format=BATTLE_FORMAT, strict=False, log_level=logging.ERROR,
    )
    opp = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration("selfplay-eval-opp", None),
        battle_format=BATTLE_FORMAT, start_listening=False,
    )
    wrapped = SingleAgentWrapper(showdown, opp)
    return ActionMasker(wrapped, mask_fn), showdown


class SnapshotCallback(BaseCallback):
    """Periodically bring the self-play opponent up to date with the learner."""

    def __init__(self, opponent, refresh_freq, verbose=1):
        super().__init__(verbose)
        self.opponent = opponent
        self.refresh_freq = refresh_freq

    def _on_step(self):
        if self.n_calls % self.refresh_freq == 0:
            self.opponent.sync_from(self.model)
            if self.verbose:
                print(f"[selfplay] refreshed opponent at {self.num_timesteps:,} steps", flush=True)
        return True


def main():
    opponent = make_self_opponent()
    env, inner = build_train_env(opponent)
    eval_env, eval_inner = build_eval_env()

    # Pick up where we left off if possible, otherwise start from the heuristic-trained
    # agent, otherwise from nothing.
    resuming = os.path.exists(MODEL_PATH)
    start_path = MODEL_PATH if resuming else (WARM_START_PATH if os.path.exists(WARM_START_PATH) else None)
    if start_path:
        kind = "CONTINUING self-play" if resuming else f"WARM-STARTING from {WARM_START_PATH}"
        print(f"{kind}.", flush=True)
        model = MaskablePPO.load(start_path, env=env, tensorboard_log=TB_DIR)
    else:
        print(f"No model found -> FRESH self-play agent ({N_FEATURES} features).", flush=True)
        model = MaskablePPO(
            "MultiInputPolicy", env, verbose=1, ent_coef=0.01, tensorboard_log=TB_DIR,
        )

    # Give the opponent a copy of the learner, so it starts out as an exact mirror.
    model.save(SNAPSHOT_PATH)
    opponent.model = MaskablePPO.load(SNAPSHOT_PATH)

    print(f"\n=== Win rate vs SimpleHeuristicsPlayer BEFORE (victory_value={VICTORY_VALUE:g}) ===", flush=True)
    before = win_rate(model, eval_env, eval_inner, EVAL_BATTLES)
    print(f"Before: {before:.0%}", flush=True)

    callbacks = CallbackList([
        SnapshotCallback(opponent, REFRESH_FREQ),
        WinRateCallback(eval_env, eval_inner, LIVE_EVAL_BATTLES, EVAL_FREQ),
        SaveCallback(MODEL_PATH, SAVE_FREQ),
    ])
    print(f"\n=== Self-play training {TRAIN_STEPS:,} steps ===", flush=True)
    model.learn(
        total_timesteps=TRAIN_STEPS,
        reset_num_timesteps=not resuming,
        tb_log_name=f"ppo_selfplay_obs{N_FEATURES}",
        callback=callbacks,
    )
    model.save(MODEL_PATH)
    print(f"Saved {MODEL_PATH}", flush=True)

    print("\n=== Win rate vs SimpleHeuristicsPlayer AFTER ===", flush=True)
    after = win_rate(model, eval_env, eval_inner, EVAL_BATTLES)
    print(f"After: {after:.0%}", flush=True)
    print(f"\nvs heuristic: {before:.0%} -> {after:.0%}  (earlier baseline ~41%)", flush=True)
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
