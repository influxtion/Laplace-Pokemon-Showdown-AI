r"""Phase 2: self-play. Train the agent against periodically-refreshed copies of itself.

Against a single fixed opponent (the heuristic) the agent plateaued near 40%: once it has the
opponent figured out, there's nothing left to learn. Self-play gives it a moving target that
gets stronger as it does, which pushes past a fixed opponent's ceiling.

The opponent (opponent.ModelPlayer) runs a snapshot of the agent's network. Every REFRESH_FREQ
steps we copy the learner's weights in, so it lags slightly behind and stays a stable rising
target -- training against the exact live policy tends to oscillate.

Curriculum:
  Phase A: leave VICTORY_VALUE at 100 and train. Learn the mirror match on the normal reward.
  Phase B: raise VICTORY_VALUE (e.g. 200) and re-run. The script resumes the saved model, so
           it keeps its skills but now values closing games over even trades. Only the one
           constant below changes between phases.

Still benchmarks vs SimpleHeuristicsPlayer every EVAL_FREQ steps so progress stays comparable
to the earlier runs (~41% baseline).

Run from the project root, with the local Showdown server running:
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
# Reuse the generic benchmarking + saving helpers from the main trainer.
from train_rl import win_rate, WinRateCallback, SaveCallback, mask_fn

BATTLE_FORMAT = "gen9randombattle"

# The one curriculum knob. Phase A: 100. Phase B: raise it (e.g. 200) and re-run.
VICTORY_VALUE = 100.0

TRAIN_STEPS = 100_000
REFRESH_FREQ = 50_000     # how often to copy the learner's weights into the opponent
EVAL_FREQ = 10_000        # how often to benchmark vs the heuristic (live graph)
EVAL_BATTLES = 200         # battles for the before/after benchmark
LIVE_EVAL_BATTLES = 50    # battles per live benchmark point
SAVE_FREQ = 100_000
TB_DIR = "tb_logs"

# Files live in the project root (paths relative to where you run from).
MODEL_PATH = f"ppo_selfplay_obs{N_FEATURES}.zip"
WARM_START_PATH = f"ppo_vs_heuristic_obs{N_FEATURES}.zip"  # the heuristic-trained agent
SNAPSHOT_PATH = "ppo_selfplay_snapshot.zip"               # scratch file to seed the opponent


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
    """A separate env that plays the heuristic, for the honest benchmark."""
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
    """Every refresh_freq steps, copy the learner's weights into the self-play opponent."""

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

    # Starting weights: a saved self-play agent (continue), else the heuristic-trained agent
    # (warm start), else fresh.
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

    # Seed the opponent with a copy of the learner so it starts as an equal mirror.
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
