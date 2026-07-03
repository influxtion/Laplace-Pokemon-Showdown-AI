r"""Self-play on top of the reward-annealed agent -- "Influxobot", the final ladder bot.

The one pre-ladder experiment we hadn't run. The shipped ~57% search agent sits on a policy
(ppo_v3_anneal_best) that only ever trained against SimpleHeuristicsPlayer. Self-play replaces
that single opponent with a refreshed snapshot of the agent itself, training it against a
moving, varied target. Self-play didn't move the heuristic benchmark before, but that's the
wrong metric here: the point is opponent diversity, which may generalize to the varied humans
on the ladder better than a policy that only saw one bot. The real signal is ladder GXE, not
the heuristic number (which may stay flat).

Warm-starts from ppo_v3_anneal_best and continues with the anneal's END reward
(train_v3_anneal.ANNEAL_END), not train_rl's larger shaping. The anneal finished at that
small-shaping/win-dominant reward, so reusing it keeps the value function calibrated; jumping
back to 10x-larger shaping would destabilize it.

Saved as ppo_v3_anneal_selfplay_obs215.zip (+ _best for the eval peak); eval_search.py /
play.py auto-pick it once it exists. The ladder display name "Influxobot" is set in ladder.py,
unrelated to the file name.

Reuses the self-play machinery (opponent, eval env, snapshot refresh, LR schedule) from
train_v3_selfplay; only the warm-start source, reward, and saved names differ.

Run from the project root, with the local Showdown server running:
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
# Reward-independent self-play helpers shared with the plain v3 self-play run.
from train_v3_selfplay import make_self_opponent, build_eval_env, linear_schedule
# The anneal's END reward, so the warm-start from ppo_v3_anneal_best is seamless.
from train_v3_anneal import ANNEAL_END

BATTLE_FORMAT = "gen9randombattle"

# --- hygiene (same as the other v3 finetunes) ---------------------------------
TARGET_KL = 0.03
LR_START = 3e-4            # decays linearly to 0 across the run

TRAIN_STEPS = 500_000
REFRESH_FREQ = 50_000      # how often to copy the learner's weights into the opponent
EVAL_FREQ = 15_000
EVAL_BATTLES = 100
LIVE_EVAL_BATTLES = 100
SAVE_FREQ = 100_000

# Warm-start from the current best (annealed) agent; fall back to the anneal latest.
WARM_START_PATHS = [
    f"ppo_v3_anneal_best_obs{N_FEATURES}.zip",
    f"ppo_v3_anneal_obs{N_FEATURES}.zip",
]
MODEL_PATH = f"ppo_v3_anneal_selfplay_obs{N_FEATURES}.zip"          # latest, for resuming
BEST_PATH = f"ppo_v3_anneal_selfplay_best_obs{N_FEATURES}.zip"      # highest win rate seen
SNAPSHOT_PATH = "ppo_v3_anneal_selfplay_snapshot.zip"              # scratch file to seed opponent
TB_LOG_NAME = f"ppo_v3_anneal_selfplay_obs{N_FEATURES}"


def build_train_env(opponent):
    """Agent vs a snapshot of itself, on the anneal's END reward so the value function stays
    on the scale it finished the anneal on."""
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

    # Continue a saved anneal-self-play agent, else warm-start from the best annealed model.
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

    # Loading preserves net_arch/gamma/reward scale; only override the optimizer.
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
    print(f"\n=== Influxobot anneal self-play: {TRAIN_STEPS:,} steps "
          f"(reward=anneal END hp{ANNEAL_END['hp_value']}/victory{ANNEAL_END['victory_value']:g}; "
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
    print(f"\nvs heuristic: {before:.0%} -> {after:.0%}  (best -> {BEST_PATH}). "
          f"The real test is ladder GXE; layer search via eval_search.py / play.py.", flush=True)
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
