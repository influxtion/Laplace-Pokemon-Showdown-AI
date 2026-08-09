r"""Teaching the agent to play to win rather than to trade.

The symptom, in every run so far: the reward kept climbing while the win rate sat flat
around 40%. That's the classic sign of rewarding the wrong thing. The small per-turn
rewards fire every single turn, so the agent gets a clear, consistent signal to chase them.
The win bonus arrives once a game, less than half the time, so its signal is sparse and
noisy. Trading evenly is, from the agent's point of view, entirely rational.

The other explanations were already ruled out. More features didn't help. Two million steps
of self-play didn't help. The reward was the one thing left untried.

The fix is to fade the small rewards out gradually. You can't simply delete them, because a
win-only reward gives a fresh agent almost nothing to learn from; that was tried and it flat
out failed. So instead we start from the already-competent agent and shrink the small
rewards towards nothing while keeping the win bonus fixed. Early on the small rewards keep
things stable; by the end, winning is the only thing that meaningfully pays. The fade is
gradual so the agent never wakes up to a completely different reward overnight.

It also adds a small penalty for switching two turns in a row, a losing pattern the
PokeLLMon paper identified. Same goal as the fade: stop running, commit, close the game out.

Plus the usual safeguards so it can't quietly train itself downhill: a cap on how far the
policy can move at once, a learning rate that decays to zero, and a saved copy of the best
version rather than just the last one.

Run from the project root, with the local server going:
    python -u src\train_v3_anneal.py
"""

import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import train_rl as base
from rl_env import N_FEATURES

TRAIN_STEPS = 1_000_000

# --- the fade ----------------------------------------------------------------
# The starting point matches exactly what the previous agent trained on, so picking it up is
# seamless. The end point shrinks the small rewards roughly tenfold while leaving the win
# bonus untouched, so by then winning is overwhelmingly the point.
ANNEAL_START = {"hp_value": 0.25, "fainted_value": 1.0, "status_value": 0.1, "victory_value": 150.0}
ANNEAL_END   = {"hp_value": 0.025, "fainted_value": 0.1, "status_value": 0.0, "victory_value": 150.0}
ANNEAL_FRACTION = 0.6      # finish the fade 60% in, then let it settle

# The cost of switching twice in a row. Sized to be meaningful once the small rewards have
# faded, but negligible against them at the start, so it doesn't disturb the handover. This
# is the main thing to tune. Zero turns it off.
SWITCH_PENALTY = 0.2

# --- safeguards, so it can't train itself downhill the way the last one did ---
TARGET_KL = 0.03
LR_START = 3e-4            # decays to zero over the run

EVAL_FREQ = 15_000
EVAL_BATTLES = 200
LIVE_EVAL_BATTLES = 200
SAVE_FREQ = 100_000

WARM_START_PATH = f"ppo_v3_obs{N_FEATURES}.zip"
MODEL_PATH = f"ppo_v3_anneal_obs{N_FEATURES}.zip"          # the latest, to resume from
BEST_PATH = f"ppo_v3_anneal_best_obs{N_FEATURES}.zip"      # the best, to actually use
TB_LOG_NAME = f"ppo_v3_anneal_obs{N_FEATURES}"


def linear_schedule(start):
    """Wind the learning rate down to zero as the run progresses."""
    return lambda progress_remaining: progress_remaining * start


def make_schedule():
    """Work out the fade schedule. Each parallel game only sees its own share of the total
    training, so the budget has to be divided up for the fade to finish at the right time."""
    horizon = int(ANNEAL_FRACTION * TRAIN_STEPS / base.N_ENVS)
    return {"start": ANNEAL_START, "end": ANNEAL_END, "horizon": horizon}


def main():
    schedule = make_schedule()
    train_env = (
        SubprocVecEnv([base.make_env(i, reward_schedule=schedule, switch_penalty=SWITCH_PENALTY)
                       for i in range(base.N_ENVS)])
        if base.N_ENVS > 1 else
        DummyVecEnv([base.make_env(0, reward_schedule=schedule, switch_penalty=SWITCH_PENALTY)])
    )
    # Measuring only counts wins, so the rewards don't matter here.
    eval_env, eval_inner = base.build_env()

    resuming = os.path.exists(MODEL_PATH)
    if resuming:
        print(f"CONTINUING the anneal finetune from {MODEL_PATH}.", flush=True)
        start_path = MODEL_PATH
    elif os.path.exists(WARM_START_PATH):
        print(f"WARM-STARTING the anneal finetune from {WARM_START_PATH}.", flush=True)
        start_path = WARM_START_PATH
    else:
        raise FileNotFoundError(
            f"Need {MODEL_PATH} (to resume) or {WARM_START_PATH} (to warm-start). "
            f"Train the v3 agent first: python -u src\\train_v3.py")
    model = MaskablePPO.load(start_path, env=train_env, tensorboard_log=base.TB_DIR)

    # Loading keeps the network shape and discount from before; we only add the safeguards.
    model.target_kl = TARGET_KL
    model.lr_schedule = linear_schedule(LR_START)

    print(f"\n=== Win rate vs {base.OPPONENT_LABEL} BEFORE ({EVAL_BATTLES} battles) ===", flush=True)
    before = base.win_rate(model, eval_env, eval_inner, EVAL_BATTLES)
    print(f"Before: {before:.0%}", flush=True)

    callbacks = CallbackList([
        base.WinRateCallback(eval_env, eval_inner, LIVE_EVAL_BATTLES, EVAL_FREQ, best_path=BEST_PATH),
        base.SaveCallback(MODEL_PATH, SAVE_FREQ),
    ])
    print(f"\n=== Reward-anneal finetune: {TRAIN_STEPS:,} steps "
          f"(shaping {ANNEAL_START['hp_value']}->{ANNEAL_END['hp_value']} hp over "
          f"{ANNEAL_FRACTION:.0%} of the run, victory {ANNEAL_END['victory_value']:g} fixed; "
          f"panic-switch penalty {SWITCH_PENALTY:g}; "
          f"target_kl={TARGET_KL}, lr {LR_START:g}->0) ===", flush=True)
    model.learn(
        total_timesteps=TRAIN_STEPS,
        reset_num_timesteps=not resuming,
        tb_log_name=TB_LOG_NAME,
        callback=callbacks,
    )
    model.save(MODEL_PATH)
    print(f"Saved {MODEL_PATH} (latest) and {BEST_PATH} (peak win rate).", flush=True)

    print(f"\n=== Win rate vs {base.OPPONENT_LABEL} AFTER ({EVAL_BATTLES} battles) ===", flush=True)
    after = base.win_rate(model, eval_env, eval_inner, EVAL_BATTLES)
    print(f"After: {after:.0%}", flush=True)
    print(f"\nvs heuristic: {before:.0%} -> {after:.0%}  (best checkpoint saved to {BEST_PATH})", flush=True)
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
