r"""Reward-anneal finetune -- the "play to WIN, not to trade" push past the ~43% ceiling.

THE PROBLEM this targets: across every earlier run the agent's reward climbed while its win
rate stayed flat at ~40-43%. That is the signature of a MISALIGNED objective -- the dense
per-turn shaping (hp/faint) fires every single turn, so its gradient is consistent and the
agent optimizes it; the victory bonus is rare (we win <half our games) so its gradient is
sparse and noisy. The agent rationally learns to TRADE Pokemon evenly rather than to CLOSE
games. We confirmed the other levers are dead: more observation features (141->215, flat) and
self-play (2M steps, flat). Reward is the one untried lever that targets the actual ceiling.

THE FIX: anneal the reward. We can't just delete the shaping -- a pure win-only reward starves
a fresh agent of any signal (we saw it flat-line at ~-99 before). So instead we WARM-START from
the already-competent v3 agent and gradually shrink the shaping toward zero while keeping the
win bonus fixed. Early on the shaping still stabilizes learning; by the end the win bonus is the
dominant consistent signal, so the agent is pushed from "trade evenly" to "play to win." The
anneal is smooth (linear over ANNEAL_FRACTION of the run) so the warm-started value function is
never hit with a sudden reward redefinition.

At frac=0 the weights EXACTLY match what ppo_v3 was trained on (hp 0.25 / victory 150), so the
warm-start is continuous; by frac=1 the shaping is ~10x smaller and victory dominates. The run
then trains at the final reward for the remaining (1 - ANNEAL_FRACTION) to settle.

This run also stacks ANTI-PANIC-SWITCH shaping (SWITCH_PENALTY): a small reward penalty for
voluntarily switching a different Pokemon out two turns in a row, a PokeLLMon-flagged losing
pattern. It pushes the same "stop fleeing, commit and close the game" behaviour the anneal targets.

Adds the same hygiene as train_v3_selfplay (target_kl, LR decay to 0, best-model checkpoint) so
it can't overtrain itself downhill. Benchmarks vs SimpleHeuristicsPlayer so the number stays
comparable to every earlier run; layer search.py on the saved model (eval_search.py) afterward.

Run from the project root, with the local Showdown server running:
    python -u v1\v3\train_v3_anneal.py
"""

import os
import sys

# This file lives in v1/v3/ but reuses v1/ modules (train_rl, rl_env). Put the parent v1/
# directory on the import path so those bare imports resolve when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import train_rl as base
from rl_env import N_FEATURES

TRAIN_STEPS = 1_000_000

# --- the anneal -------------------------------------------------------------
# START matches ppo_v3's training reward EXACTLY (hp 0.25 / victory 150, fainted/status from
# rl_env.DEFAULT_REWARD) so warm-starting is seamless. END shrinks the dense shaping ~10x while
# holding the victory bonus, so by the end of the anneal WINNING is overwhelmingly the objective.
ANNEAL_START = {"hp_value": 0.25, "fainted_value": 1.0, "status_value": 0.1, "victory_value": 150.0}
ANNEAL_END   = {"hp_value": 0.025, "fainted_value": 0.1, "status_value": 0.0, "victory_value": 150.0}
ANNEAL_FRACTION = 0.6      # finish the shift at 60% of the run, then train at the END reward

# Anti-"panic switch" penalty (see rl_env.ShowdownSinglesEnv): subtract this much reward when
# the agent voluntarily switches a DIFFERENT Pokemon out two turns running -- a PokeLLMon-flagged
# losing pattern. Chosen to roughly match a fainted_value at the END reward scale (~0.1-0.2), so
# it's a clear signal in the win-focused phase yet negligible against the START shaping (hp 0.25),
# meaning it doesn't disturb the warm-started policy early. This is the main knob to tune. 0 = off.
SWITCH_PENALTY = 0.2

# --- hygiene (so it can't overtrain downhill the way plain v3 did) ----------
TARGET_KL = 0.03
LR_START = 3e-4            # decays LINEARLY to 0 across the run

EVAL_FREQ = 15_000
EVAL_BATTLES = 200
LIVE_EVAL_BATTLES = 200
SAVE_FREQ = 100_000

WARM_START_PATH = f"ppo_v3_obs{N_FEATURES}.zip"
MODEL_PATH = f"ppo_v3_anneal_obs{N_FEATURES}.zip"          # latest (for resuming)
BEST_PATH = f"ppo_v3_anneal_best_obs{N_FEATURES}.zip"      # highest win rate seen (for eval)
TB_LOG_NAME = f"ppo_v3_anneal_obs{N_FEATURES}"


def linear_schedule(start):
    """progress_remaining goes 1 -> 0 over the run, so this decays the LR from `start` to 0."""
    return lambda progress_remaining: progress_remaining * start


def make_schedule():
    """The per-env reward schedule. horizon is PER-ENV (each of N_ENVS envs sees ~1/N of the
    timesteps), so divide the global anneal budget by N_ENVS to land frac=1 at ANNEAL_FRACTION
    of total training."""
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
    # Eval is win-rate only (reward is ignored there), so the plain heuristic env is fine.
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
            f"Train the v3 agent first: python -u v1\\v3\\train_v3.py")
    model = MaskablePPO.load(start_path, env=train_env, tensorboard_log=base.TB_DIR)

    # Loading preserves net_arch [256,256] and gamma 0.997; we only add the optimizer hygiene.
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
