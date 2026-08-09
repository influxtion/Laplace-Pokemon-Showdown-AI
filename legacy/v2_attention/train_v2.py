"""Training for the attention experiment.

Bolts the three pieces together: the structured observation, the attention network that
reads it, and the usual learning algorithm.

Measured against the same fixed opponent as everything else, so it can be compared directly
with the plain network that reached about 41%.

Needs the local server running. Then:
    python -u train_v2.py
"""

import logging
import os

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from poke_env import AccountConfiguration
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.environment.singles_env import SinglesEnv
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from rl_env_v2 import ShowdownTeamEnv, N_FEATURES
from team_net import TeamAttentionExtractor

# A switch for working out what went wrong. This experiment changed two things at once: the
# observation and the network. Turning this off feeds the new observation through the old
# plain network, which separates them. If it improves, the attention network is the problem.
# If it still stalls, the observation is. Saves under its own name so the two don't collide.
USE_ATTENTION = False

BATTLE_FORMAT = "gen9randombattle"
TRAIN_STEPS = 100_000
EVAL_BATTLES = 50
LIVE_EVAL_BATTLES = 30
EVAL_FREQ = 10_000
SAVE_FREQ = 100_000
TB_DIR = "tb_logs"
TAG = "attn" if USE_ATTENTION else "mlp"
MODEL_PATH = f"ppo_v2_{TAG}_obs{N_FEATURES}.zip"
TB_LOG_NAME = f"ppo_v2_{TAG}"


def mask_fn(env):
    return np.array(SinglesEnv.get_action_mask(env.env.battle1), dtype=bool)


def make_opponent():
    return SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration("v2-heur-opp", None),
        battle_format=BATTLE_FORMAT,
        start_listening=False,
    )


def build_env():
    showdown = ShowdownTeamEnv(
        battle_format=BATTLE_FORMAT, strict=False, log_level=logging.ERROR,
    )
    wrapped = SingleAgentWrapper(showdown, make_opponent())
    return ActionMasker(wrapped, mask_fn), showdown


def win_rate(model, env, inner, n_battles):
    """Play a batch of games and report how many the agent won."""
    wins = 0
    for _ in range(n_battles):
        obs, _ = env.reset()
        done = False
        while not done:
            masks = np.asarray(obs["action_mask"], dtype=bool)
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(np.int64(action))
            done = terminated or truncated
        if inner.battle1 is not None and inner.battle1.won:
            wins += 1
    return wins / n_battles


class WinRateCallback(BaseCallback):
    """Every so often, stop and play some games to see how the agent is doing."""

    def __init__(self, eval_env, eval_inner, n_battles, eval_freq, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_inner = eval_inner
        self.n_battles = n_battles
        self.eval_freq = eval_freq

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            wr = win_rate(self.model, self.eval_env, self.eval_inner, self.n_battles)
            self.logger.record("eval/win_rate", wr)
            if self.verbose:
                print(f"[eval] {self.num_timesteps:,} steps -> win rate {wr:.0%}", flush=True)
        return True


class SaveCallback(BaseCallback):
    """Save periodically, so an interrupted run doesn't lose everything."""

    def __init__(self, path, save_freq, verbose=1):
        super().__init__(verbose)
        self.path = path
        self.save_freq = save_freq

    def _on_step(self):
        if self.save_freq and self.n_calls % self.save_freq == 0:
            self.model.save(self.path)
            if self.verbose:
                print(f"[checkpoint] saved {self.path} at {self.num_timesteps:,} steps", flush=True)
        return True


def main():
    env, inner = build_env()
    eval_env, eval_inner = build_env()

    # Use the attention network, unless we're running the comparison, in which case fall
    # back to the plain one.
    if USE_ATTENTION:
        policy_kwargs = dict(
            features_extractor_class=TeamAttentionExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=[128, 128],
        )
    else:
        policy_kwargs = dict(net_arch=[128, 128])

    resuming = os.path.exists(MODEL_PATH)
    if resuming:
        print(f"Found {MODEL_PATH} -> CONTINUING.", flush=True)
        model = MaskablePPO.load(MODEL_PATH, env=env, tensorboard_log=TB_DIR)
    else:
        kind = "attention net" if USE_ATTENTION else "plain network (for comparison)"
        print(f"Starting a FRESH v2 agent ({kind}, {N_FEATURES} features).", flush=True)
        model = MaskablePPO(
            "MultiInputPolicy", env, verbose=1, ent_coef=0.01,
            tensorboard_log=TB_DIR, policy_kwargs=policy_kwargs,
        )

    print("\n=== Win rate vs SimpleHeuristicsPlayer BEFORE this run ===", flush=True)
    before = win_rate(model, env, inner, EVAL_BATTLES)
    print(f"Before: {before:.0%}", flush=True)

    callbacks = CallbackList([
        WinRateCallback(eval_env, eval_inner, LIVE_EVAL_BATTLES, EVAL_FREQ),
        SaveCallback(MODEL_PATH, SAVE_FREQ),
    ])
    print(f"\n=== Training {TRAIN_STEPS:,} steps ===", flush=True)
    model.learn(
        total_timesteps=TRAIN_STEPS,
        reset_num_timesteps=not resuming,
        tb_log_name=TB_LOG_NAME,
        callback=callbacks,
    )
    model.save(MODEL_PATH)
    print(f"Saved {MODEL_PATH}", flush=True)

    print("\n=== Win rate vs SimpleHeuristicsPlayer AFTER this run ===", flush=True)
    after = win_rate(model, env, inner, EVAL_BATTLES)
    print(f"After: {after:.0%}", flush=True)
    print(f"\nResult vs heuristic: {before:.0%} -> {after:.0%}  (the plain network got ~41%)", flush=True)
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
