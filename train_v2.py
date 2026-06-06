"""Phase 1 (v2) training: structured observation + team-attention network, vs the heuristic.

Ties the v2 pieces together:
  - rl_env_v2.ShowdownTeamEnv     : the 854-feature structured observation + win-weighted reward
  - team_net.TeamAttentionExtractor: per-Pokemon encoder + self-attention (with padding mask)
  - MaskablePPO                    : only ever picks legal actions

Benchmarked vs SimpleHeuristicsPlayer so we compare directly against the flat-MLP
baseline (~41%). Reuses the generic win_rate / callbacks from train_rl.

Prereq: local Showdown server running (see README). Then:
    python -u train_v2.py
"""

import logging
import os

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CallbackList

from poke_env import AccountConfiguration
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.environment.singles_env import SinglesEnv
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from rl_env_v2 import ShowdownTeamEnv, N_FEATURES
from team_net import TeamAttentionExtractor
from train_rl import win_rate, WinRateCallback, SaveCallback  # generic, env-agnostic

BATTLE_FORMAT = "gen9randombattle"
TRAIN_STEPS = 300_000
EVAL_BATTLES = 50
LIVE_EVAL_BATTLES = 30
EVAL_FREQ = 10_000
SAVE_FREQ = 100_000
TB_DIR = "tb_logs"
MODEL_PATH = f"ppo_v2_attn_obs{N_FEATURES}.zip"


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


def main():
    env, inner = build_env()
    eval_env, eval_inner = build_env()

    # Plug the team-attention network in as the feature extractor.
    policy_kwargs = dict(
        features_extractor_class=TeamAttentionExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[128, 128],
    )

    resuming = os.path.exists(MODEL_PATH)
    if resuming:
        print(f"Found {MODEL_PATH} -> CONTINUING.", flush=True)
        model = MaskablePPO.load(MODEL_PATH, env=env, tensorboard_log=TB_DIR)
    else:
        print(f"Starting a FRESH v2 agent (attention net, {N_FEATURES} features).", flush=True)
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
        tb_log_name="ppo_v2_attn",
        callback=callbacks,
    )
    model.save(MODEL_PATH)
    print(f"Saved {MODEL_PATH}", flush=True)

    print("\n=== Win rate vs SimpleHeuristicsPlayer AFTER this run ===", flush=True)
    after = win_rate(model, env, inner, EVAL_BATTLES)
    print(f"After: {after:.0%}", flush=True)
    print(f"\nResult vs heuristic: {before:.0%} -> {after:.0%}  (flat-MLP baseline was ~41%)", flush=True)
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
