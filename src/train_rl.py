"""Train an RL agent to battle, and measure whether it improved.

Prerequisite: the local Showdown server must be running (see README):
    cd server && node pokemon-showdown start --no-security

Then:
    python -u train_rl.py

Steps: build the env with an opponent, load the saved model if there is one (else start
fresh), measure the win rate, train TRAIN_STEPS more, measure again, and save.

Progress is logged to ./tb_logs. In another terminal run `tensorboard --logdir tb_logs`
and open http://localhost:6006 to watch the reward climb.

We use MaskablePPO rather than plain PPO because only some of the 26 actions are legal each
turn. poke-env gives us a mask of the legal ones, and MaskablePPO uses it so the agent only
picks legal actions, which learns much faster than flailing at illegal moves.
"""

import logging
import os

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.environment.singles_env import SinglesEnv
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from rl_env import ShowdownSinglesEnv, N_FEATURES

BATTLE_FORMAT = "gen9randombattle"

# Who the agent trains against and is benchmarked against this run:
#   "random"    -> RandomPlayer (random legal moves; the easy floor)
#   "heuristic" -> SimpleHeuristicsPlayer (type matchups, speed, switching; a real test)
OPPONENT = "heuristic"

# Parallel training envs, each in its own process with its own server connection, for a
# roughly N_ENVS-fold speedup. Set near your CPU's physical core count; 4 is a safe laptop
# default.
N_ENVS = 4

TRAIN_STEPS = 100_000     # turns of experience to add this run (bump up for a stronger agent)
TB_DIR = "tb_logs"        # TensorBoard logs
# Evals are heavy so the win rate is trustworthy: a 30-battle eval has ~+-9% noise, 100-200
# brings it to ~+-3-5%. Eval pauses training, so with parallel training we eval less often.
EVAL_FREQ = 25_000        # measure win rate every N timesteps (live graph)
LIVE_EVAL_BATTLES = 200   # battles per live measurement
EVAL_BATTLES = 200        # battles for the before/after measurement
SAVE_FREQ = 100_000       # auto-save every N timesteps

# Win-focused reward. The old default (hp 0.5 / victory 100) made the agent trade evenly but
# not close games: the dense per-turn shaping out-competed the rare victory bonus. Now that
# it's competent, halve the HP shaping and raise the victory bonus so winning is the objective.
# (Unset weights fall back to rl_env.DEFAULT_REWARD: fainted 1.0, status 0.1.)
REWARD_WEIGHTS = {"hp_value": 0.25, "victory_value": 150.0}

# Each opponent gets its own saved agent, and the filename carries the observation size: if
# you change what the network sees, old weights no longer load, so a different name keeps them
# separate instead of clobbering them.
MODEL_PATH = f"ppo_vs_{OPPONENT}_obs{N_FEATURES}.zip"
# No agent for this opponent yet? Warm-start from the random-trained one with the same
# observation, if present (transfer learning).
WARM_START_PATH = f"ppo_vs_random_obs{N_FEATURES}.zip"


def mask_fn(env):
    """Which of the 26 actions are legal in the current state.

    `env` is the SingleAgentWrapper; `env.env` is our ShowdownSinglesEnv, whose `battle1`
    is the agent's current battle.
    """
    return np.array(SinglesEnv.get_action_mask(env.env.battle1), dtype=bool)


OPPONENT_CLASSES = {"random": RandomPlayer, "heuristic": SimpleHeuristicsPlayer}
OPPONENT_LABEL = {"random": "RandomPlayer", "heuristic": "SimpleHeuristicsPlayer"}[OPPONENT]


def build_masked_env(agent_tag, opp_tag, reward_weights=None, reward_schedule=None, switch_penalty=0.0):
    """Build one masked env and return it plus the inner env.

    agent_tag/opp_tag give the server accounts unique names so parallel training envs (and the
    eval env) don't collide on the same connection.

    reward_weights overrides the module default (train_v3 injects its rescaled reward);
    reward_schedule (train_v3_anneal) anneals the weights over training; switch_penalty
    (train_v3_anneal) subtracts reward for panic switching. All three are passed as parameters,
    not read from globals, so they survive the pickle into SubprocVecEnv workers -- those
    re-import this module fresh and would otherwise see the unmodified REWARD_WEIGHTS."""
    # rand=True appends a random suffix, so accounts are unique across envs and runs. Fixed
    # names collide with ghost connections from a crashed run, and the challenge never
    # completes ("Agent is not challenging"); poke-env's per-process counter collides across
    # subprocesses too.
    showdown = ShowdownSinglesEnv(
        reward_weights=reward_weights or REWARD_WEIGHTS,
        reward_schedule=reward_schedule,
        switch_penalty=switch_penalty,
        account_configuration1=AccountConfiguration.generate(f"{agent_tag}A", rand=True),
        account_configuration2=AccountConfiguration.generate(f"{agent_tag}B", rand=True),
        battle_format=BATTLE_FORMAT,
        strict=False,                 # illegal action -> random legal move
        log_level=logging.ERROR,      # quiet the per-turn warning spam
    )
    opponent = OPPONENT_CLASSES[OPPONENT](
        account_configuration=AccountConfiguration.generate(opp_tag, rand=True),
        battle_format=BATTLE_FORMAT,
        start_listening=False,        # used only as a brain; needs no connection
    )
    wrapped = SingleAgentWrapper(showdown, opponent)
    return ActionMasker(wrapped, mask_fn), showdown


def build_env(reward_weights=None, reward_schedule=None, switch_penalty=0.0):
    """The single masked env used for evaluation (its own dedicated accounts)."""
    return build_masked_env("ev", "evopp", reward_weights, reward_schedule, switch_penalty)


def make_env(rank, reward_weights=None, reward_schedule=None, switch_penalty=0.0):
    """Factory for the training env at index `rank` (used by SubprocVecEnv).

    Wrapped in Monitor so SB3 records per-episode reward/length (ep_rew_mean, ep_len_mean in
    TensorBoard). SB3 only auto-adds Monitor for a raw env; a VecEnv you build yourself needs
    it per sub-env.

    reward_weights/reward_schedule/switch_penalty are captured in the closure so they pickle
    into each worker."""
    def _init():
        masked, _ = build_masked_env(
            f"tr{rank}", f"tropp{rank}", reward_weights, reward_schedule, switch_penalty)
        return Monitor(masked)
    return _init


def win_rate(model, env, inner, n_battles):
    """Play n_battles with the model and return the fraction it won."""
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
    """Every `eval_freq` steps, play some battles and log the win rate to TensorBoard.

    Uses a separate eval env so measuring doesn't disturb the battles the agent is learning
    from. Shows up as `eval/win_rate`.
    """

    def __init__(self, eval_env, eval_inner, n_battles, eval_freq, best_path=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_inner = eval_inner
        self.n_battles = n_battles
        self.eval_freq = eval_freq
        self.best_path = best_path     # if set, save whenever win rate hits a new high
        self.best_wr = -1.0
        self._last_bucket = None       # trigger on timestep buckets, not call count

    def _on_step(self):
        # Keyed off num_timesteps, not n_calls, so the cadence is right regardless of env count
        # and a resumed run doesn't eval every step.
        bucket = self.num_timesteps // self.eval_freq
        if bucket != self._last_bucket:
            self._last_bucket = bucket
            wr = win_rate(self.model, self.eval_env, self.eval_inner, self.n_battles)
            self.logger.record("eval/win_rate", wr)
            if self.verbose:
                print(f"[eval] {self.num_timesteps:,} steps -> win rate {wr:.0%}", flush=True)
            # Keep the peak, not the latest: PPO can drift downhill late, so the final
            # weights are often worse than an earlier checkpoint (this bit us on v3).
            if self.best_path is not None and wr > self.best_wr:
                self.best_wr = wr
                self.model.save(self.best_path)
                if self.verbose:
                    print(f"[best] new best {wr:.0%} -> saved {self.best_path}", flush=True)
        return True


class SaveCallback(BaseCallback):
    """Save the model every `save_freq` steps so an interrupted run keeps its progress."""

    def __init__(self, path, save_freq, verbose=1):
        super().__init__(verbose)
        self.path = path
        self.save_freq = save_freq
        self._last_bucket = None

    def _on_step(self):
        if not self.save_freq:
            return True
        bucket = self.num_timesteps // self.save_freq
        if bucket != self._last_bucket:
            self._last_bucket = bucket
            self.model.save(self.path)
            if self.verbose:
                print(f"[checkpoint] saved {self.path} at {self.num_timesteps:,} steps", flush=True)
        return True


def main():
    # Training runs across N_ENVS processes; eval uses one separate env.
    train_env = (
        SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
        if N_ENVS > 1 else DummyVecEnv([make_env(0)])
    )
    eval_env, eval_inner = build_env()

    def fresh_model():
        # ent_coef is an exploration bonus: it rewards keeping some randomness in the policy
        # so the agent keeps trying new strategies instead of locking in early (what made the
        # old 12-feature agent plateau).
        return MaskablePPO(
            "MultiInputPolicy", train_env, verbose=1, tensorboard_log=TB_DIR, ent_coef=0.01
        )

    def safe_load(path):
        # Loading fails if the saved network's input size no longer matches the observation;
        # fall back to a fresh agent instead of crashing.
        try:
            return MaskablePPO.load(path, env=train_env, tensorboard_log=TB_DIR), True
        except (ValueError, RuntimeError, KeyError) as e:
            print(f"Could not load {path} ({type(e).__name__}); starting fresh.", flush=True)
            return fresh_model(), False

    # Where the starting weights come from:
    #   1. a saved agent for this opponent+observation -> continue it
    #   2. else the random-trained agent (same obs)     -> warm-start (transfer learning)
    #   3. else                                         -> a fresh agent
    resuming = os.path.exists(MODEL_PATH)
    if resuming:
        print(f"Found {MODEL_PATH} -> CONTINUING training vs {OPPONENT_LABEL}.", flush=True)
        model, resuming = safe_load(MODEL_PATH)
    elif os.path.exists(WARM_START_PATH):
        print(f"Warm-starting from {WARM_START_PATH} (the random-trained agent).", flush=True)
        model, _ = safe_load(WARM_START_PATH)
    else:
        print(f"No saved model for this setup -> starting a FRESH agent ({N_FEATURES} features).", flush=True)
        model = fresh_model()

    print(f"\n=== Win rate vs {OPPONENT_LABEL} BEFORE this run ({EVAL_BATTLES} battles) ===", flush=True)
    before = win_rate(model, eval_env, eval_inner, EVAL_BATTLES)
    print(f"Before: {before:.0%}", flush=True)

    print(f"\n=== Training {TRAIN_STEPS:,} steps across {N_ENVS} envs vs {OPPONENT_LABEL} ===", flush=True)
    # reset_num_timesteps=False only when continuing the same matchup, so the TensorBoard
    # x-axis continues; a warm-start or fresh run gets its own curve.
    callbacks = CallbackList([
        WinRateCallback(eval_env, eval_inner, LIVE_EVAL_BATTLES, EVAL_FREQ),
        SaveCallback(MODEL_PATH, SAVE_FREQ),
    ])
    model.learn(
        total_timesteps=TRAIN_STEPS,
        reset_num_timesteps=not resuming,
        tb_log_name=f"ppo_vs_{OPPONENT}_obs{N_FEATURES}",
        callback=callbacks,
    )
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}", flush=True)

    print(f"\n=== Win rate vs {OPPONENT_LABEL} AFTER this run ({EVAL_BATTLES} battles) ===", flush=True)
    after = win_rate(model, eval_env, eval_inner, EVAL_BATTLES)
    print(f"After: {after:.0%}", flush=True)

    print(f"\nResult vs {OPPONENT_LABEL}: {before:.0%} -> {after:.0%} after {TRAIN_STEPS:,} steps.", flush=True)
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
