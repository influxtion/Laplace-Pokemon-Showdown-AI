"""Train a reinforcement-learning agent to battle, and measure if it improved.

Prerequisite: the local Showdown server must be running (see README):
    cd server && node pokemon-showdown start --no-security

Then:
    python -u train_rl.py

What this does:
  1. Builds the RL environment with an opponent to learn against.
  2. Loads the saved model if one exists (to CONTINUE training), else starts fresh.
  3. Measures the current win rate (a baseline for this session).
  4. Trains with MaskablePPO for TRAIN_STEPS more turns of experience.
  5. Measures the win rate again, and saves the model back to disk.

Progress is logged to ./tb_logs for live graphs. In another terminal run:
    tensorboard --logdir tb_logs
and open http://localhost:6006 to watch the reward climb in real time.

Why MaskablePPO (and not plain PPO)? At every turn only some of the 26 possible
actions are legal. poke-env gives us an "action mask" of the legal choices, and
MaskablePPO uses it so the agent ONLY ever picks legal actions -- which makes
learning much faster and avoids it flailing at illegal moves.
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

# Who the agent trains against AND is benchmarked against this run:
#   "random"    -> RandomPlayer (picks random legal moves; the easy floor)
#   "heuristic" -> SimpleHeuristicsPlayer (type matchups, speed, switching; a real test)
OPPONENT = "heuristic"

# Parallel training environments. Each runs in its OWN process with its own connection to
# the Showdown server, for a roughly N_ENVS-fold speedup. Set this near your CPU's physical
# core count; 4 is a safe laptop default. The local server handles the extra connections.
N_ENVS = 4

TRAIN_STEPS = 100_000     # turns of experience to add THIS run (bump up for a stronger agent)
TB_DIR = "tb_logs"        # TensorBoard logs go here
# Measurement is deliberately heavy so the win rate is trustworthy: a 30-battle eval has
# ~+-9% noise, 100-200 battles brings it to ~+-3-5%. Eval runs serially (it pauses
# training), so with parallel training we eval less often than before.
EVAL_FREQ = 25_000        # measure win rate every N timesteps during training (live graph)
LIVE_EVAL_BATTLES = 200   # battles per live measurement
EVAL_BATTLES = 200        # battles for the before/after headline measurement
SAVE_FREQ = 100_000       # auto-save every N timesteps (an interrupt loses at most this many)

# Win-focused reward. The old default (hp 0.5 / victory 100) made the agent trade evenly
# but not close games: the dense per-turn shaping out-competed the rare victory bonus in the
# gradient. Now that the agent is already competent, halve the HP shaping and raise the
# victory bonus so WINNING is clearly the objective. (The env fills any weight left unset
# here from rl_env.DEFAULT_REWARD, so fainted stays 1.0 and status 0.1.)
REWARD_WEIGHTS = {"hp_value": 0.25, "victory_value": 150.0}

# Each opponent gets its own saved agent. The filename also includes the observation
# size (N_FEATURES): if you change what the network "sees", the old saved weights are no
# longer compatible, so a different filename keeps them separate instead of clobbering them.
MODEL_PATH = f"ppo_vs_{OPPONENT}_obs{N_FEATURES}.zip"
# If there's no agent for THIS opponent+observation yet, warm-start from the random-trained
# one with the SAME observation, if it exists (transfer learning).
WARM_START_PATH = f"ppo_vs_random_obs{N_FEATURES}.zip"


def mask_fn(env):
    """Tell MaskablePPO which of the 26 actions are legal in the current state.

    `env` is the SingleAgentWrapper; `env.env` is our ShowdownSinglesEnv, whose
    `battle1` is the agent's current battle.
    """
    return np.array(SinglesEnv.get_action_mask(env.env.battle1), dtype=bool)


OPPONENT_CLASSES = {"random": RandomPlayer, "heuristic": SimpleHeuristicsPlayer}
OPPONENT_LABEL = {"random": "RandomPlayer", "heuristic": "SimpleHeuristicsPlayer"}[OPPONENT]


def build_masked_env(agent_tag, opp_tag, reward_weights=None, reward_schedule=None, switch_penalty=0.0):
    """Build one masked env and return it plus the inner env.

    agent_tag/opp_tag give the server accounts UNIQUE names so that parallel training envs
    (and the separate eval env) never collide on the same connection.

    reward_weights overrides the module default (train_v3 uses this to inject its rescaled
    reward). reward_schedule (train_v3_anneal) instead anneals the weights over training.
    switch_penalty (train_v3_anneal) subtracts reward for panic switching. All three are
    threaded through as parameters rather than read from the global so they survive the pickle
    into SubprocVecEnv workers, which re-import this module fresh and would otherwise see the
    unmodified REWARD_WEIGHTS."""
    # rand=True appends a random 5-char suffix to each username, so accounts are unique
    # across parallel envs AND across runs. (Fixed names collide with ghost connections
    # left by a previous crashed run -> the challenge never completes, "Agent is not
    # challenging". poke-env's per-process counter would also collide across subprocesses.)
    showdown = ShowdownSinglesEnv(
        reward_weights=reward_weights or REWARD_WEIGHTS,
        reward_schedule=reward_schedule,
        switch_penalty=switch_penalty,
        account_configuration1=AccountConfiguration.generate(f"{agent_tag}A", rand=True),
        account_configuration2=AccountConfiguration.generate(f"{agent_tag}B", rand=True),
        battle_format=BATTLE_FORMAT,
        strict=False,                 # safety net: illegal action -> random legal move
        log_level=logging.ERROR,      # quiet the per-turn warning spam
    )
    opponent = OPPONENT_CLASSES[OPPONENT](
        account_configuration=AccountConfiguration.generate(opp_tag, rand=True),
        battle_format=BATTLE_FORMAT,
        start_listening=False,        # used only as a "brain"; needs no own connection
    )
    wrapped = SingleAgentWrapper(showdown, opponent)
    return ActionMasker(wrapped, mask_fn), showdown


def build_env(reward_weights=None, reward_schedule=None, switch_penalty=0.0):
    """The single masked env used for evaluation (its own dedicated accounts)."""
    return build_masked_env("ev", "evopp", reward_weights, reward_schedule, switch_penalty)


def make_env(rank, reward_weights=None, reward_schedule=None, switch_penalty=0.0):
    """Factory for the training env at index `rank` (used by SubprocVecEnv).

    Wrapped in Monitor so SB3 records per-episode reward/length -> rollout/ep_rew_mean and
    ep_len_mean show up in TensorBoard. (SB3 only auto-adds Monitor when you pass a raw env;
    a VecEnv you build yourself needs it added per sub-env.)

    reward_weights/reward_schedule/switch_penalty are captured in the closure so they pickle
    into each SubprocVecEnv worker."""
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

    Uses a SEPARATE eval environment so measuring doesn't disturb the battles the
    agent is currently learning from. Shows up in TensorBoard as `eval/win_rate`.
    """

    def __init__(self, eval_env, eval_inner, n_battles, eval_freq, best_path=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_inner = eval_inner
        self.n_battles = n_battles
        self.eval_freq = eval_freq
        self.best_path = best_path     # if set, save the model whenever win rate hits a new high
        self.best_wr = -1.0
        self._last_bucket = None       # trigger on TIMESTEP buckets, not call count

    def _on_step(self):
        # Keyed off num_timesteps (not n_calls) so the cadence is right no matter how many
        # parallel envs there are, and so a resumed run doesn't eval every single step.
        bucket = self.num_timesteps // self.eval_freq
        if bucket != self._last_bucket:
            self._last_bucket = bucket
            wr = win_rate(self.model, self.eval_env, self.eval_inner, self.n_battles)
            self.logger.record("eval/win_rate", wr)
            if self.verbose:
                print(f"[eval] {self.num_timesteps:,} steps -> win rate {wr:.0%}", flush=True)
            # Keep the PEAK model, not the latest: PPO can drift downhill late in a run, so the
            # final weights are often worse than an earlier checkpoint (this bit us on v3).
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
    # Training runs across N_ENVS parallel processes; evaluation uses one separate env.
    train_env = (
        SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
        if N_ENVS > 1 else DummyVecEnv([make_env(0)])
    )
    eval_env, eval_inner = build_env()

    def fresh_model():
        # ent_coef adds an "exploration bonus": it rewards keeping some randomness in
        # the policy, so the agent keeps trying new strategies instead of locking into
        # one too early (which is what made the old 12-feature agent plateau).
        return MaskablePPO(
            "MultiInputPolicy", train_env, verbose=1, tensorboard_log=TB_DIR, ent_coef=0.01
        )

    def safe_load(path):
        # Loading fails if the saved network's input size no longer matches the current
        # observation. If that happens, fall back to a fresh agent instead of crashing.
        try:
            return MaskablePPO.load(path, env=train_env, tensorboard_log=TB_DIR), True
        except (ValueError, RuntimeError, KeyError) as e:
            print(f"Could not load {path} ({type(e).__name__}); starting fresh.", flush=True)
            return fresh_model(), False

    # Decide where the agent's starting weights come from:
    #   1. a saved agent for THIS opponent+observation -> continue improving it
    #   2. else the random-trained agent (same obs)     -> warm-start (transfer learning)
    #   3. else                                         -> a fresh random-weights agent
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
    # reset_num_timesteps=False only when continuing the same matchup, so the
    # TensorBoard x-axis continues; a warm-start/fresh run gets its own curve.
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
