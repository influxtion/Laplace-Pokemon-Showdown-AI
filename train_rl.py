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
from stable_baselines3.common.callbacks import BaseCallback

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

TRAIN_STEPS = 500_000     # turns of experience to add THIS run (bump up for a stronger agent)
EVAL_BATTLES = 100        # battles used to estimate win rate (start/end of run)
TB_DIR = "tb_logs"       # TensorBoard logs go here
EVAL_FREQ = 10_000       # measure win rate every N steps DURING training (for the live graph)
LIVE_EVAL_BATTLES = 30   # battles per live measurement (fewer = faster, noisier)

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


def make_opponent():
    """Build the opponent the agent trains/benchmarks against (set by OPPONENT)."""
    return OPPONENT_CLASSES[OPPONENT](
        account_configuration=AccountConfiguration(f"{OPPONENT}-opp", None),
        battle_format=BATTLE_FORMAT,
        start_listening=False,        # used only as a "brain"; needs no own connection
    )


def build_env():
    """Create the masked learning environment and return it plus the inner env."""
    showdown = ShowdownSinglesEnv(
        battle_format=BATTLE_FORMAT,
        strict=False,                 # safety net: illegal action -> random legal move
        log_level=logging.ERROR,      # quiet the per-turn warning spam
    )
    wrapped = SingleAgentWrapper(showdown, make_opponent())
    masked = ActionMasker(wrapped, mask_fn)
    return masked, showdown


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


def main():
    env, inner = build_env()
    eval_env, eval_inner = build_env()  # separate env just for measuring win rate

    def fresh_model():
        # ent_coef adds an "exploration bonus": it rewards keeping some randomness in
        # the policy, so the agent keeps trying new strategies instead of locking into
        # one too early (which is what made the old 12-feature agent plateau).
        return MaskablePPO(
            "MultiInputPolicy", env, verbose=1, tensorboard_log=TB_DIR, ent_coef=0.01
        )

    def safe_load(path):
        # Loading fails if the saved network's input size no longer matches the current
        # observation. If that happens, fall back to a fresh agent instead of crashing.
        try:
            return MaskablePPO.load(path, env=env, tensorboard_log=TB_DIR), True
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

    print(f"\n=== Win rate vs {OPPONENT_LABEL} BEFORE this run ===", flush=True)
    before = win_rate(model, env, inner, EVAL_BATTLES)
    print(f"Before: {before:.0%}", flush=True)

    print(f"\n=== Training for {TRAIN_STEPS:,} more steps vs {OPPONENT_LABEL} ===", flush=True)
    # reset_num_timesteps=False only when continuing the same matchup, so the
    # TensorBoard x-axis continues; a warm-start/fresh run gets its own curve.
    win_rate_cb = WinRateCallback(eval_env, eval_inner, LIVE_EVAL_BATTLES, EVAL_FREQ)
    model.learn(
        total_timesteps=TRAIN_STEPS,
        reset_num_timesteps=not resuming,
        tb_log_name=f"ppo_vs_{OPPONENT}_obs{N_FEATURES}",
        callback=win_rate_cb,
    )
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}", flush=True)

    print(f"\n=== Win rate vs {OPPONENT_LABEL} AFTER this run ===", flush=True)
    after = win_rate(model, env, inner, EVAL_BATTLES)
    print(f"After: {after:.0%}", flush=True)

    print(f"\nResult vs {OPPONENT_LABEL}: {before:.0%} -> {after:.0%} after {TRAIN_STEPS:,} steps.", flush=True)
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
