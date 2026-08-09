"""Train the agent, and check whether it actually got better.

You'll need the local Showdown server running first:
    cd server && node pokemon-showdown start --no-security

Then:
    python -u train_rl.py

It sets up the battles, picks up the saved agent if there is one, measures how often it
wins, trains for a while, measures again, and saves.

Progress goes to ./tb_logs. Run `tensorboard --logdir tb_logs` in another terminal and open
http://localhost:6006 to watch the curves.

We use the "maskable" flavour of PPO because only a handful of the 26 possible actions are
legal on any given turn. The library hands us a list of which ones are, and the algorithm
uses it to only ever consider legal choices. That learns far faster than letting the agent
flail at illegal moves until it works out they don't exist.
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

# Who we train against and measure against this run. "random" plays random legal moves and
# is the easy floor; "heuristic" understands type matchups, speed and switching, and is a
# genuine test.
OPPONENT = "heuristic"

# How many games to run in parallel, each in its own process with its own connection to the
# server. Roughly a proportional speedup. Set it near your core count; 4 is safe on a laptop.
N_ENVS = 4

TRAIN_STEPS = 100_000     # turns of experience to add this run
TB_DIR = "tb_logs"
# Evaluations are big on purpose. A 30-game measurement is worth about plus or minus nine
# points, which is enough noise to invent improvements that never happened. A couple of
# hundred games brings that down to three or four. They also pause training, so we don't do
# them often.
EVAL_FREQ = 25_000
LIVE_EVAL_BATTLES = 200
EVAL_BATTLES = 200        # for the before-and-after comparison
SAVE_FREQ = 100_000

# Rewards, weighted towards actually winning. The old settings taught the agent to trade
# evenly but never to close a game out: the small per-turn rewards fired constantly and
# drowned out the rare win bonus. Now that it's competent, the health reward is halved and
# the win bonus raised, so winning is unmistakably the point.
REWARD_WEIGHTS = {"hp_value": 0.25, "victory_value": 150.0}

# Each opponent gets its own saved agent, with the observation size in the filename. If you
# change what the network sees, the old weights won't load anyway, and a different name
# stops them being overwritten.
MODEL_PATH = f"ppo_vs_{OPPONENT}_obs{N_FEATURES}.zip"
# Nothing saved for this opponent yet? Start from the one trained against random opponents,
# which is a decent head start.
WARM_START_PATH = f"ppo_vs_random_obs{N_FEATURES}.zip"


def mask_fn(env):
    """Which of the 26 possible actions we're actually allowed to take right now."""
    return np.array(SinglesEnv.get_action_mask(env.env.battle1), dtype=bool)


OPPONENT_CLASSES = {"random": RandomPlayer, "heuristic": SimpleHeuristicsPlayer}
OPPONENT_LABEL = {"random": "RandomPlayer", "heuristic": "SimpleHeuristicsPlayer"}[OPPONENT]


def build_masked_env(agent_tag, opp_tag, reward_weights=None, reward_schedule=None, switch_penalty=0.0):
    """Set up one training or evaluation game.

    The two tags become server account names, kept distinct so parallel games don't fight
    over the same connection.

    The three reward arguments are passed in rather than read from the module, because the
    parallel workers re-import this file from scratch and would otherwise never see any
    changes the calling script made."""
    # The random suffix matters. Fixed account names collide with leftover ghost connections
    # from a crashed run, and then the challenge never goes through and the whole thing hangs.
    showdown = ShowdownSinglesEnv(
        reward_weights=reward_weights or REWARD_WEIGHTS,
        reward_schedule=reward_schedule,
        switch_penalty=switch_penalty,
        account_configuration1=AccountConfiguration.generate(f"{agent_tag}A", rand=True),
        account_configuration2=AccountConfiguration.generate(f"{agent_tag}B", rand=True),
        battle_format=BATTLE_FORMAT,
        strict=False,                 # an illegal choice becomes a random legal one
        log_level=logging.ERROR,      # stop the per-turn warning spam
    )
    opponent = OPPONENT_CLASSES[OPPONENT](
        account_configuration=AccountConfiguration.generate(opp_tag, rand=True),
        battle_format=BATTLE_FORMAT,
        start_listening=False,        # only used to make decisions, never connects
    )
    wrapped = SingleAgentWrapper(showdown, opponent)
    return ActionMasker(wrapped, mask_fn), showdown


def build_env(reward_weights=None, reward_schedule=None, switch_penalty=0.0):
    """The separate game used for measuring, with its own accounts."""
    return build_masked_env("ev", "evopp", reward_weights, reward_schedule, switch_penalty)


def make_env(rank, reward_weights=None, reward_schedule=None, switch_penalty=0.0):
    """Make one of the parallel training games.

    Wrapped in a Monitor so per-game reward and length show up in TensorBoard. That's
    automatic for a single game but has to be added by hand when you build the parallel
    setup yourself."""
    def _init():
        masked, _ = build_masked_env(
            f"tr{rank}", f"tropp{rank}", reward_weights, reward_schedule, switch_penalty)
        return Monitor(masked)
    return _init


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
    """Every so often, stop and play some games to see how the agent is doing.

    Uses its own separate game so that measuring doesn't disturb the ones it's learning from.
    """

    def __init__(self, eval_env, eval_inner, n_battles, eval_freq, best_path=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_inner = eval_inner
        self.n_battles = n_battles
        self.eval_freq = eval_freq
        self.best_path = best_path     # if set, save a copy whenever we hit a new high
        self.best_wr = -1.0
        self._last_bucket = None

    def _on_step(self):
        # Counted in training steps rather than calls, so the timing stays right no matter
        # how many games are running in parallel, and a resumed run doesn't evaluate
        # constantly.
        bucket = self.num_timesteps // self.eval_freq
        if bucket != self._last_bucket:
            self._last_bucket = bucket
            wr = win_rate(self.model, self.eval_env, self.eval_inner, self.n_battles)
            self.logger.record("eval/win_rate", wr)
            if self.verbose:
                print(f"[eval] {self.num_timesteps:,} steps -> win rate {wr:.0%}", flush=True)
            # Keep the best version, not the latest one. Training can wander downhill late
            # on, so the final weights are often worse than something from halfway through.
            # This caught us out once already.
            if self.best_path is not None and wr > self.best_wr:
                self.best_wr = wr
                self.model.save(self.best_path)
                if self.verbose:
                    print(f"[best] new best {wr:.0%} -> saved {self.best_path}", flush=True)
        return True


class SaveCallback(BaseCallback):
    """Save periodically, so an interrupted run doesn't lose everything."""

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
    # Training spreads across several processes; measuring gets one of its own.
    train_env = (
        SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
        if N_ENVS > 1 else DummyVecEnv([make_env(0)])
    )
    eval_env, eval_inner = build_env()

    def fresh_model():
        # That last setting is an exploration bonus. It pays the agent to stay a bit random,
        # so it keeps trying new ideas instead of settling on one early and never moving.
        # Locking in too early is exactly what stalled the first version of this agent.
        return MaskablePPO(
            "MultiInputPolicy", train_env, verbose=1, tensorboard_log=TB_DIR, ent_coef=0.01
        )

    def safe_load(path):
        # Loading fails if the saved network expects a different number of inputs. Start
        # over rather than crashing.
        try:
            return MaskablePPO.load(path, env=train_env, tensorboard_log=TB_DIR), True
        except (ValueError, RuntimeError, KeyError) as e:
            print(f"Could not load {path} ({type(e).__name__}); starting fresh.", flush=True)
            return fresh_model(), False

    # Where we start from, in order of preference: an agent already trained against this
    # opponent, then the one trained against random players, then nothing at all.
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
    # Only continue the existing graph when we're carrying on the same matchup. A fresh
    # start or a warm start gets its own line.
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
