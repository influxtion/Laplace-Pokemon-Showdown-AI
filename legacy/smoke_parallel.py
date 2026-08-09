r"""A quick check that parallel training starts up properly.

Spins up the training games in separate processes and runs a handful of learning steps on a
brand new model. It won't touch anything you've saved. If this finishes, the real run will
work on this machine too.

Run from the project root, with the local server going:
    python -u src\smoke_parallel.py
"""

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv

import train_rl as t


def main():
    n = t.N_ENVS
    print(f"Building {n} parallel envs (each its own process + server connection)...", flush=True)
    venv = SubprocVecEnv([t.make_env(i) for i in range(n)])
    model = MaskablePPO(
        "MultiInputPolicy", venv, n_steps=128, batch_size=64, verbose=1, ent_coef=0.01
    )
    steps = 128 * n
    print(f"Running {steps} smoke-test steps across {n} envs...", flush=True)
    model.learn(total_timesteps=steps)
    print(f"Parallel smoke OK: {n} envs trained {steps} steps with no errors.", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
