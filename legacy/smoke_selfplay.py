r"""A quick check of the self-play loop, before committing to a long run.

Sets the agent against a network-driven opponent, plays a few dozen turns, and confirms the
observation is the right shape and that the opponent can actually pick moves.

Run from the project root, with the local server going:
    python -u src\smoke_selfplay.py
"""

import os

import numpy as np
from sb3_contrib import MaskablePPO

import train_selfplay as t


def main():
    opponent = t.make_self_opponent()
    env, inner = t.build_train_env(opponent)

    if os.path.exists(t.WARM_START_PATH):
        print(f"Loading {t.WARM_START_PATH}", flush=True)
        model = MaskablePPO.load(t.WARM_START_PATH, env=env)
    else:
        print("No warm-start model; using a fresh one for the smoke test.", flush=True)
        model = MaskablePPO("MultiInputPolicy", env)

    # Set the opponent up exactly the way training does.
    model.save(t.SNAPSHOT_PATH)
    opponent.model = MaskablePPO.load(t.SNAPSHOT_PATH)

    obs, _ = env.reset()
    print(f"obs shape: {obs['observation'].shape} (expected ({t.N_FEATURES},))", flush=True)

    battles = 0
    for step in range(60):
        mask = np.asarray(obs["action_mask"], dtype=bool)
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(np.int64(action))
        if terminated or truncated:
            battles += 1
            print(f"  battle {battles} ended at step {step} (won={inner.battle1.won})", flush=True)
            obs, _ = env.reset()

    print(f"Self-play smoke OK: ran 60 steps, {battles} battle(s) completed.", flush=True)
    env.close()


if __name__ == "__main__":
    main()
