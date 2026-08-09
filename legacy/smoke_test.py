"""A quick check that the training setup starts up and takes a few turns without falling over."""
import numpy as np
from train_rl import build_env
from rl_env import N_FEATURES

env, _inner = build_env()
obs, _ = env.reset()
print("Observation keys:", list(obs.keys()))
print("observation shape:", obs["observation"].shape, "action_mask shape:", obs["action_mask"].shape)
assert obs["observation"].shape == (N_FEATURES,), f"expected {N_FEATURES} features"
print("observation sample:", np.round(obs["observation"], 2))

for i in range(5):
    legal = np.flatnonzero(obs["action_mask"])
    action = np.int64(np.random.choice(legal)) if len(legal) else np.int64(0)
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"step {i}: action={int(action)} reward={reward:.2f} done={terminated or truncated}")
    if terminated or truncated:
        print("battle ended; won =", env.env.battle1.won)
        break

env.close()
print("SMOKE TEST OK")
