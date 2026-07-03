"""Live smoke test for the v2 stack.

Confirms the whole chain works on a real battle:
    battle -> 854-feature observation -> attention network -> legal action -> step
"""
import numpy as np
from sb3_contrib import MaskablePPO

from rl_env_v2 import N_FEATURES
from team_net import TeamAttentionExtractor
from train_v2 import build_env

env, inner = build_env()
policy_kwargs = dict(
    features_extractor_class=TeamAttentionExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[128, 128],
)
model = MaskablePPO("MultiInputPolicy", env, n_steps=64, policy_kwargs=policy_kwargs, verbose=0)

obs, _ = env.reset()
print("observation shape:", obs["observation"].shape, "(expected", N_FEATURES, ")")
assert obs["observation"].shape == (N_FEATURES,), "observation length mismatch!"

for i in range(12):
    masks = np.asarray(obs["action_mask"], dtype=bool)
    action, _ = model.predict(obs, action_masks=masks, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(np.int64(action))
    print(f"step {i}: action={int(action)} reward={reward:.2f} done={terminated or truncated}")
    if terminated or truncated:
        print("battle ended; won =", inner.battle1.won)
        break

env.close()
print("V2 LIVE SMOKE OK")
