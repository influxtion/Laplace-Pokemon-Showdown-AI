"""Team-attention network for the v2 agent (the architecture half of phase 1).

Replaces SB3's default flat MLP. Reshapes the 854-number observation back into the
12-Pokemon grid rl_env_v2 lays out, runs a shared encoder over each Pokemon, lets them
attend to each other (unrevealed slots masked out), pools over the real Pokemon, glues on
the global field features, and outputs a vector for the policy/value heads.

The point: the network is told the team structure instead of rediscovering it, and attention
lets each Pokemon's representation depend on the others (your threat level depends on what's
in front of you). The padding mask handles imperfect information cleanly -- unrevealed slots
are excluded from attention and pooling, the way the action mask excludes illegal moves.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl_env_v2 import N_MON, PER_MON, GLOBAL

KNOWN_INDEX = 3  # position of the "known" flag inside each Pokemon's PER_MON-length token


class TeamAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim=64, n_heads=4, global_dim=64, features_dim=128):
        super().__init__(observation_space, features_dim)
        # Shared per-Pokemon encoder: PER_MON raw features -> embed_dim.
        self.mon_encoder = nn.Sequential(
            nn.Linear(PER_MON, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
        )
        # Self-attention across the 12 Pokemon tokens.
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        # Encoder for the global field features.
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL, global_dim), nn.ReLU(),
        )
        # Final head: pooled team + global -> features_dim.
        self.head = nn.Sequential(
            nn.Linear(embed_dim + global_dim, features_dim), nn.ReLU(),
        )

    def forward(self, observations):
        x = observations["observation"]                       # (B, 854)
        b = x.shape[0]
        team = x[:, :N_MON * PER_MON].reshape(b, N_MON, PER_MON)  # (B, 12, 64)
        glob = x[:, N_MON * PER_MON:]                         # (B, 86)

        # Padding mask: True where the slot is an unrevealed (all-zero) Pokemon.
        pad_mask = team[:, :, KNOWN_INDEX] < 0.5              # (B, 12), True = ignore

        enc = self.mon_encoder(team)                          # (B, 12, embed_dim)
        attn_out, _ = self.attn(enc, enc, enc, key_padding_mask=pad_mask)  # (B, 12, embed_dim)

        # Masked mean-pool over the REAL Pokemon only (unknown slots contribute nothing).
        valid = (~pad_mask).float().unsqueeze(-1)             # (B, 12, 1)
        pooled = (attn_out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)  # (B, embed_dim)

        g = self.global_encoder(glob)                         # (B, global_dim)
        return self.head(torch.cat([pooled, g], dim=1))       # (B, features_dim)
