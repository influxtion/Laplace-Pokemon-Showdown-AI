"""The attention network. This is the half of the experiment that changes the architecture.

Instead of one flat stack of layers, this folds the observation back into the twelve-Pokemon
grid it was laid out as, runs the same small encoder over each Pokemon, then lets them all
look at each other before pooling the results together with the field state.

The idea is that the network is handed the team structure rather than having to rediscover
it, and that letting each Pokemon's representation depend on the others captures something
real: how threatening something is depends entirely on what's standing opposite it.

Pokemon we haven't seen yet are masked out of both the attention and the pooling, the same
way illegal moves are masked out of the action choice.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl_env_v2 import N_MON, PER_MON, GLOBAL

KNOWN_INDEX = 3  # where the "have we seen this one?" flag sits in each Pokemon's block


class TeamAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim=64, n_heads=4, global_dim=64, features_dim=128):
        super().__init__(observation_space, features_dim)
        # One encoder, applied to every Pokemon in turn.
        self.mon_encoder = nn.Sequential(
            nn.Linear(PER_MON, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
        )
        # This is where all twelve get to look at each other.
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        # A separate small encoder for the weather, hazards and so on.
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL, global_dim), nn.ReLU(),
        )
        # And a final layer that combines the two.
        self.head = nn.Sequential(
            nn.Linear(embed_dim + global_dim, features_dim), nn.ReLU(),
        )

    def forward(self, observations):
        x = observations["observation"]
        b = x.shape[0]
        # Split the flat observation back into twelve Pokemon and the field state.
        team = x[:, :N_MON * PER_MON].reshape(b, N_MON, PER_MON)
        glob = x[:, N_MON * PER_MON:]

        # Mark the slots we haven't seen yet, so they get ignored below.
        pad_mask = team[:, :, KNOWN_INDEX] < 0.5

        enc = self.mon_encoder(team)
        attn_out, _ = self.attn(enc, enc, enc, key_padding_mask=pad_mask)

        # Average over the Pokemon we actually know about. The unknown ones count for nothing.
        valid = (~pad_mask).float().unsqueeze(-1)
        pooled = (attn_out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        g = self.global_encoder(glob)
        return self.head(torch.cat([pooled, g], dim=1))
