"""The value network itself: a small CPU-friendly MLP over engine-state features.

Split out from the trainer (``laplace.cli.train_value``) because the *bot* needs
this class at inference time -- importing it must not drag in numpy, argparse and
the training loop. Trained by ``python -m laplace.cli.train_value``; consumed by
``laplace.agent.engine_search`` as a tie-breaker on the handful of turns where the
engine's own evaluation is weak.
"""

import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, n_in, hidden=(256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)     # logits; sigmoid at inference
