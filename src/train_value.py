r"""Train the value net: V(engine-state features) ~ P(side_one wins).

Data: data_value2/chunk_*.npz from gen_value_data.py -- x float16 features, y win labels,
g per-sample game index within the chunk.

Samples within a game are strongly correlated (same teams, same outcome), so the train/val
split is BY GAME: every sample of a game lands on one side, never both. Otherwise val is
contaminated and early stopping overfits into a great-looking number that means nothing.

Model: small torch MLP, CPU-friendly at inference -- the bot scores a handful of one-ply
rollout states per decision, so the latency budget is sub-millisecond per batch.

    python -u src\train_value.py                  # -> models/value_net.pt
    python -u src\train_value.py --epochs 40
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_GLOB = os.path.join(ROOT, "data_value2", "chunk_*.npz")   # v2 features (368)
MODEL_PATH = os.path.join(ROOT, "models", "value_net.pt")


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


def load_split(val_frac=0.1, seed=7):
    """All chunks -> (train_x, train_y, val_x, val_y), split by (chunk, game) pair."""
    xs, ys, gids = [], [], []
    gid = 0
    for path in sorted(glob.glob(DATA_GLOB)):
        d = np.load(path)
        x, y, g = d["x"], d["y"], d["g"]
        if not len(y):
            continue
        xs.append(x.astype(np.float32))
        ys.append(y.astype(np.float32))
        gids.append(g.astype(np.int64) + gid)      # make game ids globally unique
        gid += int(g.max()) + 1 if len(g) else 0
    if not xs:
        raise SystemExit(f"no data found under {DATA_GLOB}")
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    g = np.concatenate(gids)

    rng = np.random.default_rng(seed)
    games = np.unique(g)
    rng.shuffle(games)
    n_val = max(1, int(len(games) * val_frac))
    val_games = set(games[:n_val].tolist())
    val_mask = np.isin(g, list(val_games))
    print(f"{len(y)} samples / {len(games)} games "
          f"({val_mask.sum()} val samples / {n_val} val games), "
          f"label mean {y.mean():.3f}")
    return x[~val_mask], y[~val_mask], x[val_mask], y[val_mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--hidden", default="256,128",
                    help="comma-separated hidden layer sizes")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--out", default=MODEL_PATH)
    ap.add_argument("--data", default=None, help="chunk dir (default data_value2/)")
    args = ap.parse_args()
    hidden = tuple(int(h) for h in args.hidden.split(","))
    global DATA_GLOB
    if args.data:
        DATA_GLOB = os.path.join(os.path.abspath(args.data), "chunk_*.npz")

    tx, ty, vx, vy = load_split()
    n_in = tx.shape[1]
    torch.manual_seed(7)
    model = ValueNet(n_in, hidden=hidden, dropout=args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    tx_t = torch.from_numpy(tx)
    ty_t = torch.from_numpy(ty)
    vx_t = torch.from_numpy(vx)
    vy_t = torch.from_numpy(vy)

    best_val = float("inf")
    best_state = None
    stale = 0
    n = len(ty_t)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(tx_t[idx]), ty_t[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vlogits = model(vx_t)
            vloss = loss_fn(vlogits, vy_t).item()
            vacc = ((vlogits > 0) == (vy_t > 0.5)).float().mean().item()
        print(f"epoch {epoch:3d}  train {tot/n:.4f}  val {vloss:.4f}  val_acc {vacc:.3f}",
              flush=True)
        if vloss < best_val - 1e-4:
            best_val, stale = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop (best val {best_val:.4f})")
                break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": best_state or model.state_dict(), "n_in": n_in,
                "hidden": list(hidden), "dropout": args.dropout,
                "val_loss": best_val}, args.out)
    print(f"saved {args.out} (val loss {best_val:.4f})")


if __name__ == "__main__":
    main()
