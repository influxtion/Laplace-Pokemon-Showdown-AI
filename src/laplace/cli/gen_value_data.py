r"""Generate (engine-state features, outcome) training data for the value net.

Two EnginePlayers battle on the local server. At every decision each player featurizes one
determinized state -- built with its live trackers (choice locks, scarf hints, pending
Wish/Future Sight), exactly like the search's worlds -- and the sample is later labelled
with that player's game outcome. 1 win / 0 loss; ties dropped.

Features live in engine-state space (value_features.py) rather than poke-env observation
space, so the same net can score states rolled one ply forward with
generate_instructions / apply_instructions at decision time.

Sharded across worker processes (poke-engine holds the GIL). Each worker writes an .npz
chunk to data_value2/; chunks accumulate across runs and train_value.py globs them all.

    python -m laplace.cli.gen_value_data --battles 250 --workers 10
"""

import argparse
import asyncio
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from laplace import paths

BATTLE_FORMAT = "gen9randombattle"
# v2: 368-feature featurizer (ability/recovery/speed-race flags), 2 worlds per decision.
DATA_DIR = paths.VALUE_DATA_DIR
WORLDS_PER_DECISION = 2


def _play_share(n, det, time_ms, idx, stamp, out_dir=None):
    import numpy as np
    from poke_env import AccountConfiguration
    from laplace.agent.engine_search import EnginePlayer
    from laplace.agent.poke_engine_adapter import build_state
    from laplace.value.value_features import featurize, N_VALUE_FEATURES

    class DataPlayer(EnginePlayer):
        """EnginePlayer that snapshots determinized world features per decision."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.samples = {}   # battle_tag -> list of feature vectors

        def choose_move(self, battle):
            try:
                recorded = self.samples.setdefault(battle.battle_tag, [])
                for _ in range(WORLDS_PER_DECISION):   # 2 hidden-set draws of the same position
                    # Every argument the search's own worlds are built with, including
                    # use_joint and the Boots hints. Leaving those two off drew the training
                    # states from the MARGINAL sampler while the shipped bot searches (and
                    # the value head scores) joint-set worlds -- the one thing this file
                    # exists to keep identical, quietly not identical.
                    state = build_state(
                        battle, use_stats=self.use_stats,
                        opp_used_since_switch=self._opp_used_since_switch(battle),
                        opp_speed_hints=self._opp_speed.get(battle.battle_tag, {}),
                        opp_item_hints=self._opp_item.get(battle.battle_tag, {}),
                        pending=self._pending.get(battle.battle_tag),
                        use_joint=self.use_joint_sets)
                    recorded.append(featurize(state).astype(np.float16))
            except Exception:
                pass
            return super().choose_move(battle)

    # Players run the LADDER root config (share averaging + mixed strategy + the current
    # net), not the engine defaults. The net's target is P(win | both sides play like the
    # DEPLOYED bot), so the data distribution has to match deployment. v3's data predates
    # the whole mixed-root era, and that mismatch is exactly why its authority had to be cut.
    # Bootstrapping on the previous net is intended.
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _net = os.path.join(_root, "models", "value_net.pt")
    ladder_kw = dict(robust_vote=False, mix_root=True, mix_frac=0.9, value_boost_margin=0,
                     value_model_path=_net if os.path.exists(_net) else None)

    async def run():
        a = DataPlayer(account_configuration=AccountConfiguration.generate(f"dga{idx}", rand=True),
                       battle_format=BATTLE_FORMAT, n_determinizations=det,
                       search_time_ms=time_ms, threads=2, **ladder_kw)
        b = DataPlayer(account_configuration=AccountConfiguration.generate(f"dgb{idx}", rand=True),
                       battle_format=BATTLE_FORMAT, n_determinizations=det,
                       search_time_ms=time_ms, threads=2, **ladder_kw)
        await a.battle_against(b, n_battles=n)

        xs, ys, gs = [], [], []
        game = 0
        for player in (a, b):
            for tag, feats in player.samples.items():
                battle = player.battles.get(tag)
                if battle is None or battle.won is None:   # drop ties / unfinished
                    continue
                label = 1.0 if battle.won else 0.0
                xs.extend(feats)
                ys.extend([label] * len(feats))
                gs.extend([game] * len(feats))
                game += 1
        if xs:
            return np.stack(xs), np.asarray(ys, np.float16), np.asarray(gs, np.int32)
        return (np.zeros((0, N_VALUE_FEATURES), np.float16),
                np.zeros(0, np.float16), np.zeros(0, np.int32))

    x, y, g = asyncio.new_event_loop().run_until_complete(run())
    out_dir = out_dir or DATA_DIR       # passed explicitly: workers re-import the module
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"chunk_{stamp}_{idx}.npz")
    np.savez_compressed(path, x=x, y=y, g=g)
    return len(y), float(y.mean()) if len(y) else 0.0


def main():
    global DATA_DIR
    ap = argparse.ArgumentParser(description="Self-play data generation for the value net.")
    ap.add_argument("--battles", type=int, default=200)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--det", type=int, default=4)
    ap.add_argument("--time-ms", type=int, default=60)
    ap.add_argument("--out-dir", default=None,
                    help="chunk output dir (default data_value2/)")
    args = ap.parse_args()
    if args.out_dir:
        DATA_DIR = os.path.abspath(args.out_dir)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    base, extra = divmod(args.battles, args.workers)
    shares = [base + (1 if i < extra else 0) for i in range(args.workers)]
    shares = [s for s in shares if s > 0]
    print(f"Generating from {args.battles} self-play games over {len(shares)} workers "
          f"(det={args.det}, {args.time_ms}ms) -> {DATA_DIR}", flush=True)

    t0 = time.time()
    total = 0
    with ProcessPoolExecutor(max_workers=len(shares)) as ex:
        futs = [ex.submit(_play_share, n, args.det, args.time_ms, i, stamp, DATA_DIR)
                for i, n in enumerate(shares)]
        for f in futs:
            n, mean = f.result()
            total += n
    dt = time.time() - t0
    print(f"Done: {total} samples from {args.battles} games in {dt:.0f}s "
          f"({total / max(dt, 1):.0f} samples/s).", flush=True)


if __name__ == "__main__":
    main()
