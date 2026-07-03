r"""Generate (engine-state features, outcome) training data for the value network.

Two EnginePlayers battle on the local server; at every decision each player featurizes ONE
determinized poke-engine state (built with its live trackers -- choice locks, scarf hints,
pending Wish/Future Sight -- exactly like the search's worlds) and the sample is later
labeled with that player's game outcome (1 win / 0 loss, ties dropped). This is the Track-A
value-model data: V(state) ~ P(side_one wins | position, both sides playing like our bot).

The features live in engine-state space (value_features.py) rather than poke-env
observation space so the same net can score states rolled one ply forward with
generate_instructions/apply_instructions at decision time.

Sharded across worker processes (poke-engine holds the GIL). Each worker writes an .npz
chunk (x features, y labels, g per-sample game index) to data_value/; chunks accumulate
across runs, train_value.py globs them all.

    python -u src\gen_value_data.py --battles 250 --workers 10
"""

import argparse
import asyncio
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

BATTLE_FORMAT = "gen9randombattle"
# v2: 368-feat featurizer (ability/recovery/speed-race flags), 2 worlds per decision.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data_value2")
WORLDS_PER_DECISION = 2


def _play_share(n, det, time_ms, idx, stamp, out_dir=None):
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from poke_env import AccountConfiguration
    from engine_search import EnginePlayer
    from poke_engine_adapter import build_state
    from value_features import featurize, N_VALUE_FEATURES

    class DataPlayer(EnginePlayer):
        """EnginePlayer that snapshots one determinized world's features per decision."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.samples = {}   # battle_tag -> list of feature vectors

        def choose_move(self, battle):
            try:
                recorded = self.samples.setdefault(battle.battle_tag, [])
                for _ in range(WORLDS_PER_DECISION):   # 2 hidden-set draws of the same position
                    state = build_state(
                        battle, use_stats=self.use_stats,
                        opp_used_since_switch=self._opp_used_since_switch(battle),
                        opp_speed_hints=self._opp_speed.get(battle.battle_tag, {}),
                        pending=self._pending.get(battle.battle_tag))
                    recorded.append(featurize(state).astype(np.float16))
            except Exception:
                pass
            return super().choose_move(battle)

    async def run():
        a = DataPlayer(account_configuration=AccountConfiguration.generate(f"dga{idx}", rand=True),
                       battle_format=BATTLE_FORMAT, n_determinizations=det,
                       search_time_ms=time_ms, threads=2)
        b = DataPlayer(account_configuration=AccountConfiguration.generate(f"dgb{idx}", rand=True),
                       battle_format=BATTLE_FORMAT, n_determinizations=det,
                       search_time_ms=time_ms, threads=2)
        await a.battle_against(b, n_battles=n)

        xs, ys, gs = [], [], []
        game = 0
        for player in (a, b):
            for tag, feats in player.samples.items():
                battle = player.battles.get(tag)
                if battle is None or battle.won is None:   # drop ties/unfinished
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
