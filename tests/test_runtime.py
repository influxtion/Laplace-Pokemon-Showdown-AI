"""Offline checks: real engine, bundled data/model, and replay regressions."""

import json
import unittest

import numpy as np
import torch
from poke_engine import State, monte_carlo_tree_search
from poke_env.battle import Pokemon, Move

from laplace import paths
from laplace.agent.knowledge import KNOWLEDGE, estimate_damage_fraction, move_nullified
from laplace.agent.engine_search import EnginePlayer  # checks the search-module imports
from laplace.analysis.mine_losses import _fold_boost, _hp_frac, wilson
from laplace.value.net import ValueNet
from laplace.value.value_features import featurize


class RuntimeTests(unittest.TestCase):
    def test_data_and_set_predictions(self):
        for path in (paths.JOINT_SETS, paths.RANDBATS_STATS, paths.RANDBATS_SETS):
            with open(path, encoding="utf-8") as stream:
                self.assertTrue(json.load(stream), path)
        moves = KNOWLEDGE.predict_moves("haxorus", {"outrage"})
        self.assertIn("outrage", moves)
        self.assertGreater(len(moves), 1)

    def test_type_immunity(self):
        attacker = Pokemon(gen=9, species="pikachu")
        defender = Pokemon(gen=9, species="garchomp")
        move = Move("thunderbolt", gen=9)
        self.assertTrue(move_nullified(attacker, move, defender))
        self.assertEqual(estimate_damage_fraction(attacker, move, defender), 0)

    def test_engine_and_shipped_model(self):
        state = State()
        # Includes the compiled Rust boundary, rather than just import checks.
        result = monte_carlo_tree_search(state, duration_ms=20, threads=1)
        self.assertGreater(result.total_visits, 0)
        features = featurize(state)
        checkpoint = torch.load(paths.VALUE_NET, map_location="cpu", weights_only=True)
        self.assertEqual(len(features), checkpoint["n_in"])
        self.assertTrue(np.isfinite(features).all())
        model = ValueNet(checkpoint["n_in"], hidden=tuple(checkpoint.get("hidden", (256, 128))),
                         dropout=checkpoint.get("dropout", 0.1))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            probability = model(torch.from_numpy(features).unsqueeze(0)).sigmoid().item()
        self.assertTrue(0 <= probability <= 1)

    def test_belly_drum_is_absolute_boost(self):
        parts = "|-setboost|p2a: Azumarill|atk|6".split("|")
        self.assertEqual(_fold_boost({"atk": 2}, parts, "-setboost", "p2"), {"atk": 6})
        self.assertEqual(_fold_boost({"atk": 2}, parts, "-setboost", "p1"), {"atk": 2})

    def test_hp_with_protocol_annotations(self):
        parts = "|-damage|p1a: Pikachu|75/100|[from] item: Life Orb".split("|")
        self.assertEqual(_hp_frac(parts), 0.75)

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson(30, 60)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)


if __name__ == "__main__":
    unittest.main()
