from __future__ import annotations

import math
import random
import unittest

from bayesian_preference_optimizer import (
    BayesianOptimizationDependencyError,
    PreferenceDataset,
    PreferentialBayesianOptimizer,
)
from sequential_plane_search import SequentialPlaneSearch


class PreferenceDatasetTest(unittest.TestCase):
    def test_choice_creates_pairwise_comparisons(self):
        chosen = (0.5, 0.5)
        rejected = tuple((idx / 10.0, idx / 11.0) for idx in range(8))
        dataset = PreferenceDataset.from_records([(chosen, rejected)])

        self.assertEqual(len(dataset.points), 9)
        self.assertEqual(len(dataset.comparisons), 8)
        self.assertTrue(all(pair[0] == 0 for pair in dataset.comparisons))


class PreferentialBayesianOptimizerTest(unittest.TestCase):
    def setUp(self):
        try:
            self.optimizer = PreferentialBayesianOptimizer(2, map_steps=20)
        except BayesianOptimizationDependencyError as exc:
            self.skipTest(str(exc))

    def test_fit_posterior_and_ei_are_finite(self):
        self.optimizer.fit_from_records([
            ((0.8, 0.8), ((0.2, 0.2), (0.3, 0.4), (0.4, 0.3))),
            ((0.7, 0.8), ((0.1, 0.2), (0.4, 0.4))),
        ])

        means, variances = self.optimizer.posterior([(0.8, 0.8), (0.2, 0.2)])
        eis = self.optimizer.expected_improvement([(0.9, 0.9), (0.1, 0.1)], (0.8, 0.8))

        self.assertEqual(len(means), 2)
        self.assertEqual(len(variances), 2)
        self.assertEqual(len(eis), 2)
        self.assertTrue(all(math.isfinite(value) for value in means + variances + eis))
        self.assertTrue(all(value >= 0.0 for value in variances + eis))


class SequentialPlaneSearchTest(unittest.TestCase):
    def setUp(self):
        try:
            self.search = SequentialPlaneSearch(2, (0.5, 0.5), seed=3)
        except BayesianOptimizationDependencyError as exc:
            self.skipTest(str(exc))

    def test_observe_records_visible_pairwise_preferences(self):
        self.search.observe(4)

        self.assertEqual(len(self.search.history), 1)
        self.assertEqual(len(self.search.history[0].rejected), 8)
        self.assertTrue(all(0.0 <= value <= 1.0 for sample in self.search.transforms() for value in sample.vector))

    def test_synthetic_preference_moves_toward_target(self):
        target = (0.82, 0.22)
        search = SequentialPlaneSearch(2, (0.5, 0.5), seed=7)

        def distance(point):
            return sum((point[idx] - target[idx]) ** 2 for idx in range(2))

        start_distance = distance(search.x_plus)
        for _ in range(5):
            samples = [sample.vector for sample in search.transforms()]
            chosen = min(range(len(samples)), key=lambda idx: distance(samples[idx]))
            search.observe(chosen)

        self.assertLess(distance(search.x_plus), start_distance)


if __name__ == "__main__":
    unittest.main()
