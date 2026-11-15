from __future__ import annotations

import math
import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, BASE_DIR)

from bayesian_preference_optimizer import (
    BayesianOptimizationDependencyError,
    PreferenceDataset,
    PreferentialBayesianOptimizer,
)
from sequential_plane_search import SequentialPlaneSearch
from sps_selection import ParticleGenomeFactory, build_sps_candidates, create_sps_search, generic_weight_slots, load_particle_config
from surrogate_scorer import SurrogateScorer


CONFIG_PATH = os.path.join(BASE_DIR, "config-generic.ini")


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


class _ZeroRng:
    def random(self):
        return 0.0


class SurrogateScorerTest(unittest.TestCase):
    def test_records_positives_and_negatives(self):
        scorer = SurrogateScorer(2)
        scorer.fit_from_records([
            ((0.8, 0.8), ((0.2, 0.2), (0.3, 0.3))),
            ((0.7, 0.8), ((0.1, 0.2),)),
        ])

        self.assertEqual(scorer.positives, [(0.8, 0.8), (0.7, 0.8)])
        self.assertEqual(scorer.negatives, [(0.2, 0.2), (0.3, 0.3), (0.1, 0.2)])

    def test_rewards_positives_and_penalizes_negatives(self):
        scorer = SurrogateScorer(2)
        scorer.fit_from_records([((0.8, 0.8), ((0.2, 0.2),))])

        self.assertGreater(
            scorer.score((0.8, 0.8), _ZeroRng()),
            scorer.score((0.2, 0.2), _ZeroRng()),
        )


class SurrogateSequentialPlaneSearchTest(unittest.TestCase):
    def test_surrogate_mode_builds_initial_plane_without_gp_fit(self):
        search = SequentialPlaneSearch(2, (0.5, 0.5), seed=3, acquisition="surrogate")

        self.assertEqual(search.acquisition, "surrogate")
        self.assertFalse(search.surrogate.fitted)
        self.assertEqual(len(search.transforms()), 9)
        self.assertGreater(len({sample.vector for sample in search.transforms()}), 1)

    def test_surrogate_observe_records_visible_preferences(self):
        search = SequentialPlaneSearch(2, (0.5, 0.5), seed=3, acquisition="surrogate")
        search.observe(4)

        self.assertEqual(search.surrogate.positives, [search.history[0].chosen])
        self.assertEqual(len(search.surrogate.negatives), 8)
        self.assertTrue(search.surrogate.fitted)

    def test_invalid_acquisition_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            SequentialPlaneSearch(2, (0.5, 0.5), acquisition="unknown")

    def test_score_tie_prefers_candidate_closest_to_center(self):
        search = SequentialPlaneSearch(2, (0.5, 0.5), seed=3, acquisition="surrogate")
        search._acquisition_score = lambda _candidate: 1.0

        best = search._best_scored_candidate([(0.8, 0.8), (0.51, 0.5), (0.7, 0.7)])

        self.assertEqual(best, (0.51, 0.5))


class SpsHelperApiTest(unittest.TestCase):
    def test_create_sps_search_supports_surrogate_acquisition(self):
        config = load_particle_config(CONFIG_PATH)
        genome = ParticleGenomeFactory(config, seed=17).create_random_genome()
        slots = generic_weight_slots(genome, config)
        search = create_sps_search(genome, slots, config, seed=3, acquisition="surrogate")
        candidates = build_sps_candidates(genome, slots, search, config)

        self.assertEqual(search.acquisition, "surrogate")
        self.assertEqual(len(candidates), 9)


if __name__ == "__main__":
    unittest.main()
