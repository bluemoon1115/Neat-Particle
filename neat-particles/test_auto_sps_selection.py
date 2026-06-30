from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from genome_targets import genome_to_target_data, load_genome_target, save_genome_target, target_data_to_genome
from bayesian_preference_optimizer import BayesianOptimizationDependencyError
from sps_selection import (
    ParticleGenomeFactory,
    build_sps_candidates,
    create_sps_search,
    generic_weight_slots,
    load_particle_config,
    run_auto_sps_selection,
    select_nearest_candidate,
)
from auto_sps_select import _compact_history


CONFIG_PATH = os.path.join(BASE_DIR, "config-generic.ini")


class GenomeTargetTest(unittest.TestCase):
    def setUp(self):
        self.config = load_particle_config(CONFIG_PATH)
        self.factory = ParticleGenomeFactory(self.config, seed=11)
        self.genome = self.factory.create_random_genome()

    def test_round_trip_preserves_genome_distance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "target.json")
            save_genome_target(path, self.genome, self.config, candidate_key=4, mode="SPS")
            loaded, data = load_genome_target(path, self.config)

        self.assertEqual(data["metadata"]["candidate_key"], 4)
        self.assertEqual(self.genome.distance(loaded, self.config.genome_config), 0.0)

    def test_config_mismatch_is_rejected(self):
        data = genome_to_target_data(self.genome, self.config)
        data["config_shape"]["num_outputs"] += 1
        with self.assertRaises(ValueError):
            target_data_to_genome(data, self.config)


class AutoSpsSelectionTest(unittest.TestCase):
    def setUp(self):
        self.config = load_particle_config(CONFIG_PATH)
        self.factory = ParticleGenomeFactory(self.config, seed=17)
        self.target = self.factory.create_random_genome()

    def _create_sps_search_or_skip(self, genome, slots, seed):
        try:
            return create_sps_search(genome, slots, self.config, seed=seed)
        except BayesianOptimizationDependencyError as exc:
            self.skipTest(str(exc))

    def test_sps_batch_has_nine_candidates_and_center_at_index_four(self):
        slots = generic_weight_slots(self.target, self.config)
        search = self._create_sps_search_or_skip(self.target, slots, seed=3)
        candidates = build_sps_candidates(self.target, slots, search, self.config)

        self.assertEqual(len(candidates), 9)
        self.assertEqual(candidates[4].search_vector.vector, search.x_plus)

    def test_select_nearest_candidate_uses_minimum_distance(self):
        slots = generic_weight_slots(self.target, self.config)
        search = self._create_sps_search_or_skip(self.target, slots, seed=5)
        candidates = build_sps_candidates(self.target, slots, search, self.config)
        exact = copy.deepcopy(self.target)
        candidates[2] = copy.copy(candidates[2])
        object.__setattr__(candidates[2], "genome", exact)

        best_index, best_distance, distances = select_nearest_candidate(
            candidates,
            self.target,
            self.config.genome_config,
        )

        self.assertEqual(best_index, 2)
        self.assertEqual(best_distance, min(distances))
        self.assertEqual(best_distance, 0.0)

    def test_auto_selection_stops_at_threshold(self):
        try:
            result = run_auto_sps_selection(
                self.target,
                self.config,
                seed=23,
                threshold=999.0,
                max_steps=10,
            )
        except BayesianOptimizationDependencyError as exc:
            self.skipTest(str(exc))

        self.assertEqual(result.stop_reason, "threshold")
        self.assertEqual(result.steps, 1)
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)

    def test_auto_selection_stops_at_max_steps(self):
        try:
            result = run_auto_sps_selection(
                self.target,
                self.config,
                seed=29,
                threshold=-1.0,
                max_steps=3,
            )
        except BayesianOptimizationDependencyError as exc:
            self.skipTest(str(exc))

        self.assertEqual(result.stop_reason, "max_steps")
        self.assertEqual(result.steps, 3)
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)


class AutoSpsReportTest(unittest.TestCase):
    def test_compact_history_keeps_every_five_steps_and_final_step(self):
        history = [{"step": step} for step in range(1, 13)]

        compact = _compact_history(history)

        self.assertEqual([record["step"] for record in compact], [5, 10, 12])


if __name__ == "__main__":
    unittest.main()
