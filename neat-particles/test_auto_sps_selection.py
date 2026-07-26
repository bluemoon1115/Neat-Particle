from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, BASE_DIR)

import sps_selection
from genome_targets import genome_to_target_data, load_genome_target, save_genome_target, target_data_to_genome
from bayesian_preference_optimizer import BayesianOptimizationDependencyError
from particle_similarity import ParticleSimilaritySettings, compare_genomes, histogram_distance, profile_genome, ssim_distance
from sps_selection import (
    ParticleGenomeFactory,
    build_sps_candidates,
    create_sps_search,
    generic_weight_slots,
    load_particle_config,
    run_auto_sps_selection,
    select_nearest_candidate,
)
from auto_sps_select import _compact_history, _export_auto_result_target, _write_run_outputs
from interactive_neat_particles import (
    INIT_CANDIDATE_INDEX,
    InteractiveBreeder,
    build_initial_iec_genomes,
    mark_initial_candidate,
)


CONFIG_PATH = os.path.join(BASE_DIR, "config-generic.ini")
FAST_SIMILARITY_SETTINGS = ParticleSimilaritySettings(simulation_steps=20, sample_stride=5, grid_size=12)


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

    def test_initial_iec_genomes_seed_center_from_export(self):
        source_factory = ParticleGenomeFactory(self.config, seed=101)
        source_genome = source_factory.create_random_genome()

        with tempfile.TemporaryDirectory() as temp_dir:
            init_path = os.path.join(temp_dir, "init.json")
            save_genome_target(init_path, source_genome, self.config, candidate_key=5, mode="IEC")
            breeder = InteractiveBreeder(self.config, seed=202)
            genomes, init_index = build_initial_iec_genomes(breeder, self.config, init_path)

        self.assertEqual(len(genomes), 9)
        self.assertEqual(init_index, INIT_CANDIDATE_INDEX)
        self.assertEqual(source_genome.distance(genomes[INIT_CANDIDATE_INDEX], self.config.genome_config), 0.0)

        non_center_distances = [
            source_genome.distance(genome, self.config.genome_config)
            for index, genome in enumerate(genomes)
            if index != INIT_CANDIDATE_INDEX
        ]
        self.assertEqual(len(non_center_distances), 8)
        self.assertTrue(all(distance > 0.0 for distance in non_center_distances))

        candidates = [SimpleNamespace(selected=False, label=None) for _genome in genomes]
        mark_initial_candidate(candidates, init_index)
        self.assertTrue(candidates[INIT_CANDIDATE_INDEX].selected)
        self.assertEqual(candidates[INIT_CANDIDATE_INDEX].label, "init")
        self.assertFalse(any(
            candidate.selected
            for index, candidate in enumerate(candidates)
            if index != INIT_CANDIDATE_INDEX
        ))

    def test_initial_iec_genomes_reject_invalid_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            init_path = os.path.join(temp_dir, "invalid.json")
            with open(init_path, "w", encoding="utf-8") as file:
                file.write("{not valid json")

            breeder = InteractiveBreeder(self.config, seed=303)
            with self.assertRaises(json.JSONDecodeError):
                build_initial_iec_genomes(breeder, self.config, init_path)


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

    def test_weight_slots_support_velocity_and_color_design_spaces(self):
        input_keys = self.config.genome_config.input_keys
        output_keys = self.config.genome_config.output_keys
        px, py, pz, distance, bias_input = input_keys
        vx, vy, vz, r, g, b = output_keys

        velocity_slots = generic_weight_slots(self.target, self.config, "velocity")
        color_slots = generic_weight_slots(self.target, self.config, "color")

        self.assertEqual(
            velocity_slots,
            [
                (distance, vx), (distance, vy), (distance, vz),
                (bias_input, vx), (bias_input, vy), (bias_input, vz),
                (px, vx), (px, vy), (px, vz),
                (py, vx), (py, vy), (py, vz),
                (pz, vx), (pz, vy), (pz, vz),
            ],
        )
        self.assertEqual(
            color_slots,
            [
                (distance, r), (distance, g), (distance, b),
                (bias_input, r), (bias_input, g), (bias_input, b),
                (px, r), (px, g), (px, b),
                (py, r), (py, g), (py, b),
                (pz, r), (pz, g), (pz, b),
            ],
        )

    def _mock_selection_distances(self, distances):
        def choose(_candidates, _target, _config, **_kwargs):
            distance = distances.pop(0)
            candidate_distances = [distance + 0.1 for _ in range(9)]
            candidate_distances[4] = distance
            return 4, distance, candidate_distances

        return choose

    def _create_fake_search(self, _bound_genome, slots, _config, seed=None):
        class FakeSearch:
            def __init__(self, dimensions):
                self.x_plus = tuple(0.5 for _ in range(dimensions))

            def transforms(self):
                return [sps_selection.SearchVector.from_vector(self.x_plus) for _ in range(9)]

            def observe(self, _chosen_index):
                return None

        return FakeSearch(len(slots))

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
            self.config,
            settings=FAST_SIMILARITY_SETTINGS,
        )

        self.assertEqual(best_index, 2)
        self.assertEqual(best_distance, min(distances))
        self.assertEqual(best_distance, 0.0)

    def test_identical_genomes_have_zero_behavior_distance(self):
        result = compare_genomes(
            self.target,
            copy.deepcopy(self.target),
            self.config,
            FAST_SIMILARITY_SETTINGS,
        )

        self.assertEqual(result.histogram_distance, 0.0)
        self.assertEqual(result.ssim_distance, 0.0)
        self.assertEqual(result.combined_distance, 0.0)

    def test_behavior_distances_are_bounded(self):
        other = self.factory.create_random_genome()

        result = compare_genomes(
            self.target,
            other,
            self.config,
            FAST_SIMILARITY_SETTINGS,
        )

        self.assertGreaterEqual(result.histogram_distance, 0.0)
        self.assertLessEqual(result.histogram_distance, 1.0)
        self.assertGreaterEqual(result.ssim_distance, 0.0)
        self.assertLessEqual(result.ssim_distance, 1.0)
        self.assertGreaterEqual(result.combined_distance, 0.0)
        self.assertLessEqual(result.combined_distance, 1.0)

    def test_compare_genomes_combines_histogram_and_ssim_distance(self):
        other = self.factory.create_random_genome()

        result = compare_genomes(
            self.target,
            other,
            self.config,
            FAST_SIMILARITY_SETTINGS,
        )

        expected = 0.75 * result.histogram_distance + 0.25 * result.ssim_distance
        self.assertAlmostEqual(result.combined_distance, expected)

    def test_histogram_similarity_includes_particle_color(self):
        dark = copy.deepcopy(self.target)
        light = copy.deepcopy(self.target)
        for output_key in self.config.genome_config.output_keys[3:6]:
            dark.nodes[output_key].bias = -30.0
            light.nodes[output_key].bias = 30.0

        spatial_only_settings = ParticleSimilaritySettings(
            simulation_steps=20,
            sample_stride=5,
            grid_size=12,
            color_weight=0.0,
        )
        color_settings = ParticleSimilaritySettings(
            simulation_steps=20,
            sample_stride=5,
            grid_size=12,
            color_weight=0.3,
        )

        dark_spatial = profile_genome(dark, self.config, spatial_only_settings, include_raster=False)
        light_spatial = profile_genome(light, self.config, spatial_only_settings, include_raster=False)
        dark_color = profile_genome(dark, self.config, color_settings, include_raster=False)
        light_color = profile_genome(light, self.config, color_settings, include_raster=False)

        self.assertEqual(histogram_distance(dark_spatial.histogram, light_spatial.histogram), 0.0)
        self.assertGreater(histogram_distance(dark_color.histogram, light_color.histogram), 0.0)

    def test_ssim_detects_particle_color_difference(self):
        dark = copy.deepcopy(self.target)
        light = copy.deepcopy(self.target)
        for output_key in self.config.genome_config.output_keys[3:6]:
            dark.nodes[output_key].bias = -30.0
            light.nodes[output_key].bias = 30.0

        dark_profile = profile_genome(dark, self.config, FAST_SIMILARITY_SETTINGS)
        light_profile = profile_genome(light, self.config, FAST_SIMILARITY_SETTINGS)

        self.assertGreater(ssim_distance(dark_profile.raster, light_profile.raster), 0.0)

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
        self.assertGreaterEqual(result.final_histogram_distance, 0.0)
        self.assertGreaterEqual(result.final_ssim_distance, 0.0)
        self.assertIn("distances", result.history[0])
        self.assertEqual(len(result.history[0]["distances"]), 9)
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)
        self.assertEqual(result.initial_design_space, "velocity")
        self.assertEqual(result.history[0]["design_space"], "velocity")

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

    def test_auto_selection_switches_when_half_max_steps_remain(self):
        distances = [0.8, 0.8, 0.8, 0.8]
        with patch.object(sps_selection, "create_sps_search", side_effect=self._create_fake_search), \
                patch.object(sps_selection, "select_nearest_candidate", side_effect=self._mock_selection_distances(distances)):
            result = run_auto_sps_selection(
                self.target,
                self.config,
                seed=31,
                threshold=-1.0,
                max_steps=4,
            )

        self.assertEqual(result.initial_design_space, "velocity")
        self.assertEqual(result.final_design_space, "color")
        self.assertEqual(result.switch_step, 2)
        self.assertEqual(result.switch_reason, "half_max_steps_remaining")
        self.assertEqual(result.history[1]["switch_to_design_space"], "color")
        self.assertEqual(result.history[2]["design_space"], "color")

    def test_auto_selection_switches_when_distance_reaches_half_initial_distance(self):
        distances = [0.8, 0.3, 0.3, 0.3]
        with patch.object(sps_selection, "create_sps_search", side_effect=self._create_fake_search), \
                patch.object(sps_selection, "select_nearest_candidate", side_effect=self._mock_selection_distances(distances)):
            result = run_auto_sps_selection(
                self.target,
                self.config,
                seed=37,
                threshold=-1.0,
                max_steps=4,
            )

        self.assertEqual(result.initial_best_distance, 0.8)
        self.assertEqual(result.final_design_space, "color")
        self.assertEqual(result.switch_step, 2)
        self.assertEqual(result.switch_reason, "half_initial_distance")

    def test_auto_selection_switches_at_most_once(self):
        distances = [0.8, 0.3, 0.2, 0.1, 0.05]
        with patch.object(sps_selection, "create_sps_search", side_effect=self._create_fake_search), \
                patch.object(sps_selection, "select_nearest_candidate", side_effect=self._mock_selection_distances(distances)):
            result = run_auto_sps_selection(
                self.target,
                self.config,
                seed=41,
                threshold=-1.0,
                max_steps=5,
            )

        switch_records = [record for record in result.history if "switch_to_design_space" in record]
        self.assertEqual(len(switch_records), 1)
        self.assertEqual(result.final_design_space, "color")


class AutoSpsReportTest(unittest.TestCase):
    def test_compact_history_keeps_every_five_steps_and_final_step(self):
        history = [{"step": step} for step in range(1, 13)]

        compact = _compact_history(history, interval=5)

        self.assertEqual([record["step"] for record in compact], [5, 10, 12])

    def test_run_report_includes_design_space_switch_metadata(self):
        config = load_particle_config(CONFIG_PATH)
        factory = ParticleGenomeFactory(config, seed=43)
        genome = factory.create_random_genome()
        result = SimpleNamespace(
            final_genome=genome,
            steps=2,
            stop_reason="max_steps",
            elapsed_seconds=0.1,
            final_distance=0.2,
            final_histogram_distance=0.2,
            final_ssim_distance=0.0,
            initial_design_space="velocity",
            final_design_space="color",
            switch_step=2,
            switch_reason="half_max_steps_remaining",
            initial_best_distance=0.8,
            history=[
                {"step": 1, "design_space": "velocity"},
                {
                    "step": 2,
                    "design_space": "velocity",
                    "switch_to_design_space": "color",
                    "switch_reason": "half_max_steps_remaining",
                },
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                output_dir=temp_dir,
                seed=43,
                target=os.path.join(temp_dir, "target.json"),
                config=CONFIG_PATH,
                threshold=-1.0,
                max_steps=2,
            )
            save_genome_target(args.target, genome, config, candidate_key=None, mode="SPS")
            _final_path, report_path = _write_run_outputs(args, config, result)
            with open(report_path, "r", encoding="utf-8") as file:
                report = json.load(file)

        self.assertEqual(report["initial_design_space"], "velocity")
        self.assertEqual(report["final_design_space"], "color")
        self.assertEqual(report["switch_step"], 2)
        self.assertEqual(report["switch_reason"], "half_max_steps_remaining")
        self.assertEqual(report["initial_best_distance"], 0.8)

    def test_auto_result_export_writes_target_metadata(self):
        config = load_particle_config(CONFIG_PATH)
        factory = ParticleGenomeFactory(config, seed=47)
        genome = factory.create_random_genome()

        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = _export_auto_result_target(
                genome,
                config,
                steps=7,
                final_distance=0.123456,
                target_dir=temp_dir,
            )
            with open(export_path, "r", encoding="utf-8") as file:
                data = json.load(file)

        self.assertTrue(os.path.basename(export_path).startswith("target_"))
        self.assertEqual(data["metadata"]["candidate_key"], None)
        self.assertEqual(data["metadata"]["mode"], "AUTO_SPS")
        self.assertEqual(data["metadata"]["generation"], 7)
        self.assertEqual(data["metadata"]["label"], "behavior_distance=0.123456")


if __name__ == "__main__":
    unittest.main()
