"""Thorough adversarial stress-testing and audit of Task 2 VLM components.

Covers:
1. MultimodalConnector:
   - Arbitrary batch sizes and dimensions (B, N, D) -> (B, N, D_out).
   - Connector calibration: scaling factor rule between connector output and text embedding L2 norms.
2. Curriculum sampling probabilities:
   - 10,000 draws for Stage 2 (target: 50% presence, 25% counting, 25% caption replay).
     Verifies 3 standard deviations and chi-squared test.
   - 10,000 draws for Stage 3 (target: 70% VQAv2, 10% presence, 10% counting, 10% caption replay).
     Verifies 3 standard deviations and frequency convergence.
3. Paired dataset split isolation:
   - Train, dev, and test sets are strictly image-disjoint.
   - Adversarial overlap injection detection.
   - Edge cases in presence and counting pair generation (no target objects, >4 objects, ambiguous counts).
4. VQA soft consensus scoring:
   - Leave-one-out consensus with 10, 3, 2, 1, and 0 annotators agreeing.
   - Text normalization invariance (case, whitespace, punctuation, articles).
   - Empty prediction behavior.
5. Bootstrap resampling:
   - 4,000 resamples at the PAIR level preserving paired correlation.
   - Zero variance, high variance, and negative mean behavior.
6. Visual token compression:
   - 2x2 average pooling from 196 to 49 tokens on batched inputs (B, 196, D) -> (B, 49, D).
   - Exact mathematical correspondence of spatial 2x2 blocks.
"""

from typing import Dict, List, Any
import unittest
import numpy as np

from common.metrics import (
    compute_vqa_accuracy,
    compute_grounding_exact_match,
    normalize_text_vqa,
    bootstrap_pair_metric_ci,
)
from task2_vlm.models import MultimodalConnector, VISUAL_DIM, LLM_DIM
from task2_vlm.dataset import (
    VLMExample,
    QuestionPair,
    CurriculumBatchSampler,
    generate_presence_pairs,
    generate_counting_pairs,
    verify_split_isolation,
    create_synthetic_task2_datasets,
)
from task2_vlm.compression import (
    compress_visual_tokens_np,
    compress_visual_tokens_torch,
    VisualTokenCompressor,
)


class TestMultimodalConnectorStress(unittest.TestCase):
    """Stress tests for MultimodalConnector module."""

    def test_dimensions_and_forward_arbitrary_batches(self):
        """Verify dimensions and forward propagation for arbitrary batch sizes and shapes."""
        connector = MultimodalConnector(in_dim=768, out_dim=1024, initial_scale=1.0)
        rng = np.random.default_rng(6304)

        test_shapes = [
            (1, 196, 768),       # Batch size 1
            (4, 196, 768),       # Batch size 4
            (16, 196, 768),      # Effective update batch 16
            (8, 49, 768),        # 49 compressed tokens
            (196, 768),          # Unbatched 2D sequence
            (2, 3, 49, 768),     # Multi-dimensional batch
        ]

        for shape in test_shapes:
            inp = rng.standard_normal(shape).astype(np.float32)
            out = connector(inp)
            expected_shape = (*shape[:-1], 1024)
            self.assertEqual(
                out.shape,
                expected_shape,
                f"Shape mismatch: expected {expected_shape}, got {out.shape} for input shape {shape}",
            )
            self.assertFalse(np.isnan(out).any(), f"NaNs detected in output for shape {shape}")
            self.assertFalse(np.isinf(out).any(), f"Infs detected in output for shape {shape}")
            self.assertTrue(np.any(out != 0.0), f"All zeros detected in output for shape {shape}")

    def test_connector_calibration_scaling_rule(self):
        """Verify connector calibration scaling factor rule:
        s = E_{w}[||e_text(w)||_2] / E_{x in D_train}[||MLP(x)||_2].

        After calibration, connector outputs must have mean L2 norm matching text embeddings.
        """
        connector = MultimodalConnector(in_dim=768, out_dim=1024, initial_scale=1.0)
        rng = np.random.default_rng(2026)

        # 32 visual feature samples of 196 tokens
        sample_vis = rng.standard_normal((32, 196, 768)).astype(np.float32)

        # Vocabulary of 2,000 text embeddings scaled with distinct norm
        target_scale = 3.75
        text_embeds = (rng.standard_normal((2000, 1024)) * target_scale).astype(np.float32)

        mean_text_norm = float(np.mean(np.linalg.norm(text_embeds, axis=-1)))

        # Uncalibrated forward pass
        unscaled_out = connector._forward_unscaled(sample_vis)
        mean_unscaled_vis_norm = float(np.mean(np.linalg.norm(unscaled_out, axis=-1)))

        # Calibrate connector
        computed_scale = connector.calibrate(sample_vis, text_embeds)
        expected_scale = mean_text_norm / mean_unscaled_vis_norm

        self.assertAlmostEqual(computed_scale, expected_scale, places=4)
        self.assertTrue(connector.calibration_info["calibrated"])
        self.assertAlmostEqual(connector.calibration_info["scale"], expected_scale, places=4)

        # Forward pass after calibration
        calibrated_out = connector(sample_vis)
        mean_calibrated_norm = float(np.mean(np.linalg.norm(calibrated_out, axis=-1)))

        # Verify mean L2 norm matches text embedding L2 norm exactly
        self.assertAlmostEqual(
            mean_calibrated_norm,
            mean_text_norm,
            places=3,
            msg="Calibrated connector output norm does not match target text embedding norm",
        )

    def test_connector_state_dict_serialization(self):
        """Verify state_dict export and reload preserves scale and weights."""
        c1 = MultimodalConnector(in_dim=768, out_dim=1024, initial_scale=2.5)
        c2 = MultimodalConnector(in_dim=768, out_dim=1024, initial_scale=1.0)

        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 10, 768)).astype(np.float32)

        out1 = c1(x)
        out2_before = c2(x)
        self.assertFalse(np.allclose(out1, out2_before))

        # Transfer state
        state = c1.state_dict()
        c2.load_state_dict(state)
        out2_after = c2(x)

        np.testing.assert_allclose(out1, out2_after, atol=1e-5)


class TestCurriculumSamplingStress(unittest.TestCase):
    """Stress tests and statistical frequency verification for CurriculumBatchSampler."""

    def setUp(self):
        self.datasets = create_synthetic_task2_datasets(seed=42)
        self.sampler = CurriculumBatchSampler(
            caption_pool=self.datasets["train_captions"],
            presence_pool=self.datasets["train_presence"],
            counting_pool=self.datasets["train_counting"],
            vqav2_pool=self.datasets["train_vqav2"],
            seed=2026,
        )

    def test_stage2_sampling_probabilities(self):
        """Simulate 10,000 random draws for Stage 2:
        Target: 50% presence, 25% counting, 25% caption replay.
        Verify empirical frequencies match targets within 3 standard deviations and pass chi-squared test.
        """
        n_draws = 10000
        counts: Dict[str, int] = {"presence": 0, "counting": 0, "caption": 0}

        for _ in range(n_draws):
            ex1, ex2 = self.sampler.draw_pair_or_two_captions(stage=2)
            self.assertEqual(ex1.example_type, ex2.example_type)
            counts[ex1.example_type] += 1

        total = sum(counts.values())
        self.assertEqual(total, n_draws)

        # Expected counts
        e_pres = n_draws * 0.50
        e_count = n_draws * 0.25
        e_cap = n_draws * 0.25

        # Standard deviations
        sigma_pres = np.sqrt(n_draws * 0.50 * 0.50)      # 50.0
        sigma_count = np.sqrt(n_draws * 0.25 * 0.75)     # 43.30
        sigma_cap = np.sqrt(n_draws * 0.25 * 0.75)       # 43.30

        # Verify within 3 standard deviations
        self.assertLessEqual(
            abs(counts["presence"] - e_pres),
            3.0 * sigma_pres,
            f"Stage 2 presence count {counts['presence']} outside 3 sigma [{e_pres - 3*sigma_pres}, {e_pres + 3*sigma_pres}]",
        )
        self.assertLessEqual(
            abs(counts["counting"] - e_count),
            3.0 * sigma_count,
            f"Stage 2 counting count {counts['counting']} outside 3 sigma [{e_count - 3*sigma_count}, {e_count + 3*sigma_count}]",
        )
        self.assertLessEqual(
            abs(counts["caption"] - e_cap),
            3.0 * sigma_cap,
            f"Stage 2 caption count {counts['caption']} outside 3 sigma [{e_cap - 3*sigma_cap}, {e_cap + 3*sigma_cap}]",
        )

        # Chi-squared test: sum((O - E)^2 / E) with df = 2
        chi2 = (
            (counts["presence"] - e_pres) ** 2 / e_pres
            + (counts["counting"] - e_count) ** 2 / e_count
            + (counts["caption"] - e_cap) ** 2 / e_cap
        )
        # Critical value for df=2 at p=0.001 is 13.82
        self.assertLess(chi2, 13.82, f"Stage 2 chi-squared test failed: chi2 = {chi2:.3f}")

    def test_stage3_sampling_probabilities(self):
        """Simulate 10,000 random draws for Stage 3:
        Target: 70% VQAv2, 10% presence, 10% counting, 10% caption replay.
        Verify empirical frequencies match targets within 3 standard deviations and pass chi-squared test.
        """
        n_draws = 10000
        counts: Dict[str, int] = {"vqav2": 0, "presence": 0, "counting": 0, "caption": 0}

        for _ in range(n_draws):
            ex1, ex2 = self.sampler.draw_pair_or_two_captions(stage=3)
            self.assertEqual(ex1.example_type, ex2.example_type)
            t = ex1.example_type
            if t.startswith("vqa_"):
                counts["vqav2"] += 1
            else:
                counts[t] += 1

        total = sum(counts.values())
        self.assertEqual(total, n_draws)

        e_vqa = n_draws * 0.70
        e_pres = n_draws * 0.10
        e_count = n_draws * 0.10
        e_cap = n_draws * 0.10

        sigma_vqa = np.sqrt(n_draws * 0.70 * 0.30)       # 45.83
        sigma_pres = np.sqrt(n_draws * 0.10 * 0.90)      # 30.0
        sigma_count = np.sqrt(n_draws * 0.10 * 0.90)     # 30.0
        sigma_cap = np.sqrt(n_draws * 0.10 * 0.90)       # 30.0

        self.assertLessEqual(
            abs(counts["vqav2"] - e_vqa),
            3.0 * sigma_vqa,
            f"Stage 3 VQAv2 count {counts['vqav2']} outside 3 sigma",
        )
        self.assertLessEqual(
            abs(counts["presence"] - e_pres),
            3.0 * sigma_pres,
            f"Stage 3 presence count {counts['presence']} outside 3 sigma",
        )
        self.assertLessEqual(
            abs(counts["counting"] - e_count),
            3.0 * sigma_count,
            f"Stage 3 counting count {counts['counting']} outside 3 sigma",
        )
        self.assertLessEqual(
            abs(counts["caption"] - e_cap),
            3.0 * sigma_cap,
            f"Stage 3 caption count {counts['caption']} outside 3 sigma",
        )

        # Chi-squared test with df = 3
        chi2 = (
            (counts["vqav2"] - e_vqa) ** 2 / e_vqa
            + (counts["presence"] - e_pres) ** 2 / e_pres
            + (counts["counting"] - e_count) ** 2 / e_count
            + (counts["caption"] - e_cap) ** 2 / e_cap
        )
        # Critical value for df=3 at p=0.001 is 16.27
        self.assertLess(chi2, 16.27, f"Stage 3 chi-squared test failed: chi2 = {chi2:.3f}")

    def test_effective_batch_composition(self):
        """Verify get_effective_batch returns exactly 16 examples (8 draws x 2)."""
        batch = self.sampler.get_effective_batch(stage=2)
        self.assertEqual(len(batch), 16)
        # Ensure pairwise grouping
        for i in range(0, 16, 2):
            self.assertEqual(batch[i].example_type, batch[i + 1].example_type)


class TestPairedDatasetSplitIsolationStress(unittest.TestCase):
    """Stress tests for split isolation and edge cases in pair generation."""

    def test_synthetic_dataset_split_isolation(self):
        """Verify train, dev, and test sets are strictly image-disjoint."""
        datasets = create_synthetic_task2_datasets(seed=6304)

        train_pool = (
            datasets["train_captions"]
            + datasets["train_presence"]
            + datasets["train_counting"]
            + datasets["train_vqav2"]
        )
        dev_pool = datasets["dev_captions"] + datasets["dev_grounding"] + datasets["dev_vqav2"]
        test_pool = datasets["test_grounding"] + datasets["test_vqav2"]

        is_disjoint, stats = verify_split_isolation(train_pool, dev_pool, test_pool)
        self.assertTrue(is_disjoint)
        self.assertEqual(stats["train_dev_overlap"], 0)
        self.assertEqual(stats["train_test_overlap"], 0)
        self.assertEqual(stats["dev_test_overlap"], 0)

    def test_adversarial_overlap_detection(self):
        """Verify verify_split_isolation detects contaminated overlapping image IDs."""
        ex_train = VLMExample("img_shared", "Q?", "A", "caption")
        ex_dev = VLMExample("img_shared", "Q?", "A", "caption")
        ex_test = VLMExample("img_clean", "Q?", "A", "caption")

        is_disjoint, stats = verify_split_isolation([ex_train], [ex_dev], [ex_test])
        self.assertFalse(is_disjoint)
        self.assertEqual(stats["train_dev_overlap"], 1)

    def test_presence_generation_edge_cases(self):
        """Test presence pair generation edge cases."""
        # 1. Empty inputs
        self.assertEqual(generate_presence_pairs({}, ["dog"], 10), [])
        self.assertEqual(generate_presence_pairs({"img_1": ["dog"]}, [], 10), [])
        self.assertEqual(generate_presence_pairs({"img_1": ["dog"]}, ["dog"], 0), [])

        # 2. Images with no target objects (all images lack 'cat')
        annos_no_cat = {
            "img_1": ["dog", "car"],
            "img_2": ["bicycle"],
            "img_3": [],
        }
        # Pos list is empty -> cannot generate pairs
        pairs = generate_presence_pairs(annos_no_cat, ["cat"], 5)
        self.assertEqual(len(pairs), 0)

        # 3. Target object present in every single image (no negative examples)
        annos_all_dog = {
            "img_1": ["dog"],
            "img_2": ["dog"],
            "img_3": ["dog"],
        }
        pairs = generate_presence_pairs(annos_all_dog, ["dog"], 5)
        self.assertEqual(len(pairs), 0)

        # 4. Valid mixed presence
        annos_mixed = {
            "img_pos": ["dog", "chair"],
            "img_neg": ["car"],
        }
        pairs = generate_presence_pairs(annos_mixed, ["dog"], 1)
        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        self.assertEqual(pair.example_a.target, "yes")
        self.assertEqual(pair.example_b.target, "no")
        self.assertEqual(pair.example_a.image_id, "img_pos")
        self.assertEqual(pair.example_b.image_id, "img_neg")

    def test_counting_generation_edge_cases(self):
        """Test counting pair generation edge cases: >4 objects, 0 objects, ambiguous counts."""
        # 1. Empty inputs
        self.assertEqual(generate_counting_pairs({}, ["dog"], 10), [])
        self.assertEqual(generate_counting_pairs({"img_1": {"dog": 2}}, [], 10), [])
        self.assertEqual(generate_counting_pairs({"img_1": {"dog": 2}}, ["dog"], 0), [])

        # 2. Images with > 4 objects (e.g. 5, 12, 100) or 0 objects
        annos_out_of_bounds = {
            "img_zero": {"dog": 0},
            "img_five": {"dog": 5},
            "img_twelve": {"dog": 12},
            "img_negative": {"dog": -1},
        }
        # None of these counts fall in [1, 4], so 0 pairs should be generated
        pairs = generate_counting_pairs(annos_out_of_bounds, ["dog"], 5)
        self.assertEqual(len(pairs), 0)

        # 3. Only 1 valid count available (e.g. all valid images have exactly 2 dogs)
        annos_single_count = {
            "img_1": {"dog": 2},
            "img_2": {"dog": 2},
            "img_3": {"dog": 2},
        }
        # Cannot form pair with distinct counts
        pairs = generate_counting_pairs(annos_single_count, ["dog"], 5)
        self.assertEqual(len(pairs), 0)

        # 4. Valid counts in [1, 4] with >4 objects present in other images
        annos_mixed = {
            "img_1": {"dog": 1},
            "img_2": {"dog": 3},
            "img_crowd": {"dog": 8},  # Should be ignored
        }
        pairs = generate_counting_pairs(annos_mixed, ["dog"], 1)
        self.assertEqual(len(pairs), 1)
        p = pairs[0]
        self.assertIn(p.example_a.target, ["1", "3"])
        self.assertIn(p.example_b.target, ["1", "3"])
        self.assertNotEqual(p.example_a.target, p.example_b.target)
        self.assertNotEqual(p.example_a.image_id, "img_crowd")
        self.assertNotEqual(p.example_b.image_id, "img_crowd")


class TestVQASoftConsensusScoringStress(unittest.TestCase):
    """Stress tests for official VQA soft consensus scoring."""

    def test_consensus_levels(self):
        """Verify consensus accuracy:
        s(a_hat) = 1/10 sum_{i=1}^10 min(1, 1/3 sum_{j != i} 1[a_hat == a_j]).
        """
        # All 10 annotators agree
        refs_10 = ["yes"] * 10
        self.assertAlmostEqual(compute_vqa_accuracy("yes", refs_10), 1.0)
        self.assertAlmostEqual(compute_vqa_accuracy("no", refs_10), 0.0)

        # Exactly 3 annotators agree:
        # 3 matching: other_matches = 2 -> 2/3
        # 7 non-matching: other_matches = 3 -> 1.0
        # Score = (3 * (2/3) + 7 * 1.0) / 10 = (2 + 7) / 10 = 0.9
        refs_3 = ["white", "white", "white", "black", "black", "blue", "red", "yellow", "green", "pink"]
        self.assertAlmostEqual(compute_vqa_accuracy("white", refs_3), 0.9, places=5)

        # Exactly 2 annotators agree:
        # 2 matching: other_matches = 1 -> 1/3
        # 8 non-matching: other_matches = 2 -> 2/3
        # Score = (2 * (1/3) + 8 * (2/3)) / 10 = (2/3 + 16/3) / 10 = 18/30 = 0.6
        refs_2 = ["car", "car", "bus", "truck", "bike", "van", "train", "boat", "plane", "scooter"]
        self.assertAlmostEqual(compute_vqa_accuracy("car", refs_2), 0.6, places=5)

        # Exactly 1 annotator agrees:
        # 1 matching: other_matches = 0 -> 0.0
        # 9 non-matching: other_matches = 1 -> 1/3
        # Score = (1 * 0 + 9 * (1/3)) / 10 = 3/10 = 0.3
        refs_1 = ["cat"] + ["dog"] * 9
        self.assertAlmostEqual(compute_vqa_accuracy("cat", refs_1), 0.3, places=5)

        # Exactly 0 annotators agree
        refs_0 = ["dog"] * 10
        self.assertAlmostEqual(compute_vqa_accuracy("cat", refs_0), 0.0, places=5)

        # 4 annotators agree:
        # 4 matching: other_matches = 3 -> 1.0
        # 6 non-matching: other_matches = 4 -> 1.0
        # Score = 1.0
        refs_4 = ["yes"] * 4 + ["no"] * 6
        self.assertAlmostEqual(compute_vqa_accuracy("yes", refs_4), 1.0, places=5)

    def test_text_normalization_invariance(self):
        """Verify capitalization, whitespace, punctuation, and article invariance."""
        refs = ["Blue sky."] * 10

        variations = [
            "blue sky",
            "Blue sky",
            "BLUE SKY",
            "  blue   sky  ",
            "the blue sky",
            "a blue sky",
            "blue sky!",
            "blue sky?",
            "blue, sky.",
        ]

        for var in variations:
            acc = compute_vqa_accuracy(var, refs)
            self.assertAlmostEqual(
                acc,
                1.0,
                places=5,
                msg=f"Failed invariance for variation: '{var}'",
            )

    def test_empty_and_degenerate_predictions(self):
        """Verify empty and whitespace-only predictions yield 0.0 score."""
        refs = ["sunny"] * 10
        degenerate_inputs = ["", "   ", "\t\n", "...", "???", "!"]

        for deg in degenerate_inputs:
            acc = compute_vqa_accuracy(deg, refs)
            self.assertEqual(
                acc,
                0.0,
                msg=f"Degenerate input '{deg}' did not yield 0.0 accuracy",
            )

        # Empty references sequence
        self.assertEqual(compute_vqa_accuracy("sunny", []), 0.0)


class TestBootstrapResamplingStress(unittest.TestCase):
    """Stress tests for pair-level bootstrap confidence interval calculation."""

    def test_pair_level_resampling_4000(self):
        """Verify 4,000 resamples are drawn at the PAIR level preserving paired correlation."""
        rng = np.random.default_rng(6304)
        n_pairs = 96
        # True mean = 0.35, std = 0.15
        pair_diffs = rng.normal(loc=0.35, scale=0.15, size=n_pairs)

        mean, lower, upper = bootstrap_pair_metric_ci(
            pair_diffs, n_resamples=4000, seed=6304, ci_level=0.95
        )

        empirical_mean = float(np.mean(pair_diffs))
        self.assertAlmostEqual(mean, empirical_mean, places=5)
        self.assertLess(lower, mean)
        self.assertGreater(upper, mean)
        # 95% CI contains true mean
        self.assertLessEqual(lower, 0.35)
        self.assertGreaterEqual(upper, 0.35)

    def test_zero_variance_behavior(self):
        """Verify bootstrap behavior when pair differences have zero variance."""
        constant_diffs = [0.42] * 64
        mean, lower, upper = bootstrap_pair_metric_ci(constant_diffs, n_resamples=1000)

        self.assertAlmostEqual(mean, 0.42, places=6)
        self.assertAlmostEqual(lower, 0.42, places=6)
        self.assertAlmostEqual(upper, 0.42, places=6)

        zero_diffs = [0.0] * 64
        m0, l0, u0 = bootstrap_pair_metric_ci(zero_diffs, n_resamples=1000)
        self.assertEqual(m0, 0.0)
        self.assertEqual(l0, 0.0)
        self.assertEqual(u0, 0.0)

    def test_negative_mean_behavior(self):
        """Verify bootstrap handles strictly negative paired differences (e.g. negative advantage)."""
        neg_diffs = [-0.85, -0.90, -0.75, -0.80, -0.95] * 10
        mean, lower, upper = bootstrap_pair_metric_ci(neg_diffs, n_resamples=1000)

        self.assertLess(mean, 0.0)
        self.assertLess(upper, 0.0)
        self.assertLessEqual(lower, mean)
        self.assertGreaterEqual(upper, mean)

    def test_high_variance_behavior(self):
        """Verify bootstrap handles high-variance differences."""
        rng = np.random.default_rng(999)
        high_var = rng.uniform(-10.0, 10.0, size=100)
        mean, lower, upper = bootstrap_pair_metric_ci(high_var, n_resamples=1000)

        self.assertLess(lower, mean)
        self.assertGreater(upper, mean)
        self.assertGreater(upper - lower, 1.0)

    def test_boundary_inputs(self):
        """Verify empty and single-element inputs."""
        self.assertEqual(bootstrap_pair_metric_ci([]), (0.0, 0.0, 0.0))
        m, l, u = bootstrap_pair_metric_ci([3.14])
        self.assertAlmostEqual(m, 3.14)
        self.assertAlmostEqual(l, 3.14)
        self.assertAlmostEqual(u, 3.14)


class TestVisualTokenCompressionStress(unittest.TestCase):
    """Stress tests for visual token spatial 2x2 average pooling (196 -> 49 tokens)."""

    def test_batched_visual_compression_shapes(self):
        """Verify 2x2 average pooling on batched inputs (B, 196, D) -> (B, 49, D)."""
        test_configs = [
            (1, 196, 768),
            (4, 196, 768),
            (16, 196, 768),
            (2, 196, 512),
            (1, 196, 1024),
            (196, 768),       # 2D unbatched
        ]
        compressor = VisualTokenCompressor(grid_size=14, pool_size=2)
        rng = np.random.default_rng(6304)

        for shape in test_configs:
            x = rng.standard_normal(shape).astype(np.float32)
            out_np = compress_visual_tokens_np(x)
            out_mod = compressor(x)

            expected_shape = (*shape[:-2], 49, shape[-1])
            self.assertEqual(out_np.shape, expected_shape)
            self.assertEqual(out_mod.shape, expected_shape)
            np.testing.assert_allclose(out_np, out_mod, atol=1e-7)

    def test_spatial_pooling_mathematical_precision(self):
        """Verify each of the 49 output tokens matches the exact 2x2 spatial average of 4 tokens."""
        rng = np.random.default_rng(42)
        d = 64
        # (14, 14, d) grid
        grid = rng.standard_normal((14, 14, d)).astype(np.float32)
        tokens_196 = grid.reshape(1, 196, d)

        tokens_49 = compress_visual_tokens_np(tokens_196)[0]  # Shape: (49, d)

        # Check all 7x7 = 49 pooled positions manually
        for r_out in range(7):
            for c_out in range(7):
                out_idx = r_out * 7 + c_out
                r_in = r_out * 2
                c_in = c_out * 2
                expected_avg = (
                    grid[r_in, c_in]
                    + grid[r_in, c_in + 1]
                    + grid[r_in + 1, c_in]
                    + grid[r_in + 1, c_in + 1]
                ) / 4.0
                np.testing.assert_allclose(
                    tokens_49[out_idx],
                    expected_avg,
                    atol=1e-6,
                    err_msg=f"Mismatch at pooled token ({r_out}, {c_out}) / index {out_idx}",
                )

    def test_invalid_visual_grid_rejection(self):
        """Verify ValueError is raised if input does not have grid_size^2 tokens."""
        bad_input = np.zeros((2, 100, 768), dtype=np.float32)
        with self.assertRaises(ValueError):
            compress_visual_tokens_np(bad_input, grid_size=14, pool_size=2)


if __name__ == "__main__":
    unittest.main()
