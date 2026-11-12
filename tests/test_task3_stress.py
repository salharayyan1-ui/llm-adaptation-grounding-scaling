"""Adversarial stress-testing and audit of task3_test_time/ and common/ utilities."""

import unittest
import numpy as np

from common.normalization import extract_boxed_answer, normalize_math_answer, normalize_mathvista_answer
from task3_test_time.data_loader import RolloutRecord, QuestionRecord
from task3_test_time.equivalence import (
    normalize_math_candidate,
    normalize_mathvista_candidate,
    are_answers_equivalent,
    group_answers,
    grade_answer,
    NO_ANSWER,
)
from task3_test_time.scoring import (
    compute_mean_logp,
    compute_self_certainty,
    compute_visual_engagement,
    compute_visconf_scores,
    compute_average_ranks,
    select_highest_score,
    select_self_consistency,
    select_rank_weighted_consensus,
    evaluate_prefix_methods,
)
from task3_test_time.signals import (
    compute_confidence_vision_correlation,
    evaluate_tokenwise_fusion_ablation,
)
from task3_test_time.adaptive import (
    check_agreement_condition,
    check_quality_aware_condition,
    simulate_adaptive_stopping,
    precompute_order_checkpoints,
    evaluate_order_fast,
    TAU_GRID,
    GAMMA_GRID,
)
from task3_test_time.opportunity import (
    evaluate_question_opportunity,
    evaluate_dataset_opportunity,
    generate_rollout_permutations,
)


class TestTask3StressTieBreaking(unittest.TestCase):
    """Adversarial stress tests for tie-breaking mechanics across consensus rules."""

    def test_rwc_vote_weight_ties_resolved_by_earliest_rollout(self):
        """When candidate answers tie in vote weights, verify resolution by earliest rollout in permutation order."""
        # 2 candidates with equal quality scores -> equal RWC weights
        r_a = RolloutRecord(dataset="math", sample_id="q0", rollout_id=0, extracted_answer="Ans_A", length=5)
        r_b = RolloutRecord(dataset="math", sample_id="q0", rollout_id=1, extracted_answer="Ans_B", length=5)

        # Order 1: [A, B]
        idx1, ans1, weights1 = select_rank_weighted_consensus([r_a, r_b], scores=[10.0, 10.0], dataset="math")
        self.assertEqual(idx1, 0)
        self.assertEqual(ans1, "Ans_A")
        self.assertAlmostEqual(weights1["Ans_A"], weights1["Ans_B"])

        # Order 2: [B, A]
        idx2, ans2, weights2 = select_rank_weighted_consensus([r_b, r_a], scores=[10.0, 10.0], dataset="math")
        self.assertEqual(idx2, 0)
        self.assertEqual(ans2, "Ans_B")
        self.assertAlmostEqual(weights2["Ans_A"], weights2["Ans_B"])

        # 4 candidates with interleaved ties: [A, B, A, B]
        # Candidates 0 & 3: score 5.0 -> ranks 3.5 -> weight (4-3.5)/3 = 0.5/3
        # Candidates 1 & 2: score 10.0 -> ranks 1.5 -> weight (4-1.5)/3 = 2.5/3
        # Weight for A = (0.5 + 2.5)/3 = 1.0. Weight for B = (2.5 + 0.5)/3 = 1.0. Exactly tied!
        cands_ab = [
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=0, extracted_answer="Ans_A", length=5),
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=1, extracted_answer="Ans_B", length=5),
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=2, extracted_answer="Ans_A", length=5),
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=3, extracted_answer="Ans_B", length=5),
        ]
        scores = [5.0, 10.0, 10.0, 5.0]
        idx_ab, ans_ab, w_ab = select_rank_weighted_consensus(cands_ab, scores=scores, dataset="math")
        self.assertAlmostEqual(w_ab["Ans_A"], w_ab["Ans_B"])
        # Earliest supporting rollout is candidate 0 ("Ans_A")
        self.assertEqual(idx_ab, 0)
        self.assertEqual(ans_ab, "Ans_A")

        # Reverse the order: [B, A, B, A]
        cands_ba = [cands_ab[1], cands_ab[0], cands_ab[3], cands_ab[2]]
        scores_ba = [10.0, 5.0, 5.0, 10.0]
        idx_ba, ans_ba, w_ba = select_rank_weighted_consensus(cands_ba, scores=scores_ba, dataset="math")
        self.assertAlmostEqual(w_ba["Ans_A"], w_ba["Ans_B"])
        # Earliest supporting rollout is now candidate 0 ("Ans_B")
        self.assertEqual(idx_ba, 0)
        self.assertEqual(ans_ba, "Ans_B")

    def test_self_consistency_ties_resolved_by_earliest_rollout(self):
        """When answer vote counts tie in Self-Consistency, verify earliest supporting rollout wins."""
        r0 = RolloutRecord(dataset="math", sample_id="q0", rollout_id=0, extracted_answer="Ans_A")
        r1 = RolloutRecord(dataset="math", sample_id="q0", rollout_id=1, extracted_answer="Ans_B")
        r2 = RolloutRecord(dataset="math", sample_id="q0", rollout_id=2, extracted_answer="Ans_C")

        # Two-way tie (1 vote each)
        idx_ab, ans_ab = select_self_consistency([r0, r1], dataset="math")
        self.assertEqual(idx_ab, 0)
        self.assertEqual(ans_ab, "Ans_A")

        idx_ba, ans_ba = select_self_consistency([r1, r0], dataset="math")
        self.assertEqual(idx_ba, 0)
        self.assertEqual(ans_ba, "Ans_B")

        # Multi-group tie with non-tied candidate: [C, A, B, A, B] -> C has 1, A has 2, B has 2
        cands = [r2, r0, r1, r0, r1]
        idx_sc, ans_sc = select_self_consistency(cands, dataset="math")
        # Tied max votes is 2. Earliest rollout supporting A is idx 1; supporting B is idx 2.
        self.assertEqual(idx_sc, 1)
        self.assertEqual(ans_sc, "Ans_A")

    def test_average_ranks_on_tied_quality_scores(self):
        """When quality scores Q_i tie, verify exact average ranks are assigned in RWC."""
        # Ties for rank 1 and 2: [10.0, 10.0, 5.0, 1.0] -> ranks [1.5, 1.5, 3.0, 4.0]
        r1 = compute_average_ranks([10.0, 10.0, 5.0, 1.0])
        np.testing.assert_allclose(r1, [1.5, 1.5, 3.0, 4.0])

        # Ties for rank 2 and 3: [10.0, 5.0, 5.0, 1.0] -> ranks [1.0, 2.5, 2.5, 4.0]
        r2 = compute_average_ranks([10.0, 5.0, 5.0, 1.0])
        np.testing.assert_allclose(r2, [1.0, 2.5, 2.5, 4.0])

        # 3-way tie for rank 1, 2, 3: [8.0, 8.0, 8.0, 2.0] -> (1+2+3)/3 = 2.0
        r3 = compute_average_ranks([8.0, 8.0, 8.0, 2.0])
        np.testing.assert_allclose(r3, [2.0, 2.0, 2.0, 4.0])

        # All scores tied: [4.0, 4.0, 4.0, 4.0] -> (1+2+3+4)/4 = 2.5
        r4 = compute_average_ranks([4.0, 4.0, 4.0, 4.0])
        np.testing.assert_allclose(r4, [2.5, 2.5, 2.5, 2.5])

        # Negative scores (e.g. logp) with tie at the top: [-0.2, -0.2, -0.8] -> [1.5, 1.5, 3.0]
        r5 = compute_average_ranks([-0.2, -0.2, -0.8])
        np.testing.assert_allclose(r5, [1.5, 1.5, 3.0])

        # Single score: [3.14] -> [1.0]
        r_single = compute_average_ranks([3.14])
        np.testing.assert_allclose(r_single, [1.0])

        # Empty scores -> []
        r_empty = compute_average_ranks([])
        self.assertEqual(len(r_empty), 0)

    def test_candidate_budget_n_equals_one_every_rule(self):
        """At candidate budget N=1, verify every scoring and consensus rule returns the single candidate."""
        # 1. MATH dataset candidate
        r_math = RolloutRecord(
            dataset="math",
            sample_id="q_math_single",
            rollout_id=0,
            extracted_answer="42",
            length=4,
            logp=[-0.15, -0.25, -0.05, -0.10],
            kl_u_p=[3.5, 4.0, 4.5, 3.8],
        )

        # Direct selectors
        idx_sc, ans_sc = select_self_consistency([r_math], dataset="math")
        self.assertEqual(idx_sc, 0)
        self.assertEqual(ans_sc, "42")

        idx_rwc, ans_rwc, w_rwc = select_rank_weighted_consensus([r_math], [4.0], dataset="math")
        self.assertEqual(idx_rwc, 0)
        self.assertEqual(ans_rwc, "42")
        self.assertEqual(w_rwc, {"42": 1.0})

        idx_hi, ans_hi = select_highest_score([r_math], [4.0])
        self.assertEqual(idx_hi, 0)
        self.assertEqual(ans_hi, "42")

        # Full prefix methods evaluation on MATH
        prefix_math = evaluate_prefix_methods([r_math], dataset="math")
        for method_name, chosen_ans in prefix_math.items():
            self.assertEqual(
                chosen_ans,
                "42",
                f"Rule {method_name} failed to return single candidate answer '42' at N=1",
            )

        # 2. MathVista dataset candidate
        r_vista = RolloutRecord(
            dataset="mathvista",
            sample_id="q_vista_single",
            rollout_id=0,
            extracted_answer="Option C",
            length=3,
            logp=[-0.1, -0.2, -0.3],
            kl_u_p=[2.0, 2.5, 3.0],
            att_image=[10.0, 15.0, 20.0],
            att_prompt=[5.0, 5.0, 5.0],
            att_generated=[5.0, 5.0, 5.0],
        )

        prefix_vista = evaluate_prefix_methods([r_vista], dataset="mathvista")
        expected_rules = {
            "first_ans",
            "self_consistency",
            "highest_logp",
            "highest_self_certainty",
            "self_certainty_rwc",
            "highest_vision",
            "vision_rwc",
            "highest_visconf",
            "visconf_rwc",
        }
        self.assertEqual(set(prefix_vista.keys()), expected_rules)
        for method_name, chosen_ans in prefix_vista.items():
            self.assertEqual(
                chosen_ans,
                "Option C",
                f"MathVista rule {method_name} failed to return single candidate answer at N=1",
            )


class TestVisConfCalibrationRobustness(unittest.TestCase):
    """Stress tests for VisConf calibration, denominator clamping, extreme values, and prefix isolation."""

    def test_flat_distribution_zero_division_guard(self):
        """Flat KL distribution where Q.05 == Q.95: clamp max(..., 1e-8) prevents zero-division."""
        # 3 rollouts all having identical KL values
        flat_kl = [5.0, 5.0, 5.0, 5.0]
        cands = [
            RolloutRecord(
                dataset="mathvista",
                sample_id="qv_flat",
                rollout_id=i,
                length=4,
                kl_u_p=list(flat_kl),
                att_image=[10.0, 20.0, 30.0, 40.0],
                att_prompt=[5.0, 5.0, 5.0, 5.0],
                att_generated=[5.0, 5.0, 5.0, 5.0],
            )
            for i in range(3)
        ]

        v_scores, c_scores, z_scores = compute_visconf_scores(cands)
        self.assertEqual(len(c_scores), 3)

        # Check no NaN, Inf, or runtime error occurred
        for i in range(3):
            self.assertFalse(np.isnan(c_scores[i]), f"Candidate {i} has NaN c_score")
            self.assertFalse(np.isinf(c_scores[i]), f"Candidate {i} has Inf c_score")
            self.assertFalse(np.isnan(z_scores[i]), f"Candidate {i} has NaN z_score")
            self.assertFalse(np.isinf(z_scores[i]), f"Candidate {i} has Inf z_score")
            # Since k_t == q05 == q95, (k_t - q05) / 1e-8 == 0.0 -> clipped to 0.0
            self.assertAlmostEqual(c_scores[i], 0.0, places=7)
            self.assertAlmostEqual(z_scores[i], 0.0, places=7)

    def test_extreme_kl_values_clipped_to_unit_interval(self):
        """Extreme negative and positive KL values are strictly clipped to [0, 1]."""
        extreme_cands = [
            RolloutRecord(
                dataset="mathvista",
                sample_id="qv_ext",
                rollout_id=0,
                length=4,
                kl_u_p=[-1e9, -1e6, 1e6, 1e9],
                att_image=[1.0, 1.0, 1.0, 1.0],
            ),
            RolloutRecord(
                dataset="mathvista",
                sample_id="qv_ext",
                rollout_id=1,
                length=4,
                kl_u_p=[0.0, 1.0, 2.0, 3.0],
                att_image=[1.0, 1.0, 1.0, 1.0],
            ),
        ]

        v_scores, c_scores, z_scores = compute_visconf_scores(extreme_cands)
        for i, c_val in enumerate(c_scores):
            self.assertGreaterEqual(c_val, 0.0, f"Candidate {i} c_score < 0.0: {c_val}")
            self.assertLessEqual(c_val, 1.0, f"Candidate {i} c_score > 1.0: {c_val}")

        for i, z_val in enumerate(z_scores):
            self.assertGreaterEqual(z_val, 0.0, f"Candidate {i} z_score < 0.0: {z_val}")
            self.assertLessEqual(z_val, 1.0, f"Candidate {i} z_score > 1.0: {z_val}")

    def test_prefix_isolation_no_future_leakage(self):
        """Percentiles Q.05 and Q.95 are strictly computed on revealed candidates without future rollout leakage."""
        r0 = RolloutRecord(
            dataset="mathvista", sample_id="q_iso", rollout_id=0, length=4,
            kl_u_p=[1.0, 2.0, 3.0, 4.0], att_image=[1.0]*4
        )
        r1 = RolloutRecord(
            dataset="mathvista", sample_id="q_iso", rollout_id=1, length=4,
            kl_u_p=[2.0, 3.0, 4.0, 5.0], att_image=[1.0]*4
        )
        # Future candidate with huge KL that alters percentiles if included
        r_future = RolloutRecord(
            dataset="mathvista", sample_id="q_iso", rollout_id=2, length=4,
            kl_u_p=[1000.0, 2000.0, 3000.0, 4000.0], att_image=[1.0]*4
        )

        # Prefix G of size 2
        _, c_prefix, z_prefix = compute_visconf_scores([r0, r1])

        # If future rollout leaked into candidate set G
        _, c_leaked, z_leaked = compute_visconf_scores([r0, r1, r_future])

        # Calibration on prefix [r0, r1] must differ from leaked [r0, r1, r_future]
        self.assertNotEqual(c_prefix[0], c_leaked[0])
        self.assertNotEqual(z_prefix[0], z_leaked[0])

        # Verify exact mathematical calibration of prefix G:
        kl_g = np.array([1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0])
        expected_q05 = float(np.percentile(kl_g, 5, method="linear"))
        expected_q95 = float(np.percentile(kl_g, 95, method="linear"))
        scale = max(expected_q95 - expected_q05, 1e-8)
        expected_c0 = float(np.mean(np.clip((np.array(r0.kl_u_p) - expected_q05) / scale, 0.0, 1.0)))
        self.assertAlmostEqual(c_prefix[0], expected_c0, places=6)


class TestTokenwiseCovarianceIdentity(unittest.TestCase):
    """Stress tests for the tokenwise covariance identity: Z_token - Z == Cov_t(c_t, v_t)."""

    def test_covariance_identity_across_diverse_lengths(self):
        """Verify Z_token - Z == Cov_t(c_t, v_t) for sequence lengths T in {10, 50, 200, 1024}."""
        lengths = (10, 50, 200, 1024)

        for T in lengths:
            rng = np.random.default_rng(6304 + T)
            kl = rng.uniform(0.5, 12.0, size=T).tolist()
            att_img = rng.uniform(1.0, 50.0, size=T).tolist()
            att_pmt = rng.uniform(1.0, 50.0, size=T).tolist()
            att_gen = rng.uniform(1.0, 50.0, size=T).tolist()

            r = RolloutRecord(
                dataset="mathvista",
                sample_id=f"q_cov_{T}",
                rollout_id=0,
                length=T,
                kl_u_p=kl,
                att_image=att_img,
                att_prompt=att_pmt,
                att_generated=att_gen,
            )

            corr, z_token, cov_val = compute_confidence_vision_correlation(r)

            # Trajectory visual engagement V
            v_val = compute_visual_engagement(r)

            # Trajectory calibrated confidence C
            k_arr = np.asarray(kl, dtype=np.float64)
            q05 = float(np.percentile(k_arr, 5, method="linear"))
            q95 = float(np.percentile(k_arr, 95, method="linear"))
            scale = max(q95 - q05, 1e-8)
            c_arr = np.clip((k_arr - q05) / scale, 0.0, 1.0)
            c_val = float(np.mean(c_arr))

            # Trajectory product Z
            z_trajectory = v_val * c_val

            # Identity verification
            diff = z_token - z_trajectory
            self.assertAlmostEqual(
                diff,
                cov_val,
                places=10,
                msg=f"Covariance identity failed at T={T}: diff={diff}, cov={cov_val}",
            )

    def test_covariance_identity_edge_cases(self):
        """Verify covariance identity under constant signals and extreme correlations."""
        # 1. Constant visual engagement -> Covariance must be 0
        T = 20
        r_const_v = RolloutRecord(
            dataset="mathvista", sample_id="q_edge_1", rollout_id=0, length=T,
            kl_u_p=list(np.linspace(1.0, 5.0, T)),
            att_image=[10.0] * T,
            att_prompt=[10.0] * T,
            att_generated=[10.0] * T,
        )
        corr, z_tok, cov = compute_confidence_vision_correlation(r_const_v)
        v = compute_visual_engagement(r_const_v)
        k_arr = np.asarray(r_const_v.kl_u_p, dtype=np.float64)
        c_arr = np.clip((k_arr - np.percentile(k_arr, 5)) / (np.percentile(k_arr, 95) - np.percentile(k_arr, 5)), 0, 1)
        z = v * float(np.mean(c_arr))
        self.assertAlmostEqual(cov, 0.0, places=10)
        self.assertAlmostEqual(z_tok - z, 0.0, places=10)

        # 2. Perfect positive correlation between c_t and v_t
        r_pos = RolloutRecord(
            dataset="mathvista", sample_id="q_edge_2", rollout_id=0, length=T,
            kl_u_p=list(np.linspace(1.0, 10.0, T)),
            att_image=list(np.linspace(1.0, 10.0, T)),
            att_prompt=[0.0] * T,
            att_generated=[1.0] * T,
        )
        corr_pos, z_tok_pos, cov_pos = compute_confidence_vision_correlation(r_pos)
        self.assertGreater(cov_pos, 0.0)
        self.assertGreater(corr_pos, 0.0)


class TestLatexBoxedAnswerParser(unittest.TestCase):
    """Stress tests for balanced-brace parsing of LaTeX boxed answers."""

    def test_deeply_nested_braces(self):
        """Deeply nested braces, e.g. fractions of square roots."""
        text1 = r"After algebraic simplification, \boxed{\frac{\sqrt{x^2+1}}{2}}."
        self.assertEqual(extract_boxed_answer(text1), r"\frac{\sqrt{x^2+1}}{2}")

        text2 = r"Solution: \boxed{\frac{1}{1 + \frac{1}{1 + \frac{1}{x}}}} is the continued fraction."
        self.assertEqual(extract_boxed_answer(text2), r"\frac{1}{1 + \frac{1}{1 + \frac{1}{x}}}")

    def test_matrix_environments_and_sub_superscripts(self):
        """Matrix environments, curly braces in environments, nested sub/superscripts."""
        text_matrix = r"The transformation matrix is \boxed{\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}}."
        self.assertEqual(extract_boxed_answer(text_matrix), r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}")

        text_sub_sup = r"The eigenvalue is \boxed{x_{i_{1}}^{y^{2_{k}}}}."
        self.assertEqual(extract_boxed_answer(text_sub_sup), r"x_{i_{1}}^{y^{2_{k}}}")

        text_bmatrix = r"\boxed{\begin{bmatrix} a & b \\ c & d \end{bmatrix}}"
        self.assertEqual(extract_boxed_answer(text_bmatrix), r"\begin{bmatrix} a & b \\ c & d \end{bmatrix}")

    def test_multiple_boxed_expressions_strictly_last(self):
        """Multiple boxed expressions: ensure strictly the LAST valid boxed answer is returned."""
        text_multi = (
            r"Initial attempt gave \boxed{10}, which had an error. "
            r"Second attempt gave \boxed{\frac{15}{2}}. "
            r"Finally, re-evaluating yields \boxed{42}."
        )
        self.assertEqual(extract_boxed_answer(text_multi), "42")

        # Mixed spacing before brace
        text_spaces = r"First \boxed {12}, next \boxed  {\sqrt{7}}, last \boxed   {99}."
        self.assertEqual(extract_boxed_answer(text_spaces), "99")

    def test_unbalanced_braces_and_malformed_text(self):
        """Unbalanced braces / malformed text returns None without raising an exception."""
        # Unclosed single brace
        self.assertIsNone(extract_boxed_answer(r"\boxed{42"))

        # Extra open braces
        self.assertIsNone(extract_boxed_answer(r"\boxed{{{42}"))

        # Missing opening brace
        self.assertIsNone(extract_boxed_answer(r"\boxed 42"))
        self.assertIsNone(extract_boxed_answer(r"No boxed answer here."))

        # None or empty
        self.assertIsNone(extract_boxed_answer(None))
        self.assertIsNone(extract_boxed_answer(""))

        # Non-string malformed inputs must return None without raising exception
        self.assertIsNone(extract_boxed_answer(12345))
        self.assertIsNone(extract_boxed_answer([r"\boxed{1}"]))
        self.assertIsNone(extract_boxed_answer({"answer": r"\boxed{2}"}))

        # Text ending with truncated/unbalanced boxed after a valid one returns last valid
        text_truncated_at_end = r"First step yields \boxed{100}. Later we started \boxed{20"
        self.assertEqual(extract_boxed_answer(text_truncated_at_end), "100")


class TestAdaptiveInferenceControllers(unittest.TestCase):
    """Stress tests for adaptive inference stopping controllers, thresholds, margins, and fallback."""

    def test_agreement_threshold_exact_behavior(self):
        """Verify exact threshold behavior for tau in {0.5, 0.625, 0.75, 0.875, 1.0}."""
        # At checkpoint n=8:
        # 1. 4 votes for leader ('A'), 3 for 'B', 1 for 'C' -> vote_share = 4/8 = 0.5
        cands_050 = [
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=i, extracted_answer="A" if i < 4 else ("B" if i < 7 else "C"), length=5)
            for i in range(8)
        ]
        self.assertTrue(check_agreement_condition(cands_050, tau=0.5)[0])
        self.assertFalse(check_agreement_condition(cands_050, tau=0.625)[0])
        self.assertFalse(check_agreement_condition(cands_050, tau=0.75)[0])
        self.assertFalse(check_agreement_condition(cands_050, tau=0.875)[0])
        self.assertFalse(check_agreement_condition(cands_050, tau=1.0)[0])

        # 2. 5 votes for leader ('A'), 2 for 'B', 1 for 'C' -> vote_share = 5/8 = 0.625
        cands_0625 = [
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=i, extracted_answer="A" if i < 5 else ("B" if i < 7 else "C"), length=5)
            for i in range(8)
        ]
        self.assertTrue(check_agreement_condition(cands_0625, tau=0.5)[0])
        self.assertTrue(check_agreement_condition(cands_0625, tau=0.625)[0])
        self.assertFalse(check_agreement_condition(cands_0625, tau=0.75)[0])
        self.assertFalse(check_agreement_condition(cands_0625, tau=0.875)[0])
        self.assertFalse(check_agreement_condition(cands_0625, tau=1.0)[0])

        # 3. 6 votes for leader ('A'), 2 for 'B' -> vote_share = 6/8 = 0.75
        cands_075 = [
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=i, extracted_answer="A" if i < 6 else "B", length=5)
            for i in range(8)
        ]
        self.assertTrue(check_agreement_condition(cands_075, tau=0.75)[0])
        self.assertFalse(check_agreement_condition(cands_075, tau=0.875)[0])
        self.assertFalse(check_agreement_condition(cands_075, tau=1.0)[0])

        # 4. 7 votes for leader ('A'), 1 for 'B' -> vote_share = 7/8 = 0.875
        cands_0875 = [
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=i, extracted_answer="A" if i < 7 else "B", length=5)
            for i in range(8)
        ]
        self.assertTrue(check_agreement_condition(cands_0875, tau=0.875)[0])
        self.assertFalse(check_agreement_condition(cands_0875, tau=1.0)[0])

        # 5. 8 votes for leader ('A') -> vote_share = 8/8 = 1.0
        cands_100 = [
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=i, extracted_answer="A", length=5)
            for i in range(8)
        ]
        self.assertTrue(check_agreement_condition(cands_100, tau=1.0)[0])

        # 6. Non-unique leader: 4 votes for 'A', 4 votes for 'B'
        cands_tied = [
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=i, extracted_answer="A" if i < 4 else "B", length=5)
            for i in range(8)
        ]
        # Even though vote share is 4/8 = 0.5, leader is not unique -> False
        self.assertFalse(check_agreement_condition(cands_tied, tau=0.5)[0])

        # 7. Leader is NO_ANSWER: 5 votes NO_ANSWER, 3 votes 'A'
        cands_no_ans = [
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=i, extracted_answer=NO_ANSWER if i < 5 else "A", length=5)
            for i in range(8)
        ]
        self.assertFalse(check_agreement_condition(cands_no_ans, tau=0.5)[0])

    def test_quality_aware_winner_agreement_and_margin(self):
        """Verify both winner agreement (SC == RWC) and vote margin >= gamma are satisfied."""
        # Case A: Agreement satisfied, SC == RWC ("A"), margin >= gamma -> Success
        # 3 A's with high KL, 1 B with low KL out of 4
        cands_ok = [
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=0, extracted_answer="A", length=5, kl_u_p=[5.0]*5),
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=1, extracted_answer="A", length=5, kl_u_p=[5.0]*5),
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=2, extracted_answer="A", length=5, kl_u_p=[5.0]*5),
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=3, extracted_answer="B", length=5, kl_u_p=[1.0]*5),
        ]
        ok_a, win_a = check_quality_aware_condition(cands_ok, tau=0.75, gamma=0.25, dataset="math")
        self.assertTrue(ok_a)
        self.assertEqual(win_a, "A")

        # Case B: Agreement satisfied, but SC != RWC
        # SC winner is "A" (3 votes vs 2 votes for "B")
        # BUT "B" candidates have huge KL (e.g. 100.0) while "A" candidates have tiny KL (e.g. 1.0)
        # RWC winner becomes "B"
        cands_disagree = [
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=0, extracted_answer="A", length=5, kl_u_p=[1.0]*5),
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=1, extracted_answer="A", length=5, kl_u_p=[1.0]*5),
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=2, extracted_answer="A", length=5, kl_u_p=[1.0]*5),
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=3, extracted_answer="B", length=5, kl_u_p=[100.0]*5),
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=4, extracted_answer="B", length=5, kl_u_p=[100.0]*5),
        ]
        # SC has 3/5 = 0.60 vote share for "A"
        agree_ok, leader = check_agreement_condition(cands_disagree, tau=0.6, dataset="math")
        self.assertTrue(agree_ok)
        self.assertEqual(leader, "A")

        # Quality-aware must fail because SC winner ("A") != RWC winner ("B")
        qual_ok, qual_winner = check_quality_aware_condition(cands_disagree, tau=0.6, gamma=0.0, dataset="math")
        self.assertFalse(qual_ok)
        self.assertIsNone(qual_winner)

        # Case C: SC == RWC, but margin < gamma
        # 3 candidates: 2 A's (scores 6.0, 5.0), 1 B (score 10.0)
        # Ranks: B gets 1.0 (w=1.0), A1 gets 2.0 (w=0.5), A2 gets 3.0 (w=0.0)
        # Total weight: W(B) = 1.0, W(A) = 0.5. Here RWC winner is B.
        # Let's adjust so SC == RWC:
        # 4 candidates: A (score 10), A (score 8), B (score 9), B (score 7)
        # Ranks: A0 (1), B0 (2), A1 (3), B1 (4).
        # Weights (N=4, w=(4-r)/3):
        # A0: 3/3=1.0, B0: 2/3, A1: 1/3, B1: 0/3.
        # W(A) = 1.0 + 1/3 = 4/3. W(B) = 2/3 + 0 = 2/3.
        # Margin = (4/3 - 2/3) / (6/3) = (2/3) / 2 = 1/3 ~ 0.3333.
        cands_margin = [
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=0, extracted_answer="A", length=5, kl_u_p=[10.0]*5),
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=1, extracted_answer="A", length=5, kl_u_p=[8.0]*5),
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=2, extracted_answer="B", length=5, kl_u_p=[9.0]*5),
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=3, extracted_answer="C", length=5, kl_u_p=[7.0]*5),
        ]
        # A has 2/4 = 0.5 vote share (unique leader since B has 1, C has 1).
        # SC winner is A. RWC winner is A with margin ~ 0.333.
        # For gamma = 0.25 <= 0.333: True
        ok_low_gamma, _ = check_quality_aware_condition(cands_margin, tau=0.5, gamma=0.25, dataset="math")
        self.assertTrue(ok_low_gamma)
        # For gamma = 0.50 > 0.333: False
        ok_high_gamma, _ = check_quality_aware_condition(cands_margin, tau=0.5, gamma=0.50, dataset="math")
        self.assertFalse(ok_high_gamma)

    def test_rwc_vote_margin_with_single_answer_group(self):
        """Single answer group: margin calculation (largest - runner_up)/total_weight correctly handles runner_up=0."""
        # 4 identical answers
        cands_single_grp = [
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=i, extracted_answer="42", length=5, kl_u_p=[5.0]*5)
            for i in range(4)
        ]

        # Margin with single group: W_total = W_1, runner_up = 0.0 -> margin = (W_1 - 0) / W_1 = 1.0
        # Condition should pass for any gamma in {0.0, 0.1, 0.25, 0.5}
        for gamma in (0.0, 0.1, 0.25, 0.5):
            ok, winner = check_quality_aware_condition(cands_single_grp, tau=0.5, gamma=gamma, dataset="math")
            self.assertTrue(ok, f"Failed for gamma={gamma} with single answer group")
            self.assertEqual(winner, "42")

    def test_fallback_to_32_when_stopping_conditions_not_reached(self):
        """Verify fallback to 32 candidates when stopping conditions are never reached at checkpoints 4, 8, 16."""
        # 32 candidates with 3 alternating answers so no answer ever reaches tau=0.75
        rollouts_32 = [
            RolloutRecord(
                dataset="math",
                sample_id="q_fallback",
                rollout_id=i,
                extracted_answer=f"Ans_{i % 3}",
                length=15,
                kl_u_p=[3.0] * 15,
            )
            for i in range(32)
        ]

        # Simulate agreement-based policy with high tau (0.75)
        dec_agree = simulate_adaptive_stopping(
            rollouts_32,
            policy_type="agreement",
            tau=0.75,
            dataset="math",
            gold_answer="Ans_0",
        )
        self.assertEqual(dec_agree.stopped_at, 32)
        self.assertEqual(dec_agree.policy_name, "fixed_32")
        self.assertEqual(dec_agree.tokens_consumed, 15 * 32)

        # Simulate quality-aware policy with high tau (0.75)
        dec_qual = simulate_adaptive_stopping(
            rollouts_32,
            policy_type="quality",
            tau=0.75,
            gamma=0.25,
            dataset="math",
            gold_answer="Ans_0",
        )
        self.assertEqual(dec_qual.stopped_at, 32)
        self.assertEqual(dec_qual.policy_name, "fixed_32")
        self.assertEqual(dec_qual.tokens_consumed, 15 * 32)


class TestOfflineOpportunityAnalysis(unittest.TestCase):
    """Stress tests for offline opportunity analysis and failure decomposition."""

    def test_failure_decomposition_sampling_vs_selection(self):
        """Verify exact failure decomposition: (1 - FirstAns) == Sampling Failure + Selection Failure."""
        # Scenario 1: Sampling Failure (no correct candidate in full 32-rollout pool)
        # All 32 candidates have wrong answer
        q_samp_fail = QuestionRecord(
            dataset="math",
            sample_id="q_samp_fail",
            gold_answer="42",
            rollouts=[
                RolloutRecord(dataset="math", sample_id="q_samp_fail", rollout_id=i, extracted_answer="0", length=10)
                for i in range(32)
            ],
        )
        identity_order = [list(range(32))]
        res_samp = evaluate_question_opportunity(q_samp_fail, identity_order, budgets=[32])
        self.assertEqual(res_samp[32]["first_ans_accuracy"], 0.0)
        self.assertEqual(res_samp[32]["oracle_coverage"], 0.0)
        self.assertEqual(res_samp[32]["sampling_failure"], 1.0)
        self.assertEqual(res_samp[32]["selection_failure"], 0.0)
        # Identity: 1 - 0 = 1 + 0
        self.assertAlmostEqual(1.0 - res_samp[32]["first_ans_accuracy"], res_samp[32]["sampling_failure"] + res_samp[32]["selection_failure"])

        # Scenario 2: Selection Failure (correct candidate exists in pool, but FirstAns picked wrong)
        # Rollout 0 is wrong, Rollout 1 is correct ("42"), rest wrong
        q_sel_fail = QuestionRecord(
            dataset="math",
            sample_id="q_sel_fail",
            gold_answer="42",
            rollouts=[
                RolloutRecord(
                    dataset="math",
                    sample_id="q_sel_fail",
                    rollout_id=i,
                    extracted_answer="42" if i == 1 else "0",
                    length=10,
                )
                for i in range(32)
            ],
        )
        res_sel = evaluate_question_opportunity(q_sel_fail, identity_order, budgets=[32])
        self.assertEqual(res_sel[32]["first_ans_accuracy"], 0.0)
        self.assertEqual(res_sel[32]["oracle_coverage"], 1.0)
        self.assertEqual(res_sel[32]["sampling_failure"], 0.0)
        self.assertEqual(res_sel[32]["selection_failure"], 1.0)
        # Identity: 1 - 0 = 0 + 1
        self.assertAlmostEqual(1.0 - res_sel[32]["first_ans_accuracy"], res_sel[32]["sampling_failure"] + res_sel[32]["selection_failure"])

        # Scenario 3: Success (First candidate is correct)
        q_success = QuestionRecord(
            dataset="math",
            sample_id="q_success",
            gold_answer="42",
            rollouts=[
                RolloutRecord(
                    dataset="math",
                    sample_id="q_success",
                    rollout_id=i,
                    extracted_answer="42" if i == 0 else "0",
                    length=10,
                )
                for i in range(32)
            ],
        )
        res_succ = evaluate_question_opportunity(q_success, identity_order, budgets=[32])
        self.assertEqual(res_succ[32]["first_ans_accuracy"], 1.0)
        self.assertEqual(res_succ[32]["oracle_coverage"], 1.0)
        self.assertEqual(res_succ[32]["sampling_failure"], 0.0)
        self.assertEqual(res_succ[32]["selection_failure"], 0.0)
        # Identity: 1 - 1 = 0 + 0
        self.assertAlmostEqual(1.0 - res_succ[32]["first_ans_accuracy"], res_succ[32]["sampling_failure"] + res_succ[32]["selection_failure"])

    def test_dataset_opportunity_decomposition_across_all_budgets(self):
        """Verify (1 - FirstAns) == Sampling Failure + Selection Failure across all budgets N in {1, 2, 4, 8, 16, 32}."""
        # Create dataset with 3 questions
        q_dict = {
            "q1": QuestionRecord(
                dataset="math", sample_id="q1", gold_answer="42",
                rollouts=[RolloutRecord("math", "q1", i, extracted_answer="0", length=10) for i in range(32)]
            ),
            "q2": QuestionRecord(
                dataset="math", sample_id="q2", gold_answer="42",
                rollouts=[RolloutRecord("math", "q2", i, extracted_answer="42" if i == 5 else "0", length=10) for i in range(32)]
            ),
            "q3": QuestionRecord(
                dataset="math", sample_id="q3", gold_answer="42",
                rollouts=[RolloutRecord("math", "q3", i, extracted_answer="42" if i < 16 else "0", length=10) for i in range(32)]
            ),
        }
        qids = ["q1", "q2", "q3"]
        perms = generate_rollout_permutations(qids, num_permutations=10, seed=6304)

        eval_res = evaluate_dataset_opportunity(q_dict, qids, permutations=perms, seed=6304)

        for N in (1, 2, 4, 8, 16, 32):
            summary_n = eval_res["summary"][N]
            acc = summary_n["first_ans_accuracy"]["mean"]
            oracle = summary_n["oracle_coverage"]["mean"]
            samp_fail = summary_n["sampling_failure"]["mean"]
            sel_fail = summary_n["selection_failure"]["mean"]

            # FirstAns accuracy must not exceed oracle coverage
            self.assertLessEqual(acc, oracle + 1e-8)

            # Failure decomposition identity
            err = 1.0 - acc
            self.assertAlmostEqual(
                err,
                samp_fail + sel_fail,
                places=6,
                msg=f"Failure decomposition failed at budget N={N}: err={err}, samp+sel={samp_fail + sel_fail}",
            )


if __name__ == "__main__":
    unittest.main()
