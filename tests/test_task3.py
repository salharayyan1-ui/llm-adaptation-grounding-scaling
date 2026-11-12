"""Comprehensive Unit and Verification Tests for Task 3: Test-Time Compute."""

import unittest
import numpy as np

from task3_test_time.split import generate_splits
from task3_test_time.equivalence import (
    normalize_math_candidate,
    normalize_mathvista_candidate,
    are_answers_equivalent,
    group_answers,
    grade_answer,
    NO_ANSWER,
)
from task3_test_time.data_loader import (
    RolloutRecord,
    QuestionRecord,
    verify_token_alignment,
    generate_synthetic_data,
)
from task3_test_time.opportunity import (
    generate_rollout_permutations,
    evaluate_question_opportunity,
    evaluate_dataset_opportunity,
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
    evaluate_quartile_differences,
    evaluate_shortest_rollout_baseline,
)
from task3_test_time.live_generation import (
    QwenLiveGenerator,
    run_live_generation_experiment,
    summarize_live_results,
)
from task3_test_time.adaptive import (
    check_agreement_condition,
    check_quality_aware_condition,
    simulate_adaptive_stopping,
    search_development_policies,
    evaluate_held_out_policy,
)


class TestTask3Pipeline(unittest.TestCase):

    def test_split_determinism_and_properties(self):
        """Verify lexicographical sort + default_rng(6304) permutation."""
        # 1000 simulated question IDs
        raw_ids = [f"sample_{i:04d}" for i in range(1000)]
        np.random.default_rng(1234).shuffle(raw_ids)  # shuffle input order

        splits1 = generate_splits(raw_ids, seed=6304)
        splits2 = generate_splits(raw_ids, seed=6304)

        # Exact reproducibility
        self.assertEqual(splits1["sorted"], sorted(raw_ids))
        self.assertEqual(splits1["permuted"], splits2["permuted"])
        self.assertEqual(len(splits1["dev"]), 200)
        self.assertEqual(len(splits1["eval"]), 800)
        self.assertEqual(len(splits1["live_math"]), 50)

        # Disjoint dev and eval
        dev_set = set(splits1["dev"])
        eval_set = set(splits1["eval"])
        self.assertEqual(len(dev_set.intersection(eval_set)), 0)
        self.assertEqual(len(dev_set.union(eval_set)), 1000)

        # Live math is prefix of eval
        self.assertEqual(splits1["live_math"], splits1["eval"][:50])

    def test_equivalence_math_and_vista(self):
        """Test equivalence, normalization rules, and NO_ANSWER behavior."""
        # Math normalization
        self.assertEqual(normalize_math_candidate(" $ \\frac{1}{2} $ "), "\\frac{1}{2}")
        self.assertEqual(normalize_math_candidate("\\text{42}"), "42")
        self.assertEqual(normalize_math_candidate(""), NO_ANSWER)
        self.assertEqual(normalize_math_candidate(None), NO_ANSWER)

        # Math equivalence
        self.assertTrue(are_answers_equivalent("\\frac{1}{2}", "0.5", dataset="math"))
        self.assertTrue(are_answers_equivalent("2 + 2", "4", dataset="math"))
        self.assertTrue(are_answers_equivalent("x^2 - 1", "(x-1)*(x+1)", dataset="math"))

        # MathVista normalization
        self.assertEqual(normalize_mathvista_candidate("(B)", question_type="multi_choice"), "B")
        self.assertEqual(normalize_mathvista_candidate("Option C.", question_type="multi_choice"), "C")
        self.assertEqual(normalize_mathvista_candidate("12.5000", question_type="float"), "12.5")
        self.assertEqual(normalize_mathvista_candidate("14.0", question_type="integer"), "14")
        self.assertEqual(normalize_mathvista_candidate(""), NO_ANSWER)

        # Grading: NO_ANSWER must always be False
        self.assertFalse(grade_answer("NO_ANSWER", "4", dataset="math"))
        self.assertFalse(grade_answer(None, "4", dataset="math"))
        self.assertTrue(grade_answer("4", "4", dataset="math"))

    def test_answer_grouping_without_gold(self):
        """Verify grouping candidates into equivalence classes without gold answers."""
        candidates = ["\\frac{2}{4}", "0.5", "1/2", "3", "NO_ANSWER", "NO_ANSWER", "3.0"]
        labels, groups = group_answers(candidates, dataset="math")

        # \\frac{2}{4}, 0.5, 1/2 must belong to the same group
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[0], labels[2])

        # 3 and 3.0 must belong to the same group
        self.assertEqual(labels[3], labels[6])

        # NO_ANSWER must form its own single group containing indices 4 and 5
        self.assertIn(NO_ANSWER, groups)
        self.assertEqual(groups[NO_ANSWER], [4, 5])

    def test_data_loader_and_token_alignment(self):
        """Test RolloutRecord token alignment verification and synthetic data."""
        valid_rec = RolloutRecord(
            dataset="math",
            sample_id="q1",
            rollout_id=0,
            length=3,
            logp=[-0.1, -0.2, -0.3],
            kl_u_p=[4.0, 4.2, 4.1],
        )
        self.assertTrue(verify_token_alignment(valid_rec))

        invalid_rec = RolloutRecord(
            dataset="math",
            sample_id="q1",
            rollout_id=0,
            length=4,  # mismatch with logp length 3
            logp=[-0.1, -0.2, -0.3],
            kl_u_p=[4.0, 4.2, 4.1],
        )
        self.assertFalse(verify_token_alignment(invalid_rec))

        synth = generate_synthetic_data(dataset_name="mathvista", n_questions=5, n_rollouts=32, seed=6304)
        self.assertEqual(len(synth), 5)
        for q in synth.values():
            self.assertEqual(len(q.rollouts), 32)
            for r in q.rollouts:
                self.assertTrue(verify_token_alignment(r))

    def test_opportunity_metrics_and_failure_decomposition(self):
        """Verify opportunity metrics and failure decomposition."""
        synth = generate_synthetic_data(dataset_name="math", n_questions=3, n_rollouts=32, seed=6304)
        qids = list(synth.keys())
        perms = generate_rollout_permutations(qids, num_permutations=10, seed=6304)

        for qid in qids:
            self.assertEqual(len(perms[qid]), 10)
            self.assertEqual(len(perms[qid][0]), 32)

        res = evaluate_dataset_opportunity(synth, qids, permutations=perms, seed=6304)
        for N in (1, 2, 4, 8, 16, 32):
            summary_n = res["summary"][N]
            first_acc = summary_n["first_ans_accuracy"]["mean"]
            oracle_cov = summary_n["oracle_coverage"]["mean"]
            samp_fail = summary_n["sampling_failure"]["mean"]
            sel_fail = summary_n["selection_failure"]["mean"]

            # FirstAns accuracy <= Oracle coverage
            self.assertLessEqual(first_acc, oracle_cov + 1e-6)

            # Failure decomposition identity: (1 - FirstAns) == Sampling Failure + Selection Failure
            err = 1.0 - first_acc
            self.assertAlmostEqual(err, samp_fail + sel_fail, places=5)

    def test_scoring_and_rwc_weighting(self):
        """Test RWC rank calculation, weights, and tie-breaking."""
        # Descending average ranks
        scores = [10.0, 5.0, 5.0, 1.0]
        ranks = compute_average_ranks(scores)
        np.testing.assert_allclose(ranks, [1.0, 2.5, 2.5, 4.0])

        # Weights: (N - r_i) / (N - 1)
        # N=4: w = (4 - ranks) / 3
        expected_weights = [(4.0 - 1.0)/3.0, (4.0 - 2.5)/3.0, (4.0 - 2.5)/3.0, (4.0 - 4.0)/3.0]
        actual_weights = (4.0 - ranks) / 3.0
        np.testing.assert_allclose(actual_weights, expected_weights)

        # Single candidate case N=1
        r_single = RolloutRecord(dataset="math", sample_id="q0", rollout_id=0, extracted_answer="42")
        idx, ans, weights = select_rank_weighted_consensus([r_single], [5.0])
        self.assertEqual(idx, 0)
        self.assertEqual(ans, "42")
        self.assertEqual(weights["42"], 1.0)

    def test_visconf_percentile_calibration_and_covariance_identity(self):
        """Test VisConf prefix calibration and tokenwise covariance identity."""
        r1 = RolloutRecord(
            dataset="mathvista",
            sample_id="qv",
            rollout_id=0,
            length=4,
            kl_u_p=[2.0, 4.0, 6.0, 8.0],
            att_image=[10.0, 20.0, 30.0, 40.0],
            att_prompt=[10.0, 10.0, 10.0, 10.0],
            att_generated=[10.0, 10.0, 10.0, 10.0],
        )
        r2 = RolloutRecord(
            dataset="mathvista",
            sample_id="qv",
            rollout_id=1,
            length=4,
            kl_u_p=[3.0, 5.0, 7.0, 9.0],
            att_image=[5.0, 15.0, 25.0, 35.0],
            att_prompt=[5.0, 5.0, 5.0, 5.0],
            att_generated=[5.0, 5.0, 5.0, 5.0],
        )

        v_scores, c_scores, z_scores = compute_visconf_scores([r1, r2])
        self.assertEqual(len(z_scores), 2)
        # Z_i == V_i * C_i
        self.assertAlmostEqual(z_scores[0], v_scores[0] * c_scores[0], places=6)
        self.assertAlmostEqual(z_scores[1], v_scores[1] * c_scores[1], places=6)

        # Test covariance identity on rollout: Z_token - Z_rollout == Cov(c, v)
        corr, z_tok, cov_val = compute_confidence_vision_correlation(r1)
        v_1 = compute_visual_engagement(r1)
        k_1 = np.asarray(r1.kl_u_p, dtype=np.float64)
        c_1 = (k_1 - np.percentile(k_1, 5, method="linear")) / max(np.percentile(k_1, 95, method="linear") - np.percentile(k_1, 5, method="linear"), 1e-8)
        c_1 = np.clip(c_1, 0.0, 1.0)
        z_rollout = v_1 * float(np.mean(c_1))
        self.assertAlmostEqual(z_tok - z_rollout, cov_val, places=6)

    def test_live_generation_dry_run_and_transitions(self):
        """Test live generation strategies and transition accounting in dry run."""
        generator = QwenLiveGenerator(dry_run=True)
        questions = [
            {"sample_id": "q_001", "problem": "What is 20 + 22?", "gold_answer": "42"},
            {"sample_id": "q_002", "problem": "What is 10 + 10?", "gold_answer": "20"},
        ]
        results = run_live_generation_experiment(questions, generator, base_seed=6304)
        self.assertEqual(len(results), 2)

        # Verify strategy outputs
        for r in results:
            self.assertEqual(r.strategy_single.output_tokens, 150)
            self.assertEqual(len(r.strategy_parallel), 3)
            self.assertEqual(r.strategy_parallel[0].output_tokens, 150)
            self.assertIn(r.revision_transition, [
                "correct_to_correct", "wrong_to_correct", "correct_to_wrong", "wrong_to_wrong"
            ])

        summary = summarize_live_results(results)
        self.assertEqual(summary["count"], 2)
        self.assertIn("accuracy", summary)
        self.assertIn("mean_output_tokens", summary)
        self.assertIn("transitions_count", summary)

    def test_adaptive_stopping_and_policy_search(self):
        """Test adaptive checkpoints (4, 8, 16, 32) and policy search."""
        synth = generate_synthetic_data(dataset_name="math", n_questions=10, n_rollouts=32, seed=6304)
        qids = list(synth.keys())
        perms = generate_rollout_permutations(qids, num_permutations=4, seed=6304)

        # Agreement condition test
        cand_agree = [
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=i, extracted_answer="42", length=10)
            for i in range(4)
        ]
        ok, leader = check_agreement_condition(cand_agree, tau=0.75, dataset="math")
        self.assertTrue(ok)
        self.assertEqual(leader, "42")

        # Disagreement condition test
        cand_disagree = [
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=0, extracted_answer="1", length=10),
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=1, extracted_answer="2", length=10),
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=2, extracted_answer="3", length=10),
            RolloutRecord(dataset="math", sample_id="q0", rollout_id=3, extracted_answer="4", length=10),
        ]
        ok_no, _ = check_agreement_condition(cand_disagree, tau=0.5, dataset="math")
        self.assertFalse(ok_no)

        # Simulation
        dec = simulate_adaptive_stopping(
            cand_agree * 8, policy_type="agreement", tau=0.75, dataset="math", gold_answer="42"
        )
        self.assertEqual(dec.stopped_at, 4)
        self.assertEqual(dec.chosen_answer, "42")
        self.assertTrue(dec.is_correct)

        # Dev search on small set
        dev_search = search_development_policies(synth, qids[:6], perms)
        self.assertIn("agreement", dev_search)
        self.assertIn("quality_aware", dev_search)
        best_tau = dev_search["agreement"]["best_params"]["tau"]

        # Held-out evaluation
        held_out_res = evaluate_held_out_policy(
            synth, qids[6:], perms, policy_type="agreement", tau=best_tau, seed=6304
        )
        self.assertIn("stopping_distribution", held_out_res)
        self.assertIn("early_stop_analysis", held_out_res)
        self.assertIn(4, held_out_res["stopping_counts"])
        self.assertIn(32, held_out_res["stopping_counts"])


if __name__ == "__main__":
    unittest.main()
