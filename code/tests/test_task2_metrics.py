"""Pair-level metric semantics of Task 2 (swap behaviour categories, correct-image advantages, bootstrap coupling)."""

import unittest

from task2_vlm.evaluate import build_mismatched_image_mapping, compute_suite_metrics_and_ci
from task2_vlm.trainer import validate_stage_config


def rec(pid, cond, pred_a, pred_b, ref_a="yes", ref_b="no", nll=1.0):
    """One pair record shaped like ``run_condition`` output."""
    norm = lambda s: s.strip().lower()
    return {
        "pair_id": pid, "example_type": "presence", "condition": cond, "question": "Q?",
        "image_a": "a", "image_b": "b", "reference_a": ref_a, "reference_b": ref_b,
        "pred_a": pred_a, "pred_b": pred_b,
        "score_a": float(norm(pred_a) == ref_a), "score_b": float(norm(pred_b) == ref_b),
        "mean_score": (float(norm(pred_a) == ref_a) + float(norm(pred_b) == ref_b)) / 2,
        "nll_a": nll, "nll_b": nll, "mean_nll": nll,
        "pair_accuracy": float(norm(pred_a) == ref_a and norm(pred_b) == ref_b),
        "identical_answer": float(norm(pred_a) == norm(pred_b)), "empty_rate": 0.0, "gen_time": 0.01,
        "follows_partner_a": float(norm(pred_a) == ref_b), "follows_partner_b": float(norm(pred_b) == ref_a),
    }


class TestSwapBehaviour(unittest.TestCase):
    def test_categories_are_exclusive_and_sum_to_one(self):
        # pair 1: correct images answer (yes, no); swap flips both -> follows the partner image (2 answers)
        # pair 2: model always says "no": correct (no, no); swap (no, no) -> unchanged (2 answers) although the
        #         swapped answer of side a ("no") ALSO equals the partner's reference - must count as unchanged only
        # pair 3: correct (yes, no); swap (maybe, no) -> a changed to something else ("other"), b unchanged
        correct = [rec("p1", "correct", "yes", "no"), rec("p2", "correct", "no", "no"), rec("p3", "correct", "yes", "no")]
        swap = [rec("p1", "swap", "no", "yes"), rec("p2", "swap", "no", "no"), rec("p3", "swap", "maybe", "no")]
        out = compute_suite_metrics_and_ci({"correct": correct, "swap": swap}, n_resamples=50, seed=1)["swap_behavior"]
        self.assertAlmostEqual(out["follows_partner_answer_rate"], 2 / 6)
        self.assertAlmostEqual(out["unchanged_vs_correct_image_rate"], 3 / 6)
        self.assertAlmostEqual(out["other_rate"], 1 / 6)
        self.assertAlmostEqual(out["follows_partner_answer_rate"] + out["unchanged_vs_correct_image_rate"] + out["other_rate"], 1.0)
        # the raw (overlapping) rate counts p1 (2) + p2 side a (1) = 3 answers equal to the partner's reference
        self.assertAlmostEqual(out["matches_partner_reference_rate"], 3 / 6)

    def test_correct_image_advantage_signs(self):
        correct = [rec("p1", "correct", "yes", "no", nll=1.0), rec("p2", "correct", "yes", "no", nll=1.0)]
        swap = [rec("p1", "swap", "no", "yes", nll=3.0), rec("p2", "swap", "no", "yes", nll=3.0)]
        out = compute_suite_metrics_and_ci({"correct": correct, "swap": swap}, n_resamples=50, seed=1)
        adv = out["correct_advantage_over_swap"]
        self.assertAlmostEqual(adv["delta_accuracy"]["mean"], 1.0)      # score(correct) - score(swap)
        self.assertAlmostEqual(adv["delta_nll_penalty"]["mean"], 2.0)   # NLL(swap) - NLL(correct) > 0 means the image helps
        self.assertEqual(out["correct"]["n_pairs"], 2)


class _Ex:
    def __init__(self, image_id):
        self.image_id = image_id


class _Pair:
    def __init__(self, pid, ia, ib):
        self.pair_id, self.example_a, self.example_b = pid, _Ex(ia), _Ex(ib)


class TestMismatchedMapping(unittest.TestCase):
    PAIRS = [_Pair(f"p{i}", f"img{2 * i}", f"img{2 * i + 1}") for i in range(12)]

    def test_each_example_gets_its_own_unrelated_image(self):
        m = build_mismatched_image_mapping(self.PAIRS, seed=6304)
        for p in self.PAIRS:
            a, b = m[p.pair_id]["a"], m[p.pair_id]["b"]
            self.assertNotEqual(a, b)                                          # distinct, so answers may differ
            self.assertNotIn(a, (p.example_a.image_id, p.example_b.image_id))  # unrelated: never an image of its own pair
            self.assertNotIn(b, (p.example_a.image_id, p.example_b.image_id))

    def test_mapping_is_deterministic_and_seeded(self):
        m1 = build_mismatched_image_mapping(self.PAIRS, seed=6304)
        self.assertEqual(m1, build_mismatched_image_mapping(self.PAIRS, seed=6304))
        self.assertNotEqual(m1, build_mismatched_image_mapping(self.PAIRS, seed=1))


class TestStageConfig(unittest.TestCase):
    BASE = {"stages": {"stage1": {"updates": 2250, "connector_peak_lr": 1e-3, "lora_peak_lr": None,
                                  "mix_probabilities": {"captions": 1.0}},
                       "stage2": {"updates": 1500, "connector_peak_lr": 1e-4, "lora_peak_lr": 2e-5, "lora_rank": 16, "lora_alpha": 32,
                                  "lora_dropout": 0.05, "mix_probabilities": {"presence": 0.5, "counting": 0.25, "caption_replay": 0.25}},
                       "stage3": {"updates": 750, "connector_peak_lr": 5e-5, "lora_peak_lr": 1e-5, "lora_rank": 16, "lora_alpha": 32,
                                  "lora_dropout": 0.05, "mix_probabilities": {"natural_vqa": 0.7, "presence_replay": 0.1,
                                                                              "counting_replay": 0.1, "caption_replay": 0.1}}},
            "training": {"effective_batch_size": 16, "micro_batch_size": 8, "gradient_accumulation_steps": 2},
            "evaluation": {"checkpoint_interval": 250, "dev_eval_interval": 500}}

    def test_matching_config_returns_budgets(self):
        self.assertEqual(validate_stage_config(self.BASE), {1: 2250, 2: 1500, 3: 750})

    def test_mismatching_values_raise_instead_of_being_ignored(self):
        import copy
        for path, value in ((("stages", "stage2", "lora_peak_lr"), 3e-5), (("stages", "stage3", "lora_rank"), 8),
                            (("training", "micro_batch_size"), 4), (("evaluation", "dev_eval_interval"), 100)):
            cfg = copy.deepcopy(self.BASE)
            d = cfg
            for k in path[:-1]:
                d = d[k]
            d[path[-1]] = value
            with self.assertRaises(ValueError, msg=str(path)):
                validate_stage_config(cfg)

    def test_budgets_can_be_overridden_for_smoke_runs(self):
        import copy
        cfg = copy.deepcopy(self.BASE)
        for i, n in ((1, 12), (2, 8), (3, 6)):
            cfg["stages"][f"stage{i}"]["updates"] = n
        self.assertEqual(validate_stage_config(cfg), {1: 12, 2: 8, 3: 6})


if __name__ == "__main__":
    unittest.main()
