"""Tests for task1_llm_ft.offline_analysis (extraction sensitivity, paired bootstrap, BoolQ bias)."""
import itertools
import json
import os
import random
import tempfile
import unittest

import numpy as np

from task1_llm_ft import offline_analysis as oa

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestGsm8kExtraction(unittest.TestCase):
    def test_dollar_dollar_artifact(self):
        """A response ending in a closing LaTeX '$$' is 'extracted' as '$$' by the stock filter (the answer is lost)."""
        resp = "Total = 160 + 80 + 20 = 260\n\n### Final Answer:\n$$\n\\boxed{260}\n$$"
        self.assertEqual(oa.stock_flexible_extract(resp), "$$")
        self.assertFalse(oa.stock_correct(oa.stock_flexible_extract(resp), "260"))
        self.assertEqual(oa.digit_flexible_extract(resp), "260")
        self.assertTrue(oa.stock_correct(oa.digit_flexible_extract(resp), "260"))

    def test_stock_filter_normal_cases(self):
        self.assertEqual(oa.stock_flexible_extract("so the answer is: 18"), "18")
        self.assertEqual(oa.stock_flexible_extract("costs -5 then 7"), "7")
        self.assertEqual(oa.stock_flexible_extract("no numbers here"), oa.INVALID)
        # stock normalisation ignores ',', '$' and a trailing '.'
        self.assertTrue(oa.stock_correct(oa.stock_flexible_extract("He earns $1,000."), "1000"))
        # a digit-free tail only changes the answer when the stock rule would have returned it
        for r in ("The answer is: 42", "x = 3, y = 4 so 12", "approx 3.5."):
            self.assertEqual(oa.stock_flexible_extract(r), oa.digit_flexible_extract(r))

    def test_digit_rule_skips_every_digitless_match(self):
        self.assertEqual(oa.digit_flexible_extract("answer 9 ... , $$ .."), "9")
        self.assertEqual(oa.digit_flexible_extract("$$ .."), oa.INVALID)

    def test_last_boxed_balanced_and_nested(self):
        self.assertEqual(oa.last_boxed("a \\boxed{1} b \\boxed{\\frac{3}{4}}"), "\\frac{3}{4}")
        self.assertIsNone(oa.last_boxed("no box"))
        self.assertIsNone(oa.last_boxed("\\boxed{unbalanced"))

    def test_boxed_first_numeric_compare_and_fallback(self):
        dp = oa.digit_flexible_extract
        self.assertTrue(oa.boxed_first_correct("\\boxed{\\$16.00}", "16", dp("\\boxed{\\$16.00}")))
        self.assertTrue(oa.boxed_first_correct("\\boxed{60\\%}", "60", dp("\\boxed{60\\%}")))
        self.assertTrue(oa.boxed_first_correct("\\boxed{18 \\text{ apples}}", "18", dp("x")))
        self.assertFalse(oa.boxed_first_correct("\\boxed{17}", "18", dp("\\boxed{17}")))
        # non-numeric box -> fall back to the digit-required stock rule on the whole response
        resp = "\\boxed{\\text{unknown}} the answer is 18"
        self.assertTrue(oa.boxed_first_correct(resp, "18", dp(resp)))


class TestStatistics(unittest.TestCase):
    def test_auc_matches_pair_counting_with_ties(self):
        rnd = random.Random(0)
        scores = [rnd.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) for _ in range(60)]
        labels = [rnd.randint(0, 1) for _ in range(60)]
        pos = [s for s, y in zip(scores, labels) if y == 1]
        neg = [s for s, y in zip(scores, labels) if y == 0]
        brute = sum((p > q) + 0.5 * (p == q) for p, q in itertools.product(pos, neg)) / (len(pos) * len(neg))
        self.assertAlmostEqual(oa.roc_auc(scores, labels), brute, places=12)
        self.assertEqual(oa.roc_auc([1, 2, 3, 4], [0, 0, 1, 1]), 1.0)
        self.assertTrue(np.isnan(oa.roc_auc([1, 2], [1, 1])))

    def test_paired_bootstrap(self):
        rng = np.random.default_rng(6304)
        same = oa.paired_bootstrap(np.ones(50), np.ones(50), rng, n_boot=200)
        self.assertEqual((same["diff_pp"], same["ci_lo_pp"], same["ci_hi_pp"], same["significant"]), (0.0, 0.0, 0.0, False))
        a = np.array([1.0] * 80 + [0.0] * 20)
        b = np.zeros(100)
        res = oa.paired_bootstrap(a, b, rng, n_boot=500)
        self.assertAlmostEqual(res["diff_pp"], 80.0)
        self.assertTrue(res["significant"] and res["ci_lo_pp"] > 60.0 and res["ci_hi_pp"] < 95.0)
        # deterministic given the generator state
        r1 = oa.paired_bootstrap(a, b, np.random.default_rng(1), n_boot=300)
        r2 = oa.paired_bootstrap(a, b, np.random.default_rng(1), n_boot=300)
        self.assertEqual(r1, r2)
        with self.assertRaises(ValueError):
            oa.paired_bootstrap(np.ones(3), np.ones(4), rng)

    def test_analyse_boolq_bias_and_accuracy_check(self):
        # choices ["no","yes"]; target 1 == gold yes. A model that always says "no".
        rows = [{"doc_id": str(i), "target": str(t), "loglikelihoods": [-1.0, -2.0 - 0.1 * i],
                 "acc": float(t == 0)} for i, t in enumerate([1, 1, 0, 0, 1])]
        s, ok = oa.analyse_boolq(rows)
        self.assertEqual(s["pred_yes_pct"], 0.0)
        self.assertAlmostEqual(s["acc_gold_yes_pct"], 0.0)
        self.assertAlmostEqual(s["acc_gold_no_pct"], 100.0)
        self.assertAlmostEqual(s["balanced_accuracy_pct"], 50.0)
        self.assertAlmostEqual(s["accuracy_pct"], 40.0)
        self.assertEqual(s["accuracy_check_mismatches"], 0)
        self.assertEqual(ok.tolist(), [0, 0, 1, 1, 0])


def _write_samples(samples_dir, cond, n_gsm=24, n_ll=40, seed=0, shuffle_ids=False):
    rnd = random.Random(seed)
    gsm = []
    for i in range(n_gsm):
        target = str(rnd.randint(1, 99))
        tail = "$$" if i % 3 == 0 else ""
        resp = f"work... The answer is {target if rnd.random() < 0.6 else 7}\n{tail}"
        ext = oa.stock_flexible_extract(resp)
        gsm.append({"doc_id": str(i), "target": target, "response": resp, "extracted_flexible-extract": ext,
                    "correct_flexible-extract": oa.stock_correct(ext, target), "n_tokens": 10 + i, "truncated": False})
    ids = list(range(n_ll))
    if shuffle_ids:
        ids = [i + 1000 for i in ids]
    mk = lambda i, k: {"doc_id": str(i), "target": str(rnd.randint(0, 1)), "acc": float(rnd.random() < 0.5),
                       "acc_norm": float(rnd.random() < 0.5), "loglikelihoods": [-rnd.random(), -rnd.random()]}
    boolq = []
    for i in ids:
        r = mk(i, 2)
        r["acc"] = float(int(np.argmax(r["loglikelihoods"])) == int(r["target"]))
        boolq.append(r)
    payload = {"gsm8k": gsm, "hellaswag": [mk(i, 4) for i in ids], "boolq": boolq, "arc_easy": [mk(i, 4) for i in ids]}
    for b, rows in payload.items():
        with open(os.path.join(samples_dir, f"{cond}_{b}.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f)


class TestRunOfflineAnalysis(unittest.TestCase):
    def _cfg(self, root, conds):
        return {"checkpoints": {"output_dir": os.path.join(root, "outputs")}, "results_dir": os.path.join(root, "results"),
                "model": {"seed": 6304}, "evaluation": {"eval_conditions": conds}}

    def test_end_to_end_synthetic(self):
        with tempfile.TemporaryDirectory() as root:
            sd = os.path.join(root, "outputs", "eval_samples")
            os.makedirs(sd)
            for k, c in enumerate(["baseline", "lora_4_final", "lora_16_final"]):
                _write_samples(sd, c, seed=k)
            # make the doc ids of the two LL benchmarks / gsm identical across conditions (they are: same ranges)
            res = oa.run_offline_analysis(self._cfg(root, ["baseline", "lora_4_final", "lora_16_final", "oplora_16_final"]), n_boot=50)
            self.assertEqual(res["settings"]["conditions"], ["baseline", "lora_4_final", "lora_16_final"])  # incomplete one skipped
            with open(os.path.join(root, "results", "offline_analysis.json"), encoding="utf-8") as f:
                on_disk = json.load(f)
            self.assertEqual(on_disk["gsm8k_extraction"].keys(), res["gsm8k_extraction"].keys())
            for c, s in res["gsm8k_extraction"].items():
                self.assertEqual(s["replication_correctness_mismatches"], 0)
                self.assertEqual(s["replication_extraction_mismatches"], 0)
                # the digit rule can only recover answers hidden behind digit-free extractions
                self.assertGreaterEqual(s["digit_flexible_pct"], s["stock_flexible_pct"])
                self.assertGreater(s["no_digit_extraction_pct"], 0.0)
            ids = {p["id"] for p in res["paired"]}
            self.assertIn("rank", ids)                      # lora_16 vs lora_4 are both present
            self.assertIn("lora_16_final_vs_baseline", ids)
            self.assertNotIn("orthogonal", ids)             # oplora_16 not available
            for p in res["paired"]:
                self.assertEqual(set(p["benchmarks"]), set(oa.PAIRED_COLUMNS))

    def test_mismatched_doc_ids_refuse_to_pair(self):
        with tempfile.TemporaryDirectory() as root:
            sd = os.path.join(root, "outputs", "eval_samples")
            os.makedirs(sd)
            _write_samples(sd, "baseline", seed=0)
            _write_samples(sd, "lora_4_final", seed=1, shuffle_ids=True)
            with self.assertRaises(ValueError):
                oa.run_offline_analysis(self._cfg(root, ["baseline", "lora_4_final"]), n_boot=10)

    def test_missing_baseline_raises(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "outputs", "eval_samples"))
            with self.assertRaises(FileNotFoundError):
                oa.run_offline_analysis(self._cfg(root, ["baseline"]), n_boot=10)


@unittest.skipUnless(os.path.exists(os.path.join(REPO, "outputs", "task1", "eval_samples", "baseline_gsm8k.json")),
                     "real per-example outputs not present (outputs/ is git-ignored)")
class TestRealData(unittest.TestCase):
    """The replicated stock filter must reproduce lm-eval's stored extraction exactly on the real outputs."""

    def test_replication_and_headline_numbers(self):
        sd = os.path.join(REPO, "outputs", "task1", "eval_samples")
        for cond in oa.ALL_CONDITIONS:
            rows = oa.load_condition_rows(sd, cond, "gsm8k")
            s, vecs = oa.analyse_gsm8k(rows)
            self.assertEqual(s["replication_correctness_mismatches"], 0, cond)
            self.assertEqual(s["replication_extraction_mismatches"], 0, cond)
            ev = os.path.join(REPO, "results", "task1", "eval", f"{cond}.json")
            if os.path.exists(ev):
                with open(ev, encoding="utf-8") as f:
                    sc = json.load(f)["scores"]
                self.assertAlmostEqual(s["stock_flexible_pct"], 100.0 * sc["gsm8k"]["exact_match_flexible"], places=6)
                brows = oa.load_condition_rows(sd, cond, "boolq")
                b, _ = oa.analyse_boolq(brows)
                self.assertEqual(b["accuracy_check_mismatches"], 0, cond)
                self.assertAlmostEqual(b["accuracy_pct"], 100.0 * sc["boolq"]["acc"], places=6)


if __name__ == "__main__":
    unittest.main()
