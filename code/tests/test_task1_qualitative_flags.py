"""Tests for the qualitative math-framing flags and the annotation-subset bookkeeping."""
import json
import os
import tempfile
import unittest

from task1_llm_ft import qualitative_flags as qf
from task1_llm_ft.evaluate import generate_annotation_subset
from task1_llm_ft.offline_analysis import analyse_qualitative_math_intrusion

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestMathFramingSignals(unittest.TestCase):
    def test_training_artefacts_are_flagged(self):
        cases = {
            "answer_is": ["I want to travel.\nThe answer is: 1", "ok. The answer is: humans, values", "THE ANSWER IS: x"],
            "gsm8k_marker": ["blah\n#### 66\n#### 67", "#### -4", "  #### $12"],
            "boxed": ["so \\boxed{18}", "\\boxed {7}"],
            "latex": ["total $$x$$", "\\(a+b\\)", "\\[ y \\]", "\\frac{1}{2} of it", "3 \\times 4"],
            "calc_annotation": ["He pays <<48/2=24>>24 dollars"],
        }
        for name, texts in cases.items():
            for t in texts:
                self.assertIn(name, qf.math_framing_signals(t), (name, t))
                self.assertTrue(qf.has_math_artefact(t), t)

    def test_ordinary_prose_is_not_flagged(self):
        for t in ("The answer is yes, you can.",                 # no colon: an ordinary sentence
                  "I think the answer is simple.",
                  "#### Summary\nBe kind to yourself.",          # a markdown heading, not the GSM8K marker
                  "### Step 1: talk to him\n- listen",
                  "It costs $5 and $6 to start.",                # single dollar signs
                  "Use #hashtags and # comments in code.",
                  "wish = # Your wish\nreason = # Why you want it",
                  ""):
            self.assertEqual(qf.math_framing_signals(t), [], t)

    def test_vocabulary_is_separate_from_artefacts(self):
        s = qf.math_framing_signals("Let x = 5 then the matrix is large")
        self.assertEqual(s, ["vocabulary"])
        self.assertFalse(qf.has_math_artefact("Let x = 5 then the matrix is large"))
        # 'proof:' / 'q.e.d.' end in a non-word character: the old trailing \b made them unmatchable
        self.assertIn("vocabulary", qf.math_framing_signals("Proof: trivial."))
        self.assertIn("vocabulary", qf.math_framing_signals("... Q.E.D."))
        self.assertEqual(qf.math_framing_signals("a variablehood of things"), [])    # word boundaries respected

    def test_signal_order_is_stable(self):
        s = qf.math_framing_signals("The answer is: 3\n#### 3\n\\boxed{3}\n$$ \\frac{1}{2}\nequation")
        self.assertEqual(s, ["answer_is", "gsm8k_marker", "boxed", "latex", "vocabulary"])
        self.assertEqual(tuple(s), tuple(n for n in qf.SIGNAL_NAMES if n in s))


def _responses(extra=None):
    out = []
    for q in ("what_is_your_wish", "enough_of_my_husband"):
        out += [{"id": q, "format": "plain", "response": "Take time to talk.", "truncated": False},
                {"id": f"{q}_json", "format": "json", "response": '{"a": 1}', "truncated": False},
                {"id": f"{q}_template", "format": "template", "response": "wish = x\nreason = y", "truncated": False}]
    for r in out:
        r.update((extra or {}).get(r["id"], {}))
    return out


class TestAnnotationSubset(unittest.TestCase):
    CONDS = ["baseline", "full_ft_midpoint", "full_ft_final", "full_ft_l2_sp_final", "lora_4_final", "lora_16_final",
             "oplora_16_final"]

    def _all(self, intruded=None):
        return {c: _responses({"what_is_your_wish_template": {"response": "wish = x\nThe answer is: 1"}}
                              if c == intruded else None) for c in self.CONDS}

    def test_flags_and_signal_field(self):
        with tempfile.TemporaryDirectory() as d:
            res = generate_annotation_subset(self._all(intruded="full_ft_final"), output_dir=d)
            rec = {(r["condition"], r["question_id"]): r for r in res["records"]}
            hit = rec[("full_ft_final", "what_is_your_wish_template")]
            self.assertTrue(hit["unnecessary_math_framing"])
            self.assertEqual(hit["math_framing_signals"], ["answer_is"])
            self.assertEqual(sum(r["unnecessary_math_framing"] for r in res["records"]), 1)
            self.assertEqual(rec[("baseline", "what_is_your_wish_template")]["math_framing_signals"], [])
            with open(os.path.join(d, "annotation_subset_42.md"), encoding="utf-8") as f:
                md = f.read()
            self.assertIn("Yes (answer_is)", md)

    def test_regeneration_keeps_manual_annotations(self):
        with tempfile.TemporaryDirectory() as d:
            generate_annotation_subset(self._all(), output_dir=d)
            path = os.path.join(d, "annotation_subset_42.json")
            with open(path, encoding="utf-8") as f:
                rows = json.load(f)
            rows[3]["inappropriate_content"] = False
            rows[3]["manual_notes"] = "polite but off-topic"
            rows[10]["inappropriate_content"] = True
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f)
            res = generate_annotation_subset(self._all(intruded="lora_4_final"), output_dir=d)
            again = res["records"]
            self.assertIs(again[3]["inappropriate_content"], False)
            self.assertEqual(again[3]["manual_notes"], "polite but off-topic")
            self.assertIs(again[10]["inappropriate_content"], True)
            # nothing else was invented, and the automatic flags were refreshed
            others = [r for i, r in enumerate(again) if i not in (3, 10)]
            self.assertTrue(all(r["inappropriate_content"] is None and r["manual_notes"] == "" for r in others))
            self.assertEqual(sum(r["unnecessary_math_framing"] for r in again), 1)


class TestQualitativeIntrusionCounts(unittest.TestCase):
    def test_counts(self):
        rows = _responses({"what_is_your_wish": {"response": "x\n#### 5"},
                           "enough_of_my_husband_json": {"response": '{"a": 1}\nThe answer is: {"a": 1}'},
                           "what_is_your_wish_template": {"response": "the matrix"}})
        res = analyse_qualitative_math_intrusion(rows)
        self.assertEqual((res["n_responses"], res["n_flagged"], res["n_with_artefact"]), (6, 3, 2))
        self.assertEqual(res["by_signal"]["gsm8k_marker"], 1)
        self.assertEqual(res["by_signal"]["answer_is"], 1)
        self.assertEqual(res["by_signal"]["vocabulary"], 1)
        self.assertEqual(res["by_format"]["json"], {"n": 2, "flagged": 1})
        self.assertEqual(set(res["flagged_ids"]), {"what_is_your_wish", "enough_of_my_husband_json",
                                                   "what_is_your_wish_template"})


@unittest.skipUnless(os.path.exists(os.path.join(REPO, "results", "task1", "qualitative", "baseline.json")),
                     "saved qualitative responses not present")
class TestRealQualitative(unittest.TestCase):
    def test_baseline_clean_and_fine_tuned_models_intrude(self):
        d = os.path.join(REPO, "results", "task1", "qualitative")
        counts = {}
        for c in ("baseline", "full_ft_midpoint", "full_ft_final", "full_ft_l2_sp_final", "lora_4_final",
                  "lora_16_final", "oplora_16_final"):
            with open(os.path.join(d, f"{c}.json"), encoding="utf-8") as f:
                rows = json.load(f)
            self.assertEqual(len(rows), 24, c)
            counts[c] = analyse_qualitative_math_intrusion(rows)["n_with_artefact"]
        self.assertEqual(counts["baseline"], 0)
        for c, n in counts.items():
            if c != "baseline":
                self.assertGreater(n, 0, c)


if __name__ == "__main__":
    unittest.main()
