"""Tests for the real-data pipeline of Task 2 using small COCO / VQAv2-format fixtures (no downloads)."""

import json
import os
import random
import shutil
import tempfile
import unittest

from task2_vlm.data_prep import (
    build_task2_datasets,
    collect_image_keys,
    image_key,
    load_datasets,
    pluralize,
    save_datasets,
    verify_task2_isolation,
    with_article,
)

CATS = ["dog", "cat", "person", "bus", "sheep", "bench", "car", "chair"]

SMALL_QUOTAS = {
    "train_caption_images": 120, "captions_per_train_image": 3, "dev_caption_images": 8,
    "train_presence_pairs": 40, "train_counting_pairs": 40,
    "dev_presence_pairs": 3, "dev_counting_pairs": 3, "test_presence_pairs": 4, "test_counting_pairs": 4,
    "vqa": {
        "train": {"yes/no": 12, "number": 6, "other": 12},
        "dev": {"yes/no": 2, "number": 2, "other": 2},
        "test": {"yes/no": 3, "number": 2, "other": 3},
    },
}


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def make_fixture(root, n_train=400, n_val=160, seed=1):
    rng = random.Random(seed)
    coco, vqa = os.path.join(root, "coco"), os.path.join(root, "vqa")
    ann_id = 1
    qid = 1
    for split, n, base in (("train", n_train, 1000), ("val", n_val, 900000)):
        images, anns, caps = [], [], []
        for i in range(n):
            iid = base + i
            images.append({"id": iid})
            for c_idx, name in enumerate(CATS):
                for _ in range(rng.choice([0, 0, 1, 2, 3, 5])):
                    anns.append({"id": ann_id, "image_id": iid, "category_id": c_idx + 1, "iscrowd": 0})
                    ann_id += 1
            for k in range(5):
                caps.append({"id": iid * 10 + k, "image_id": iid, "caption": f"a photo number {k} of image {iid}."})
        cats = [{"id": i + 1, "name": n_} for i, n_ in enumerate(CATS)]
        _write(os.path.join(coco, "annotations", f"instances_{split}2014.json"), {"images": images, "annotations": anns, "categories": cats})
        _write(os.path.join(coco, "annotations", f"captions_{split}2014.json"), {"annotations": caps})

        questions, annotations, pairs = [], [], []
        for p in range(n // 2):
            kind = ["yes/no", "number", "other"][p % 3]
            text = {"yes/no": "Is it sunny?", "number": "How many animals?", "other": "What color is the wall?"}[kind] + f" #{p}"
            if kind == "yes/no":
                a1, a2 = "yes", "no"
            elif kind == "number":
                a1, a2 = "two", "3"
            else:
                a1, a2 = "white", "blue"
            q1, q2 = qid, qid + 1
            qid += 2
            i1, i2 = base + 2 * p, base + 2 * p + 1
            for q, i, a in ((q1, i1, a1), (q2, i2, a2)):
                questions.append({"question_id": q, "image_id": i, "question": text})
                answers = [{"answer": a} for _ in range(9)] + [{"answer": a if p % 4 else "maybe"}]
                annotations.append({"question_id": q, "image_id": i, "answer_type": kind, "answers": answers})
            pairs.append([q1, q2])
        _write(os.path.join(vqa, f"v2_OpenEnded_mscoco_{split}2014_questions.json"), {"questions": questions})
        _write(os.path.join(vqa, f"v2_mscoco_{split}2014_annotations.json"), {"annotations": annotations})
        _write(os.path.join(vqa, f"v2_mscoco_{split}2014_complementary_pairs.json"), pairs)
    return coco, vqa


class TestTask2DataPrep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="tmp_test_t2prep_")
        cls.coco, cls.vqa = make_fixture(cls.root)
        cls.ds = build_task2_datasets(cls.coco, cls.vqa, seed=6304, quotas=SMALL_QUOTAS)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_sizes_follow_the_quotas(self):
        ds = self.ds
        self.assertEqual(len(ds["train_captions"]), 120 * 3)
        self.assertEqual(len(ds["dev_captions"]), 8)
        self.assertEqual(len(ds["train_presence"]), 40)
        self.assertEqual(len(ds["train_counting"]), 40)
        self.assertEqual(len(ds["dev_grounding"]), 6)
        self.assertEqual(len(ds["test_grounding"]), 8)
        self.assertEqual(len(ds["train_vqav2"]), 30)
        self.assertEqual(len(ds["dev_vqav2"]), 6)
        self.assertEqual(len(ds["test_vqav2"]), 8)

    def test_vqa_type_quotas_and_consensus_rules(self):
        by_type = {}
        for p in self.ds["test_vqav2"]:
            by_type[p.example_type] = by_type.get(p.example_type, 0) + 1
            self.assertEqual(p.example_a.question, p.example_b.question)
            self.assertEqual(len(p.example_a.all_references), 10)
            self.assertNotEqual(p.example_a.target.lower(), p.example_b.target.lower())
        self.assertEqual(by_type, {"vqa_yes_no": 3, "vqa_number": 2, "vqa_other": 3})

    def test_splits_are_image_disjoint(self):
        stats = verify_task2_isolation(self.ds)
        self.assertTrue(stats["is_disjoint"])
        val_sets = {
            "dev": [i for c in ("dev_captions", "dev_grounding", "dev_vqav2") for i in self._imgs(self.ds[c])],
            "test": [i for c in ("test_grounding", "test_vqav2") for i in self._imgs(self.ds[c])],
        }
        self.assertEqual(len(val_sets["dev"]), len(set(val_sets["dev"])))      # unique inside dev
        self.assertEqual(len(val_sets["test"]), len(set(val_sets["test"])))    # unique inside test
        self.assertFalse(set(val_sets["dev"]) & set(val_sets["test"]))         # dev vs test
        self.assertTrue(all(k.startswith("val2014_") for k in val_sets["dev"] + val_sets["test"]))
        train_imgs = set(self._imgs(self.ds["train_captions"] + self.ds["train_presence"] + self.ds["train_counting"] + self.ds["train_vqav2"]))
        self.assertTrue(all(k.startswith("train2014_") for k in train_imgs))

    @staticmethod
    def _imgs(items):
        out = []
        for it in items:
            if hasattr(it, "example_a"):
                out += [it.example_a.image_id, it.example_b.image_id]
            else:
                out.append(it.image_id)
        return out

    def test_grounding_labels_are_consistent_with_annotations(self):
        inst = json.load(open(os.path.join(self.coco, "annotations", "instances_val2014.json")))
        counts = {}
        for a in inst["annotations"]:
            counts.setdefault(image_key("val", a["image_id"]), {}).setdefault(CATS[a["category_id"] - 1], 0)
            counts[image_key("val", a["image_id"])][CATS[a["category_id"] - 1]] += 1
        for p in self.ds["test_grounding"]:
            if p.example_type == "presence":
                cat = p.question.replace("Does this image contain ", "").rstrip("?")
                cat = cat.split(" ", 1)[1] if cat.startswith(("a ", "an ")) else cat
                self.assertGreater(counts.get(p.example_a.image_id, {}).get(cat, 0), 0)
                self.assertEqual(counts.get(p.example_b.image_id, {}).get(cat, 0), 0)
                self.assertEqual((p.example_a.target, p.example_b.target), ("yes", "no"))
            else:
                c1, c2 = int(p.example_a.target), int(p.example_b.target)
                self.assertNotEqual(c1, c2)
                self.assertTrue(1 <= c1 <= 4 and 1 <= c2 <= 4)

    def test_captions_have_three_per_image(self):
        per_image = {}
        for ex in self.ds["train_captions"]:
            per_image[ex.image_id] = per_image.get(ex.image_id, 0) + 1
        self.assertEqual(set(per_image.values()), {3})
        self.assertEqual(len({ex.image_id for ex in self.ds["dev_captions"]}), 8)

    def test_determinism_and_roundtrip(self):
        again = build_task2_datasets(self.coco, self.vqa, seed=6304, quotas=SMALL_QUOTAS)
        self.assertEqual(collect_image_keys(again), collect_image_keys(self.ds))
        self.assertEqual([p.pair_id for p in again["test_vqav2"]], [p.pair_id for p in self.ds["test_vqav2"]])
        other = build_task2_datasets(self.coco, self.vqa, seed=1, quotas=SMALL_QUOTAS)
        self.assertNotEqual([p.pair_id for p in other["test_vqav2"]], [p.pair_id for p in self.ds["test_vqav2"]])

        path, ids = os.path.join(self.root, "d.json"), os.path.join(self.root, "ids.json")
        save_datasets(self.ds, path, ids)
        loaded = load_datasets(path)
        self.assertEqual([p.pair_id for p in loaded["dev_grounding"]], [p.pair_id for p in self.ds["dev_grounding"]])
        self.assertEqual(loaded["train_captions"][0].target, self.ds["train_captions"][0].target)
        self.assertIn("test_vqav2", json.load(open(ids)))

    def test_wording_helpers(self):
        self.assertEqual(pluralize("person"), "people")
        self.assertEqual(pluralize("bus"), "buses")
        self.assertEqual(pluralize("dog"), "dogs")
        self.assertEqual(pluralize("skis"), "skis")
        self.assertEqual(with_article("apple"), "an apple")
        self.assertEqual(with_article("dog"), "a dog")
        self.assertEqual(with_article("scissors"), "scissors")


if __name__ == "__main__":
    unittest.main()
