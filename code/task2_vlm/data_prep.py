"""Real-data builders for Task 2 (COCO 2014 captions / instances and VQAv2 complementary pairs).

Expected layout (see scripts/download_assets.py)::

    data/coco2014/train2014/COCO_train2014_<12 digit id>.jpg      data/coco2014/val2014/...
    data/coco2014/annotations/{captions,instances}_{train,val}2014.json
    data/vqav2/v2_OpenEnded_mscoco_{train,val}2014_questions.json
    data/vqav2/v2_mscoco_{train,val}2014_annotations.json
    data/vqav2/v2_mscoco_{train,val}2014_complementary_pairs.json

Protocol implemented here (seed 6304 throughout):
- training images come from train2014; development and final-test images from val2014;
- development and test sets (captions, grounding, VQA) are image-disjoint from each other,
  every evaluation image is used once, and no evaluation image can enter training or replay;
- COCO captions: 6,000 train images x 3 captions, 96 dev images x 1 caption;
- annotation-derived grounding pairs: presence ("Does this image contain a dog?") pairs an image that
  has the annotated category with one without any annotation of it (yes / no); counting
  ("How many dogs can you see?") pairs images with different annotated counts in 1..4 (crowd
  annotations make a category's count ambiguous, so such images are skipped for that category);
  training uses reusable pools that the curriculum samples with replacement;
- natural VQAv2 pairs: same question on two images, consensus answers differ after normalization,
  >= 8/10 annotators agree on each image, target = an original answer string of the consensus group,
  all ten answers kept for scoring; quotas yes/no : number : other = 640:256:640 (train),
  12:8:12 (dev), 36:24:36 (test).
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import json
import os
import random

from common.metrics import normalize_text_vqa
from task2_vlm.dataset import VLMExample, QuestionPair, verify_split_isolation

DEFAULT_QUOTAS: Dict[str, Any] = {
    "train_caption_images": 6000,
    "captions_per_train_image": 3,
    "dev_caption_images": 96,
    "train_presence_pairs": 2000,
    "train_counting_pairs": 2000,
    "dev_presence_pairs": 12,
    "dev_counting_pairs": 12,
    "test_presence_pairs": 32,
    "test_counting_pairs": 32,
    "vqa": {
        "train": {"yes/no": 640, "number": 256, "other": 640},
        "dev": {"yes/no": 12, "number": 8, "other": 12},
        "test": {"yes/no": 36, "number": 24, "other": 36},
    },
}

_IRREGULAR_PLURALS = {"person": "people", "knife": "knives", "mouse": "mice", "sheep": "sheep",
                      "skis": "skis", "scissors": "scissors", "broccoli": "broccoli"}
_NO_ARTICLE = {"skis", "scissors", "broccoli"}
CAPTION_PROMPT = "Describe this image."


def image_key(split: str, image_id: int) -> str:
    """Stable key used for feature caching and split bookkeeping, e.g. ``val2014_000000000042``."""
    return f"{split}2014_{int(image_id):012d}"


def image_path(coco_root: str, key: str) -> str:
    split = key.split("_")[0]
    image_id = int(key.split("_")[1])
    for name in (f"COCO_{split}_{image_id:012d}.jpg", f"{image_id:012d}.jpg"):
        path = os.path.join(coco_root, split, name)
        if os.path.exists(path):
            return path
    return os.path.join(coco_root, split, f"COCO_{split}_{image_id:012d}.jpg")


def pluralize(name: str) -> str:
    if name in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[name]
    if name.endswith(("s", "x", "z", "ch", "sh")):
        return name + "es"
    return name + "s"


def with_article(name: str) -> str:
    if name in _NO_ARTICLE:
        return name
    return ("an " if name[0] in "aeiou" else "a ") + name


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------
# COCO
# ----------------------------------------------------------------------------------------------

def load_coco_split(coco_root: str, split: str) -> Dict[str, Any]:
    """Captions, per-image category sets/counts and crowd flags for ``split`` in {'train', 'val'}."""
    ann_dir = os.path.join(coco_root, "annotations")
    caps = _load_json(os.path.join(ann_dir, f"captions_{split}2014.json"))
    inst = _load_json(os.path.join(ann_dir, f"instances_{split}2014.json"))

    captions: Dict[str, List[str]] = {}
    for a in sorted(caps["annotations"], key=lambda a: a["id"]):
        captions.setdefault(image_key(split, a["image_id"]), []).append(a["caption"].strip())

    categories = {c["id"]: c["name"] for c in inst["categories"]}
    counts: Dict[str, Dict[str, int]] = {image_key(split, im["id"]): {} for im in inst["images"]}
    crowd: Dict[str, Set[str]] = {}
    for a in inst["annotations"]:
        key = image_key(split, a["image_id"])
        name = categories[a["category_id"]]
        if a.get("iscrowd", 0):
            crowd.setdefault(key, set()).add(name)
        counts.setdefault(key, {})
        counts[key][name] = counts[key].get(name, 0) + 1
    return {"captions": captions, "counts": counts, "crowd": crowd, "categories": sorted(categories.values())}


# ----------------------------------------------------------------------------------------------
# Annotation-derived grounding pairs
# ----------------------------------------------------------------------------------------------

def _pick_unused(rng: random.Random, pool: Sequence[str], used: Set[str], unique: bool) -> Optional[str]:
    if not unique:
        return rng.choice(pool) if pool else None
    for _ in range(60):
        if not pool:
            return None
        cand = rng.choice(pool)
        if cand not in used:
            return cand
    free = [p for p in pool if p not in used]
    return rng.choice(free) if free else None


def build_presence_pairs(
    counts: Dict[str, Dict[str, int]], categories: Sequence[str], n: int, rng: random.Random,
    prefix: str, unique_images: bool, blocked: Set[str],
) -> List[QuestionPair]:
    images = [k for k in sorted(counts) if k not in blocked]
    with_cat = {c: [k for k in images if counts[k].get(c, 0) > 0] for c in categories}
    without_cat = {c: [k for k in images if counts[k].get(c, 0) == 0] for c in categories}
    cats = [c for c in categories if with_cat[c] and without_cat[c]]
    rng.shuffle(cats)
    used: Set[str] = set()
    pairs: List[QuestionPair] = []
    guard = 0
    while len(pairs) < n and guard < n * 200:
        guard += 1
        cat = cats[guard % len(cats)]
        pos = _pick_unused(rng, with_cat[cat], used, unique_images)
        neg = _pick_unused(rng, without_cat[cat], used, unique_images)
        if pos is None or neg is None or pos == neg:
            continue
        question = f"Does this image contain {with_article(cat)}?"
        pid = f"{prefix}_presence_{len(pairs)}"
        pairs.append(QuestionPair(
            pid, question, "presence",
            VLMExample(pos, question, "yes", "presence", ["yes"], pid),
            VLMExample(neg, question, "no", "presence", ["no"], pid)))
        used.update((pos, neg))
    if len(pairs) < n:
        raise RuntimeError(f"Could only build {len(pairs)}/{n} presence pairs for '{prefix}'")
    return pairs


def build_counting_pairs(
    counts: Dict[str, Dict[str, int]], crowd: Dict[str, Set[str]], categories: Sequence[str], n: int,
    rng: random.Random, prefix: str, unique_images: bool, blocked: Set[str],
) -> List[QuestionPair]:
    buckets: Dict[Tuple[str, int], List[str]] = {}
    for key in sorted(counts):
        if key in blocked:
            continue
        for cat, cnt in counts[key].items():
            if 1 <= cnt <= 4 and cat not in crowd.get(key, set()):
                buckets.setdefault((cat, cnt), []).append(key)
    cats = [c for c in categories if sum(1 for k in range(1, 5) if buckets.get((c, k))) >= 2]
    rng.shuffle(cats)
    used: Set[str] = set()
    pairs: List[QuestionPair] = []
    guard = 0
    while len(pairs) < n and guard < n * 200:
        guard += 1
        cat = cats[guard % len(cats)]
        avail = [k for k in range(1, 5) if buckets.get((cat, k))]
        c1, c2 = rng.sample(avail, 2)
        a = _pick_unused(rng, buckets[(cat, c1)], used, unique_images)
        b = _pick_unused(rng, buckets[(cat, c2)], used, unique_images)
        if a is None or b is None or a == b:
            continue
        question = f"How many {pluralize(cat)} can you see?"
        pid = f"{prefix}_counting_{len(pairs)}"
        pairs.append(QuestionPair(
            pid, question, "counting",
            VLMExample(a, question, str(c1), "counting", [str(c1)], pid),
            VLMExample(b, question, str(c2), "counting", [str(c2)], pid)))
        used.update((a, b))
    if len(pairs) < n:
        raise RuntimeError(f"Could only build {len(pairs)}/{n} counting pairs for '{prefix}'")
    return pairs


# ----------------------------------------------------------------------------------------------
# Natural VQAv2 complementary pairs
# ----------------------------------------------------------------------------------------------

def load_consensus_vqa_pairs(vqa_root: str, split: str) -> Dict[str, List[QuestionPair]]:
    """All complementary pairs of ``split`` that satisfy the consensus filter, grouped by answer type."""
    qs = {q["question_id"]: q for q in _load_json(
        os.path.join(vqa_root, f"v2_OpenEnded_mscoco_{split}2014_questions.json"))["questions"]}
    anns = {a["question_id"]: a for a in _load_json(
        os.path.join(vqa_root, f"v2_mscoco_{split}2014_annotations.json"))["annotations"]}
    pair_ids = _load_json(os.path.join(vqa_root, f"v2_mscoco_{split}2014_complementary_pairs.json"))

    out: Dict[str, List[QuestionPair]] = {"yes/no": [], "number": [], "other": []}
    for q1, q2 in pair_ids:
        if q1 not in anns or q2 not in anns or qs[q1]["question"] != qs[q2]["question"]:
            continue
        a1, a2 = anns[q1], anns[q2]
        if a1["answer_type"] != a2["answer_type"] or a1["answer_type"] not in out:
            continue
        ans1 = [x["answer"] for x in a1["answers"]]
        ans2 = [x["answer"] for x in a2["answers"]]
        if len(ans1) < 10 or len(ans2) < 10:
            continue
        n1, n2 = [normalize_text_vqa(a) for a in ans1], [normalize_text_vqa(a) for a in ans2]
        m1, m2 = max(sorted(set(n1)), key=n1.count), max(sorted(set(n2)), key=n2.count)
        if m1 == m2 or n1.count(m1) < 8 or n2.count(m2) < 8:
            continue
        t1 = next(a for a, na in zip(ans1, n1) if na == m1)
        t2 = next(a for a, na in zip(ans2, n2) if na == m2)
        etype = "vqa_" + a1["answer_type"].replace("/", "_")
        pid = f"{split}_{q1}_{q2}"
        question = qs[q1]["question"]
        out[a1["answer_type"]].append(QuestionPair(
            pid, question, etype,
            VLMExample(image_key(split, a1["image_id"]), question, t1, etype, ans1, pid),
            VLMExample(image_key(split, a2["image_id"]), question, t2, etype, ans2, pid)))
    return out


def select_vqa_pairs(
    pools: Dict[str, List[QuestionPair]], quotas: Dict[str, int], rng: random.Random,
    used_images: Optional[Set[str]],
) -> List[QuestionPair]:
    """Draw quota pairs per type; with ``used_images`` the two images must be new and are then reserved."""
    selected: List[QuestionPair] = []
    for atype in ("yes/no", "number", "other"):
        pool = sorted(pools[atype], key=lambda p: p.pair_id)
        rng.shuffle(pool)
        taken = 0
        for pair in pool:
            if taken >= quotas[atype]:
                break
            ia, ib = pair.example_a.image_id, pair.example_b.image_id
            if used_images is not None:
                if ia == ib or ia in used_images or ib in used_images:
                    continue
                used_images.update((ia, ib))
            selected.append(pair)
            taken += 1
        if taken < quotas[atype]:
            raise RuntimeError(f"Only {taken}/{quotas[atype]} usable '{atype}' VQA pairs")
    return selected


# ----------------------------------------------------------------------------------------------
# Full assembly
# ----------------------------------------------------------------------------------------------

def _caption_example(key: str, caption: str) -> VLMExample:
    return VLMExample(key, CAPTION_PROMPT, caption, "caption", [caption])


def build_task2_datasets(
    coco_root: str, vqa_root: str, seed: int = 6304, quotas: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    q = quotas or DEFAULT_QUOTAS
    rng = random.Random(seed)
    train, val = load_coco_split(coco_root, "train"), load_coco_split(coco_root, "val")
    categories = train["categories"]

    # ---- evaluation sets (val2014): image-disjoint from each other and from training ----
    used: Set[str] = set()
    vqa_val = load_consensus_vqa_pairs(vqa_root, "val")
    test_vqa = select_vqa_pairs(vqa_val, q["vqa"]["test"], rng, used)
    dev_vqa = select_vqa_pairs(vqa_val, q["vqa"]["dev"], rng, used)

    def grounding(prefix: str, n_pres: int, n_count: int) -> List[QuestionPair]:
        pres = build_presence_pairs(val["counts"], categories, n_pres, rng, prefix, True, used)
        for p in pres:
            used.update((p.example_a.image_id, p.example_b.image_id))
        cnt = build_counting_pairs(val["counts"], val["crowd"], categories, n_count, rng, prefix, True, used)
        for p in cnt:
            used.update((p.example_a.image_id, p.example_b.image_id))
        return pres + cnt

    test_grounding = grounding("test", q["test_presence_pairs"], q["test_counting_pairs"])
    dev_grounding = grounding("dev", q["dev_presence_pairs"], q["dev_counting_pairs"])

    free = [k for k in sorted(val["captions"]) if k not in used]
    dev_cap_keys = rng.sample(free, q["dev_caption_images"])
    used.update(dev_cap_keys)
    dev_captions = [_caption_example(k, rng.choice(val["captions"][k])) for k in dev_cap_keys]

    # ---- training sets (train2014) ----
    eligible = [k for k in sorted(train["captions"]) if len(train["captions"][k]) >= q["captions_per_train_image"]]
    cap_keys = rng.sample(eligible, q["train_caption_images"])
    train_captions = [
        _caption_example(k, c) for k in cap_keys for c in rng.sample(train["captions"][k], q["captions_per_train_image"])
    ]
    vqa_train = load_consensus_vqa_pairs(vqa_root, "train")
    train_vqa = select_vqa_pairs(vqa_train, q["vqa"]["train"], rng, None)

    pool_images = set(cap_keys) | {p.example_a.image_id for p in train_vqa} | {p.example_b.image_id for p in train_vqa}
    pool_counts = {k: v for k, v in train["counts"].items() if k in pool_images}
    train_presence = build_presence_pairs(pool_counts, categories, q["train_presence_pairs"], rng, "train", False, set())
    train_counting = build_counting_pairs(pool_counts, train["crowd"], categories, q["train_counting_pairs"], rng, "train", False, set())

    datasets = {
        "train_captions": train_captions, "dev_captions": dev_captions,
        "train_presence": train_presence, "train_counting": train_counting, "train_vqav2": train_vqa,
        "dev_grounding": dev_grounding, "test_grounding": test_grounding,
        "dev_vqav2": dev_vqa, "test_vqav2": test_vqa,
    }
    datasets["stats"] = dataset_statistics(datasets)
    verify_task2_isolation(datasets)
    return datasets


def _images_of(items: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for it in items:
        if isinstance(it, QuestionPair):
            out += [it.example_a.image_id, it.example_b.image_id]
        else:
            out.append(it.image_id)
    return out


def verify_task2_isolation(datasets: Dict[str, Any]) -> Dict[str, Any]:
    """Raise unless train / dev / test images are disjoint and dev & test components do not overlap."""
    train = datasets["train_captions"] + datasets["train_presence"] + datasets["train_counting"] + datasets["train_vqav2"]
    dev = datasets["dev_captions"] + datasets["dev_grounding"] + datasets["dev_vqav2"]
    test = datasets["test_grounding"] + datasets["test_vqav2"]
    ok, stats = verify_split_isolation(train, dev, test)
    if not ok:
        raise RuntimeError(f"Train/dev/test images overlap: {stats}")
    for name in ("dev", "test"):
        comps = {"dev": ["dev_captions", "dev_grounding", "dev_vqav2"], "test": ["test_grounding", "test_vqav2"]}[name]
        imgs = [i for c in comps for i in _images_of(datasets[c])]
        if len(imgs) != len(set(imgs)):
            raise RuntimeError(f"{name} images are not unique across components")
    return stats


def dataset_statistics(datasets: Dict[str, Any]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for name, items in datasets.items():
        if name == "stats":
            continue
        stats[name] = {"items": len(items), "questions": len(items) * (2 if items and isinstance(items[0], QuestionPair) else 1)}
        if items and isinstance(items[0], QuestionPair):
            kinds: Dict[str, int] = {}
            for p in items:
                kinds[p.example_type] = kinds.get(p.example_type, 0) + 1
            stats[name]["by_type"] = kinds
    return stats


def collect_image_keys(datasets: Dict[str, Any]) -> List[str]:
    keys: Set[str] = set()
    for name, items in datasets.items():
        if name != "stats":
            keys.update(_images_of(items))
    return sorted(keys)


# ----------------------------------------------------------------------------------------------
# (De)serialisation
# ----------------------------------------------------------------------------------------------

def save_datasets(datasets: Dict[str, Any], path: str, ids_path: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {k: [asdict(x) for x in v] if k != "stats" else v for k, v in datasets.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    if ids_path:  # compact id lists for the repository
        ids: Dict[str, Any] = {"stats": datasets.get("stats", {})}
        for name, items in datasets.items():
            if name == "stats":
                continue
            if items and isinstance(items[0], QuestionPair):
                ids[name] = [{"pair_id": p.pair_id, "images": [p.example_a.image_id, p.example_b.image_id]} for p in items]
            else:
                ids[name] = [{"image": x.image_id} for x in items]
        os.makedirs(os.path.dirname(os.path.abspath(ids_path)), exist_ok=True)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(ids, f)


def load_datasets(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, Any] = {}
    for name, items in raw.items():
        if name == "stats":
            out[name] = items
        elif items and "example_a" in items[0]:
            out[name] = [QuestionPair(
                pair_id=i["pair_id"], question=i["question"], example_type=i["example_type"],
                example_a=VLMExample(**i["example_a"]), example_b=VLMExample(**i["example_b"])) for i in items]
        else:
            out[name] = [VLMExample(**i) for i in items]
    return out
