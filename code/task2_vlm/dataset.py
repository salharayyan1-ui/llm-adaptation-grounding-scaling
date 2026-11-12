"""Datasets, programmatic pair generation, and curriculum sampling for Task 2.

Handles:
1. COCO 2014 Captions (18,000 train examples, 96 dev examples).
2. Programmatic Grounding Pairs (Presence & Counting from COCO annotations).
3. Natural VQAv2 Complementary Pairs (640/256/640 train, 12/8/12 dev, 36/24/36 test).
4. Strict split isolation: train, dev, and test image sets are strictly disjoint.
5. Curriculum Sampler with exact per-stage draw probabilities preserving pairs.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
import os
import json
import random
import logging
from dataclasses import dataclass, field
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, IterableDataset
    HAS_TORCH = True
except ImportError:
    torch = None
    Dataset = object
    IterableDataset = object
    HAS_TORCH = False

from common.metrics import normalize_text_vqa
from task2_vlm.caching import FeatureCache
from task2_vlm.models import build_qwen_chat_prompt, IMAGE_PAD_TOKEN_ID, NUM_VISUAL_TOKENS

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SEED = 6304
QUESTION_TOKEN_CAP = 80
CAPTION_TOKEN_CAP = 64
ANSWER_TOKEN_CAP = 16


@dataclass
class VLMExample:
    """A single multimodal QA or captioning example."""
    image_id: Union[str, int]
    question: str
    target: str
    example_type: str  # 'caption', 'presence', 'counting', 'vqa_yesno', 'vqa_number', 'vqa_other'
    all_references: List[str] = field(default_factory=list)
    pair_id: Optional[str] = None
    image_path: Optional[str] = None


@dataclass
class QuestionPair:
    """A complementary pair: the same question applied to two distinct images."""
    pair_id: str
    question: str
    example_type: str
    example_a: VLMExample
    example_b: VLMExample


# -------------------------------------------------------------------------
# Programmatic Grounding Generation (Presence & Counting)
# -------------------------------------------------------------------------

def generate_presence_pairs(
    image_annotations: Dict[str, List[str]],
    categories: List[str],
    num_pairs: int,
    seed: int = DEFAULT_SEED,
) -> List[QuestionPair]:
    """Generate presence question pairs programmatically.
    
    Question template: "Does this image contain [object]?"
    Image A has the category (answer: "yes").
    Image B does NOT have the category (answer: "no").
    """
    if not categories or not image_annotations or num_pairs <= 0:
        return []
    rng = random.Random(seed)
    pairs = []
    
    # Inverted index: category -> list of image_ids containing it
    cat_to_images: Dict[str, List[str]] = {cat: [] for cat in categories}
    cat_to_absent: Dict[str, List[str]] = {cat: [] for cat in categories}
    
    for img_id, cats in image_annotations.items():
        present_set = set(cats)
        for cat in categories:
            if cat in present_set:
                cat_to_images[cat].append(img_id)
            else:
                cat_to_absent[cat].append(img_id)
                
    shuffled_cats = list(categories)
    rng.shuffle(shuffled_cats)
    
    pair_idx = 0
    cat_cycle_idx = 0
    while len(pairs) < num_pairs and cat_cycle_idx < len(shuffled_cats) * 20:
        cat = shuffled_cats[cat_cycle_idx % len(shuffled_cats)]
        cat_cycle_idx += 1
        
        pos_list = cat_to_images.get(cat, [])
        neg_list = cat_to_absent.get(cat, [])
        if not pos_list or not neg_list:
            continue
            
        img_pos = rng.choice(pos_list)
        img_neg = rng.choice(neg_list)
        if img_pos == img_neg:
            continue
            
        question = f"Does this image contain {cat}?"
        pair_id = f"presence_{pair_idx}_{cat}"
        
        ex_a = VLMExample(
            image_id=img_pos,
            question=question,
            target="yes",
            example_type="presence",
            all_references=["yes"],
            pair_id=pair_id,
        )
        ex_b = VLMExample(
            image_id=img_neg,
            question=question,
            target="no",
            example_type="presence",
            all_references=["no"],
            pair_id=pair_id,
        )
        
        pairs.append(QuestionPair(
            pair_id=pair_id,
            question=question,
            example_type="presence",
            example_a=ex_a,
            example_b=ex_b,
        ))
        pair_idx += 1
        
    return pairs[:num_pairs]


def generate_counting_pairs(
    image_cat_counts: Dict[str, Dict[str, int]],
    categories: List[str],
    num_pairs: int,
    seed: int = DEFAULT_SEED,
) -> List[QuestionPair]:
    """Generate counting question pairs programmatically.
    
    Question template: "How many [objects] can you see?"
    Images A and B have different annotated counts between 1 and 4.
    """
    if not categories or not image_cat_counts or num_pairs <= 0:
        return []
    rng = random.Random(seed)
    pairs = []
    
    # Bucket by category and count: (cat, count) -> list of image_ids
    count_buckets: Dict[Tuple[str, int], List[str]] = {}
    for img_id, cat_counts in image_cat_counts.items():
        for cat, cnt in cat_counts.items():
            if 1 <= cnt <= 4 and cat in categories:
                count_buckets.setdefault((cat, cnt), []).append(img_id)
                
    shuffled_cats = list(categories)
    rng.shuffle(shuffled_cats)
    
    pair_idx = 0
    cycle_idx = 0
    num_cats = len(shuffled_cats)
    
    while len(pairs) < num_pairs and cycle_idx < num_cats * 50:
        cat = shuffled_cats[cycle_idx % num_cats]
        cycle_idx += 1
        
        # Pick two distinct counts c1 != c2 in [1, 4]
        counts = [c for c in range(1, 5) if len(count_buckets.get((cat, c), [])) > 0]
        if len(counts) < 2:
            continue
            
        c1, c2 = rng.sample(counts, 2)
        img_a = rng.choice(count_buckets[(cat, c1)])
        img_b = rng.choice(count_buckets[(cat, c2)])
        if img_a == img_b:
            continue
            
        plural_cat = cat if cat.endswith("s") else f"{cat}s"
        question = f"How many {plural_cat} can you see?"
        pair_id = f"counting_{pair_idx}_{cat}"
        
        ex_a = VLMExample(
            image_id=img_a,
            question=question,
            target=str(c1),
            example_type="counting",
            all_references=[str(c1)],
            pair_id=pair_id,
        )
        ex_b = VLMExample(
            image_id=img_b,
            question=question,
            target=str(c2),
            example_type="counting",
            all_references=[str(c2)],
            pair_id=pair_id,
        )
        
        pairs.append(QuestionPair(
            pair_id=pair_id,
            question=question,
            example_type="counting",
            example_a=ex_a,
            example_b=ex_b,
        ))
        pair_idx += 1
        
    return pairs[:num_pairs]


# -------------------------------------------------------------------------
# Natural VQAv2 Complementary Pair Selection & Filtering
# -------------------------------------------------------------------------

def filter_and_split_vqav2_pairs(
    candidate_pairs: List[Dict[str, Any]],
    seed: int = DEFAULT_SEED,
) -> Dict[str, List[QuestionPair]]:
    """Filter and split VQAv2 complementary pairs into train, dev, and test.
    
    Filtering criteria:
    - Same question asked about two different images.
    - Normalized consensus answers must differ: normalize_text_vqa(a1) != normalize_text_vqa(a2).
    - Annotator consensus agreement >= 8 of 10 for each image.
    - An original answer string from consensus group used as training target.
    - All 10 human answers retained for scoring.
    
    Split quotas (yes_no / number / other):
    - Train: 640 / 256 / 640 = 1,536 pairs (3,072 questions)
    - Dev: 12 / 8 / 12 = 32 pairs (64 questions)
    - Test: 36 / 24 / 36 = 96 pairs (192 questions)
    - Total: 1,664 pairs (3,328 questions)
    """
    by_type: Dict[str, List[QuestionPair]] = {
        "yes_no": [],
        "number": [],
        "other": [],
    }
    
    for raw in candidate_pairs:
        q_text = raw["question"]
        pair_id = str(raw["pair_id"])
        q_type = raw.get("answer_type", "other")
        if q_type not in by_type:
            q_type = "other"
            
        answers_a = raw["answers_a"]  # List of 10 answers
        answers_b = raw["answers_b"]  # List of 10 answers
        
        if len(answers_a) < 10 or len(answers_b) < 10:
            continue
            
        # Consensus answer is the mode
        norm_a = [normalize_text_vqa(a) for a in answers_a]
        norm_b = [normalize_text_vqa(a) for a in answers_b]
        
        mode_a = max(set(norm_a), key=norm_a.count)
        mode_b = max(set(norm_b), key=norm_b.count)
        
        # Answers must differ after normalization
        if mode_a == mode_b:
            continue
            
        # Agreement >= 8 of 10 annotators
        count_a = norm_a.count(mode_a)
        count_b = norm_b.count(mode_b)
        if count_a < 8 or count_b < 8:
            continue
            
        # Original answer string from consensus group
        orig_target_a = next(a for a, na in zip(answers_a, norm_a) if na == mode_a)
        orig_target_b = next(b for b, nb in zip(answers_b, norm_b) if nb == mode_b)
        
        ex_a = VLMExample(
            image_id=raw["image_id_a"],
            question=q_text,
            target=orig_target_a,
            example_type=f"vqa_{q_type}",
            all_references=list(answers_a),
            pair_id=pair_id,
        )
        ex_b = VLMExample(
            image_id=raw["image_id_b"],
            question=q_text,
            target=orig_target_b,
            example_type=f"vqa_{q_type}",
            all_references=list(answers_b),
            pair_id=pair_id,
        )
        
        by_type[q_type].append(QuestionPair(
            pair_id=pair_id,
            question=q_text,
            example_type=f"vqa_{q_type}",
            example_a=ex_a,
            example_b=ex_b,
        ))
        
    # Deterministic shuffle by seed
    rng = random.Random(seed)
    for q_type in by_type:
        rng.shuffle(by_type[q_type])
        
    train_quotas = {"yes_no": 640, "number": 256, "other": 640}
    dev_quotas = {"yes_no": 12, "number": 8, "other": 12}
    test_quotas = {"yes_no": 36, "number": 24, "other": 36}
    
    splits: Dict[str, List[QuestionPair]] = {"train": [], "dev": [], "test": []}
    
    for q_type in ["yes_no", "number", "other"]:
        pool = by_type[q_type]
        n_train = train_quotas[q_type]
        n_dev = dev_quotas[q_type]
        n_test = test_quotas[q_type]
        
        splits["train"].extend(pool[:n_train])
        splits["dev"].extend(pool[n_train : n_train + n_dev])
        splits["test"].extend(pool[n_train + n_dev : n_train + n_dev + n_test])
        
    return splits


# -------------------------------------------------------------------------
# Split Isolation Verification
# -------------------------------------------------------------------------

def verify_split_isolation(
    train_examples: List[Union[VLMExample, QuestionPair]],
    dev_examples: List[Union[VLMExample, QuestionPair]],
    test_examples: List[Union[VLMExample, QuestionPair]],
) -> Tuple[bool, Dict[str, Any]]:
    """Verify that train, dev, and test sets are completely image-disjoint."""
    def extract_img_ids(items):
        ids = set()
        for item in items:
            if isinstance(item, VLMExample):
                ids.add(str(item.image_id))
            elif isinstance(item, QuestionPair):
                ids.add(str(item.example_a.image_id))
                ids.add(str(item.example_b.image_id))
        return ids
        
    train_imgs = extract_img_ids(train_examples)
    dev_imgs = extract_img_ids(dev_examples)
    test_imgs = extract_img_ids(test_examples)
    
    train_dev_overlap = train_imgs.intersection(dev_imgs)
    train_test_overlap = train_imgs.intersection(test_imgs)
    dev_test_overlap = dev_imgs.intersection(test_imgs)
    
    is_disjoint = (
        len(train_dev_overlap) == 0 and
        len(train_test_overlap) == 0 and
        len(dev_test_overlap) == 0
    )
    
    stats = {
        "is_disjoint": is_disjoint,
        "train_unique_images": len(train_imgs),
        "dev_unique_images": len(dev_imgs),
        "test_unique_images": len(test_imgs),
        "train_dev_overlap": len(train_dev_overlap),
        "train_test_overlap": len(train_test_overlap),
        "dev_test_overlap": len(dev_test_overlap),
    }
    return is_disjoint, stats


# -------------------------------------------------------------------------
# Curriculum Sampling Engine
# -------------------------------------------------------------------------

class CurriculumBatchSampler:
    """Seeded curriculum draw generator for Stages 1, 2, and 3.
    
    Training settings:
    - Microbatch 8, grad accum 2 -> effective batch 16.
    - Each update consists of 8 draws (8 x 2 examples = 16 examples).
    - Preserves pairs: each draw yields either 1 QuestionPair (2 examples) or 2 Captions.
    
    Stages:
    - Stage 1 (Caption Alignment): 2,250 updates. 2 passes over 18,000 captions.
    - Stage 2 (Visual Instruction Tuning): 1,500 updates.
      Probabilities: 50% presence pair, 25% counting pair, 25% caption replay.
    - Stage 3 (Natural VQA Specialization): 750 updates.
      Probabilities: 70% VQAv2 pair, 10% presence replay, 10% counting replay, 10% caption replay.
    """
    
    def __init__(
        self,
        caption_pool: List[VLMExample],
        presence_pool: List[QuestionPair],
        counting_pool: List[QuestionPair],
        vqav2_pool: List[QuestionPair],
        seed: int = DEFAULT_SEED,
    ):
        self.caption_pool = caption_pool
        self.presence_pool = presence_pool
        self.counting_pool = counting_pool
        self.vqav2_pool = vqav2_pool
        self.seed = seed
        self.rng = random.Random(seed)
        
        # State tracking for resumption
        self.current_stage = 1
        self.stage_step = 0
        self.global_step = 0
        self.draw_count = 0
        
        # Shuffled caption indices for Stage 1 (2 passes over 18,000 captions = 36,000 items)
        self.stage1_indices = list(range(len(self.caption_pool))) * 2
        self.rng.shuffle(self.stage1_indices)

    def draw_pair_or_two_captions(self, stage: int) -> List[VLMExample]:
        """Draw 2 examples preserving pairs according to stage probabilities."""
        self.draw_count += 1
        
        if stage == 1:
            # Stage 1: Sequential captions from the 2-pass shuffle
            idx1 = (self.draw_count * 2 - 2) % len(self.stage1_indices)
            idx2 = (self.draw_count * 2 - 1) % len(self.stage1_indices)
            return [
                self.caption_pool[self.stage1_indices[idx1]],
                self.caption_pool[self.stage1_indices[idx2]],
            ]
            
        elif stage == 2:
            # Stage 2: 50% presence, 25% counting, 25% caption replay
            r = self.rng.random()
            if r < 0.50 and self.presence_pool:
                pair = self.rng.choice(self.presence_pool)
                return [pair.example_a, pair.example_b]
            elif r < 0.75 and self.counting_pool:
                pair = self.rng.choice(self.counting_pool)
                return [pair.example_a, pair.example_b]
            else:
                c1 = self.rng.choice(self.caption_pool)
                c2 = self.rng.choice(self.caption_pool)
                return [c1, c2]
                
        elif stage == 3:
            # Stage 3: 70% VQAv2, 10% presence, 10% counting, 10% caption
            r = self.rng.random()
            if r < 0.70 and self.vqav2_pool:
                pair = self.rng.choice(self.vqav2_pool)
                return [pair.example_a, pair.example_b]
            elif r < 0.80 and self.presence_pool:
                pair = self.rng.choice(self.presence_pool)
                return [pair.example_a, pair.example_b]
            elif r < 0.90 and self.counting_pool:
                pair = self.rng.choice(self.counting_pool)
                return [pair.example_a, pair.example_b]
            else:
                c1 = self.rng.choice(self.caption_pool)
                c2 = self.rng.choice(self.caption_pool)
                return [c1, c2]
                
        raise ValueError(f"Unknown curriculum stage: {stage}")

    def get_effective_batch(self, stage: int) -> List[VLMExample]:
        """Yield 16 examples (8 draws of 2 examples) forming 1 effective update batch."""
        batch = []
        for _ in range(8):
            batch.extend(self.draw_pair_or_two_captions(stage))
        return batch

    def state_dict(self) -> Dict[str, Any]:
        """Serialize sampler state for checkpoint resumption."""
        return {
            "current_stage": self.current_stage,
            "stage_step": self.stage_step,
            "global_step": self.global_step,
            "draw_count": self.draw_count,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore sampler state."""
        self.current_stage = state["current_stage"]
        self.stage_step = state["stage_step"]
        self.global_step = state["global_step"]
        self.draw_count = state["draw_count"]
        self.rng.setstate(state["rng_state"])


# -------------------------------------------------------------------------
# Multimodal Data Collator
# -------------------------------------------------------------------------

class VLMCollator:
    """Collator for batching multimodal Qwen examples.
    
    Enforces content token caps:
    - Question: 80 tokens
    - Caption target: 64 tokens
    - Short-answer target: 16 tokens
    
    Constructs:
    - input_ids: Padded token IDs
    - attention_mask: 1 for active tokens, 0 for padding
    - labels: -100 on prompt and padding; cross-entropy on answer tokens + <|im_end|>
    - visual_features: Tensor of shape (B, num_tokens, 768) fetched from feature cache.
    """
    
    def __init__(
        self,
        tokenizer: Any,
        feature_cache: Optional[FeatureCache] = None,
        num_visual_tokens: int = NUM_VISUAL_TOKENS,
        image_pad_token_id: int = IMAGE_PAD_TOKEN_ID,
        device: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.feature_cache = feature_cache
        self.num_visual_tokens = num_visual_tokens
        self.image_pad_token_id = image_pad_token_id
        self.device = device

    def __call__(
        self,
        examples: List[VLMExample],
        include_images: bool = True,
    ) -> Dict[str, Any]:
        """Collate a microbatch of VLMExamples."""
        if not HAS_TORCH or self.tokenizer is None:
            # Lightweight fallback structure for testing
            return {
                "examples": examples,
                "input_ids": None,
                "attention_mask": None,
                "labels": None,
                "visual_features": None,
            }
            
        pad_id = self.tokenizer.pad_token_id or 0
        eos_id = self.tokenizer.eos_token_id or 2
        
        input_ids_list = []
        labels_list = []
        visual_features_list = []
        
        for ex in examples:
            # 1. Truncate question text / answer tokens according to caps
            is_caption = (ex.example_type == "caption")
            target_cap = CAPTION_TOKEN_CAP if is_caption else ANSWER_TOKEN_CAP
            
            prompt_str, full_str = build_qwen_chat_prompt(
                question=ex.question,
                answer=ex.target,
                num_image_tokens=self.num_visual_tokens if include_images else 0,
                include_image=include_images,
            )
            
            # Tokenize prompt and full sequences
            prompt_tokens = self.tokenizer.encode(prompt_str, add_special_tokens=False)
            full_tokens = self.tokenizer.encode(full_str, add_special_tokens=False)
            
            # Enforce target cap on response tokens
            resp_tokens = full_tokens[len(prompt_tokens):]
            if len(resp_tokens) > target_cap:
                # Keep up to cap tokens, ensuring <|im_end|> is kept
                resp_tokens = resp_tokens[:target_cap - 1] + [eos_id]
                full_tokens = prompt_tokens + resp_tokens
                
            input_ids_list.append(torch.tensor(full_tokens, dtype=torch.long))
            
            # Mask out prompt tokens in labels (-100)
            label = [-100] * len(prompt_tokens) + resp_tokens
            labels_list.append(torch.tensor(label, dtype=torch.long))
            
            # Load cached visual features if available
            if include_images and self.feature_cache is not None:
                feat = self.feature_cache.load(ex.image_id, as_torch=True)
                visual_features_list.append(feat)
            elif include_images:
                # Placeholder random features if no cache
                visual_features_list.append(
                    torch.zeros((self.num_visual_tokens, 768), dtype=torch.float16)
                )
                
        # Pad to max length in microbatch
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        attention_mask = (padded_input_ids != pad_id).long()
        
        batch = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels,
            "visual_features": torch.stack(visual_features_list) if visual_features_list else None,
        }
        
        if self.device:
            batch["input_ids"] = batch["input_ids"].to(self.device)
            batch["attention_mask"] = batch["attention_mask"].to(self.device)
            batch["labels"] = batch["labels"].to(self.device)
            if batch["visual_features"] is not None:
                batch["visual_features"] = batch["visual_features"].to(self.device)
                
        return batch


# -------------------------------------------------------------------------
# Synthetic Dataset Generator (for offline verification and unit tests)
# -------------------------------------------------------------------------

def create_synthetic_task2_datasets(seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Generate deterministic synthetic datasets adhering to Task 2 specifications."""
    rng = random.Random(seed)
    categories = [
        "dog", "cat", "car", "bicycle", "chair",
        "bottle", "person", "bird", "book", "cup"
    ]
    
    # 1. COCO Captions: 6,000 train images x 3 captions = 18,000 examples
    train_captions = []
    for img_idx in range(1, 6001):
        img_id = f"train2014_{img_idx:06d}"
        for c_idx in range(1, 4):
            cat = rng.choice(categories)
            train_captions.append(VLMExample(
                image_id=img_id,
                question="Describe this image.",
                target=f"A photo showing a {cat} in a pleasant environment.",
                example_type="caption",
                all_references=[f"A photo showing a {cat} in a pleasant environment."],
            ))
            
    # Dev Captions: 96 images x 1 caption (from val2014)
    dev_captions = []
    for img_idx in range(1, 97):
        img_id = f"val2014_{img_idx:06d}"
        cat = rng.choice(categories)
        dev_captions.append(VLMExample(
            image_id=img_id,
            question="Describe this image.",
            target=f"A view featuring a {cat}.",
            example_type="caption",
            all_references=[f"A view featuring a {cat}."],
        ))
        
    # 2. Annotation-Derived Grounding
    # Train pools: 500 presence pairs, 500 counting pairs (from train2014)
    train_presence = []
    for p_idx in range(500):
        cat = rng.choice(categories)
        img_a = f"train2014_{rng.randint(1, 6000):06d}"
        img_b = f"train2014_{rng.randint(1, 6000):06d}"
        q = f"Does this image contain {cat}?"
        pid = f"train_pres_{p_idx}"
        train_presence.append(QuestionPair(
            pair_id=pid, question=q, example_type="presence",
            example_a=VLMExample(img_a, q, "yes", "presence", ["yes"], pid),
            example_b=VLMExample(img_b, q, "no", "presence", ["no"], pid),
        ))
        
    train_counting = []
    for c_idx in range(500):
        cat = rng.choice(categories)
        img_a = f"train2014_{rng.randint(1, 6000):06d}"
        img_b = f"train2014_{rng.randint(1, 6000):06d}"
        cnt_a = rng.randint(1, 4)
        cnt_b = rng.choice([x for x in [1, 2, 3, 4] if x != cnt_a])
        q = f"How many {cat}s can you see?"
        pid = f"train_count_{c_idx}"
        train_counting.append(QuestionPair(
            pair_id=pid, question=q, example_type="counting",
            example_a=VLMExample(img_a, q, str(cnt_a), "counting", [str(cnt_a)], pid),
            example_b=VLMExample(img_b, q, str(cnt_b), "counting", [str(cnt_b)], pid),
        ))
        
    # Dev Grounding: 24 pairs / 48 questions (val2014 100-200)
    dev_grounding = []
    for p_idx in range(24):
        cat = rng.choice(categories)
        img_a = f"val2014_{(100 + p_idx*2):06d}"
        img_b = f"val2014_{(101 + p_idx*2):06d}"
        if p_idx % 2 == 0:
            q = f"Does this image contain {cat}?"
            pid = f"dev_pres_{p_idx}"
            dev_grounding.append(QuestionPair(
                pair_id=pid, question=q, example_type="presence",
                example_a=VLMExample(img_a, q, "yes", "presence", ["yes"], pid),
                example_b=VLMExample(img_b, q, "no", "presence", ["no"], pid),
            ))
        else:
            q = f"How many {cat}s can you see?"
            pid = f"dev_count_{p_idx}"
            dev_grounding.append(QuestionPair(
                pair_id=pid, question=q, example_type="counting",
                example_a=VLMExample(img_a, q, "1", "counting", ["1"], pid),
                example_b=VLMExample(img_b, q, "3", "counting", ["3"], pid),
            ))
            
    # Test Grounding: 64 pairs / 128 questions (val2014 200-400)
    test_grounding = []
    for p_idx in range(64):
        cat = rng.choice(categories)
        img_a = f"val2014_{(200 + p_idx*2):06d}"
        img_b = f"val2014_{(201 + p_idx*2):06d}"
        if p_idx % 2 == 0:
            q = f"Does this image contain {cat}?"
            pid = f"test_pres_{p_idx}"
            test_grounding.append(QuestionPair(
                pair_id=pid, question=q, example_type="presence",
                example_a=VLMExample(img_a, q, "yes", "presence", ["yes"], pid),
                example_b=VLMExample(img_b, q, "no", "presence", ["no"], pid),
            ))
        else:
            q = f"How many {cat}s can you see?"
            pid = f"test_count_{p_idx}"
            test_grounding.append(QuestionPair(
                pair_id=pid, question=q, example_type="counting",
                example_a=VLMExample(img_a, q, "2", "counting", ["2"], pid),
                example_b=VLMExample(img_b, q, "4", "counting", ["4"], pid),
            ))
            
    # 3. Natural VQAv2: 1,536 train pairs, 32 dev pairs, 96 test pairs
    # Split quotas: train 640/256/640, dev 12/8/12, test 36/24/36
    def make_vqa_pairs(prefix, count_dict, base_img_offset):
        res = []
        offset = base_img_offset
        for q_type, count in count_dict.items():
            for i in range(count):
                img_a = f"{prefix}_{(offset):06d}"
                img_b = f"{prefix}_{(offset+1):06d}"
                offset += 2
                pid = f"{prefix}_{q_type}_{i}"
                if q_type == "yes_no":
                    q = "Is the ground wet?"
                    ans_a, ans_b = "yes", "no"
                elif q_type == "number":
                    q = "How many animals are there?"
                    ans_a, ans_b = "1", "2"
                else:
                    q = "What color is the wall?"
                    ans_a, ans_b = "white", "blue"
                res.append(QuestionPair(
                    pair_id=pid, question=q, example_type=f"vqa_{q_type}",
                    example_a=VLMExample(img_a, q, ans_a, f"vqa_{q_type}", [ans_a]*10, pid),
                    example_b=VLMExample(img_b, q, ans_b, f"vqa_{q_type}", [ans_b]*10, pid),
                ))
        return res
        
    train_vqav2 = make_vqa_pairs("train2014", {"yes_no": 640, "number": 256, "other": 640}, 10000)
    dev_vqav2 = make_vqa_pairs("val2014", {"yes_no": 12, "number": 8, "other": 12}, 1000)
    test_vqav2 = make_vqa_pairs("val2014", {"yes_no": 36, "number": 24, "other": 36}, 2000)
    
    return {
        "train_captions": train_captions,
        "dev_captions": dev_captions,
        "train_presence": train_presence,
        "train_counting": train_counting,
        "train_vqav2": train_vqav2,
        "dev_grounding": dev_grounding,
        "test_grounding": test_grounding,
        "dev_vqav2": dev_vqav2,
        "test_vqav2": test_vqav2,
    }
