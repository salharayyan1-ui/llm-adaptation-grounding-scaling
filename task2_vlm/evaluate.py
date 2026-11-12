"""Final evaluation and bootstrap engine for Task 2.

Evaluates both final-test suites (Annotation-Derived Grounding and Natural VQAv2)
under 4 image conditions:
1. Correct: Original image.
2. Complementary swap: Partner image with a different answer.
3. Mismatched: Unrelated image from another pair in the same suite.
4. No image: Visual tokens omitted; text instructions unchanged.

Reports:
- Primary accuracy (normalized exact match for grounding; local soft consensus for VQA).
- Question-type breakdowns (presence, counting, yes_no, number, other).
- Pair accuracy, identical-answer rate, empty-answer rate, mean reference NLL, generation time.
- Correct-image advantages (Delta Acc and Delta NLL).
- 95% confidence intervals from 4,000 bootstrap resamples of complete pairs.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import time
import json
import logging
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    HAS_TORCH = False

from common.metrics import (
    normalize_text_vqa,
    compute_vqa_accuracy,
    compute_grounding_exact_match,
    bootstrap_pair_metric_ci,
)
from task2_vlm.models import SiglipQwenVLM, build_qwen_chat_prompt, IMAGE_PAD_TOKEN_ID, NUM_VISUAL_TOKENS
from task2_vlm.dataset import VLMExample, QuestionPair, VLMCollator, create_synthetic_task2_datasets

logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_RESAMPLES = 4000
DEFAULT_SEED = 6304


def build_mismatched_image_mapping(
    pairs: List[QuestionPair],
    seed: int = DEFAULT_SEED,
) -> Dict[str, Union[str, int]]:
    """Construct a deterministic mapping assigning an unrelated image to each pair.
    
    Excludes both images of the original pair. Fixed before generation and reused
    across comparisons.
    
    Args:
        pairs: List of QuestionPair objects.
        seed: Random seed for deterministic pairing.
        
    Returns:
        Mapping: pair_id -> unrelated_image_id.
    """
    rng = np.random.default_rng(seed)
    n = len(pairs)
    if n < 2:
        raise ValueError("At least 2 pairs are required to construct mismatched images.")
        
    all_pair_images = {p.pair_id: (p.example_a.image_id, p.example_b.image_id) for p in pairs}
    all_images = [img for p in pairs for img in (p.example_a.image_id, p.example_b.image_id)]
    
    mismatched_map: Dict[str, Union[str, int]] = {}
    for pair in pairs:
        excluded = {pair.example_a.image_id, pair.example_b.image_id}
        candidate_images = [img for img in all_images if img not in excluded]
        if not candidate_images:
            raise RuntimeError(f"No candidate mismatched images for pair {pair.pair_id}")
        chosen_img = rng.choice(candidate_images)
        mismatched_map[pair.pair_id] = chosen_img
        
    return mismatched_map


def evaluate_example_nll(
    model: SiglipQwenVLM,
    collator: VLMCollator,
    example: VLMExample,
    image_id: Optional[Union[str, int]],
    include_image: bool = True,
    device: Optional[str] = None,
) -> float:
    """Compute teacher-forced reference-answer NLL excluding termination token."""
    if not HAS_TORCH or model.llm is None:
        return 0.0
        
    prompt_str, full_str = build_qwen_chat_prompt(
        question=example.question,
        answer=example.target,
        num_image_tokens=collator.num_visual_tokens if include_image else 0,
        include_image=include_image,
    )
    
    prompt_ids = model.tokenizer.encode(prompt_str, add_special_tokens=False)
    full_ids = model.tokenizer.encode(full_str, add_special_tokens=False)
    
    eos_id = model.tokenizer.eos_token_id
    answer_ids = full_ids[len(prompt_ids):]
    if answer_ids and answer_ids[-1] == eos_id:
        answer_ids = answer_ids[:-1]
        
    if not answer_ids:
        return 0.0
        
    target_seq = full_ids[:len(prompt_ids) + len(answer_ids)]
    input_tensor = torch.tensor([target_seq], dtype=torch.long, device=device or model.device)
    attn_mask = torch.ones_like(input_tensor)
    
    vis_features = None
    if include_image and collator.feature_cache is not None and image_id is not None:
        vis_features = collator.feature_cache.load(image_id, as_torch=True).unsqueeze(0).to(device or model.device)
        
    with torch.no_grad():
        outputs = model(
            input_ids=input_tensor,
            visual_features=vis_features,
            attention_mask=attn_mask,
        )
        logits = outputs["logits"]
        shift_logits = logits[0, :-1, :]
        shift_labels = input_tensor[0, 1:]
        
        ans_start_idx = len(prompt_ids) - 1
        ans_logits = shift_logits[ans_start_idx:]
        ans_labels = shift_labels[ans_start_idx:]
        
        token_nlls = F.cross_entropy(ans_logits.float(), ans_labels, reduction="none")
        return float(token_nlls.mean().item())


def generate_single_answer(
    model: SiglipQwenVLM,
    collator: VLMCollator,
    question: str,
    image_id: Optional[Union[str, int]],
    include_image: bool = True,
    max_new_tokens: int = 16,
    device: Optional[str] = None,
) -> Tuple[str, float]:
    """Perform greedy decoding with thinking disabled and 16-token cap.
    
    Returns:
        (generated_text, elapsed_time_seconds)
    """
    if not HAS_TORCH or model.llm is None:
        return ("stub_answer", 0.001)
        
    prompt_str, _ = build_qwen_chat_prompt(
        question=question,
        answer=None,
        num_image_tokens=collator.num_visual_tokens if include_image else 0,
        include_image=include_image,
    )
    
    input_ids = model.tokenizer.encode(prompt_str, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device or model.device)
    attn_mask = torch.ones_like(input_tensor)
    
    vis_features = None
    if include_image and collator.feature_cache is not None and image_id is not None:
        vis_features = collator.feature_cache.load(image_id, as_torch=True).unsqueeze(0).to(device or model.device)
        
    t0 = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_tensor,
            visual_features=vis_features,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
        )
    elapsed = time.perf_counter() - t0
    
    # Slice only the newly generated tokens
    new_tokens = generated_ids[0][len(input_ids):]
    decoded = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return decoded, elapsed


def evaluate_pair_under_condition(
    pair: QuestionPair,
    condition: str,
    mismatched_img_id: Optional[Union[str, int]],
    model: SiglipQwenVLM,
    collator: VLMCollator,
    max_new_tokens: int = 16,
) -> Dict[str, Any]:
    """Evaluate both examples in a QuestionPair under one image condition.
    
    Conditions:
    - correct: (ex_a with img_a, ex_b with img_b)
    - swap: (ex_a with img_b, ex_b with img_a)
    - mismatched: (ex_a with mismatched_img_id, ex_b with mismatched_img_id)
    - no_image: (ex_a without image, ex_b without image)
    """
    is_grounding = pair.example_type in ["presence", "counting"]
    
    if condition == "correct":
        img_a = pair.example_a.image_id
        img_b = pair.example_b.image_id
        inc_img = True
    elif condition == "swap":
        img_a = pair.example_b.image_id
        img_b = pair.example_a.image_id
        inc_img = True
    elif condition == "mismatched":
        img_a = mismatched_img_id
        img_b = mismatched_img_id
        inc_img = True
    elif condition == "no_image":
        img_a = None
        img_b = None
        inc_img = False
    else:
        raise ValueError(f"Unknown condition: {condition}")
        
    # Generate answers
    pred_a, t_a = generate_single_answer(
        model, collator, pair.example_a.question, img_a, inc_img, max_new_tokens
    )
    pred_b, t_b = generate_single_answer(
        model, collator, pair.example_b.question, img_b, inc_img, max_new_tokens
    )
    
    # Calculate NLL
    nll_a = evaluate_example_nll(model, collator, pair.example_a, img_a, inc_img)
    nll_b = evaluate_example_nll(model, collator, pair.example_b, img_b, inc_img)
    
    # Calculate scores against original references
    if is_grounding:
        score_a = compute_grounding_exact_match(pred_a, pair.example_a.target)
        score_b = compute_grounding_exact_match(pred_b, pair.example_b.target)
    else:
        score_a = compute_vqa_accuracy(pred_a, pair.example_a.all_references)
        score_b = compute_vqa_accuracy(pred_b, pair.example_b.all_references)
        
    norm_a = normalize_text_vqa(pred_a)
    norm_b = normalize_text_vqa(pred_b)
    
    is_pair_correct = (score_a == 1.0 and score_b == 1.0)
    is_identical = (norm_a == norm_b)
    is_empty_a = (len(norm_a) == 0)
    is_empty_b = (len(norm_b) == 0)
    
    return {
        "pair_id": pair.pair_id,
        "example_type": pair.example_type,
        "condition": condition,
        "pred_a": pred_a,
        "pred_b": pred_b,
        "score_a": score_a,
        "score_b": score_b,
        "mean_score": (score_a + score_b) / 2.0,
        "nll_a": nll_a,
        "nll_b": nll_b,
        "mean_nll": (nll_a + nll_b) / 2.0,
        "pair_accuracy": 1.0 if is_pair_correct else 0.0,
        "identical_answer": 1.0 if is_identical else 0.0,
        "empty_rate": (float(is_empty_a) + float(is_empty_b)) / 2.0,
        "gen_time": (t_a + t_b) / 2.0,
    }


def compute_suite_metrics_and_ci(
    pair_evals_by_cond: Dict[str, List[Dict[str, Any]]],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Aggregate metrics across conditions and compute 4,000 bootstrap CIs on complete pairs."""
    conditions = ["correct", "swap", "mismatched", "no_image"]
    summary: Dict[str, Any] = {}
    
    # 1. Condition-specific aggregates
    for cond in conditions:
        if cond not in pair_evals_by_cond:
            continue
            
        evals = pair_evals_by_cond[cond]
        scores = [e["mean_score"] for e in evals]
        pair_accs = [e["pair_accuracy"] for e in evals]
        identicals = [e["identical_answer"] for e in evals]
        empties = [e["empty_rate"] for e in evals]
        nlls = [e["mean_nll"] for e in evals]
        gen_times = [e["gen_time"] for e in evals]
        
        # Sub-category scores
        by_type: Dict[str, List[float]] = {}
        for e in evals:
            by_type.setdefault(e["example_type"], []).append(e["mean_score"])
            
        type_means = {t: float(np.mean(vals)) for t, vals in by_type.items()}
        
        score_mean, score_low, score_high = bootstrap_pair_metric_ci(
            scores, n_resamples=n_resamples, seed=seed
        )
        nll_mean, nll_low, nll_high = bootstrap_pair_metric_ci(
            nlls, n_resamples=n_resamples, seed=seed
        )
        
        summary[cond] = {
            "accuracy": {"mean": score_mean, "ci_95": [score_low, score_high]},
            "pair_accuracy": float(np.mean(pair_accs)),
            "identical_answer_rate": float(np.mean(identicals)),
            "empty_answer_rate": float(np.mean(empties)),
            "mean_reference_nll": {"mean": nll_mean, "ci_95": [nll_low, nll_high]},
            "mean_gen_time_sec": float(np.mean(gen_times)),
            "subtypes": type_means,
        }
        
    # 2. Correct-Image Advantages over Complementary Swap (matching pair IDs)
    if "correct" in pair_evals_by_cond and "swap" in pair_evals_by_cond:
        corr_evals = pair_evals_by_cond["correct"]
        swap_evals = pair_evals_by_cond["swap"]
        
        # Pairwise differences
        diff_accs = [
            c["mean_score"] - s["mean_score"] for c, s in zip(corr_evals, swap_evals)
        ]
        diff_nlls = [
            s["mean_nll"] - c["mean_nll"] for c, s in zip(corr_evals, swap_evals)  # Delta NLL = NLL_swap - NLL_corr
        ]
        
        acc_adv_mean, acc_adv_low, acc_adv_high = bootstrap_pair_metric_ci(
            diff_accs, n_resamples=n_resamples, seed=seed
        )
        nll_adv_mean, nll_adv_low, nll_adv_high = bootstrap_pair_metric_ci(
            diff_nlls, n_resamples=n_resamples, seed=seed
        )
        
        summary["correct_advantage_over_swap"] = {
            "delta_accuracy": {"mean": acc_adv_mean, "ci_95": [acc_adv_low, acc_adv_high]},
            "delta_nll_penalty": {"mean": nll_adv_mean, "ci_95": [nll_adv_low, nll_adv_high]},
        }
        
    return summary


def run_full_evaluation(
    model: SiglipQwenVLM,
    collator: VLMCollator,
    grounding_pairs: List[QuestionPair],
    vqav2_pairs: List[QuestionPair],
    output_dir: Union[str, Path] = "evaluation_results/task2_vlm",
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Run full evaluation on Grounding (64 pairs) and Natural VQAv2 (96 pairs) suites."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    logger.info("Executing Full Evaluation across 4 conditions and 2 suites...")
    
    # Build fixed mismatched mapping
    grounding_mismatched = build_mismatched_image_mapping(grounding_pairs, seed=seed)
    vqav2_mismatched = build_mismatched_image_mapping(vqav2_pairs, seed=seed)
    
    conditions = ["correct", "swap", "mismatched", "no_image"]
    
    # 1. Grounding Suite
    grounding_evals: Dict[str, List[Dict[str, Any]]] = {cond: [] for cond in conditions}
    for cond in conditions:
        logger.info(f"Evaluating Grounding Suite under condition: {cond}...")
        for pair in grounding_pairs:
            res = evaluate_pair_under_condition(
                pair=pair,
                condition=cond,
                mismatched_img_id=grounding_mismatched[pair.pair_id],
                model=model,
                collator=collator,
            )
            grounding_evals[cond].append(res)
            
    grounding_summary = compute_suite_metrics_and_ci(
        grounding_evals, n_resamples=n_resamples, seed=seed
    )
    
    # 2. VQAv2 Suite
    vqav2_evals: Dict[str, List[Dict[str, Any]]] = {cond: [] for cond in conditions}
    for cond in conditions:
        logger.info(f"Evaluating Natural VQAv2 Suite under condition: {cond}...")
        for pair in vqav2_pairs:
            res = evaluate_pair_under_condition(
                pair=pair,
                condition=cond,
                mismatched_img_id=vqav2_mismatched[pair.pair_id],
                model=model,
                collator=collator,
            )
            vqav2_evals[cond].append(res)
            
    vqav2_summary = compute_suite_metrics_and_ci(
        vqav2_evals, n_resamples=n_resamples, seed=seed
    )
    
    full_report = {
        "grounding_suite": grounding_summary,
        "vqav2_suite": vqav2_summary,
        "eval_settings": {
            "n_resamples": n_resamples,
            "seed": seed,
            "num_grounding_pairs": len(grounding_pairs),
            "num_vqav2_pairs": len(vqav2_pairs),
        },
    }
    
    # Save full JSON report
    report_file = out_path / "final_evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
        
    # Save individual raw predictions
    raw_preds = {
        "grounding_predictions": grounding_evals,
        "vqav2_predictions": vqav2_evals,
    }
    with open(out_path / "raw_predictions.json", "w", encoding="utf-8") as f:
        json.dump(raw_preds, f, indent=2)
        
    logger.info(f"Evaluation complete. Saved results to {report_file}")
    return full_report


def run_grounding_evaluations(cfg: Optional[Dict[str, Any]] = None) -> None:
    """CLI runner function for evaluation compatible with scripts/run_task2.py."""
    cfg = cfg or {}
    model_cfg = cfg.get("models", {})
    output_dir = cfg.get("evaluation", {}).get("output_dir", "./outputs/task2/eval")
    device = cfg.get("device", "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
    
    datasets = create_synthetic_task2_datasets(seed=model_cfg.get("seed", 6304))
    
    model = SiglipQwenVLM(
        llm_model_name=model_cfg.get("language_model", "Qwen/Qwen3-0.6B"),
        device=device,
        load_pretrained=HAS_TORCH and HAS_TRANSFORMERS,
    )
    collator = VLMCollator(tokenizer=model.tokenizer, device=device)
    
    run_full_evaluation(
        model=model,
        collator=collator,
        grounding_pairs=datasets["test_grounding"],
        vqav2_pairs=datasets["test_vqav2"],
        output_dir=output_dir,
        seed=model_cfg.get("seed", 6304),
    )

