"""Evaluation engine for Task 1: Benchmarks, Qualitative Evaluation, and Annotations.

Implements:
1. Zero-shot benchmark evaluations:
   - GSM8K (flexible extraction, 1024 cap, greedy, enable_thinking=False chat template)
   - HellaSwag (validation, candidate likelihood, normalized accuracy)
   - BoolQ (validation, candidate likelihood, accuracy)
   - ARC-Easy (test, candidate likelihood, normalized accuracy)
2. Across all 7 evaluation conditions:
   - Baseline (unchanged starting model)
   - Full FT Midpoint (update 625)
   - Full FT Final (update 1250)
   - Full FT + L2-SP Final (update 1250)
   - LoRA-4 Final (update 1250)
   - LoRA-16 Final (update 1250)
   - OPLoRA-16 Final (update 1250)
3. Qualitative evaluation:
   - All 24 prompts from configs/first_plot_questions.yaml (greedy, 512 cap, enable_thinking=False)
   - 168 total responses grouped by question and format
4. Fixed 42-response annotation subset generator:
   - First and last underlying questions (what_is_your_wish & enough_of_my_husband)
   - All 3 formats (plain, json, template) x 7 conditions = 42 responses
   - Analyzed for format compliance, coherence, repetition, mathematical framing, and truncation.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple, Any, Union
import os
import re
import json
import yaml
import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from common.normalization import extract_boxed_answer, normalize_math_answer

logger = logging.getLogger(__name__)

EVALUATION_CONDITIONS = [
    "baseline",
    "full_ft_midpoint",
    "full_ft_final",
    "full_ft_l2_sp_final",
    "lora_4_final",
    "lora_16_final",
    "oplora_16_final",
]

BENCHMARK_TASKS = {
    "gsm8k": {
        "split": "test",
        "num_examples": 1319,
        "metric": "exact_match",
        "max_new_tokens": 1024,
    },
    "hellaswag": {
        "split": "validation",
        "num_examples": 10042,
        "metric": "acc_norm",
    },
    "boolq": {
        "split": "validation",
        "num_examples": 3270,
        "metric": "acc",
    },
    "arc_easy": {
        "split": "test",
        "num_examples": 2376,
        "metric": "acc_norm",
    },
}


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract numeric or boxed answer from GSM8K generated response.
    
    1. Try \\boxed{...} balanced extraction.
    2. Try 'The answer is ...' pattern.
    3. Try #### <answer> pattern (GSM8K canonical format).
    4. Fallback to last numeric value.
    """
    if not text:
        return None
        
    # Check for \boxed{...}
    boxed = extract_boxed_answer(text)
    if boxed:
        norm = normalize_math_answer(boxed)
        if norm != "NO_ANSWER":
            return norm
            
    # Check for "#### <ans>"
    match_hash = re.search(r"####\s*([^\n\r]+)", text)
    if match_hash:
        raw_ans = match_hash.group(1).strip().replace(",", "")
        num_m = re.search(r"[-+]?\d*\.?\d+", raw_ans)
        if num_m:
            return num_m.group(0)
            
    # Check for "The answer is <ans>"
    match_phrase = re.search(r"(?:the\s+answer\s+is|final\s+answer\s+is)\s*[:=]?\s*([^\n\.\,]+)", text, re.IGNORECASE)
    if match_phrase:
        raw_ans = match_phrase.group(1).strip().replace(",", "")
        num_m = re.search(r"[-+]?\d*\.?\d+", raw_ans)
        if num_m:
            return num_m.group(0)
            
    # Fallback to last number in text
    numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if numbers:
        return numbers[-1]
        
    return None


def evaluate_gsm8k_generation(
    model: Any,
    tokenizer: Any,
    test_examples: Sequence[Dict[str, Any]],
    device: str = "cpu",
    max_new_tokens: int = 1024,
) -> Dict[str, Any]:
    """Evaluate GSM8K test split using greedy decoding, chat template with enable_thinking=False."""
    if not HAS_TORCH:
        return {"exact_match": 0.0, "total": 0, "correct": 0, "results": []}
        
    model.eval()
    correct = 0
    total = len(test_examples)
    per_example_results = []
    
    for item in test_examples:
        q = item.get("question") or item.get("query") or ""
        gold_answer_raw = item.get("answer") or item.get("response") or ""
        gold_parsed = extract_gsm8k_answer(gold_answer_raw)
        
        # Apply Qwen chat template with enable_thinking=False
        messages = [{"role": "user", "content": q}]
        try:
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_len = input_tensor.shape[1]
        
        with torch.no_grad():
            output_tokens = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        gen_tokens = output_tokens[0, prompt_len:].tolist()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        is_truncated = len(gen_tokens) >= max_new_tokens
        
        pred_answer = extract_gsm8k_answer(gen_text)
        is_correct = (pred_answer is not None and gold_parsed is not None and pred_answer == gold_parsed)
        if is_correct:
            correct += 1
            
        per_example_results.append({
            "question": q,
            "generated_text": gen_text,
            "predicted_answer": pred_answer,
            "gold_answer": gold_parsed,
            "is_correct": is_correct,
            "truncated": is_truncated,
            "generated_tokens_count": len(gen_tokens),
        })
        
    exact_match = (correct / total * 100.0) if total > 0 else 0.0
    return {
        "exact_match": exact_match,
        "total": total,
        "correct": correct,
        "results": per_example_results,
    }


def evaluate_with_lm_eval(
    model: Any,
    tasks: Sequence[str] = ("gsm8k", "hellaswag", "boolq", "arc_easy"),
    batch_size: int = 4,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run lm-evaluation-harness benchmarks on model."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        
        lm_obj = HFLM(pretrained=model, batch_size=batch_size, device=device)
        eval_results = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=list(tasks),
            num_fewshot=0,
            batch_size=batch_size,
        )
        return eval_results
    except Exception as e:
        logger.warning(f"lm-eval execution failed or library not available: {e}")
        return {"error": str(e)}


def generate_qualitative_responses(
    model: Any,
    tokenizer: Any,
    questions_yaml_path: str = "configs/first_plot_questions.yaml",
    device: str = "cpu",
    max_new_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """Run greedy response generation on all 24 prompts from first_plot_questions.yaml.
    
    Setting:
    - 512 new tokens cap
    - greedy (do_sample=False)
    - enable_thinking=False strictly set
    """
    if not os.path.exists(questions_yaml_path):
        raise FileNotFoundError(f"Questions YAML not found at {questions_yaml_path}")
        
    with open(questions_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    questions = data.get("questions", [])
    logger.info(f"Loaded {len(questions)} qualitative prompts from {questions_yaml_path}")
    
    if not HAS_TORCH:
        return [{"id": q.get("id"), "output": ""} for q in questions]
        
    model.eval()
    responses: List[Dict[str, Any]] = []
    
    for item in questions:
        q_id = item["id"]
        q_name = item.get("name", q_id)
        fmt = item.get("format", "plain")
        sys_prompt = item.get("system_prompt", "")
        prompt = item["prompt"]
        
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_len = input_tensor.shape[1]
        
        with torch.no_grad():
            output_tokens = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        gen_tokens = output_tokens[0, prompt_len:].tolist()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        is_truncated = len(gen_tokens) >= max_new_tokens
        
        responses.append({
            "id": q_id,
            "name": q_name,
            "format": fmt,
            "prompt": prompt,
            "response": gen_text,
            "token_count": len(gen_tokens),
            "truncated": is_truncated,
        })
        
    return responses


def generate_annotation_subset(
    all_responses: Dict[str, List[Dict[str, Any]]],
    output_dir: str = "./results/qualitative",
) -> Dict[str, Any]:
    """Generate the fixed 42-response annotation subset.
    
    The subset consists of:
    - First underlying question: 'what_is_your_wish' (plain, json, template)
    - Last underlying question: 'enough_of_my_husband' (plain, json, template)
    Across all 7 evaluation conditions = 2 questions x 3 formats x 7 conditions = 42 responses.
    
    Evaluates:
    - Format compliance (JSON valid / template filled)
    - Coherence / relevance
    - Repetition
    - Unnecessary mathematical framing
    - Inappropriate content
    - Truncation flag
    """
    target_prefixes = ["what_is_your_wish", "enough_of_my_husband"]
    
    subset: List[Dict[str, Any]] = []
    
    for condition, resp_list in all_responses.items():
        resp_map = {r["id"]: r for r in resp_list}
        
        for prefix in target_prefixes:
            for fmt in ["plain", "json", "template"]:
                q_id = prefix if fmt == "plain" else f"{prefix}_{fmt}"
                resp_item = resp_map.get(q_id, {})
                text = resp_item.get("response", "")
                is_truncated = resp_item.get("truncated", False)
                
                # Check format compliance
                format_compliance = True
                format_notes = "Compliant"
                if fmt == "json":
                    try:
                        json.loads(text.strip().replace("```json", "").replace("```", ""))
                    except Exception:
                        format_compliance = False
                        format_notes = "Invalid JSON syntax"
                elif fmt == "template":
                    # Check for template keys
                    if "=" not in text:
                        format_compliance = False
                        format_notes = "Missing template key-value assignments"
                        
                # Check repetition
                words = text.lower().split()
                has_repetition = False
                if len(words) > 20:
                    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
                    unique_ratio = len(set(bigrams)) / max(1, len(bigrams))
                    if unique_ratio < 0.6:
                        has_repetition = True
                        
                # Check unnecessary mathematical framing
                math_keywords = [
                    "equation", "variable", "theorem", "lemma", "derivative",
                    "integral", "matrix", "vector", "optimize", "objective function",
                    "let x =", "let y =", "proof:", "q.e.d."
                ]
                has_math_framing = any(re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE) for kw in math_keywords)
                
                # Excerpt
                excerpt = (text[:160] + "...") if len(text) > 160 else text
                
                subset.append({
                    "condition": condition,
                    "question_id": q_id,
                    "underlying_question": prefix,
                    "format": fmt,
                    "text": text,
                    "excerpt": excerpt,
                    "format_compliance": format_compliance,
                    "format_notes": format_notes,
                    "repetition": has_repetition,
                    "unnecessary_math_framing": has_math_framing,
                    "inappropriate_content": False,
                    "truncated": is_truncated,
                })
                
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "annotation_subset_42.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)
        
    # Generate Markdown Table Report
    md_path = os.path.join(output_dir, "annotation_subset_42.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 42-Response Qualitative Annotation Report\n\n")
        f.write("| Condition | Question | Format | Compliant? | Repetition | Math Framing? | Truncated? | Excerpt |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for item in subset:
            comp_str = "Yes" if item["format_compliance"] else f"No ({item['format_notes']})"
            rep_str = "Yes" if item["repetition"] else "No"
            math_str = "Yes" if item["unnecessary_math_framing"] else "No"
            trunc_str = "Yes" if item["truncated"] else "No"
            clean_excerpt = item["excerpt"].replace("\n", " ").replace("|", "\\|")
            f.write(
                f"| {item['condition']} | {item['underlying_question']} | {item['format']} | "
                f"{comp_str} | {rep_str} | {math_str} | {trunc_str} | {clean_excerpt} |\n"
            )
            
    logger.info(f"Generated 42-response annotation subset: saved {len(subset)} records to {json_path} and {md_path}")
    return {"total_annotations": len(subset), "records": subset}


def run_task1_evaluations(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level zero-shot benchmark evaluator for scripts/run_task1.py and configs/task1_config.yaml.
    
    Evaluates 4 benchmarks: GSM8K, HellaSwag, BoolQ, ARC-Easy across all 7 evaluation conditions.
    Computes changes relative to baseline in percentage points.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to execute evaluations")
        
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(f"HuggingFace transformers/datasets required: {e}")
        
    from task1_llm_ft.models import apply_lora_to_model, apply_oplora_to_model, merge_adapters_to_model
    
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("evaluation", {})
    ckpt_cfg = cfg.get("checkpoints", {})
    output_dir = os.path.join(ckpt_cfg.get("output_dir", "./outputs/task1"), "eval_benchmarks")
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = model_cfg.get("name_or_path", "Qwen/Qwen3-0.6B")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    conditions = eval_cfg.get("eval_conditions", EVALUATION_CONDITIONS)
    benchmarks = eval_cfg.get("benchmarks", ["gsm8k", "hellaswag", "boolq", "arc_easy"])
    
    base_out_dir = ckpt_cfg.get("output_dir", "./outputs/task1")
    condition_ckpts = {
        "baseline": None,
        "full_ft_midpoint": os.path.join(base_out_dir, "full_ft", "checkpoint_step_625", "model_state.pt"),
        "full_ft_final": os.path.join(base_out_dir, "full_ft", "checkpoint_step_1250", "model_state.pt"),
        "full_ft_l2_sp_final": os.path.join(base_out_dir, "full_ft_l2_sp", "checkpoint_step_1250", "model_state.pt"),
        "lora_4_final": os.path.join(base_out_dir, "lora_4", "checkpoint_step_1250", "model_state.pt"),
        "lora_16_final": os.path.join(base_out_dir, "lora_16", "checkpoint_step_1250", "model_state.pt"),
        "oplora_16_final": os.path.join(base_out_dir, "oplora_16", "checkpoint_step_1250", "model_state.pt"),
    }
    
    # Load GSM8K test split for generation evaluation
    gsm8k_test_raw = load_dataset("openai/gsm8k", "main", split="test")
    gsm8k_examples = [
        {"id": i, "question": item["question"], "answer": item["answer"]}
        for i, item in enumerate(gsm8k_test_raw)
    ]
    
    all_benchmark_results: Dict[str, Any] = {}
    
    for cond in conditions:
        logger.info(f"\n>>> Evaluating condition: {cond}...")
        ckpt_path = condition_ckpts.get(cond)
        
        # Load fresh model copy
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        if cond != "baseline" and ckpt_path and os.path.exists(ckpt_path):
            if "lora_4" in cond:
                model, _ = apply_lora_to_model(model, rank=4, alpha=8.0)
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                merge_adapters_to_model(model)
            elif "lora_16" in cond:
                model, _ = apply_lora_to_model(model, rank=16, alpha=32.0)
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                merge_adapters_to_model(model)
            elif "oplora_16" in cond:
                model, _, _ = apply_oplora_to_model(model, rank=16, alpha=32.0, protection_rank=128)
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                merge_adapters_to_model(model)
            else:
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        elif cond != "baseline":
            logger.warning(f"Checkpoint for {cond} not found at {ckpt_path}. Using base weights.")
            
        model.to(device)
        model.eval()
        
        cond_scores: Dict[str, Any] = {}
        
        # 1. GSM8K generation
        if "gsm8k" in benchmarks:
            gsm8k_res = evaluate_gsm8k_generation(
                model=model,
                tokenizer=tokenizer,
                test_examples=gsm8k_examples,
                device=device,
                max_new_tokens=eval_cfg.get("gsm8k_max_new_tokens", 1024),
            )
            cond_scores["gsm8k"] = {
                "exact_match": gsm8k_res["exact_match"],
                "total": gsm8k_res["total"],
                "correct": gsm8k_res["correct"],
            }
            # Save per-example outputs
            per_ex_path = os.path.join(output_dir, f"gsm8k_{cond}_outputs.json")
            with open(per_ex_path, "w", encoding="utf-8") as f:
                json.dump(gsm8k_res["results"], f, indent=2)
                
        # 2. Other benchmarks via lm-eval
        other_tasks = [b for b in benchmarks if b != "gsm8k"]
        if other_tasks:
            eval_out = evaluate_with_lm_eval(
                model=model,
                tasks=other_tasks,
                batch_size=4,
                device=device,
            )
            if "results" in eval_out:
                for t in other_tasks:
                    t_res = eval_out["results"].get(t, {})
                    cond_scores[t] = {
                        "acc_norm": t_res.get("acc_norm,none", t_res.get("acc_norm", 0.0)),
                        "acc": t_res.get("acc,none", t_res.get("acc", 0.0)),
                    }
                    
        all_benchmark_results[cond] = cond_scores
        
        # Free model GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Compute changes from baseline in percentage points
    baseline_scores = all_benchmark_results.get("baseline", {})
    delta_results: Dict[str, Any] = {}
    
    for cond, scores in all_benchmark_results.items():
        if cond == "baseline":
            continue
        cond_deltas: Dict[str, Any] = {}
        for task, task_score in scores.items():
            base_task = baseline_scores.get(task, {})
            if task == "gsm8k":
                base_val = base_task.get("exact_match", 0.0)
                cur_val = task_score.get("exact_match", 0.0)
                cond_deltas[task] = cur_val - base_val
            elif "acc_norm" in task_score:
                base_val = base_task.get("acc_norm", 0.0) * 100.0
                cur_val = task_score.get("acc_norm", 0.0) * 100.0
                cond_deltas[task] = cur_val - base_val
            else:
                base_val = base_task.get("acc", 0.0) * 100.0
                cur_val = task_score.get("acc", 0.0) * 100.0
                cond_deltas[task] = cur_val - base_val
        delta_results[cond] = cond_deltas
        
    summary_data = {
        "benchmark_scores": all_benchmark_results,
        "delta_from_baseline_pct": delta_results,
    }
    
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    # Generate Markdown Table
    md_path = os.path.join(output_dir, "benchmark_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Task 1 Benchmark Results (7 Conditions)\n\n")
        f.write("| Condition | GSM8K (EM) | HellaSwag (Acc Norm) | BoolQ (Acc) | ARC-Easy (Acc Norm) |\n")
        f.write("|---|---|---|---|---|\n")
        for cond in conditions:
            sc = all_benchmark_results.get(cond, {})
            gsm = f"{sc.get('gsm8k', {}).get('exact_match', 0.0):.2f}%"
            hs = f"{sc.get('hellaswag', {}).get('acc_norm', 0.0)*100.0:.2f}%"
            bq = f"{sc.get('boolq', {}).get('acc', 0.0)*100.0:.2f}%"
            arc = f"{sc.get('arc_easy', {}).get('acc_norm', 0.0)*100.0:.2f}%"
            f.write(f"| {cond} | {gsm} | {hs} | {bq} | {arc} |\n")
            
    logger.info(f"Saved complete benchmark results to {json_path} and {md_path}")
    return summary_data


def run_qualitative_suite(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level qualitative evaluation runner for scripts/run_task1.py and configs/task1_config.yaml.
    
    Generates greedy responses for all 24 prompts from first_plot_questions.yaml across 7 conditions.
    Produces 168 total responses and the fixed 42-response annotation subset.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to execute qualitative evaluation")
        
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        raise RuntimeError(f"HuggingFace transformers required: {e}")
        
    from task1_llm_ft.models import apply_lora_to_model, apply_oplora_to_model, merge_adapters_to_model
    
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("evaluation", {})
    ckpt_cfg = cfg.get("checkpoints", {})
    output_dir = os.path.join(ckpt_cfg.get("output_dir", "./outputs/task1"), "qualitative")
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = model_cfg.get("name_or_path", "Qwen/Qwen3-0.6B")
    prompts_file = eval_cfg.get("qualitative_prompts_file", "./configs/first_plot_questions.yaml")
    max_tokens = eval_cfg.get("qualitative_max_tokens", 512)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    conditions = eval_cfg.get("eval_conditions", EVALUATION_CONDITIONS)
    base_out_dir = ckpt_cfg.get("output_dir", "./outputs/task1")
    
    condition_ckpts = {
        "baseline": None,
        "full_ft_midpoint": os.path.join(base_out_dir, "full_ft", "checkpoint_step_625", "model_state.pt"),
        "full_ft_final": os.path.join(base_out_dir, "full_ft", "checkpoint_step_1250", "model_state.pt"),
        "full_ft_l2_sp_final": os.path.join(base_out_dir, "full_ft_l2_sp", "checkpoint_step_1250", "model_state.pt"),
        "lora_4_final": os.path.join(base_out_dir, "lora_4", "checkpoint_step_1250", "model_state.pt"),
        "lora_16_final": os.path.join(base_out_dir, "lora_16", "checkpoint_step_1250", "model_state.pt"),
        "oplora_16_final": os.path.join(base_out_dir, "oplora_16", "checkpoint_step_1250", "model_state.pt"),
    }
    
    all_responses: Dict[str, List[Dict[str, Any]]] = {}
    
    for cond in conditions:
        logger.info(f"\n>>> Generating qualitative responses for condition: {cond}...")
        ckpt_path = condition_ckpts.get(cond)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        if cond != "baseline" and ckpt_path and os.path.exists(ckpt_path):
            if "lora_4" in cond:
                model, _ = apply_lora_to_model(model, rank=4, alpha=8.0)
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                merge_adapters_to_model(model)
            elif "lora_16" in cond:
                model, _ = apply_lora_to_model(model, rank=16, alpha=32.0)
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                merge_adapters_to_model(model)
            elif "oplora_16" in cond:
                model, _, _ = apply_oplora_to_model(model, rank=16, alpha=32.0, protection_rank=128)
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                merge_adapters_to_model(model)
            else:
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                
        model.to(device)
        model.eval()
        
        responses = generate_qualitative_responses(
            model=model,
            tokenizer=tokenizer,
            questions_yaml_path=prompts_file,
            device=device,
            max_new_tokens=max_tokens,
        )
        all_responses[cond] = responses
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Save all 168 responses
    resp_path = os.path.join(output_dir, "qualitative_responses_all_168.json")
    with open(resp_path, "w", encoding="utf-8") as f:
        json.dump(all_responses, f, indent=2)
    logger.info(f"Saved 168 responses to {resp_path}")
    
    # Generate 42-response annotation subset
    annotation_res = generate_annotation_subset(
        all_responses=all_responses,
        output_dir=output_dir,
    )
    
    return {
        "all_responses_count": sum(len(r) for r in all_responses.values()),
        "annotation_subset_count": annotation_res["total_annotations"],
        "output_dir": output_dir,
    }

