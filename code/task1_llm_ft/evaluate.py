"""Evaluation engine for Task 1: benchmarks, qualitative evaluation and annotation subset.

Benchmarks (all through lm-evaluation-harness, zero-shot, fixed splits):
  GSM8K     task ``gsm8k_cot``, generation, chat template with enable_thinking=False, greedy,
            1,024 new tokens, stock stop sequences and answer filters; primary metric =
            ``exact_match,flexible-extract``. Unextractable answers count as incorrect.
  HellaSwag / BoolQ / ARC-Easy  candidate log-likelihood, stock prompts WITHOUT chat template;
            primary metrics acc_norm / acc / acc_norm (raw accuracy is reported as well).

Every condition is scored by a fresh FP16 backbone (adapters merged and unwrapped), one result file
per condition, so conditions can be distributed over several GPUs and aggregated afterwards.
Failures raise: nothing is replaced by zeros.
"""

from typing import Any, Dict, List, Optional, Sequence
import functools
import json
import logging
import os
import re
import time

import yaml

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from common.normalization import extract_boxed_answer, normalize_math_answer
from task1_llm_ft.checkpoints import ALL_CONDITIONS, load_condition_model, make_merge_check_batch
from task1_llm_ft.qualitative_flags import math_framing_signals

logger = logging.getLogger(__name__)

EVALUATION_CONDITIONS = list(ALL_CONDITIONS)

BENCHMARK_TASKS = {
    "gsm8k": {"lm_eval_task": "gsm8k_cot", "split": "test", "num_examples": 1319,
              "primary": "exact_match,flexible-extract", "max_new_tokens": 1024},
    "hellaswag": {"lm_eval_task": "hellaswag", "split": "validation", "num_examples": 10042, "primary": "acc_norm,none"},
    "boolq": {"lm_eval_task": "boolq", "split": "validation", "num_examples": 3270, "primary": "acc,none"},
    "arc_easy": {"lm_eval_task": "arc_easy", "split": "test", "num_examples": 2376, "primary": "acc_norm,none"},
}
# ----------------------------------------------------------------------------------------------
# Local answer extraction (diagnostic helper; the reported GSM8K score comes from lm-eval)
# ----------------------------------------------------------------------------------------------


def _clean_number(raw: str) -> Optional[str]:
    """Strip currency, thousands separators, units and trailing punctuation from a numeric answer."""
    text = re.sub(r"\\text\{[^}]*\}|\\[a-zA-Z]+", " ", raw)
    text = text.replace("$", "").replace(",", "").strip()
    m = re.search(r"[-+]?\d*\.?\d+", text)
    if not m:
        return None
    num = m.group(0)
    if "." in num:
        num = num.rstrip("0").rstrip(".") or "0"
    return num.lstrip("+")


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract a numeric answer: \\boxed{...}, ``#### x``, ``The answer is x``, else the last number."""
    if not text:
        return None

    boxed = extract_boxed_answer(text)
    if boxed:
        num = _clean_number(boxed)
        if num is not None:
            return num
        norm = normalize_math_answer(boxed)
        if norm != "NO_ANSWER":
            return norm

    match_hash = re.search(r"####\s*([^\n\r]+)", text)
    if match_hash:
        num = _clean_number(match_hash.group(1))
        if num is not None:
            return num

    match_phrase = re.search(r"(?:the\s+answer\s+is|final\s+answer\s+is)\s*[:=]?\s*([^\n]+)", text, re.IGNORECASE)
    if match_phrase:
        num = _clean_number(match_phrase.group(1))
        if num is not None:
            return num

    numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if numbers:
        return _clean_number(numbers[-1])
    return None


# ----------------------------------------------------------------------------------------------
# Benchmarks via lm-evaluation-harness
# ----------------------------------------------------------------------------------------------


def _has_digit(x: Any) -> bool:
    return isinstance(x, str) and any(ch.isdigit() for ch in x)


def _sample_rows(task: str, samples: List[Dict[str, Any]], tokenizer: Any, max_gen_toks: int) -> List[Dict[str, Any]]:
    """Compact per-example records from lm-eval ``log_samples`` output."""
    rows: Dict[int, Dict[str, Any]] = {}
    for s in samples:
        row = rows.setdefault(s["doc_id"], {"doc_id": s["doc_id"], "target": s.get("target")})
        f = s.get("filter", "none")
        if task == "gsm8k_cot":
            raw = s["resps"][0][0] if s.get("resps") else ""
            row["response"] = raw
            row[f"extracted_{f}"] = s.get("filtered_resps", [None])[0]
            row[f"correct_{f}"] = bool(s.get("exact_match", 0))
            if "n_tokens" not in row:
                n_tok = len(tokenizer.encode(raw, add_special_tokens=False))
                row["n_tokens"] = n_tok
                row["truncated"] = n_tok >= max_gen_toks - 24  # re-encoding tolerance; stop strings end earlier
        else:
            row["loglikelihoods"] = [r[0][0] if isinstance(r[0], (list, tuple)) else r[0] for r in s.get("resps", [])]
            for k in ("acc", "acc_norm"):
                if k in s:
                    row[k] = s[k]
    return list(rows.values())


def _safe_sample_rows(task: str, samples: List[Dict[str, Any]], tokenizer: Any, max_gen_toks: int) -> List[Dict[str, Any]]:
    """Per-example logging must never discard an already computed benchmark score."""
    try:
        return _sample_rows(task, samples, tokenizer, max_gen_toks)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not condense lm-eval samples for {task}: {e}; storing raw samples instead")
        return [{"raw": s} for s in samples]


def evaluate_with_lm_eval(
    model: Any,
    tokenizer: Any,
    tasks: Sequence[str],
    batch_size: int,
    chat_template: bool = False,
    max_gen_toks: Optional[int] = None,
    seed: int = 6304,
) -> Dict[str, Any]:
    """Run lm-evaluation-harness zero-shot on an in-memory model; raises on any failure."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    if chat_template:
        # Qwen3 non-thinking mode: the harness must render the template with enable_thinking=False
        original = tokenizer.apply_chat_template
        tokenizer.apply_chat_template = functools.partial(original, enable_thinking=False)
    oom_error = getattr(torch, "OutOfMemoryError", None) or getattr(getattr(torch, "cuda", None), "OutOfMemoryError", RuntimeError)
    try:
        bs = batch_size
        while True:
            try:
                lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=bs)
                out = lm_eval.simple_evaluate(
                    model=lm,
                    tasks=list(tasks),
                    num_fewshot=0,
                    batch_size=bs,
                    apply_chat_template=chat_template,
                    gen_kwargs=f"max_gen_toks={max_gen_toks}" if max_gen_toks else None,
                    log_samples=True,
                    random_seed=seed,
                    numpy_random_seed=seed,
                    torch_random_seed=seed,
                    fewshot_random_seed=seed,
                    bootstrap_iters=1000,
                )
                break
            except oom_error:
                # Greedy decoding / log-likelihood scores do not depend on the batch size (up to fp16 padding noise),
                # so halving it and restarting this task group is safe.
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                logger.warning(f"CUDA OOM while evaluating {list(tasks)}; retrying with batch size {bs}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        if chat_template:
            tokenizer.apply_chat_template = original
    if out is None or "results" not in out:
        raise RuntimeError(f"lm-eval returned no results for {tasks}")
    return out


def run_benchmarks_for_condition(
    cond: str,
    cfg: Dict[str, Any],
    device: str,
    tokenizer: Any,
) -> Dict[str, Any]:
    """Evaluate one condition on all four benchmarks and write its result files."""
    eval_cfg = cfg.get("evaluation", {})
    ckpt_cfg = cfg.get("checkpoints", {})
    base_out = ckpt_cfg.get("output_dir", "./outputs/task1")
    res_dir = os.path.join(cfg.get("results_dir", "./results/task1"), "eval")
    samples_dir = os.path.join(base_out, "eval_samples")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    t0 = time.time()
    merge_batch = make_merge_check_batch(tokenizer, [
        "The capital of France is", "Question: What is 17 * 23? Answer:",
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many in May.",
        "A man is sitting on a roof. He"])
    model, info = load_condition_model(
        cond, base_out, cfg.get("model", {}).get("name_or_path", "Qwen/Qwen3-0.6B"), device,
        merge_check_inputs=merge_batch)

    max_new = eval_cfg.get("gsm8k_max_new_tokens", 1024)
    scores: Dict[str, Any] = {}
    raw_results: Dict[str, Any] = {}

    benchmarks = eval_cfg.get("benchmarks", list(BENCHMARK_TASKS))
    if "gsm8k" in benchmarks:
        # The ~20 min generation is cached the moment it finishes, so a failure in a later benchmark (or a retry of the
        # queue) never repeats it. Delete {cond}_gsm8k_raw.json to force a fresh run.
        raw_path = os.path.join(samples_dir, f"{cond}_gsm8k_raw.json")
        cached = None
        if os.path.exists(raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("max_new") != max_new:
                cached = None
        if cached is not None:
            logger.info(f"Reusing cached GSM8K results for {cond} ({raw_path})")
            raw_results["gsm8k_cot"], rows = cached["raw"], cached["rows"]
        else:
            out = evaluate_with_lm_eval(
                model, tokenizer, ["gsm8k_cot"], batch_size=eval_cfg.get("gsm8k_batch_size", 32),
                chat_template=True, max_gen_toks=max_new)
            raw_results["gsm8k_cot"] = out["results"]["gsm8k_cot"]
            rows = _safe_sample_rows("gsm8k_cot", out["samples"]["gsm8k_cot"], tokenizer, max_new)
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump({"raw": raw_results["gsm8k_cot"], "rows": rows, "max_new": max_new}, f, default=str)
        with open(os.path.join(samples_dir, f"{cond}_gsm8k.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, default=str)
        # NB: read from raw_results, which holds the lm-eval result in BOTH branches above (``out`` is unbound when the
        # cached generation is reused).
        gsm8k_res = raw_results["gsm8k_cot"]
        scores["gsm8k"] = {
            "exact_match_flexible": gsm8k_res["exact_match,flexible-extract"],
            "exact_match_strict": gsm8k_res.get("exact_match,strict-match"),
            "truncated_fraction": sum(r.get("truncated", False) for r in rows) / max(1, len(rows)),
            # Only the harness's own '[invalid]' fallback / empty string. The stock flexible regex also matches
            # number-free strings such as "$$", which are NOT counted here: see the next key.
            "unextractable_fraction_flexible": sum(
                1 for r in rows if r.get("extracted_flexible-extract") in (None, "", "[invalid]")) / max(1, len(rows)),
            # Extraction that contains no digit at all ("$$", "[invalid]", ...): the answer was not actually read.
            "no_digit_extraction_fraction_flexible": sum(
                1 for r in rows if not _has_digit(r.get("extracted_flexible-extract"))) / max(1, len(rows)),
            "n": len(rows),
        }

    ll_tasks = [t for t in ("hellaswag", "boolq", "arc_easy") if t in benchmarks]
    if ll_tasks:
        out = evaluate_with_lm_eval(
            model, tokenizer, ll_tasks, batch_size=eval_cfg.get("loglik_batch_size", 32), chat_template=False)
        for t in ll_tasks:
            r = out["results"][t]
            raw_results[t] = r
            scores[t] = {
                "acc": r.get("acc,none"),
                "acc_norm": r.get("acc_norm,none"),
                "n": len(out["samples"][t]) if t in out.get("samples", {}) else None,
            }
            with open(os.path.join(samples_dir, f"{cond}_{t}.json"), "w", encoding="utf-8") as f:
                json.dump(_safe_sample_rows(t, out["samples"][t], tokenizer, max_new), f, default=str)

    result = {
        "condition": cond,
        "model_info": info,
        "scores": scores,
        "lm_eval_results": raw_results,
        "settings": {"num_fewshot": 0, "dtype": "float16", "gsm8k_max_new_tokens": max_new,
                     "gsm8k_chat_template": True, "enable_thinking": False},
        "elapsed_sec": time.time() - t0,
        "peak_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0,
    }
    try:
        import lm_eval
        result["lm_eval_version"] = getattr(lm_eval, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        pass
    with open(os.path.join(res_dir, f"{cond}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _primary(scores: Dict[str, Any], bench: str) -> Optional[float]:
    s = scores.get(bench)
    if not s:
        return None
    v = s["exact_match_flexible"] if bench == "gsm8k" else (s["acc"] if bench == "boolq" else s["acc_norm"])
    return None if v is None else 100.0 * float(v)


def aggregate_benchmark_results(cfg: Dict[str, Any], conditions: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Combine per-condition files into one table with percentage-point changes from the baseline."""
    res_dir = os.path.join(cfg.get("results_dir", "./results/task1"), "eval")
    conditions = list(conditions or cfg.get("evaluation", {}).get("eval_conditions", EVALUATION_CONDITIONS))
    per_cond: Dict[str, Any] = {}
    for c in conditions:
        path = os.path.join(res_dir, f"{c}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                per_cond[c] = json.load(f)["scores"]
    if "baseline" not in per_cond:
        raise FileNotFoundError("baseline evaluation missing: cannot compute changes from baseline")

    table: Dict[str, Any] = {}
    deltas: Dict[str, Any] = {}
    for c, sc in per_cond.items():
        table[c] = {b: _primary(sc, b) for b in ("gsm8k", "hellaswag", "boolq", "arc_easy")}
        table[c]["raw_acc"] = {b: (100.0 * sc[b]["acc"] if sc.get(b, {}).get("acc") is not None else None)
                               for b in ("hellaswag", "boolq", "arc_easy")}
        if c != "baseline":
            deltas[c] = {b: (None if table[c][b] is None or table["baseline"][b] is None
                             else table[c][b] - table["baseline"][b])
                         for b in ("gsm8k", "hellaswag", "boolq", "arc_easy")}

    out_dir = os.path.join(cfg.get("results_dir", "./results/task1"))
    summary = {"primary_scores_pct": table, "delta_from_baseline_pp": deltas,
               "missing_conditions": [c for c in conditions if c not in per_cond]}
    with open(os.path.join(out_dir, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    def fmt(v: Optional[float], d: bool = False) -> str:
        return "n/a" if v is None else (f"{v:+.2f}" if d else f"{v:.2f}")

    with open(os.path.join(out_dir, "benchmark_results.md"), "w", encoding="utf-8") as f:
        f.write("# Task 1 benchmark results (primary scores, %; deltas in percentage points vs baseline)\n\n")
        f.write("| Condition | GSM8K (flex EM) | HellaSwag (acc_norm) | BoolQ (acc) | ARC-Easy (acc_norm) |\n|---|---|---|---|---|\n")
        for c in per_cond:
            row = table[c]
            cells = []
            for b in ("gsm8k", "hellaswag", "boolq", "arc_easy"):
                cell = fmt(row[b])
                if c != "baseline":
                    cell += f" ({fmt(deltas[c][b], True)})"
                cells.append(cell)
            f.write(f"| {c} | " + " | ".join(cells) + " |\n")
    return summary


def run_task1_evaluations(cfg: Dict[str, Any], conditions: Optional[Sequence[str]] = None, aggregate: bool = True) -> Dict[str, Any]:
    """Evaluate the requested conditions (default: all seven) and aggregate the table."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to execute evaluations")
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = cfg.get("model", {}).get("name_or_path", "Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    conds = list(conditions or cfg.get("evaluation", {}).get("eval_conditions", EVALUATION_CONDITIONS))
    for cond in conds:
        logger.info(f">>> Evaluating condition: {cond}")
        run_benchmarks_for_condition(cond, cfg, device, tokenizer)
    return aggregate_benchmark_results(cfg) if aggregate else {}


# ----------------------------------------------------------------------------------------------
# Qualitative evaluation
# ----------------------------------------------------------------------------------------------


def load_qualitative_prompts(path: str) -> List[Dict[str, str]]:
    """Read first_plot_questions.yaml. Accepts both this repo's layout (``questions:`` list with
    ``prompt``/``system_prompt``/``format``) and the original layout (top-level list with
    ``paraphrases`` and ``system``; JSON/template variants identified by their id suffix)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Questions YAML not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    items = data.get("questions", []) if isinstance(data, dict) else data
    out: List[Dict[str, str]] = []
    for it in items:
        qid = it["id"]
        prompt = it.get("prompt")
        if prompt is None:
            paras = it.get("paraphrases") or []
            if not paras:
                raise KeyError(f"Question {qid} has neither 'prompt' nor 'paraphrases'")
            prompt = paras[0]
        fmt = it.get("format") or ("json" if qid.endswith("_json") else "template" if qid.endswith("_template") else "plain")
        out.append({
            "id": qid,
            "name": it.get("name", qid),
            "format": fmt,
            "system_prompt": it.get("system_prompt", it.get("system", "")) or "",
            "prompt": prompt,
        })
    return out


def generate_qualitative_responses(
    model: Any,
    tokenizer: Any,
    questions_yaml_path: str = "configs/first_plot_questions.yaml",
    device: str = "cpu",
    max_new_tokens: int = 512,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """Greedy response for every prompt (thinking disabled, 512-token cap), batched with left padding."""
    questions = load_qualitative_prompts(questions_yaml_path)
    logger.info(f"Loaded {len(questions)} qualitative prompts from {questions_yaml_path}")
    model.eval()

    texts = []
    for q in questions:
        messages = []
        if q["system_prompt"]:
            messages.append({"role": "system", "content": q["system_prompt"]})
        messages.append({"role": "user", "content": q["prompt"]})
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    gen_eos = model.generation_config.eos_token_id
    eos_ids = set([gen_eos] if isinstance(gen_eos, int) else (gen_eos or [])) | {tokenizer.eos_token_id}
    eos_ids.discard(None)

    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    responses: List[Dict[str, Any]] = []
    try:
        for start in range(0, len(questions), batch_size):
            chunk = questions[start:start + batch_size]
            enc = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                    pad_token_id=tokenizer.pad_token_id)
            for i, q in enumerate(chunk):
                new = out[i, enc["input_ids"].shape[1]:].tolist()
                end = next((j for j, t in enumerate(new) if t in eos_ids), None)
                gen = new if end is None else new[:end]
                responses.append({
                    "id": q["id"], "name": q["name"], "format": q["format"], "prompt": q["prompt"],
                    "system_prompt": q["system_prompt"],
                    "response": tokenizer.decode(gen, skip_special_tokens=True),
                    "token_count": len(gen),
                    "truncated": end is None and len(gen) >= max_new_tokens,
                })
    finally:
        tokenizer.padding_side = old_side
    return responses


def generate_annotation_subset(
    all_responses: Dict[str, List[Dict[str, Any]]],
    output_dir: str = "./results/task1/qualitative",
) -> Dict[str, Any]:
    """Build the fixed 42-response annotation subset.

    The first and last *underlying* questions are taken from the file's plain-question order
    (2 questions x 3 formats x 7 conditions = 42). Automatic flags are only a first pass for the
    manual annotation (format compliance, coherence, repetition, mathematical framing, inappropriate
    content, truncation): ``inappropriate_content`` and ``manual_notes`` are left for the annotator.
    """
    first_cond = next(iter(all_responses.values()))
    plain_ids = [r["id"] for r in first_cond
                 if r.get("format", "plain") == "plain" and not r["id"].endswith(("_json", "_template"))]
    if len(plain_ids) < 2:
        raise ValueError("Need at least two plain questions to pick the first and last underlying question")
    target_prefixes = [plain_ids[0], plain_ids[-1]]

    subset: List[Dict[str, Any]] = []
    for condition, resp_list in all_responses.items():
        resp_map = {r["id"]: r for r in resp_list}
        for prefix in target_prefixes:
            for fmt in ("plain", "json", "template"):
                q_id = prefix if fmt == "plain" else f"{prefix}_{fmt}"
                item = resp_map.get(q_id, {})
                text = item.get("response", "")
                is_truncated = item.get("truncated", False)

                compliant, notes = True, "Compliant"
                if fmt == "json":
                    body = text.strip()
                    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body)
                    try:
                        json.loads(body)
                    except Exception:  # noqa: BLE001
                        compliant, notes = False, "Invalid JSON syntax"
                elif fmt == "template":
                    if not re.search(r"^\s*\w+\s*=", text, re.MULTILINE):
                        compliant, notes = False, "Missing template key = value lines"

                words = text.lower().split()
                has_repetition = False
                if len(words) > 20:
                    bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
                    has_repetition = len(set(bigrams)) / max(1, len(bigrams)) < 0.6

                # training-format artefacts ("The answer is: x", "#### x", \boxed, LaTeX, <<calc>>) and math vocabulary;
                # see task1_llm_ft/qualitative_flags.py
                math_signals = math_framing_signals(text)

                subset.append({
                    "condition": condition,
                    "question_id": q_id,
                    "underlying_question": prefix,
                    "format": fmt,
                    "text": text,
                    "excerpt": (text[:160] + "...") if len(text) > 160 else text,
                    "format_compliance": compliant,
                    "format_notes": notes,
                    "repetition": has_repetition,
                    "unnecessary_math_framing": bool(math_signals),
                    "math_framing_signals": math_signals,
                    "inappropriate_content": None,   # to be filled in by manual annotation
                    "manual_notes": "",
                    "truncated": is_truncated,
                })

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "annotation_subset_42.json")
    # Regenerating the automatic flags must never wipe hand-written annotations from an earlier pass.
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            previous = {(p.get("condition"), p.get("question_id")): p for p in json.load(f)}
        for it in subset:
            old = previous.get((it["condition"], it["question_id"]))
            if old is not None:
                if old.get("inappropriate_content") is not None:
                    it["inappropriate_content"] = old["inappropriate_content"]
                if old.get("manual_notes"):
                    it["manual_notes"] = old["manual_notes"]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)

    with open(os.path.join(output_dir, "annotation_subset_42.md"), "w", encoding="utf-8") as f:
        f.write("# 42-response qualitative annotation subset (automatic first-pass flags; annotate manually)\n\n")
        f.write("| Condition | Question | Format | Compliant? | Repetition | Math framing? | Truncated? | Excerpt |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for it in subset:
            comp = "Yes" if it["format_compliance"] else f"No ({it['format_notes']})"
            excerpt = it["excerpt"].replace("\n", " ").replace("|", "\\|")
            math_cell = f"Yes ({', '.join(it['math_framing_signals'])})" if it["unnecessary_math_framing"] else "No"
            f.write(f"| {it['condition']} | {it['underlying_question']} | {it['format']} | {comp} | "
                    f"{'Yes' if it['repetition'] else 'No'} | {math_cell} | "
                    f"{'Yes' if it['truncated'] else 'No'} | {excerpt} |\n")
    return {"total_annotations": len(subset), "records": subset}


def run_qualitative_suite(cfg: Dict[str, Any], conditions: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Generate qualitative responses per condition; build the 168-response file and the 42-subset once all exist."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to execute qualitative evaluation")
    from transformers import AutoTokenizer

    eval_cfg = cfg.get("evaluation", {})
    model_name = cfg.get("model", {}).get("name_or_path", "Qwen/Qwen3-0.6B")
    base_out = cfg.get("checkpoints", {}).get("output_dir", "./outputs/task1")
    qual_dir = os.path.join(cfg.get("results_dir", "./results/task1"), "qualitative")
    os.makedirs(qual_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_conds = list(eval_cfg.get("eval_conditions", EVALUATION_CONDITIONS))
    for cond in list(conditions or all_conds):
        logger.info(f">>> Qualitative generation: {cond}")
        model, _ = load_condition_model(cond, base_out, model_name, device)
        responses = generate_qualitative_responses(
            model, tokenizer, eval_cfg.get("qualitative_prompts_file", "./configs/first_plot_questions.yaml"),
            device=device, max_new_tokens=eval_cfg.get("qualitative_max_tokens", 512))
        with open(os.path.join(qual_dir, f"{cond}.json"), "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=2)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return aggregate_qualitative(cfg)


def aggregate_qualitative(cfg: Dict[str, Any]) -> Dict[str, Any]:
    qual_dir = os.path.join(cfg.get("results_dir", "./results/task1"), "qualitative")
    all_conds = list(cfg.get("evaluation", {}).get("eval_conditions", EVALUATION_CONDITIONS))
    have = {c: os.path.join(qual_dir, f"{c}.json") for c in all_conds if os.path.exists(os.path.join(qual_dir, f"{c}.json"))}
    if len(have) != len(all_conds):
        return {"complete": False, "missing": [c for c in all_conds if c not in have]}
    all_responses = {}
    for c, path in have.items():
        with open(path, "r", encoding="utf-8") as f:
            all_responses[c] = json.load(f)
    with open(os.path.join(qual_dir, "qualitative_responses_all.json"), "w", encoding="utf-8") as f:
        json.dump(all_responses, f, indent=2)
    res = generate_annotation_subset(all_responses, output_dir=qual_dir)
    return {"complete": True, "all_responses_count": sum(len(r) for r in all_responses.values()),
            "annotation_subset_count": res["total_annotations"], "output_dir": qual_dir}


# Legacy name kept for API compatibility (the reported GSM8K number comes from lm-eval).
def evaluate_gsm8k_generation(model: Any, tokenizer: Any, test_examples: Sequence[Dict[str, Any]],
                              device: str = "cpu", max_new_tokens: int = 1024) -> Dict[str, Any]:
    """Quick local GSM8K check (greedy, non-thinking chat template) using the numeric extractor above."""
    if not HAS_TORCH:
        return {"exact_match": 0.0, "total": 0, "correct": 0, "results": []}
    model.eval()
    results, correct = [], 0
    for item in test_examples:
        q = item.get("question") or item.get("query") or ""
        gold = extract_gsm8k_answer(item.get("answer") or item.get("response") or "")
        text = tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        enc = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id)
        gen = out[0, enc["input_ids"].shape[1]:].tolist()
        gen_text = tokenizer.decode(gen, skip_special_tokens=True)
        pred = extract_gsm8k_answer(gen_text)
        ok = pred is not None and gold is not None and pred == gold
        correct += int(ok)
        results.append({"question": q, "generated_text": gen_text, "predicted_answer": pred, "gold_answer": gold,
                        "is_correct": ok, "truncated": len(gen) >= max_new_tokens, "generated_tokens_count": len(gen)})
    total = len(test_examples)
    return {"exact_match": 100.0 * correct / total if total else 0.0, "total": total, "correct": correct, "results": results}
