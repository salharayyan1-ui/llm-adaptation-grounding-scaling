"""Offline re-analysis of the saved per-example Task 1 outputs (no GPU, no model, no re-evaluation).

Reads ``outputs/task1/eval_samples/<condition>_<benchmark>.json`` (written by ``evaluate.py``) and writes the small,
committed file ``results/task1/offline_analysis.json``. Nothing here changes a reported benchmark score: the primary
metrics stay the stock lm-evaluation-harness numbers; this module adds evidence *about* those numbers.

1. GSM8K answer-extraction sensitivity.
   The stock ``flexible-extract`` filter of ``gsm8k_cot`` takes the LAST match of ``(-?[$0-9.,]{2,})|(-?[0-9]+)``.
   That pattern also matches number-free strings such as ``$$`` (the closing LaTeX display delimiter), so a response
   that ends in ``\\boxed{18} $$`` is "extracted" as ``$$`` and graded wrong, while the answer is present. The stock
   filter is replicated here (and checked against the stored extractions), then re-scored with
     * ``digit_flexible``: the same rule, but matches without any digit are skipped;
     * ``boxed_first``: the last numeric ``\\boxed{}`` if present, else ``digit_flexible``.
   These are sensitivity analyses next to the stock score, not replacements for it.
2. Paired bootstrap confidence intervals for the controlled comparisons and the changes from the baseline (the same
   examples are scored under both conditions, so the examples are resampled jointly).
3. BoolQ yes/no bias: predicted-yes rate, per-class accuracy, balanced accuracy and ROC-AUC of the yes-vs-no
   log-likelihood gap (AUC does not depend on the decision threshold, so it separates discrimination from a
   shifted yes/no prior). These are descriptive point estimates without intervals.
4. Math-style intrusion in the qualitative responses. Counts, over all 24 saved responses of each condition (not only
   the 42-response annotation subset), of responses carrying MetaMathQA / GSM8K training artefacts (``The answer is:``,
   ``#### x``, ``\\boxed``, LaTeX, ``<<calc>>``; see ``qualitative_flags.py``). Descriptive; the manual annotation of
   the subset remains the annotator's own.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import logging
import os
import re

import numpy as np

from task1_llm_ft.checkpoints import ALL_CONDITIONS
from task1_llm_ft.qualitative_flags import SIGNAL_NAMES, math_framing_signals

logger = logging.getLogger(__name__)

INVALID = "[invalid]"
# primary per-example field of each benchmark in the eval_samples files
PRIMARY_FIELD = {"hellaswag": "acc_norm", "boolq": "acc", "arc_easy": "acc_norm"}
BENCHMARKS = ("gsm8k", "hellaswag", "boolq", "arc_easy")
# columns of the paired tables: (key, source benchmark / variant)
PAIRED_COLUMNS = ("gsm8k", "gsm8k_digit_flexible", "hellaswag", "boolq", "arc_easy")

CONTROLLED_PAIRS: List[Tuple[str, str, str, str]] = [
    ("rank", "lora_16_final", "lora_4_final", "Rank: LoRA-16 − LoRA-4"),
    ("regularisation", "full_ft_l2_sp_final", "full_ft_final", "Explicit regularisation: Full FT + L2-SP − Full FT"),
    ("orthogonal", "oplora_16_final", "lora_16_final", "Orthogonal protection: OPLoRA-16 − LoRA-16"),
    ("method", "full_ft_final", "lora_16_final", "Method: Full FT − LoRA-16"),
    ("progress", "full_ft_final", "full_ft_midpoint", "Progress: Full FT final − midpoint"),
]


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------
def _truthy(x: Any) -> bool:
    return x is True or str(x).strip().lower() in ("true", "1", "1.0")


def _as_list(x: Any) -> List[float]:
    if isinstance(x, str):
        x = json.loads(x)
    return [float(v) for v in x]


def load_condition_rows(samples_dir: str, cond: str, bench: str) -> List[Dict[str, Any]]:
    """Per-example rows of one condition/benchmark, ordered by integer ``doc_id``."""
    path = os.path.join(samples_dir, f"{cond}_{bench}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing per-example file {path}. These live in outputs/ (git-ignored): run the evaluation, or copy "
            f"them back from the GPU machine with scripts/pull_results.sh.")
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    by_id = {str(r["doc_id"]): r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"{path}: duplicate doc_id values")
    return [by_id[i] for i in sorted(by_id, key=int)]


# ----------------------------------------------------------------------------------------------
# GSM8K extraction (stock replication + digit-required variants)
# ----------------------------------------------------------------------------------------------
_FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")          # gsm8k_cot "flexible-extract", group_select = -1
_STOCK_IGNORE = (",", r"\$", r"(?s).*#### ", r"\.$")          # gsm8k_cot exact_match regexes_to_ignore


def _match_token(match: Tuple[str, ...]) -> str:
    return next((g for g in match if g), "")


def stock_flexible_extract(response: str) -> str:
    """lm-eval ``RegexFilter`` of gsm8k_cot's flexible-extract: last regex match, ``[invalid]`` if none."""
    matches = _FLEX.findall(response)
    if not matches:
        return INVALID
    tok = _match_token(matches[-1]).strip()
    return tok if tok else INVALID


def digit_flexible_extract(response: str) -> str:
    """Same rule as the stock filter, but matches without a digit (``$$``, ``..``, ``,`` ...) are skipped."""
    last = None
    for m in _FLEX.findall(response):
        tok = _match_token(m)
        if any(ch.isdigit() for ch in tok):
            last = tok
    return last.strip() if last else INVALID


def stock_normalise(s: str) -> str:
    for pat in _STOCK_IGNORE:
        s = re.sub(pat, "", s)
    return s


def stock_correct(pred: str, target: str) -> bool:
    """lm-eval ``exact_match`` with gsm8k_cot's ``regexes_to_ignore`` on both sides."""
    return stock_normalise(pred) == stock_normalise(str(target))


def has_digit(s: Any) -> bool:
    return isinstance(s, str) and any(ch.isdigit() for ch in s)


def last_boxed(text: str) -> Optional[str]:
    """Content of the last balanced ``\\boxed{...}``; None if absent or unbalanced."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    depth, k = 1, i + len("\\boxed{")
    start = k
    while k < len(text) and depth > 0:
        depth += (text[k] == "{") - (text[k] == "}")
        k += 1
    return text[start:k - 1] if depth == 0 else None


def _to_float(s: str) -> Optional[float]:
    t = re.sub(r"\\text\{[^}]*\}", "", str(s))
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    t = re.sub(r"[\\${}%,\s]", "", t)
    return float(t) if re.fullmatch(r"-?\d+(?:\.\d+)?|-?\.\d+", t) else None


def boxed_first_correct(response: str, target: str, digit_pred: str) -> bool:
    """Numeric comparison of the last numeric ``\\boxed{}`` if there is one, else the digit-required stock rule."""
    b = last_boxed(response)
    if b is not None:
        v, tv = _to_float(b), _to_float(target)
        if v is not None and tv is not None:
            return abs(v - tv) < 1e-6
    return stock_correct(digit_pred, target)


def analyse_gsm8k(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Per-condition extraction summary and the per-example correctness vectors used for pairing."""
    n = len(rows)
    stored_ok = np.array([_truthy(r["correct_flexible-extract"]) for r in rows], dtype=float)
    stock_pred = [stock_flexible_extract(r["response"]) for r in rows]
    digit_pred = [digit_flexible_extract(r["response"]) for r in rows]
    stock_ok = np.array([stock_correct(p, r["target"]) for p, r in zip(stock_pred, rows)], dtype=float)
    digit_ok = np.array([stock_correct(p, r["target"]) for p, r in zip(digit_pred, rows)], dtype=float)
    boxed_ok = np.array([boxed_first_correct(r["response"], r["target"], p) for p, r in zip(digit_pred, rows)], dtype=float)
    stored_extract = [str(r.get("extracted_flexible-extract")) for r in rows]

    summary = {
        "n": n,
        "stock_flexible_pct": 100.0 * float(stored_ok.mean()),
        "replicated_stock_pct": 100.0 * float(stock_ok.mean()),
        # replication check: both must be 0 for the replicated filter to be trusted
        "replication_correctness_mismatches": int((stock_ok != stored_ok).sum()),
        "replication_extraction_mismatches": int(sum(a != b for a, b in zip(stock_pred, stored_extract))),
        "digit_flexible_pct": 100.0 * float(digit_ok.mean()),
        "boxed_first_pct": 100.0 * float(boxed_ok.mean()),
        # stock extraction returned a string without any digit ("$$", "[invalid]", ...): answer not actually read
        "no_digit_extraction_pct": 100.0 * float(np.mean([not has_digit(p) for p in stock_pred])),
        "no_digit_extraction_correct_by_digit_rule_pct": 100.0 * float(
            np.sum([(not has_digit(p)) and bool(d) for p, d in zip(stock_pred, digit_ok)]) / n),
        "boxed_present_pct": 100.0 * float(np.mean([last_boxed(r["response"]) is not None for r in rows])),
        "answer_is_phrase_pct": 100.0 * float(np.mean(["answer is" in r["response"].lower() for r in rows])),
        "mean_tokens": float(np.mean([float(r["n_tokens"]) for r in rows])),
        "truncated_pct": 100.0 * float(np.mean([_truthy(r["truncated"]) for r in rows])),
    }
    if summary["replication_correctness_mismatches"] or summary["replication_extraction_mismatches"]:
        logger.warning(
            "Stock-filter replication differs from the stored lm-eval output (correctness: %d, extraction: %d examples)",
            summary["replication_correctness_mismatches"], summary["replication_extraction_mismatches"])
    return summary, {"gsm8k": stored_ok, "gsm8k_digit_flexible": digit_ok}


# ----------------------------------------------------------------------------------------------
# BoolQ yes/no bias
# ----------------------------------------------------------------------------------------------
def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic with average ranks for ties."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0           # 1-based average rank of each distinct value
    ranks = avg_rank[inv.reshape(-1)]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def analyse_boolq(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], np.ndarray]:
    """BoolQ choices are ["no", "yes"]; ``target`` 1 means the gold answer is yes."""
    ll = np.array([_as_list(r["loglikelihoods"]) for r in rows], dtype=float)
    gold = np.array([int(float(r["target"])) for r in rows], dtype=int)
    pred = ll.argmax(axis=1)                                      # same tie rule as lm-eval's acc
    ok = (pred == gold).astype(float)
    stored = np.array([float(r["acc"]) for r in rows])
    acc_yes = float((pred[gold == 1] == 1).mean())
    acc_no = float((pred[gold == 0] == 0).mean())
    return {
        "n": len(rows),
        "accuracy_pct": 100.0 * float(ok.mean()),
        "accuracy_check_mismatches": int((ok != stored).sum()),
        "pred_yes_pct": 100.0 * float(pred.mean()),
        "gold_yes_pct": 100.0 * float(gold.mean()),
        "acc_gold_yes_pct": 100.0 * acc_yes,
        "acc_gold_no_pct": 100.0 * acc_no,
        "balanced_accuracy_pct": 100.0 * (acc_yes + acc_no) / 2.0,
        "auc_yes_vs_no_gap": roc_auc(ll[:, 1] - ll[:, 0], gold),
        "mean_loglik_gap_yes_minus_no": float((ll[:, 1] - ll[:, 0]).mean()),
    }, ok


# ----------------------------------------------------------------------------------------------
# Paired bootstrap
# ----------------------------------------------------------------------------------------------
def paired_bootstrap(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int = 2000,
                     ci: float = 95.0) -> Dict[str, Any]:
    """Mean difference a-b in percentage points with a percentile interval over jointly resampled examples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired vectors differ in length: {a.shape} vs {b.shape}")
    d = a - b
    n = len(d)
    means = np.empty(n_boot)
    chunk = max(1, 2_000_000 // n)
    for s in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s)
        idx = rng.integers(0, n, size=(k, n))
        means[s:s + k] = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(100 - ci) / 2.0, 100 - (100 - ci) / 2.0])
    return {"n": n, "diff_pp": 100.0 * float(d.mean()), "ci_lo_pp": 100.0 * float(lo), "ci_hi_pp": 100.0 * float(hi),
            "significant": bool(lo > 0 or hi < 0)}


def pair_definitions(conditions: Sequence[str]) -> List[Dict[str, str]]:
    have = set(conditions)
    defs = [{"id": i, "label": lab, "kind": "controlled", "a": a, "b": b}
            for i, a, b, lab in CONTROLLED_PAIRS if a in have and b in have]
    defs += [{"id": f"{c}_vs_baseline", "label": f"{c} − baseline", "kind": "vs_baseline", "a": c, "b": "baseline"}
             for c in conditions if c != "baseline"]
    return defs


# ----------------------------------------------------------------------------------------------
# Math-style intrusion in the qualitative responses
# ----------------------------------------------------------------------------------------------
def analyse_qualitative_math_intrusion(responses: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Math-framing signal counts over one condition's saved qualitative responses (all prompts, all formats)."""
    by_signal = {name: 0 for name in SIGNAL_NAMES}
    by_format: Dict[str, Dict[str, int]] = {}
    flagged: Dict[str, List[str]] = {}
    artefact = 0
    for r in responses:
        sig = math_framing_signals(r.get("response", ""))
        fmt = r.get("format", "plain")
        cell = by_format.setdefault(fmt, {"n": 0, "flagged": 0})
        cell["n"] += 1
        for name in sig:
            by_signal[name] += 1
        if sig:
            cell["flagged"] += 1
            flagged[r["id"]] = sig
        artefact += int(any(n != "vocabulary" for n in sig))
    return {"n_responses": len(responses), "n_flagged": len(flagged), "n_with_artefact": artefact,
            "by_signal": by_signal, "by_format": by_format, "flagged_ids": flagged}


def load_qualitative_responses(res_dir: str, condition: str) -> Optional[List[Dict[str, Any]]]:
    path = os.path.join(res_dir, "qualitative", f"{condition}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------
def run_offline_analysis(cfg: Dict[str, Any], n_boot: int = 2000) -> Dict[str, Any]:
    """Compute the three analyses for every condition with complete per-example files; write the JSON."""
    base_out = cfg.get("checkpoints", {}).get("output_dir", "./outputs/task1")
    res_dir = cfg.get("results_dir", "./results/task1")
    samples_dir = os.path.join(base_out, "eval_samples")
    seed = int(cfg.get("model", {}).get("seed", 6304))
    wanted = list(cfg.get("evaluation", {}).get("eval_conditions", ALL_CONDITIONS))
    conditions = [c for c in wanted
                  if all(os.path.exists(os.path.join(samples_dir, f"{c}_{b}.json")) for b in BENCHMARKS)]
    if "baseline" not in conditions:
        raise FileNotFoundError(f"baseline per-example files missing in {samples_dir}: nothing to compare against")
    skipped = [c for c in wanted if c not in conditions]
    if skipped:
        logger.warning("Skipping conditions without complete per-example files: %s", skipped)

    vectors: Dict[str, Dict[str, np.ndarray]] = {c: {} for c in conditions}   # per-example 0/1 (or accuracy) vectors
    doc_ids: Dict[str, List[str]] = {}
    gsm: Dict[str, Any] = {}
    boolq: Dict[str, Any] = {}
    for c in conditions:
        rows = load_condition_rows(samples_dir, c, "gsm8k")
        gsm[c], vecs = analyse_gsm8k(rows)
        vectors[c].update(vecs)
        ids = [str(r["doc_id"]) for r in rows]
        if doc_ids.setdefault("gsm8k", ids) != ids:
            raise ValueError(f"{c}: GSM8K doc_ids differ from the baseline's; pairing would be invalid")

        for b, field in PRIMARY_FIELD.items():
            rows = load_condition_rows(samples_dir, c, b)
            ids = [str(r["doc_id"]) for r in rows]
            if doc_ids.setdefault(b, ids) != ids:
                raise ValueError(f"{c}: {b} doc_ids differ from the baseline's; pairing would be invalid")
            vectors[c][b] = np.array([float(r[field]) for r in rows])
            if b == "boolq":
                boolq[c], _ = analyse_boolq(rows)

    base = gsm["baseline"]
    gain = {c: {k: gsm[c][k] - base[k] for k in ("stock_flexible_pct", "digit_flexible_pct", "boxed_first_pct")}
            for c in conditions if c != "baseline"}

    rng = np.random.default_rng(seed)
    paired = []
    for p in pair_definitions(conditions):
        cols = {col: paired_bootstrap(vectors[p["a"]][col], vectors[p["b"]][col], rng, n_boot=n_boot)
                for col in PAIRED_COLUMNS}
        paired.append({**p, "benchmarks": cols})

    qualitative: Dict[str, Any] = {}
    for c in wanted:
        qrows = load_qualitative_responses(res_dir, c)
        if qrows is not None:
            qualitative[c] = analyse_qualitative_math_intrusion(qrows)

    result = {
        "settings": {
            "seed": seed, "n_bootstrap": n_boot, "ci": "95% percentile interval, examples resampled jointly (paired)",
            "samples_dir": samples_dir, "conditions": conditions,
            "primary_fields": {"gsm8k": "exact_match,flexible-extract (stored per-example flag)", **PRIMARY_FIELD},
        },
        "definitions": {
            "stock_flexible": "gsm8k_cot flexible-extract: last match of (-?[$0-9.,]{2,})|(-?[0-9]+)",
            "digit_flexible": "same rule, matches without a digit (e.g. '$$') are skipped",
            "boxed_first": "last numeric \\boxed{} if present (numeric compare), else digit_flexible",
            "no_digit_extraction": "stock extraction returned a string without any digit",
            "gsm8k_digit_flexible (paired column)": "per-example correctness under digit_flexible",
            "qualitative_math_intrusion": "responses (of 24 per condition) containing math-training artefacts or math "
                                          "vocabulary; see task1_llm_ft/qualitative_flags.py",
        },
        "gsm8k_extraction": gsm,
        "gsm8k_gain_vs_baseline_pp": gain,
        "boolq_bias": boolq,
        "qualitative_math_intrusion": qualitative,
        "paired": paired,
    }
    os.makedirs(res_dir, exist_ok=True)
    out_path = os.path.join(res_dir, "offline_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved %s", out_path)
    return result
