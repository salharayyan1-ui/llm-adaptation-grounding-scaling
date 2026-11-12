"""Signal Analysis module for Task 3.

Implements Step 3:
- At N=32: length vs score vs correctness within questions.
- Top vs bottom quartile correctness differences within questions.
- Shortest-rollout baseline as length-only control.
- Confidence-vision correlation within each rollout for MathVista.
- Tokenwise fusion ablation:
    Z_token = (1 / T) sum_t c_t v_t vs Z = V * C.
    Covariance identity: Z_token - Z = Cov_t(c_t, v_t).
    Evaluation under highest-score and RWC aggregations.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from task3_test_time.data_loader import QuestionRecord, RolloutRecord
from task3_test_time.equivalence import grade_answer
from task3_test_time.scoring import (
    compute_mean_logp,
    compute_self_certainty,
    compute_visual_engagement,
    compute_visconf_scores,
    select_highest_score,
    select_rank_weighted_consensus,
)
from common.metrics import bootstrap_pair_metric_ci


def compute_rollout_metrics(
    rollouts: Sequence[RolloutRecord],
    dataset: str = "math",
) -> Dict[str, np.ndarray]:
    """Compute all rollout-level scores for a set of candidates (e.g. N=32)."""
    N = len(rollouts)
    is_vista = "vista" in dataset.lower()
    
    lengths = np.array([r.T for r in rollouts], dtype=np.float64)
    logp = np.array([compute_mean_logp(r) for r in rollouts], dtype=np.float64)
    self_cert = np.array([compute_self_certainty(r) for r in rollouts], dtype=np.float64)

    scores = {
        "length": lengths,
        "logp": logp,
        "self_certainty": self_cert,
    }

    if is_vista:
        v_scores, c_scores, z_scores = compute_visconf_scores(rollouts)
        scores["visual_engagement"] = np.array(v_scores, dtype=np.float64)
        scores["calibrated_confidence"] = np.array(c_scores, dtype=np.float64)
        scores["visconf"] = np.array(z_scores, dtype=np.float64)

    return scores


def evaluate_quartile_differences(
    question: QuestionRecord,
) -> Dict[str, Dict[str, float]]:
    """Compute correctness in top vs bottom quartiles (8 rollouts each out of 32) within question.
    
    Returns dict mapping score_name -> {'top': float, 'bottom': float, 'diff': float}.
    """
    rollouts = question.rollouts
    N = len(rollouts)
    assert N >= 8, f"Need at least 8 rollouts for quartiles, found {N}"
    k = N // 4  # 8 for N=32

    correctness = np.array([
        1.0 if grade_answer(r.extracted_answer, question.gold_answer, question.dataset, question.metadata) else 0.0
        for r in rollouts
    ], dtype=np.float64)

    scores = compute_rollout_metrics(rollouts, question.dataset)
    results: Dict[str, Dict[str, float]] = {}

    for score_name, vals in scores.items():
        # Quartile ranking:
        # Note: for length, smaller is often shorter; we check highest vs lowest
        sorted_order = np.argsort(vals)  # ascending
        bottom_idxs = sorted_order[:k]
        top_idxs = sorted_order[-k:]

        top_corr = float(np.mean(correctness[top_idxs]))
        bottom_corr = float(np.mean(correctness[bottom_idxs]))
        results[score_name] = {
            "top": top_corr,
            "bottom": bottom_corr,
            "diff": top_corr - bottom_corr,
        }

    return results


def evaluate_shortest_rollout_baseline(question: QuestionRecord) -> bool:
    """Select candidate with shortest rollout length (breaking ties by rollout order)."""
    best_idx = 0
    min_len = question.rollouts[0].T
    for i in range(1, len(question.rollouts)):
        if question.rollouts[i].T < min_len:
            min_len = question.rollouts[i].T
            best_idx = i
    chosen = question.rollouts[best_idx].extracted_answer
    return grade_answer(chosen, question.gold_answer, question.dataset, question.metadata)


def compute_confidence_vision_correlation(
    rollout: RolloutRecord,
    q05: Optional[float] = None,
    q95: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Compute confidence-vision correlation, tokenwise fusion, and covariance identity.
    
    Returns:
        (pearson_r, z_token, covariance)
    """
    if rollout.att_image is None or len(rollout.att_image) == 0:
        return 0.0, 0.0, 0.0

    T = rollout.T
    img = np.asarray(rollout.att_image, dtype=np.float64)
    pmt = np.asarray(rollout.att_prompt, dtype=np.float64) if rollout.att_prompt is not None else np.zeros_like(img)
    gen = np.asarray(rollout.att_generated, dtype=np.float64) if rollout.att_generated is not None else np.zeros_like(img)
    denom = img + pmt + gen
    v_t = np.where(denom > 0, img / np.where(denom > 0, denom, 1.0), 0.0)

    k_t = np.asarray(rollout.kl_u_p, dtype=np.float64)
    if q05 is None or q95 is None:
        q05 = float(np.percentile(k_t, 5, method="linear"))
        q95 = float(np.percentile(k_t, 95, method="linear"))
    scale = max(q95 - q05, 1e-8)
    c_t = np.clip((k_t - q05) / scale, 0.0, 1.0)

    # Pearson correlation
    std_c = np.std(c_t)
    std_v = np.std(v_t)
    if std_c > 1e-8 and std_v > 1e-8:
        cov_matrix = np.cov(c_t, v_t, ddof=0)
        cov_val = float(cov_matrix[0, 1])
        corr = float(cov_val / (std_c * std_v))
    else:
        cov_val = 0.0
        corr = 0.0

    z_token = float(np.mean(c_t * v_t))
    z_rollout = float(np.mean(c_t)) * float(np.mean(v_t))
    # Identity: z_token - z_rollout = cov_val
    return corr, z_token, cov_val


def evaluate_tokenwise_fusion_ablation(
    candidates: Sequence[RolloutRecord],
    dataset: str = "mathvista",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Compute tokenwise fusion scores Z_token and compare with trajectory fusion Z.
    
    Returns answers selected under:
        'visconf_score', 'visconf_rwc', 'visconf_token_score', 'visconf_token_rwc'
    """
    N = len(candidates)
    if N == 0:
        return {}

    # Pool KL for candidates to get q05, q95
    all_kl: List[float] = []
    for c in candidates:
        all_kl.extend(c.kl_u_p)
    kl_arr = np.asarray(all_kl, dtype=np.float64)
    q05 = float(np.percentile(kl_arr, 5, method="linear"))
    q95 = float(np.percentile(kl_arr, 95, method="linear"))

    z_scores: List[float] = []
    z_token_scores: List[float] = []

    for c in candidates:
        _, z_tok, _ = compute_confidence_vision_correlation(c, q05=q05, q95=q95)
        V_i = compute_visual_engagement(c)
        k_t = np.asarray(c.kl_u_p, dtype=np.float64)
        c_t = np.clip((k_t - q05) / max(q95 - q05, 1e-8), 0.0, 1.0)
        C_i = float(np.mean(c_t))
        Z_i = V_i * C_i

        z_scores.append(Z_i)
        z_token_scores.append(z_tok)

    chosen: Dict[str, str] = {}
    _, chosen["visconf_score"] = select_highest_score(candidates, z_scores)
    _, chosen["visconf_rwc"], _ = select_rank_weighted_consensus(candidates, z_scores, dataset, metadata)
    _, chosen["visconf_token_score"] = select_highest_score(candidates, z_token_scores)
    _, chosen["visconf_token_rwc"], _ = select_rank_weighted_consensus(candidates, z_token_scores, dataset, metadata)

    return chosen


def run_signals_analysis(
    questions: Dict[str, QuestionRecord],
    question_ids: Sequence[str],
    seed: int = 6304,
) -> Dict[str, Any]:
    """Run full signal analysis across specified question IDs."""
    is_vista = "vista" in questions[question_ids[0]].dataset.lower()

    # 1. Quartile differences
    quartile_diffs: Dict[str, List[float]] = {}
    shortest_accs: List[float] = []

    for qid in question_ids:
        q = questions[qid]
        q_res = evaluate_quartile_differences(q)
        for s_name, data in q_res.items():
            if s_name not in quartile_diffs:
                quartile_diffs[s_name] = []
            quartile_diffs[s_name].append(data["diff"])

        shortest_correct = evaluate_shortest_rollout_baseline(q)
        shortest_accs.append(1.0 if shortest_correct else 0.0)

    quartile_summary = {}
    for s_name, diffs in quartile_diffs.items():
        m, low, high = bootstrap_pair_metric_ci(diffs, n_resamples=1000, seed=seed)
        quartile_summary[s_name] = {"mean_diff": m, "ci_lower": low, "ci_upper": high}

    shortest_mean, s_low, s_high = bootstrap_pair_metric_ci(shortest_accs, n_resamples=1000, seed=seed)
    control_summary = {"shortest_baseline_acc": shortest_mean, "ci_lower": s_low, "ci_upper": s_high}

    # 2. MathVista correlations and fusion ablation
    correlations_summary = {}
    ablation_summary = {}

    if is_vista:
        all_corrs: List[float] = []
        cov_diff_errors: List[float] = []
        token_acc: Dict[str, List[float]] = {
            "visconf_score": [],
            "visconf_rwc": [],
            "visconf_token_score": [],
            "visconf_token_rwc": [],
        }

        for qid in question_ids:
            q = questions[qid]
            # Pool all kl for question to calibrate
            all_kl = []
            for r in q.rollouts:
                all_kl.extend(r.kl_u_p)
            kl_arr = np.asarray(all_kl, dtype=np.float64)
            q05 = float(np.percentile(kl_arr, 5, method="linear"))
            q95 = float(np.percentile(kl_arr, 95, method="linear"))

            for r in q.rollouts:
                corr, z_tok, cov_val = compute_confidence_vision_correlation(r, q05, q95)
                all_corrs.append(corr)
                
                # Check covariance identity: z_tok - z_rollout == cov_val
                V_i = compute_visual_engagement(r)
                k_t = np.asarray(r.kl_u_p, dtype=np.float64)
                c_t = np.clip((k_t - q05) / max(q95 - q05, 1e-8), 0.0, 1.0)
                z_roll = V_i * float(np.mean(c_t))
                cov_diff_errors.append(abs((z_tok - z_roll) - cov_val))

            # Full N=32 ablation
            chosen = evaluate_tokenwise_fusion_ablation(q.rollouts, q.dataset, q.metadata)
            for k_m, ans in chosen.items():
                c_flag = grade_answer(ans, q.gold_answer, q.dataset, q.metadata)
                token_acc[k_m].append(1.0 if c_flag else 0.0)

        correlations_summary = {
            "mean_correlation": float(np.mean(all_corrs)),
            "std_correlation": float(np.std(all_corrs)),
            "max_covariance_identity_error": float(np.max(cov_diff_errors)),
        }

        for k_m, vals in token_acc.items():
            m, low, high = bootstrap_pair_metric_ci(vals, n_resamples=1000, seed=seed)
            ablation_summary[k_m] = {"mean": m, "ci_lower": low, "ci_upper": high}

    return {
        "quartile_summary": quartile_summary,
        "control_summary": control_summary,
        "correlations_summary": correlations_summary,
        "ablation_summary": ablation_summary,
    }
