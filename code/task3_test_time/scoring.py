"""Scoring and Consensus Aggregations module for Task 3.

Implements Step 2:
- Mean log-probability: L_i = (1 / T_i) sum_t log p_{i,t}.
- Self-certainty: K_i = (1 / T_i) sum_t k_{i,t}, where k_{i,t} = D_KL(U || p_{i,t}).
- Visual engagement (MathVista):
    v_{i,t} = A^image / (A^image + A^prompt + A^generated)
    V_i = (1 / T_i) sum_t v_{i,t}.
- VisConf:
    c_{i,t} = clip_{[0, 1]}((k_{i,t} - Q_{.05}(G)) / max(Q_{.95}(G) - Q_{.05}(G), 10^-8))
    C_i = (1 / T_i) sum_t c_{i,t}
    Z_i = V_i * C_i
    (Recomputed per prefix G with linear interpolation percentiles).
- Rank-Weighted Consensus (RWC):
    w_i = (N - r_i) / (N - 1), W(a) = sum_{i: a_i = a} w_i, a_hat = argmax_a W(a).
    Average ranks for tied scores. At N=1 returns single candidate.
    Ties broken by earliest supporting rollout in current order.
- Selectors evaluated across budgets N in {1, 2, 4, 8, 16, 32}.
- 1,000 paired question bootstrap resamples for 95% CIs.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from task3_test_time.data_loader import QuestionRecord, RolloutRecord
from task3_test_time.equivalence import grade_answer, group_answers, NO_ANSWER
from common.metrics import bootstrap_pair_metric_ci

BUDGETS = (1, 2, 4, 8, 16, 32)


def compute_mean_logp(rollout: RolloutRecord) -> float:
    """Compute mean log-probability L_i = (1 / T_i) sum_t log p_{i,t}."""
    if len(rollout.logp) == 0:
        return 0.0
    return float(np.mean(rollout.logp))


def compute_self_certainty(rollout: RolloutRecord) -> float:
    """Compute self-certainty K_i = (1 / T_i) sum_t k_{i,t}."""
    if len(rollout.kl_u_p) == 0:
        return 0.0
    return float(np.mean(rollout.kl_u_p))


def compute_visual_engagement(rollout: RolloutRecord) -> float:
    """Compute visual engagement V_i = (1 / T_i) sum_t v_{i,t}."""
    if rollout.att_image is None or len(rollout.att_image) == 0:
        return 0.0
    
    img = np.asarray(rollout.att_image, dtype=np.float64)
    pmt = np.asarray(rollout.att_prompt, dtype=np.float64) if rollout.att_prompt is not None else np.zeros_like(img)
    gen = np.asarray(rollout.att_generated, dtype=np.float64) if rollout.att_generated is not None else np.zeros_like(img)
    
    denom = img + pmt + gen
    safe_denom = np.where(denom > 0, denom, 1.0)
    v_t = np.where(denom > 0, img / safe_denom, 0.0)
    return float(np.mean(v_t))


def compute_visconf_scores(
    candidates: Sequence[RolloutRecord],
) -> Tuple[List[float], List[float], List[float]]:
    """Compute VisConf scores C_i, V_i, and Z_i = V_i * C_i for candidate prefix G.
    
    Recomputed strictly for candidate set G using linear-interpolation percentiles.
    
    Returns:
        (V_scores, C_scores, Z_scores)
    """
    N = len(candidates)
    if N == 0:
        return [], [], []

    # Pool token KL values across prefix G
    all_kl: List[float] = []
    for c in candidates:
        all_kl.extend(c.kl_u_p)

    if len(all_kl) == 0:
        return [0.0] * N, [0.0] * N, [0.0] * N

    kl_arr = np.asarray(all_kl, dtype=np.float64)
    q05 = float(np.percentile(kl_arr, 5, method="linear"))
    q95 = float(np.percentile(kl_arr, 95, method="linear"))
    scale = max(q95 - q05, 1e-8)

    v_scores = []
    c_scores = []
    z_scores = []

    for c in candidates:
        V_i = compute_visual_engagement(c)
        if len(c.kl_u_p) > 0:
            k_i = np.asarray(c.kl_u_p, dtype=np.float64)
            c_t = np.clip((k_i - q05) / scale, 0.0, 1.0)
            C_i = float(np.mean(c_t))
        else:
            C_i = 0.0

        Z_i = V_i * C_i
        v_scores.append(V_i)
        c_scores.append(C_i)
        z_scores.append(Z_i)

    return v_scores, c_scores, z_scores


def compute_average_ranks(scores: Sequence[float]) -> np.ndarray:
    """Compute 1-based descending average ranks for scores (rank 1 is highest score)."""
    scores_arr = np.asarray(scores, dtype=np.float64)
    N = len(scores_arr)
    if N == 0:
        return np.array([], dtype=np.float64)
    if N == 1:
        return np.array([1.0], dtype=np.float64)

    # Sort descending
    order = np.argsort(-scores_arr, kind="mergesort")
    sorted_scores = scores_arr[order]
    
    ranks = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # Indices i to j-1 are tied. Their 1-based ranks are (i+1) to j.
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
        
    return ranks


def select_highest_score(
    candidates: Sequence[RolloutRecord],
    scores: Sequence[float],
) -> Tuple[int, str]:
    """Select candidate with highest score; tie-broken by earliest candidate in order.
    
    Returns:
        (selected_index, selected_extracted_answer)
    """
    best_idx = 0
    best_score = scores[0]
    for idx in range(1, len(candidates)):
        if scores[idx] > best_score:
            best_score = scores[idx]
            best_idx = idx
    return best_idx, candidates[best_idx].extracted_answer


def select_self_consistency(
    candidates: Sequence[RolloutRecord],
    dataset: str = "math",
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[int, str]:
    """Self-consistency selection: 1 vote per rollout, ties broken by earliest supporting rollout.
    
    Returns:
        (representative_rollout_idx, selected_answer)
    """
    N = len(candidates)
    if N == 1:
        return 0, candidates[0].extracted_answer

    answers = [c.extracted_answer for c in candidates]
    cand_labels, groups = group_answers(answers, dataset=dataset, metadata=metadata)

    # Count votes per canonical group
    group_votes = {grp: len(idxs) for grp, idxs in groups.items()}
    max_votes = max(group_votes.values())

    # Find earliest candidate whose group has max_votes
    for idx, grp in enumerate(cand_labels):
        if group_votes[grp] == max_votes:
            return idx, candidates[idx].extracted_answer

    return 0, candidates[0].extracted_answer


def select_rank_weighted_consensus(
    candidates: Sequence[RolloutRecord],
    scores: Sequence[float],
    dataset: str = "math",
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[int, str, Dict[str, float]]:
    """Rank-Weighted Consensus (RWC):
    w_i = (N - r_i) / (N - 1), W(a) = sum_{i: a_i = a} w_i.
    
    At N=1, returns the single candidate.
    Ties broken by earliest supporting rollout in current order.
    
    Returns:
        (best_cand_idx, best_answer, group_weights)
    """
    N = len(candidates)
    if N == 1:
        return 0, candidates[0].extracted_answer, {candidates[0].extracted_answer: 1.0}

    ranks = compute_average_ranks(scores)
    weights = (N - ranks) / (N - 1.0)

    answers = [c.extracted_answer for c in candidates]
    cand_labels, groups = group_answers(answers, dataset=dataset, metadata=metadata)

    group_weights: Dict[str, float] = {}
    for grp, idxs in groups.items():
        group_weights[grp] = float(sum(weights[i] for i in idxs))

    max_weight = max(group_weights.values())

    # Earliest supporting rollout in current order
    for idx, grp in enumerate(cand_labels):
        if abs(group_weights[grp] - max_weight) < 1e-9:
            return idx, candidates[idx].extracted_answer, group_weights

    return 0, candidates[0].extracted_answer, group_weights


def evaluate_prefix_methods(
    candidates: Sequence[RolloutRecord],
    dataset: str = "math",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Run all relevant selectors on candidate prefix and return their chosen answer."""
    N = len(candidates)
    is_vista = "vista" in dataset.lower()

    # Precompute candidate scores
    logp_scores = [compute_mean_logp(c) for c in candidates]
    kl_scores = [compute_self_certainty(c) for c in candidates]

    chosen: Dict[str, str] = {}
    
    # 1. FirstAns
    chosen["first_ans"] = candidates[0].extracted_answer

    # 2. Self-Consistency
    _, chosen["self_consistency"] = select_self_consistency(candidates, dataset, metadata)

    # 3. Highest mean log-probability
    _, chosen["highest_logp"] = select_highest_score(candidates, logp_scores)

    # 4. Highest self-certainty
    _, chosen["highest_self_certainty"] = select_highest_score(candidates, kl_scores)

    # 5. Self-certainty + RWC
    _, chosen["self_certainty_rwc"], _ = select_rank_weighted_consensus(
        candidates, kl_scores, dataset, metadata
    )

    if is_vista:
        v_scores, _, z_scores = compute_visconf_scores(candidates)
        # Vision-only highest score
        _, chosen["highest_vision"] = select_highest_score(candidates, v_scores)
        # Vision-only + RWC
        _, chosen["vision_rwc"], _ = select_rank_weighted_consensus(
            candidates, v_scores, dataset, metadata
        )
        # VisConf highest score
        _, chosen["highest_visconf"] = select_highest_score(candidates, z_scores)
        # VisConf + RWC
        _, chosen["visconf_rwc"], _ = select_rank_weighted_consensus(
            candidates, z_scores, dataset, metadata
        )

    return chosen


def evaluate_question_scoring(
    question: QuestionRecord,
    permutations: List[List[int]],
    budgets: Sequence[int] = BUDGETS,
) -> Dict[int, Dict[str, float]]:
    """Evaluate all scoring and consensus aggregations for a single question."""
    results: Dict[int, Dict[str, float]] = {}
    is_vista = "vista" in question.dataset.lower()

    method_names = [
        "first_ans",
        "self_consistency",
        "highest_logp",
        "highest_self_certainty",
        "self_certainty_rwc",
    ]
    if is_vista:
        method_names.extend([
            "highest_vision",
            "vision_rwc",
            "highest_visconf",
            "visconf_rwc",
        ])

    for N in budgets:
        method_correct: Dict[str, List[float]] = {m: [] for m in method_names}
        for p in permutations:
            prefix = [question.rollouts[idx] for idx in p[:N]]
            answers = evaluate_prefix_methods(prefix, question.dataset, question.metadata)
            for m in method_names:
                corr = grade_answer(answers[m], question.gold_answer, question.dataset, question.metadata)
                method_correct[m].append(1.0 if corr else 0.0)

        results[N] = {m: float(np.mean(method_correct[m])) for m in method_names}

    return results


def evaluate_dataset_scoring(
    questions: Dict[str, QuestionRecord],
    question_ids: Sequence[str],
    permutations: Dict[str, List[List[int]]],
    budgets: Sequence[int] = BUDGETS,
    seed: int = 6304,
) -> Dict[str, Any]:
    """Evaluate scoring and consensus methods across questions with bootstrap CIs."""
    per_question: Dict[str, Dict[int, Dict[str, float]]] = {}
    for qid in question_ids:
        q = questions[qid]
        per_question[qid] = evaluate_question_scoring(q, permutations[qid], budgets=budgets)

    first_qid = question_ids[0]
    sample_methods = list(per_question[first_qid][budgets[0]].keys())

    summary: Dict[int, Dict[str, Any]] = {}
    for N in budgets:
        summary[N] = {}
        for m in sample_methods:
            vals = [per_question[qid][N][m] for qid in question_ids]
            mean_val, low_ci, high_ci = bootstrap_pair_metric_ci(vals, n_resamples=1000, seed=seed)
            summary[N][m] = {
                "mean": mean_val,
                "ci_lower": low_ci,
                "ci_upper": high_ci,
            }

    return {
        "summary": summary,
        "per_question": per_question,
    }
