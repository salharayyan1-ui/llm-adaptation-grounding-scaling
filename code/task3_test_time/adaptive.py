"""Adaptive Inference module for Task 3.

Implements Step 5:
- Candidate checkpoints n in {4, 8, 16}, fallback at 32.
- Both controllers return current self-consistency answer.
- Agreement-based policy:
    Stop if unique non-NO_ANSWER leader has >= tau vote share of n candidates.
- Quality-aware policy:
    Agreement condition + SC and RWC winners agree + RWC margin >= gamma.
    RWC margin = (W(a_1) - W(a_2)) / sum_a W(a).
    Recomputed on revealed candidates only (self-certainty for MATH, VisConf for MathVista).
- Development search over tau in {0.5, 0.625, 0.75, 0.875, 1.0} and gamma in {0, 0.1, 0.25, 0.5}.
    Lowest token cost within 1% dev accuracy of fixed-32 SC.
- Freeze policies and evaluate on 800 held-out evaluation questions.
- Detailed stopping distributions, token costs, and early-stop error analysis.
- Precomputed checkpoint representation for high-speed simulation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from task3_test_time.data_loader import QuestionRecord, RolloutRecord
from task3_test_time.equivalence import grade_answer, group_answers, are_answers_equivalent, NO_ANSWER
from task3_test_time.scoring import (
    compute_self_certainty,
    compute_visconf_scores,
    select_self_consistency,
    select_rank_weighted_consensus,
)
from common.metrics import bootstrap_pair_metric_ci

CHECKPOINTS = (4, 8, 16)
TAU_GRID = (0.5, 0.625, 0.75, 0.875, 1.0)
GAMMA_GRID = (0.0, 0.1, 0.25, 0.5)


@dataclass
class StoppingDecision:
    stopped_at: int  # 4, 8, 16, or 32
    chosen_answer: str
    tokens_consumed: int
    is_correct: bool
    policy_name: str


@dataclass
class CheckpointInfo:
    n: int
    tokens: int
    sc_answer: str
    is_correct: bool
    is_unique_non_no_answer: bool
    vote_share: float
    sc_rwc_agree: bool
    rwc_margin: float


def precompute_order_checkpoints(
    rollouts_order: Sequence[RolloutRecord],
    dataset: str,
    gold_answer: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[List[CheckpointInfo], StoppingDecision]:
    """Precompute all checkpoint features for a single rollout permutation order.
    
    Allows instantaneous evaluation of all tau and gamma parameter configurations.
    """
    checkpoints: List[CheckpointInfo] = []
    is_vista = "vista" in dataset.lower()

    for n in CHECKPOINTS:
        revealed = rollouts_order[:n]
        tokens = sum(c.T for c in revealed)
        answers = [c.extracted_answer for c in revealed]
        cand_labels, groups = group_answers(answers, dataset=dataset, metadata=metadata)

        # Leader vote share and uniqueness
        sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
        is_unique_leader = False
        vote_share = 0.0
        if sorted_groups:
            lead_grp, lead_idxs = sorted_groups[0]
            if lead_grp != NO_ANSWER:
                if len(sorted_groups) == 1 or len(lead_idxs) > len(sorted_groups[1][1]):
                    is_unique_leader = True
                    vote_share = len(lead_idxs) / float(n)

        # SC winner
        _, sc_ans = select_self_consistency(revealed, dataset, metadata)
        is_corr = grade_answer(sc_ans, gold_answer, dataset, metadata)

        # RWC winner & margin
        if is_vista:
            _, _, z_scores = compute_visconf_scores(revealed)
            scores = z_scores
        else:
            scores = [compute_self_certainty(c) for c in revealed]

        _, rwc_ans, group_weights = select_rank_weighted_consensus(
            revealed, scores, dataset, metadata
        )
        sc_rwc_agree = are_answers_equivalent(sc_ans, rwc_ans, dataset=dataset, metadata=metadata)

        sorted_w = sorted(group_weights.values(), reverse=True)
        tot_w = sum(sorted_w)
        if tot_w > 0:
            w1 = sorted_w[0]
            w2 = sorted_w[1] if len(sorted_w) > 1 else 0.0
            margin = (w1 - w2) / tot_w
        else:
            margin = 0.0

        checkpoints.append(CheckpointInfo(
            n=n,
            tokens=tokens,
            sc_answer=sc_ans,
            is_correct=is_corr,
            is_unique_non_no_answer=is_unique_leader,
            vote_share=vote_share,
            sc_rwc_agree=sc_rwc_agree,
            rwc_margin=margin,
        ))

    # Fallback at 32
    revealed_32 = rollouts_order[:32]
    _, sc_ans32 = select_self_consistency(revealed_32, dataset, metadata)
    tokens_32 = sum(c.T for c in revealed_32)
    is_corr32 = grade_answer(sc_ans32, gold_answer, dataset, metadata)
    fallback_32 = StoppingDecision(
        stopped_at=32,
        chosen_answer=sc_ans32,
        tokens_consumed=tokens_32,
        is_correct=is_corr32,
        policy_name="fixed_32",
    )

    return checkpoints, fallback_32


def evaluate_order_fast(
    checkpoints: List[CheckpointInfo],
    fallback_32: StoppingDecision,
    policy_type: str,
    tau: float,
    gamma: float = 0.0,
) -> StoppingDecision:
    """Evaluate stopping rule instantly against precomputed checkpoints."""
    for chk in checkpoints:
        if policy_type == "agreement":
            if chk.is_unique_non_no_answer and chk.vote_share >= tau:
                return StoppingDecision(
                    stopped_at=chk.n,
                    chosen_answer=chk.sc_answer,
                    tokens_consumed=chk.tokens,
                    is_correct=chk.is_correct,
                    policy_name=f"agreement(tau={tau})",
                )
        elif policy_type == "quality":
            if (
                chk.is_unique_non_no_answer
                and chk.vote_share >= tau
                and chk.sc_rwc_agree
                and chk.rwc_margin >= gamma
            ):
                return StoppingDecision(
                    stopped_at=chk.n,
                    chosen_answer=chk.sc_answer,
                    tokens_consumed=chk.tokens,
                    is_correct=chk.is_correct,
                    policy_name=f"quality(tau={tau},gamma={gamma})",
                )
    return fallback_32


def check_agreement_condition(
    candidates: Sequence[RolloutRecord],
    tau: float,
    dataset: str = "math",
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """Check if there is a unique, non-NO_ANSWER leader with >= tau vote share."""
    n = len(candidates)
    answers = [c.extracted_answer for c in candidates]
    _, groups = group_answers(answers, dataset=dataset, metadata=metadata)

    sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    if not sorted_groups:
        return False, None

    leader_grp, leader_idxs = sorted_groups[0]
    leader_votes = len(leader_idxs)

    if leader_grp == NO_ANSWER:
        return False, None

    if len(sorted_groups) > 1:
        runner_up_votes = len(sorted_groups[1][1])
        if leader_votes == runner_up_votes:
            return False, None

    vote_share = leader_votes / float(n)
    if vote_share >= tau:
        return True, leader_grp

    return False, None


def check_quality_aware_condition(
    candidates: Sequence[RolloutRecord],
    tau: float,
    gamma: float,
    dataset: str = "math",
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """Check agreement condition + SC & RWC winner agreement + RWC margin >= gamma."""
    agree_ok, leader_grp = check_agreement_condition(candidates, tau, dataset, metadata)
    if not agree_ok or leader_grp is None:
        return False, None

    _, sc_ans = select_self_consistency(candidates, dataset, metadata)

    is_vista = "vista" in dataset.lower()
    if is_vista:
        _, _, z_scores = compute_visconf_scores(candidates)
        scores = z_scores
    else:
        scores = [compute_self_certainty(c) for c in candidates]

    _, rwc_ans, group_weights = select_rank_weighted_consensus(
        candidates, scores, dataset, metadata
    )

    if not are_answers_equivalent(sc_ans, rwc_ans, dataset=dataset, metadata=metadata):
        return False, None

    sorted_weights = sorted(group_weights.values(), reverse=True)
    total_weight = sum(sorted_weights)
    if total_weight <= 0:
        return False, None

    w1 = sorted_weights[0]
    w2 = sorted_weights[1] if len(sorted_weights) > 1 else 0.0
    rwc_margin = (w1 - w2) / total_weight

    if rwc_margin >= gamma:
        return True, sc_ans

    return False, None


def simulate_adaptive_stopping(
    rollouts_order: Sequence[RolloutRecord],
    policy_type: str,
    tau: float,
    gamma: float = 0.0,
    dataset: str = "math",
    gold_answer: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> StoppingDecision:
    """Simulate adaptive stopping along a single rollout permutation order."""
    chks, fb = precompute_order_checkpoints(rollouts_order, dataset, gold_answer, metadata)
    return evaluate_order_fast(chks, fb, policy_type, tau, gamma)


def search_development_policies(
    questions: Dict[str, QuestionRecord],
    dev_qids: Sequence[str],
    permutations: Dict[str, List[List[int]]],
) -> Dict[str, Any]:
    """Search grid for agreement and quality-aware policies on development split.
    
    Criterion: lowest mean token cost with dev accuracy >= Acc(fixed-32 SC) - 0.01.
    Ties broken by: higher accuracy, smaller tau, smaller gamma.
    """
    first_q = questions[dev_qids[0]]
    dataset = first_q.dataset

    # Precompute all development sequences
    dev_sequences = []
    sc32_accs = []
    sc32_tokens = []

    for qid in dev_qids:
        q = questions[qid]
        for p in permutations[qid]:
            order = [q.rollouts[idx] for idx in p]
            chks, fb32 = precompute_order_checkpoints(order, dataset, q.gold_answer, q.metadata)
            dev_sequences.append((chks, fb32))
            sc32_accs.append(1.0 if fb32.is_correct else 0.0)
            sc32_tokens.append(fb32.tokens_consumed)

    baseline_acc = float(np.mean(sc32_accs))
    baseline_tokens = float(np.mean(sc32_tokens))
    target_acc = baseline_acc - 0.01

    # 1. Search agreement-based policies
    agreement_grid_results = []
    for tau in TAU_GRID:
        accs = []
        toks = []
        for chks, fb32 in dev_sequences:
            dec = evaluate_order_fast(chks, fb32, policy_type="agreement", tau=tau)
            accs.append(1.0 if dec.is_correct else 0.0)
            toks.append(dec.tokens_consumed)

        mean_acc = float(np.mean(accs))
        mean_tok = float(np.mean(toks))
        agreement_grid_results.append({
            "tau": tau,
            "mean_accuracy": mean_acc,
            "mean_tokens": mean_tok,
            "meets_target": mean_acc >= target_acc,
        })

    valid_agree = [r for r in agreement_grid_results if r["meets_target"]]
    if not valid_agree:
        valid_agree = agreement_grid_results

    best_agree = sorted(
        valid_agree,
        key=lambda r: (r["mean_tokens"], -r["mean_accuracy"], r["tau"]),
    )[0]

    # 2. Search quality-aware policies
    quality_grid_results = []
    for tau in TAU_GRID:
        for gamma in GAMMA_GRID:
            accs = []
            toks = []
            for chks, fb32 in dev_sequences:
                dec = evaluate_order_fast(chks, fb32, policy_type="quality", tau=tau, gamma=gamma)
                accs.append(1.0 if dec.is_correct else 0.0)
                toks.append(dec.tokens_consumed)

            mean_acc = float(np.mean(accs))
            mean_tok = float(np.mean(toks))
            quality_grid_results.append({
                "tau": tau,
                "gamma": gamma,
                "mean_accuracy": mean_acc,
                "mean_tokens": mean_tok,
                "meets_target": mean_acc >= target_acc,
            })

    valid_quality = [r for r in quality_grid_results if r["meets_target"]]
    if not valid_quality:
        valid_quality = quality_grid_results

    best_quality = sorted(
        valid_quality,
        key=lambda r: (r["mean_tokens"], -r["mean_accuracy"], r["tau"], r["gamma"]),
    )[0]

    return {
        "baseline_sc32": {
            "mean_accuracy": baseline_acc,
            "mean_tokens": baseline_tokens,
            "target_accuracy": target_acc,
        },
        "agreement": {
            "best_params": best_agree,
            "grid": agreement_grid_results,
        },
        "quality_aware": {
            "best_params": best_quality,
            "grid": quality_grid_results,
        },
    }


def evaluate_held_out_policy(
    questions: Dict[str, QuestionRecord],
    eval_qids: Sequence[str],
    permutations: Dict[str, List[List[int]]],
    policy_type: str,
    tau: float,
    gamma: float = 0.0,
    seed: int = 6304,
) -> Dict[str, Any]:
    """Evaluate frozen adaptive policy on the held-out evaluation questions."""
    first_q = questions[eval_qids[0]]
    dataset = first_q.dataset

    per_question_accs = []
    per_question_tokens = []
    per_question_rollouts = []
    stop_counts = {4: 0, 8: 0, 16: 0, 32: 0}

    incorrect_early_stops = 0
    sc32_recoverable = 0

    for qid in eval_qids:
        q = questions[qid]
        q_accs = []
        q_toks = []
        q_rolls = []

        for p in permutations[qid]:
            order = [q.rollouts[idx] for idx in p]
            chks, fb32 = precompute_order_checkpoints(order, dataset, q.gold_answer, q.metadata)
            dec = evaluate_order_fast(chks, fb32, policy_type=policy_type, tau=tau, gamma=gamma)
            
            q_accs.append(1.0 if dec.is_correct else 0.0)
            q_toks.append(dec.tokens_consumed)
            q_rolls.append(dec.stopped_at)
            stop_counts[dec.stopped_at] += 1

            if dec.stopped_at < 32 and not dec.is_correct:
                incorrect_early_stops += 1
                if fb32.is_correct:
                    sc32_recoverable += 1

        per_question_accs.append(float(np.mean(q_accs)))
        per_question_tokens.append(float(np.mean(q_toks)))
        per_question_rollouts.append(float(np.mean(q_rolls)))

    acc_mean, acc_low, acc_high = bootstrap_pair_metric_ci(
        per_question_accs, n_resamples=1000, seed=seed
    )

    total_decisions = sum(stop_counts.values())
    stop_dist = {n: float(cnt / total_decisions) for n, cnt in stop_counts.items()}

    return {
        "policy": f"{policy_type}(tau={tau},gamma={gamma})",
        "accuracy": {"mean": acc_mean, "ci_lower": acc_low, "ci_upper": acc_high},
        "mean_tokens": float(np.mean(per_question_tokens)),
        "mean_rollouts": float(np.mean(per_question_rollouts)),
        "stopping_distribution": stop_dist,
        "stopping_counts": stop_counts,
        "early_stop_analysis": {
            "total_early_incorrect_stops": incorrect_early_stops,
            "recovered_by_fixed32_count": sc32_recoverable,
            "recovered_by_fixed32_rate": float(sc32_recoverable / max(incorrect_early_stops, 1)),
        },
    }
