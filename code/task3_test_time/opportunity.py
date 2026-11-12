"""Opportunity from Repeated Sampling module for Task 3.

Implements Step 1:
- Candidate budgets N in {1, 2, 4, 8, 16, 32}.
- 10 seeded random permutations of the 32 rollout IDs per question.
- Nested prefixes of each permutation.
- FirstAns accuracy, oracle coverage, distinct answer groups count.
- Failure decomposition:
    * sampling failure: no correct candidate was generated in candidate set G.
    * selection failure: correct candidate was available in G but selector picked incorrect.
- Paired question-level bootstrap confidence intervals.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from task3_test_time.data_loader import QuestionRecord, RolloutRecord
from task3_test_time.equivalence import grade_answer, group_answers, NO_ANSWER
from common.metrics import bootstrap_pair_metric_ci

BUDGETS = (1, 2, 4, 8, 16, 32)


def generate_rollout_permutations(
    question_ids: Sequence[str],
    num_permutations: int = 10,
    num_rollouts: int = 32,
    seed: int = 6304,
) -> Dict[str, List[List[int]]]:
    """Generate 10 seeded random permutations of 32 rollout IDs per question.
    
    Reused across opportunity, scoring, and adaptive stopping.
    """
    rng = np.random.default_rng(seed)
    perms_by_qid: Dict[str, List[List[int]]] = {}
    
    # Sort question IDs for deterministic assignment
    sorted_qids = sorted(list(question_ids))
    for qid in sorted_qids:
        perms_by_qid[qid] = [
            rng.permutation(num_rollouts).tolist() for _ in range(num_permutations)
        ]
    return perms_by_qid


def evaluate_question_opportunity(
    question: QuestionRecord,
    permutations: List[List[int]],
    budgets: Sequence[int] = BUDGETS,
) -> Dict[int, Dict[str, float]]:
    """Evaluate repeated sampling opportunity for a single question across budgets.
    
    Averages metrics across the permutations.
    """
    results: Dict[int, Dict[str, float]] = {}
    n_perms = len(permutations)
    
    # Pre-grade all rollouts once
    rollout_correctness = [
        grade_answer(r.extracted_answer, question.gold_answer, question.dataset, question.metadata)
        for r in question.rollouts
    ]

    for N in budgets:
        first_ans_acc_list = []
        oracle_cov_list = []
        num_groups_list = []
        sampling_fail_list = []
        selection_fail_list = []
        first_ans_tokens_list = []
        total_tokens_list = []

        for p in permutations:
            prefix_indices = p[:N]
            candidates = [question.rollouts[idx] for idx in prefix_indices]
            cand_correct = [rollout_correctness[idx] for idx in prefix_indices]

            # FirstAns
            first_correct = cand_correct[0]
            first_ans_acc_list.append(1.0 if first_correct else 0.0)
            first_ans_tokens_list.append(candidates[0].T)
            total_tokens_list.append(sum(c.T for c in candidates))

            # Oracle coverage
            oracle_has_correct = any(cand_correct)
            oracle_cov_list.append(1.0 if oracle_has_correct else 0.0)

            # Answer groups
            answers = [c.extracted_answer for c in candidates]
            _, groups = group_answers(answers, dataset=question.dataset, metadata=question.metadata)
            num_groups_list.append(len(groups))

            # Failure decomposition for FirstAns
            if not first_correct:
                if not oracle_has_correct:
                    sampling_fail_list.append(1.0)
                    selection_fail_list.append(0.0)
                else:
                    sampling_fail_list.append(0.0)
                    selection_fail_list.append(1.0)
            else:
                sampling_fail_list.append(0.0)
                selection_fail_list.append(0.0)

        results[N] = {
            "first_ans_accuracy": float(np.mean(first_ans_acc_list)),
            "first_ans_tokens": float(np.mean(first_ans_tokens_list)),
            "oracle_coverage": float(np.mean(oracle_cov_list)),
            "distinct_groups": float(np.mean(num_groups_list)),
            "sampling_failure": float(np.mean(sampling_fail_list)),
            "selection_failure": float(np.mean(selection_fail_list)),
            "total_candidate_tokens": float(np.mean(total_tokens_list)),
        }

    return results


def evaluate_dataset_opportunity(
    questions: Dict[str, QuestionRecord],
    question_ids: Sequence[str],
    permutations: Optional[Dict[str, List[List[int]]]] = None,
    budgets: Sequence[int] = BUDGETS,
    seed: int = 6304,
) -> Dict[str, Any]:
    """Evaluate opportunity from repeated sampling across a set of questions.
    
    Returns aggregated metrics and 95% bootstrap confidence intervals per budget.
    """
    if permutations is None:
        permutations = generate_rollout_permutations(question_ids, seed=seed)

    per_question_results: Dict[str, Dict[int, Dict[str, float]]] = {}
    for qid in question_ids:
        q = questions[qid]
        p_list = permutations[qid]
        per_question_results[qid] = evaluate_question_opportunity(q, p_list, budgets=budgets)

    # Aggregate across questions
    summary_by_budget: Dict[int, Dict[str, Any]] = {}
    
    for N in budgets:
        metric_keys = [
            "first_ans_accuracy",
            "first_ans_tokens",
            "oracle_coverage",
            "distinct_groups",
            "sampling_failure",
            "selection_failure",
            "total_candidate_tokens",
        ]
        summary_by_budget[N] = {}
        for m_key in metric_keys:
            vals = [per_question_results[qid][N][m_key] for qid in question_ids]
            mean_val, low_ci, high_ci = bootstrap_pair_metric_ci(vals, n_resamples=1000, seed=seed)
            summary_by_budget[N][m_key] = {
                "mean": mean_val,
                "ci_lower": low_ci,
                "ci_upper": high_ci,
            }

    return {
        "summary": summary_by_budget,
        "per_question": per_question_results,
    }
