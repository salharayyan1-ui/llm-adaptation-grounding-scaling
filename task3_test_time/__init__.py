"""Task 3: Test-Time Compute: Sampling, Selection, and Adaptive Inference.

Modular package for ATML PA3 Task 3.
"""

from task3_test_time.split import (
    generate_splits,
    save_splits,
    load_splits,
)
from task3_test_time.equivalence import (
    are_answers_equivalent,
    group_answers,
    grade_answer,
    normalize_math_candidate,
    normalize_mathvista_candidate,
    NO_ANSWER,
    SCORER_VERSION,
)
from task3_test_time.data_loader import (
    load_dataset,
    verify_token_alignment,
    generate_synthetic_data,
    QuestionRecord,
    RolloutRecord,
)
from task3_test_time.opportunity import (
    evaluate_dataset_opportunity,
    generate_rollout_permutations,
    BUDGETS,
)
from task3_test_time.scoring import (
    evaluate_dataset_scoring,
    compute_mean_logp,
    compute_self_certainty,
    compute_visual_engagement,
    compute_visconf_scores,
    select_highest_score,
    select_self_consistency,
    select_rank_weighted_consensus,
)
from task3_test_time.signals import (
    run_signals_analysis,
    compute_confidence_vision_correlation,
    evaluate_tokenwise_fusion_ablation,
    evaluate_shortest_rollout_baseline,
)
from task3_test_time.live_generation import (
    QwenLiveGenerator,
    run_live_generation_experiment,
    summarize_live_results,
    save_live_results,
)
from task3_test_time.adaptive import (
    search_development_policies,
    evaluate_held_out_policy,
    simulate_adaptive_stopping,
    CHECKPOINTS,
    TAU_GRID,
    GAMMA_GRID,
)

__all__ = [
    "generate_splits",
    "save_splits",
    "load_splits",
    "are_answers_equivalent",
    "group_answers",
    "grade_answer",
    "normalize_math_candidate",
    "normalize_mathvista_candidate",
    "NO_ANSWER",
    "SCORER_VERSION",
    "load_dataset",
    "verify_token_alignment",
    "generate_synthetic_data",
    "QuestionRecord",
    "RolloutRecord",
    "evaluate_dataset_opportunity",
    "generate_rollout_permutations",
    "BUDGETS",
    "evaluate_dataset_scoring",
    "compute_mean_logp",
    "compute_self_certainty",
    "compute_visual_engagement",
    "compute_visconf_scores",
    "select_highest_score",
    "select_self_consistency",
    "select_rank_weighted_consensus",
    "run_signals_analysis",
    "compute_confidence_vision_correlation",
    "evaluate_tokenwise_fusion_ablation",
    "evaluate_shortest_rollout_baseline",
    "QwenLiveGenerator",
    "run_live_generation_experiment",
    "summarize_live_results",
    "save_live_results",
    "search_development_policies",
    "evaluate_held_out_policy",
    "simulate_adaptive_stopping",
    "CHECKPOINTS",
    "TAU_GRID",
    "GAMMA_GRID",
]
