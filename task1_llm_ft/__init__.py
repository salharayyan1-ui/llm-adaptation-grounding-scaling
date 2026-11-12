"""Task 1: Large Language Model Fine-Tuning Module.

Provides:
- Dataset filtering, formatting, decontamination, and effective-batch collation (`dataset`)
- Custom LoRA and OPLoRA adapter implementations and factor projection (`models`)
- Starting-point L2 regularization (L2-SP) and displacement tracking (`l2_sp`)
- Training engine with cosine schedule, FP16 AMP, and recovery checkpoints (`trainer`)
- Weight-space (\\rho_k, \\eta) and activation diagnostics R(z) (`diagnostics`)
- Zero-shot benchmarks, qualitative evaluation, and 42-response annotation subset (`evaluate`)
"""

from task1_llm_ft.dataset import (
    MetaMathQADataset,
    EffectiveBatchCollator,
    create_dataloader,
    filter_metamath_dataset,
    build_gsm8k_decontamination_set,
    format_qwen_chat,
)

from task1_llm_ft.models import (
    LoRALinear,
    OPLoRALinear,
    apply_lora_to_model,
    apply_oplora_to_model,
    merge_adapters_to_model,
    unmerge_adapters_from_model,
    export_all_projected_factors,
    verify_merged_equivalence,
    count_parameters,
    DEFAULT_TARGET_MODULES,
)

from task1_llm_ft.l2_sp import (
    L2SPRegularizer,
    compute_parameter_displacement,
)

from task1_llm_ft.trainer import (
    Task1Trainer,
    TrainingConfig,
    create_optimizer,
    get_cosine_schedule_with_warmup_lr,
    run_task1_training,
)

from task1_llm_ft.diagnostics import (
    compute_weight_diagnostics,
    ActivationProbeCacher,
    run_activation_diagnostics,
    sample_probe_set,
    run_task1_diagnostics,
    DIAGNOSTIC_LAYERS,
    PROTECTION_RANK_K,
)

from task1_llm_ft.evaluate import (
    evaluate_gsm8k_generation,
    evaluate_with_lm_eval,
    generate_qualitative_responses,
    generate_annotation_subset,
    extract_gsm8k_answer,
    run_task1_evaluations,
    run_qualitative_suite,
    EVALUATION_CONDITIONS,
    BENCHMARK_TASKS,
)

__all__ = [
    # Dataset
    "MetaMathQADataset",
    "EffectiveBatchCollator",
    "create_dataloader",
    "filter_metamath_dataset",
    "build_gsm8k_decontamination_set",
    "format_qwen_chat",
    # Models
    "LoRALinear",
    "OPLoRALinear",
    "apply_lora_to_model",
    "apply_oplora_to_model",
    "merge_adapters_to_model",
    "unmerge_adapters_from_model",
    "export_all_projected_factors",
    "verify_merged_equivalence",
    "count_parameters",
    "DEFAULT_TARGET_MODULES",
    # L2-SP
    "L2SPRegularizer",
    "compute_parameter_displacement",
    # Trainer
    "Task1Trainer",
    "TrainingConfig",
    "create_optimizer",
    "get_cosine_schedule_with_warmup_lr",
    "run_task1_training",
    # Diagnostics
    "compute_weight_diagnostics",
    "ActivationProbeCacher",
    "run_activation_diagnostics",
    "sample_probe_set",
    "run_task1_diagnostics",
    "DIAGNOSTIC_LAYERS",
    "PROTECTION_RANK_K",
    # Evaluation
    "evaluate_gsm8k_generation",
    "evaluate_with_lm_eval",
    "generate_qualitative_responses",
    "generate_annotation_subset",
    "extract_gsm8k_answer",
    "run_task1_evaluations",
    "run_qualitative_suite",
    "EVALUATION_CONDITIONS",
    "BENCHMARK_TASKS",
]
