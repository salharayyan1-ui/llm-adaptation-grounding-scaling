"""Task 2: Vision-Language Models and Visual Grounding.

Package exports for SigLIP-Qwen VLM architecture, caching, dataset sampling,
curriculum training, evaluation, and visual-token compression.
"""

from task2_vlm.caching import (
    SigLIPFeatureExtractor,
    FeatureCache,
    deterministic_siglip_preprocess,
    cache_images_batch,
    cache_siglip_features,
)

from task2_vlm.compression import (
    compress_visual_tokens_np,
    compress_visual_tokens_torch,
    VisualTokenCompressor,
    run_compression_experiment,
)

from task2_vlm.models import (
    MultimodalConnector,
    SiglipQwenVLM,
    build_qwen_chat_prompt,
    VISUAL_DIM,
    LLM_DIM,
    NUM_VISUAL_TOKENS,
    COMPRESSED_TOKENS,
    IMAGE_PAD_TOKEN_ID,
)

from task2_vlm.dataset import (
    VLMExample,
    QuestionPair,
    CurriculumBatchSampler,
    VLMCollator,
    generate_presence_pairs,
    generate_counting_pairs,
    filter_and_split_vqav2_pairs,
    verify_split_isolation,
    create_synthetic_task2_datasets,
    QUESTION_TOKEN_CAP,
    CAPTION_TOKEN_CAP,
    ANSWER_TOKEN_CAP,
)

from task2_vlm.trainer import (
    VLMCurriculumTrainer,
    CosineWarmupTo10PercentScheduler,
    run_curriculum_training,
)

from task2_vlm.evaluate import (
    build_mismatched_image_mapping,
    evaluate_pair_under_condition,
    compute_suite_metrics_and_ci,
    run_full_evaluation,
    run_grounding_evaluations,
)

__all__ = [
    # Caching
    "SigLIPFeatureExtractor",
    "FeatureCache",
    "deterministic_siglip_preprocess",
    "cache_images_batch",
    "cache_siglip_features",
    # Compression
    "compress_visual_tokens_np",
    "compress_visual_tokens_torch",
    "VisualTokenCompressor",
    "run_compression_experiment",
    # Models
    "MultimodalConnector",
    "SiglipQwenVLM",
    "build_qwen_chat_prompt",
    "VISUAL_DIM",
    "LLM_DIM",
    "NUM_VISUAL_TOKENS",
    "COMPRESSED_TOKENS",
    "IMAGE_PAD_TOKEN_ID",
    # Dataset
    "VLMExample",
    "QuestionPair",
    "CurriculumBatchSampler",
    "VLMCollator",
    "generate_presence_pairs",
    "generate_counting_pairs",
    "filter_and_split_vqav2_pairs",
    "verify_split_isolation",
    "create_synthetic_task2_datasets",
    "QUESTION_TOKEN_CAP",
    "CAPTION_TOKEN_CAP",
    "ANSWER_TOKEN_CAP",
    # Trainer
    "VLMCurriculumTrainer",
    "CosineWarmupTo10PercentScheduler",
    "run_curriculum_training",
    # Evaluation
    "build_mismatched_image_mapping",
    "evaluate_pair_under_condition",
    "compute_suite_metrics_and_ci",
    "run_full_evaluation",
    "run_grounding_evaluations",
]
