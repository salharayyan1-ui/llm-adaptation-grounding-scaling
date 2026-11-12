"""Comprehensive adversarial stress tests and edge-case audit for Task 1: LLM Fine-Tuning.

Tests:
1. MetaMathQA data filtering:
   - Pathological inputs: empty questions, empty responses, whitespace-only, None, malformed dicts.
   - Sequence length boundary conditions: 1 token, 2047 tokens, exactly 2048 tokens, 2049 tokens, 4000 tokens.
   - GSM8K decontamination filtering on query and original_question.
   - Target count stopping boundary condition and source order preservation.
2. GSM8K decontamination normalizer:
   - LaTeX notation: inline math $16$, \\(16\\), \\$15, \\frac{1}{2}, \\text{days}, \\textbf{Calculate}, \\mathbf{X}.
   - Punctuation differences: straight vs escaped vs unicode curly apostrophes, quotes, hyphens vs en/em dashes.
   - Whitespace variants: consecutive spaces, tabs, newlines, carriage returns, non-breaking space \\u00a0.
   - Pathological inputs: empty, whitespace-only, punctuation-only, None.
3. Effective batch token loss normalization:
   - Verify mathematical gradient equivalence between a single large batch vs 16 accumulated microbatches with varying response lengths.
   - Contrast with naive microbatch normalization showing breakdown in gradient equivalence.
   - Edge cases: microbatch with single valid token, microbatch with 0 valid tokens, extreme response length imbalance.
4. Custom LoRA and OPLoRA linear layers:
   - Rank-deficient matrices W0 (e.g., rank 5 matrix with protection rank k=16).
   - Asymmetric shapes: (768, 1024), (1024, 768), (512, 2048).
   - Strict numerical verification: (Uk^T * B_tilde) == 0 and (A_tilde * Vk) == 0 down to 1e-6 tolerance.
   - Merged weights vs unmerged forward output equivalence, idempotence of merge and unmerge.
   - Exported projected factor loading equivalence.
5. L2-SP penalty:
   - Tied parameter handling: memory sharing (data_ptr) counted exactly once in parameters and penalty.
   - Inclusion of biases and 1D normalization weights in reference parameters and penalty.
   - Exact Euclidean distance verification of full-model displacement norm ||theta - theta_0||_2.
6. Diagnostic metrics (rho_k, eta, R(z)):
   - Edge cases when delta_w is zero (initialization).
   - Edge cases with zero-activation inputs z (1D and 2D).
   - Numerical stability with zero-weight matrices W0 and extreme inputs.
"""

import unittest
import math
import numpy as np
import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure mock PyTorch is installed BEFORE importing task1 modules
from tests.mock_torch import setup_mock_torch, MockTensor, MockParameter, MockLinear, MockModule
setup_mock_torch()

import importlib
for mod_name in ["task1_llm_ft.dataset", "task1_llm_ft.models", "task1_llm_ft.l2_sp", "task1_llm_ft.trainer", "task1_llm_ft.diagnostics"]:
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])

from common.normalization import (
    normalize_question_for_contamination,
    extract_boxed_answer,
    normalize_math_answer,
)
from common.metrics import (
    compute_subspace_interference,
    compute_relative_update_magnitude,
    compute_activation_ratio,
)
from task1_llm_ft.dataset import (
    build_gsm8k_decontamination_set,
    format_qwen_chat,
    filter_metamath_dataset,
    EffectiveBatchCollator,
)
from task1_llm_ft.models import (
    LoRALinear,
    OPLoRALinear,
    apply_lora_to_model,
    apply_oplora_to_model,
    merge_adapters_to_model,
    unmerge_adapters_from_model,
    export_all_projected_factors,
    count_parameters,
)
from task1_llm_ft.l2_sp import (
    L2SPRegularizer,
    compute_parameter_displacement,
)
from task1_llm_ft.trainer import get_cosine_schedule_with_warmup_lr


class ControllableMockTokenizer:
    """Mock tokenizer allowing exact control of token count for boundary testing."""
    pad_token_id = 0
    eos_token_id = 151643

    def __init__(self, fixed_length: int = 0):
        self.fixed_length = fixed_length

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        if self.fixed_length > 0:
            if add_generation_prompt:
                prompt_len = min(self.fixed_length // 2, self.fixed_length - 1)
                return [100] * prompt_len
            else:
                return [100] * self.fixed_length

        tokens = [151644]
        for m in messages:
            content = m.get("content", "")
            tokens.extend([abs(hash(w)) % 1000 + 10 for w in content.split()])
        if add_generation_prompt:
            tokens.extend([151645])
        else:
            tokens.extend([self.eos_token_id])
        return tokens

    def encode(self, text, add_special_tokens=False):
        if self.fixed_length > 0:
            return [100] * self.fixed_length
        return [abs(hash(w)) % 1000 + 10 for w in text.split()]


# =====================================================================
# 1. MetaMathQA Data Filtering Stress Tests
# =====================================================================
class TestMetaMathQADataFilteringStress(unittest.TestCase):
    """Adversarial stress tests for MetaMathQA filtering and formatting."""

    def test_pathological_inputs_filtering(self):
        """Test handling of empty, whitespace, None, and malformed inputs."""
        decontam_set = {"what is two plus two"}
        tokenizer = ControllableMockTokenizer()

        pathological_data = [
            # 1. Empty string query
            {"query": "", "response": "Valid answer"},
            # 2. Whitespace-only query
            {"query": "   \t\n   ", "response": "Valid answer"},
            # 3. None query
            {"query": None, "response": "Valid answer"},
            # 4. Missing query key
            {"response": "Valid answer"},
            # 5. Empty string response
            {"query": "Valid question?", "response": ""},
            # 6. Whitespace-only response
            {"query": "Valid question?", "response": "\r\n  \t"},
            # 7. None response
            {"query": "Valid question?", "response": None},
            # 8. Missing response key
            {"query": "Valid question?"},
            # 9. Non-dict items (should be cleanly ignored without crash)
            None,
            "corrupted string line",
            12345,
            {},
            # 10. Item with original_question set to None (must not convert to string 'None')
            {"query": "Valid question 1", "response": "Valid answer 1", "original_question": None},
            # 11. Valid clean item
            {"query": "Find the derivative of x^2", "response": "2x"},
        ]

        filtered = filter_metamath_dataset(
            pathological_data,
            tokenizer=tokenizer,
            gsm8k_decontamination_set=decontam_set,
            target_count=100,
            max_seq_length=2048,
        )

        self.assertEqual(len(filtered), 2, "Only the 2 valid items should survive pathological filtering")
        self.assertEqual(filtered[0]["query"], "Valid question 1")
        self.assertEqual(filtered[1]["query"], "Find the derivative of x^2")

    def test_sequence_length_boundary_conditions(self):
        """Test boundary conditions: exactly 2047, 2048, 2049, and 4000 tokens."""
        raw_items = [
            {"query": "Short Q", "response": "Short A", "id": 1},
            {"query": "Exact 2047 Q", "response": "A", "id": 2047},
            {"query": "Exact 2048 Q", "response": "A", "id": 2048},
            {"query": "Exceed 2049 Q", "response": "A", "id": 2049},
            {"query": "Exceed 4000 Q", "response": "A", "id": 4000},
        ]

        class BoundaryTokenizer:
            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, enable_thinking=False):
                content = messages[0]["content"] if messages else ""
                if "Short" in content:
                    tokens = [1] * 10
                elif "2047" in content:
                    tokens = [1] * 2047
                elif "2048" in content:
                    tokens = [1] * 2048
                elif "2049" in content:
                    tokens = [1] * 2049
                elif "4000" in content:
                    tokens = [1] * 4000
                else:
                    tokens = [1] * 50
                if add_generation_prompt:
                    return tokens[: max(1, len(tokens) // 2)]
                return tokens

        tok = BoundaryTokenizer()
        filtered = filter_metamath_dataset(
            raw_items,
            tokenizer=tok,
            target_count=100,
            max_seq_length=2048,
        )

        # 10 tokens (Short), 2047 tokens, and 2048 tokens are <= 2048 -> Retained (3 items)
        # 2049 tokens and 4000 tokens are > 2048 -> Filtered out
        self.assertEqual(len(filtered), 3)
        retained_queries = [x["query"] for x in filtered]
        self.assertIn("Short Q", retained_queries)
        self.assertIn("Exact 2047 Q", retained_queries)
        self.assertIn("Exact 2048 Q", retained_queries)
        self.assertNotIn("Exceed 2049 Q", retained_queries)
        self.assertNotIn("Exceed 4000 Q", retained_queries)

    def test_target_count_stopping_and_order_preservation(self):
        """Test that filtering halts immediately upon reaching target_count and preserves order."""
        raw_items = [{"query": f"Question {i}", "response": f"Answer {i}"} for i in range(50)]
        target = 7
        filtered = filter_metamath_dataset(
            raw_items,
            tokenizer=ControllableMockTokenizer(),
            target_count=target,
            max_seq_length=2048,
        )

        self.assertEqual(len(filtered), target)
        # Verify strict source order
        for idx, item in enumerate(filtered):
            self.assertEqual(item["query"], f"Question {idx}")
            self.assertEqual(item["response"], f"Answer {idx}")

    def test_decontamination_matching_sources(self):
        """Test decontamination matching against normalized query and original_question."""
        decontam_set = {"a train leaves station a at 50 mph"}
        items = [
            # Contaminated via query
            {"query": "A train leaves station A at 50 mph?", "response": "100 miles"},
            # Contaminated via original_question
            {"query": "Rephrased math question", "response": "100", "original_question": "A train leaves station A at 50 mph."},
            # Clean item
            {"query": "What is the area of a circle with radius 4?", "response": "16 pi"},
        ]

        filtered = filter_metamath_dataset(
            items,
            tokenizer=ControllableMockTokenizer(),
            gsm8k_decontamination_set=decontam_set,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["query"], "What is the area of a circle with radius 4?")


# =====================================================================
# 2. GSM8K Decontamination Normalizer Stress Tests
# =====================================================================
class TestGSM8KDecontaminationNormalizerStress(unittest.TestCase):
    """Adversarial stress tests for GSM8K question text normalization."""

    def test_latex_notation_normalization(self):
        """Test LaTeX notation normalization: inline math, fractions, formatting commands."""
        # Inline math delimiters $...$ and \(...\)
        t1 = "Janet's ducks lay $16$ eggs per day."
        t2 = "Janet's ducks lay \\(16\\) eggs per day."
        t3 = "Janet's ducks lay 16 eggs per day."
        self.assertEqual(
            normalize_question_for_contamination(t1),
            normalize_question_for_contamination(t3),
        )
        self.assertEqual(
            normalize_question_for_contamination(t2),
            normalize_question_for_contamination(t3),
        )

        # Dollar sign amounts
        d1 = "The price of the book is $15."
        d2 = "The price of the book is 15."
        self.assertEqual(
            normalize_question_for_contamination(d1),
            normalize_question_for_contamination(d2),
        )

        # Formatting commands: \textbf{...}, \mathrm{...}, \text{...}, \mathbf{...}
        cmd_text = "\\textbf{Calculate} the \\mathrm{total} value."
        norm_cmd = normalize_question_for_contamination(cmd_text)
        self.assertEqual(norm_cmd, "calculate the total value")

        # Fractions: \frac{1}{2} -> 12
        frac_text = "Find \\frac{1}{2} of 10."
        self.assertEqual(normalize_question_for_contamination(frac_text), "find 12 of 10")

    def test_punctuation_and_unicode_variants(self):
        """Test straight vs escaped vs unicode curly apostrophes, hyphens vs dashes."""
        # Apostrophe variants
        q_straight = "Janet's ducks lay 16 eggs."
        q_escaped = "Janet\\'s ducks lay 16 eggs."
        q_curly = "Janet’s ducks lay 16 eggs."  # \u2019
        norm_straight = normalize_question_for_contamination(q_straight)
        norm_escaped = normalize_question_for_contamination(q_escaped)
        norm_curly = normalize_question_for_contamination(q_curly)
        self.assertEqual(norm_straight, "janets ducks lay 16 eggs")
        self.assertEqual(norm_straight, norm_escaped)
        self.assertEqual(norm_straight, norm_curly)

        # Quotes variants: straight vs curly double quotes
        q_q1 = 'He said "hello" to everyone.'
        q_q2 = "He said “hello” to everyone."  # \u201c, \u201d
        self.assertEqual(
            normalize_question_for_contamination(q_q1),
            normalize_question_for_contamination(q_q2),
        )

        # Math operators with whitespace: 2 + 3 = 5 vs 2  +  3  =  5
        m1 = "Find if 2 + 3 = 5 is true."
        m2 = "Find if 2  +  3  =  5 is true."
        self.assertEqual(
            normalize_question_for_contamination(m1),
            normalize_question_for_contamination(m2),
        )
        self.assertEqual(normalize_question_for_contamination(m1), "find if 2 3 5 is true")

        # Sentence punctuation variants (?, !, ., ..., ;, :)
        s1 = "How many cows remain?"
        s2 = "How many cows remain."
        s3 = "How many cows remain!!!"
        s4 = "How many cows remain..."
        self.assertEqual(normalize_question_for_contamination(s1), "how many cows remain")
        self.assertEqual(normalize_question_for_contamination(s1), normalize_question_for_contamination(s2))
        self.assertEqual(normalize_question_for_contamination(s1), normalize_question_for_contamination(s3))
        self.assertEqual(normalize_question_for_contamination(s1), normalize_question_for_contamination(s4))

    def test_whitespace_variants_and_pathological_strings(self):
        """Test tabs, newlines, non-breaking space \\u00a0, and empty inputs."""
        # Non-breaking spaces and tabs
        ws_text = "\t\nJanet's   ducks\n\nlay   16   eggs\u00a0per day.\r\n"
        norm = normalize_question_for_contamination(ws_text)
        self.assertEqual(norm, "janets ducks lay 16 eggs per day")

        # Pathological inputs
        self.assertEqual(normalize_question_for_contamination(""), "")
        self.assertEqual(normalize_question_for_contamination("   \t\n  "), "")
        self.assertEqual(normalize_question_for_contamination("!@#$%^&*()_+=-[]{};':\",.<>/?\\|"), "")
        self.assertEqual(normalize_question_for_contamination(None), "")


# =====================================================================
# 3. Effective Batch Token Loss Normalization Stress Tests
# =====================================================================
class TestEffectiveBatchLossNormalizationStress(unittest.TestCase):
    """Verify gradient equivalence between a single large batch vs 16 microbatches with varying lengths."""

    def test_gradient_equivalence_single_batch_vs_16_microbatches(self):
        """Strict numerical proof and test of gradient equivalence under varying response lengths.
        
        Setting:
        - 16 microbatches, each with different response token counts N_m.
        - Linear layer W: d_in -> vocab_size.
        - Method A: Single large batch normalized by total response tokens N_total = sum(N_m).
        - Method B: 16 microbatches accumulated, each normalized by N_total.
        - Contrast Method C: Naive microbatch normalization (dividing by N_m or by 16).
        """
        rng = np.random.default_rng(6304)
        d_in = 32
        vocab_size = 64
        num_microbatches = 16

        # Fixed linear layer weight
        W = rng.standard_normal((vocab_size, d_in), dtype=np.float64)

        # Create 16 microbatches with varying response token counts (e.g. from 5 to 45 tokens)
        microbatch_data = []
        total_valid_tokens = 0

        for m in range(num_microbatches):
            batch_size = 2
            seq_len = rng.integers(15, 50)
            valid_tokens_count = rng.integers(5, seq_len * batch_size - 5)
            
            total_tokens = batch_size * seq_len
            X = rng.standard_normal((total_tokens, d_in), dtype=np.float64)

            labels = np.full(total_tokens, -100, dtype=np.int64)
            valid_indices = rng.choice(total_tokens, size=valid_tokens_count, replace=False)
            labels[valid_indices] = rng.integers(0, vocab_size, size=valid_tokens_count)

            microbatch_data.append({
                "X": X,
                "labels": labels,
                "valid_indices": valid_indices,
                "num_valid_tokens": len(valid_indices),
            })
            total_valid_tokens += len(valid_indices)

        self.assertGreater(total_valid_tokens, 0)

        # -------------------------------------------------------------
        # Method A: Single Large Batch
        # -------------------------------------------------------------
        X_all = np.vstack([mb["X"] for mb in microbatch_data])
        labels_all = np.concatenate([mb["labels"] for mb in microbatch_data])
        valid_mask_all = (labels_all != -100)
        self.assertEqual(np.sum(valid_mask_all), total_valid_tokens)

        logits_all = X_all @ W.T
        max_logits_all = np.max(logits_all, axis=-1, keepdims=True)
        exp_logits_all = np.exp(logits_all - max_logits_all)
        probs_all = exp_logits_all / np.sum(exp_logits_all, axis=-1, keepdims=True)

        grad_logits_all = np.zeros_like(logits_all)
        for i in range(len(labels_all)):
            if valid_mask_all[i]:
                target = labels_all[i]
                grad_logits_all[i] = probs_all[i].copy()
                grad_logits_all[i, target] -= 1.0
                grad_logits_all[i] /= total_valid_tokens

        grad_W_single = grad_logits_all.T @ X_all

        # -------------------------------------------------------------
        # Method B: 16 Accumulated Microbatches (Effective Batch Token Normalization)
        # -------------------------------------------------------------
        grad_W_accum = np.zeros_like(W)

        for mb in microbatch_data:
            X_m = mb["X"]
            labels_m = mb["labels"]
            valid_mask_m = (labels_m != -100)

            logits_m = X_m @ W.T
            max_m = np.max(logits_m, axis=-1, keepdims=True)
            exp_m = np.exp(logits_m - max_m)
            probs_m = exp_m / np.sum(exp_m, axis=-1, keepdims=True)

            grad_logits_m = np.zeros_like(logits_m)
            for i in range(len(labels_m)):
                if valid_mask_m[i]:
                    target = labels_m[i]
                    grad_logits_m[i] = probs_m[i].copy()
                    grad_logits_m[i, target] -= 1.0
                    grad_logits_m[i] /= total_valid_tokens

            grad_W_m = grad_logits_m.T @ X_m
            grad_W_accum += grad_W_m

        # Verify strict mathematical equivalence down to machine epsilon (1e-12)
        diff_accum = np.max(np.abs(grad_W_single - grad_W_accum))
        self.assertLess(
            diff_accum,
            1e-12,
            f"Accumulated microbatch gradient must match single large batch gradient exactly. Diff: {diff_accum}",
        )

        # -------------------------------------------------------------
        # Contrast Method C: Naive Microbatch Normalization
        # -------------------------------------------------------------
        grad_W_naive = np.zeros_like(W)
        for mb in microbatch_data:
            X_m = mb["X"]
            labels_m = mb["labels"]
            valid_mask_m = (labels_m != -100)
            n_m = max(1, mb["num_valid_tokens"])

            logits_m = X_m @ W.T
            max_m = np.max(logits_m, axis=-1, keepdims=True)
            exp_m = np.exp(logits_m - max_m)
            probs_m = exp_m / np.sum(exp_m, axis=-1, keepdims=True)

            grad_logits_m = np.zeros_like(logits_m)
            for i in range(len(labels_m)):
                if valid_mask_m[i]:
                    target = labels_m[i]
                    grad_logits_m[i] = probs_m[i].copy()
                    grad_logits_m[i, target] -= 1.0
                    grad_logits_m[i] /= (n_m * num_microbatches)

            grad_W_naive += (grad_logits_m.T @ X_m)

        diff_naive = np.max(np.abs(grad_W_single - grad_W_naive))
        self.assertGreater(
            diff_naive,
            1e-3,
            "Naive normalization must show gradient discrepancy when response lengths vary",
        )

    def test_loss_normalization_edge_cases(self):
        """Test edge cases: single valid token, microbatch with 0 valid tokens, empty batch."""
        collator = EffectiveBatchCollator(pad_token_id=0)

        # Microbatch where all tokens are masked (-100)
        batch_all_masked = [
            {"id": 0, "input_ids": [10, 20, 30], "labels": [-100, -100, -100]},
            {"id": 1, "input_ids": [40, 50], "labels": [-100, -100]},
        ]
        out_masked = collator(batch_all_masked)
        self.assertEqual(out_masked["num_valid_tokens"], 0)

        # Microbatch with exactly 1 valid token
        batch_single = [
            {"id": 0, "input_ids": [10, 20, 30], "labels": [-100, -100, 42]},
        ]
        out_single = collator(batch_single)
        self.assertEqual(out_single["num_valid_tokens"], 1)

        # Empty batch handling
        out_empty = collator([])
        self.assertEqual(out_empty["num_valid_tokens"], 0)


# =====================================================================
# 4. Custom LoRA and OPLoRA Linear Layers Stress Tests
# =====================================================================
class TestLoRAAndOPLoRALinearLayersStress(unittest.TestCase):
    """Stress tests for LoRA and OPLoRA linear layers: rank-deficient W0, asymmetric shapes, orthogonality."""

    def test_rank_deficient_w0_matrices(self):
        """Test OPLoRA SVD projection when W0 is rank-deficient (rank 5 in 128x64)."""
        d_out, d_in = 128, 64
        true_rank = 5
        protection_rank = 16
        r = 8
        alpha = 16.0

        rng = np.random.default_rng(6304)
        A_rand = rng.standard_normal((d_out, true_rank))
        B_rand = rng.standard_normal((true_rank, d_in))
        W0 = (A_rand @ B_rand).astype(np.float32)
        self.assertLessEqual(np.linalg.matrix_rank(W0), true_rank)

        base_layer = MockLinear(d_in, d_out, bias=False)
        base_layer.weight.data = MockTensor(W0)

        oplora = OPLoRALinear(
            base_layer=base_layer,
            r=r,
            lora_alpha=alpha,
            protection_rank=protection_rank,
            lora_dropout=0.0,
        )

        # 1. At initialization: B is 0, A is random
        Uk = oplora.Uk.numpy()
        Vk = oplora.Vk.numpy()
        A_tilde_init, B_tilde_init = oplora.get_projected_factors()
        self.assertLess(np.max(np.abs(Uk.T @ B_tilde_init.numpy())), 1e-12)
        self.assertLess(np.max(np.abs(A_tilde_init.numpy() @ Vk)), 1e-6)

        # 2. With non-zero trained adapter weights (standard adapter magnitude std 0.05)
        oplora.lora_A.data = MockTensor((rng.standard_normal((r, d_in)) * 0.05).astype(np.float32))
        oplora.lora_B.data = MockTensor((rng.standard_normal((d_out, r)) * 0.05).astype(np.float32))

        A_tilde, B_tilde = oplora.get_projected_factors()
        A_tilde_np = A_tilde.numpy()
        B_tilde_np = B_tilde.numpy()

        err_B = np.max(np.abs(Uk.T @ B_tilde_np))
        err_A = np.max(np.abs(A_tilde_np @ Vk))
        self.assertLess(err_B, 1e-6, f"Uk^T B_tilde must be zero for rank-deficient W0. Got {err_B}")
        self.assertLess(err_A, 1e-6, f"A_tilde Vk must be zero for rank-deficient W0. Got {err_A}")

        delta_w = oplora.get_delta_weight().numpy()
        rho_k = compute_subspace_interference(delta_w, Uk)
        self.assertLess(rho_k, 1e-6, f"OPLoRA subspace interference rho_k must be 0. Got {rho_k}")

    def test_asymmetric_shapes_orthogonality(self):
        """Test asymmetric shapes: 768x1024, 1024x768, and 512x2048."""
        test_shapes = [
            (768, 1024, 16, 64),
            (1024, 768, 16, 64),
            (512, 2048, 8, 32),
        ]

        rng = np.random.default_rng(6304)

        for d_out, d_in, r, k in test_shapes:
            with self.subTest(shape=(d_out, d_in), r=r, k=k):
                W0 = rng.standard_normal((d_out, d_in)).astype(np.float32)
                base = MockLinear(d_in, d_out, bias=False)
                base.weight.data = MockTensor(W0)

                oplora = OPLoRALinear(
                    base_layer=base,
                    r=r,
                    lora_alpha=float(2 * r),
                    protection_rank=k,
                    lora_dropout=0.0,
                )

                Uk = oplora.Uk.numpy()
                Vk = oplora.Vk.numpy()
                self.assertEqual(Uk.shape, (d_out, k))
                self.assertEqual(Vk.shape, (d_in, k))

                # Standard adapter magnitude (std 0.05)
                oplora.lora_A.data = MockTensor((rng.standard_normal((r, d_in)) * 0.05).astype(np.float32))
                oplora.lora_B.data = MockTensor((rng.standard_normal((d_out, r)) * 0.05).astype(np.float32))

                A_tilde, B_tilde = oplora.get_projected_factors()
                A_tilde_np = A_tilde.numpy()
                B_tilde_np = B_tilde.numpy()

                err_B = np.max(np.abs(Uk.T @ B_tilde_np))
                err_A = np.max(np.abs(A_tilde_np @ Vk))
                self.assertLess(err_B, 1e-6, f"Uk^T B_tilde failed for shape ({d_out}, {d_in}): {err_B}")
                self.assertLess(err_A, 1e-6, f"A_tilde Vk failed for shape ({d_out}, {d_in}): {err_A}")

    def test_merged_weights_vs_unmerged_forward_output_equivalence(self):
        """Strict verification that merged model produces identical output to unmerged model."""
        d_out, d_in, r, k = 64, 32, 8, 16
        rng = np.random.default_rng(6304)

        base_unmerged = MockLinear(d_in, d_out, bias=True)
        w_init = rng.standard_normal((d_out, d_in)).astype(np.float32)
        b_init = rng.standard_normal(d_out).astype(np.float32)
        base_unmerged.weight.data = MockTensor(w_init)
        base_unmerged.bias.data = MockTensor(b_init)

        oplora = OPLoRALinear(base_unmerged, r=r, lora_alpha=16.0, protection_rank=k, lora_dropout=0.0)
        oplora.lora_A.data = MockTensor((rng.standard_normal((r, d_in)) * 0.1).astype(np.float32))
        oplora.lora_B.data = MockTensor((rng.standard_normal((d_out, r)) * 0.1).astype(np.float32))

        x = MockTensor(rng.standard_normal((4, d_in)).astype(np.float32))

        # 1. Forward pass before merge
        out_unmerged = oplora(x).numpy()

        # 2. Merge adapter into base weights
        oplora.merge()
        self.assertTrue(oplora.merged)

        # 3. Forward pass after merge
        out_merged = oplora(x).numpy()

        # Float32 matrix associativity tolerance (atol <= 2e-5)
        diff_merge = np.max(np.abs(out_unmerged - out_merged))
        self.assertLess(diff_merge, 2e-5, f"Merged output must match unmerged forward pass: diff {diff_merge}")

        # 4. Calling merge again should be idempotent
        oplora.merge()
        out_merged_twice = oplora(x).numpy()
        self.assertLess(np.max(np.abs(out_merged - out_merged_twice)), 1e-12)

        # 5. Unmerge restores base weights
        oplora.unmerge()
        self.assertFalse(oplora.merged)
        restored_weight = oplora.base_layer.weight.numpy()
        self.assertLess(np.max(np.abs(restored_weight - w_init)), 1e-6, "Unmerge must restore initial base weights")

        # 6. Unmerge idempotence
        oplora.unmerge()
        self.assertLess(np.max(np.abs(oplora.base_layer.weight.numpy() - w_init)), 1e-6)

    def test_export_projected_factors_equivalence(self):
        """Test exporting projected factors and loading into standard LoRALinear."""
        d_out, d_in, r, k = 48, 24, 4, 8
        rng = np.random.default_rng(6304)

        base1 = MockLinear(d_in, d_out, bias=False)
        base1.weight.data = MockTensor(rng.standard_normal((d_out, d_in)).astype(np.float32))

        oplora = OPLoRALinear(base1, r=r, lora_alpha=8.0, protection_rank=k, lora_dropout=0.0)
        oplora.lora_A.data = MockTensor((rng.standard_normal((r, d_in)) * 0.1).astype(np.float32))
        oplora.lora_B.data = MockTensor((rng.standard_normal((d_out, r)) * 0.1).astype(np.float32))

        exported = oplora.export_projected_factors()
        self.assertIn("lora_A", exported)
        self.assertIn("lora_B", exported)

        # Load into standard LoRALinear with identical base weights
        base2 = MockLinear(d_in, d_out, bias=False)
        base2.weight.data = MockTensor(base1.weight.numpy().copy())

        standard_lora = LoRALinear(base2, r=r, lora_alpha=8.0, lora_dropout=0.0)
        standard_lora.lora_A.data = exported["lora_A"]
        standard_lora.lora_B.data = exported["lora_B"]

        x = MockTensor(rng.standard_normal((2, d_in)).astype(np.float32))
        out_oplora = oplora(x).numpy()
        out_standard = standard_lora(x).numpy()

        diff = np.max(np.abs(out_oplora - out_standard))
        self.assertLess(diff, 1e-6, "Exported projected factors in standard LoRA must match OPLoRA output")


# =====================================================================
# 5. L2-SP Penalty Stress Tests
# =====================================================================
class TestL2SPPenaltyStress(unittest.TestCase):
    """Stress tests for L2-SP proximity penalty, tied parameters, 1D weights, and displacement."""

    def test_tied_parameters_handled_exactly_once(self):
        """Verify that tied parameters (e.g. embed_tokens and lm_head sharing memory) are counted once."""
        class MockTiedModel(MockModule):
            def __init__(self):
                super().__init__()
                self.embed = MockLinear(10, 20, bias=False)
                self.head = MockLinear(20, 10, bias=False)
                self.head.weight = self.embed.weight
                self.norm = MockParameter(np.ones(10, dtype=np.float32))
                self.bias = MockParameter(np.zeros(10, dtype=np.float32))

        model = MockTiedModel()
        regularizer = L2SPRegularizer(model, lambda_reg=1e-3)

        weight_params = [name for name in regularizer.param_names if "weight" in name]
        self.assertEqual(len(weight_params), 1, "Tied parameter must appear only once in reference param names")

        # Total elements: embed weight (200) + norm (10) + bias (10) = 220
        self.assertEqual(regularizer.total_elements, 220)

        # Initial penalty must be exactly 0
        pen_tensor, omega_0 = regularizer.compute_penalty(model)
        self.assertEqual(omega_0, 0.0)

        # Perturb the tied parameter: 200 elements by 0.5 each -> sum = 50.0, Omega = 25.0
        model.embed.weight._data += 0.5

        pen_tensor, omega_perturbed = regularizer.compute_penalty(model)
        self.assertAlmostEqual(omega_perturbed, 25.0, places=6, msg="Tied parameter must not be double counted in Omega")

        disp = regularizer.compute_displacement(model)
        self.assertAlmostEqual(disp, math.sqrt(50.0), places=6)

    def test_biases_and_1d_normalization_weights_included(self):
        """Verify biases and 1D normalization weights are captured in L2-SP penalty."""
        class MockModelWith1D(MockModule):
            def __init__(self):
                super().__init__()
                self.linear = MockLinear(16, 32, bias=True)
                self.layernorm_weight = MockParameter(np.ones(16, dtype=np.float32))
                self.layernorm_bias = MockParameter(np.zeros(16, dtype=np.float32))

        model = MockModelWith1D()
        reg = L2SPRegularizer(model, lambda_reg=1e-3)

        self.assertIn("linear.weight", reg.param_names)
        self.assertIn("linear.bias", reg.param_names)
        self.assertIn("layernorm_weight", reg.param_names)
        self.assertIn("layernorm_bias", reg.param_names)

        # Perturb ONLY the 1D layernorm_weight: 16 elements by 0.2 each -> sum = 0.64, Omega = 0.32
        model.layernorm_weight._data += 0.2
        _, omega = reg.compute_penalty(model)
        self.assertAlmostEqual(omega, 0.32, places=6)

    def test_full_model_displacement_norm_euclidean(self):
        """Verify full-model displacement norm ||theta - theta_0||_2 matches exact Euclidean distance."""
        rng = np.random.default_rng(6304)

        class MultiParamModel(MockModule):
            def __init__(self):
                super().__init__()
                self.w1 = MockParameter(rng.standard_normal((30, 20)).astype(np.float32))
                self.b1 = MockParameter(rng.standard_normal(30).astype(np.float32))
                self.norm = MockParameter(np.ones(20, dtype=np.float32))

        model = MultiParamModel()
        reg = L2SPRegularizer(model, lambda_reg=1e-3)

        initial_state = {k: v.numpy().copy() for k, v in model.named_parameters()}

        diff_sq_sum = 0.0
        for name, param in model.named_parameters():
            delta = rng.standard_normal(param.shape).astype(np.float32) * 0.1
            param._data += delta
            diff_sq_sum += np.sum(delta ** 2)

        expected_euclidean = math.sqrt(diff_sq_sum)

        disp_reg = reg.compute_displacement(model)
        self.assertAlmostEqual(disp_reg, expected_euclidean, places=5)

        state_dict_mock = {k: MockTensor(v) for k, v in initial_state.items()}
        disp_direct = compute_parameter_displacement(model, state_dict_mock)
        self.assertAlmostEqual(disp_direct, expected_euclidean, places=5)

    def test_parameter_count_deduplication(self):
        """Verify count_parameters deduplicates tied weights."""
        class TiedModel(MockModule):
            def __init__(self):
                super().__init__()
                self.embed = MockLinear(50, 100, bias=False)
                self.head = MockLinear(100, 50, bias=False)
                self.head.weight = self.embed.weight

        m = TiedModel()
        counts = count_parameters(m)
        self.assertEqual(counts["total"], 5000, "Tied parameters must be counted only once in parameter total")


# =====================================================================
# 6. Diagnostic Metrics Stress Tests (rho_k, eta, R(z))
# =====================================================================
class TestDiagnosticMetricsStress(unittest.TestCase):
    """Stress tests for diagnostic metrics: zero delta_w, zero activation z, scale invariance."""

    def test_subspace_interference_rho_k_edge_cases(self):
        """Test rho_k when delta_w is zero, fully protected, or fully aligned."""
        d_out, d_in, k = 64, 32, 16
        rng = np.random.default_rng(6304)

        Q, _ = np.linalg.qr(rng.standard_normal((d_out, k)))
        Uk = Q[:, :k]

        # 1. Delta_W is zero (at initialization)
        zero_delta = np.zeros((d_out, d_in))
        rho_zero = compute_subspace_interference(zero_delta, Uk)
        self.assertEqual(rho_zero, 0.0, "rho_k must be 0.0 when Delta W is 0")

        # 2. Delta_W is entirely in the column space of Uk: Delta_W = Uk @ M
        M = rng.standard_normal((k, d_in))
        aligned_delta = Uk @ M
        rho_aligned = compute_subspace_interference(aligned_delta, Uk)
        self.assertAlmostEqual(rho_aligned, 1.0, places=9, msg="rho_k must be 1.0 when Delta W is aligned with Uk")

        # 3. Delta_W is entirely orthogonal to Uk: Delta_W = (I - Uk Uk^T) M
        M_full = rng.standard_normal((d_out, d_in))
        ortho_delta = M_full - Uk @ (Uk.T @ M_full)
        rho_ortho = compute_subspace_interference(ortho_delta, Uk)
        self.assertLess(rho_ortho, 1e-12, "rho_k must be 0.0 when Delta W is orthogonal to Uk")

        # 4. Scale invariance: rho_k(2 * Delta_W) == rho_k(Delta_W)
        mixed_delta = rng.standard_normal((d_out, d_in))
        rho_1 = compute_subspace_interference(mixed_delta, Uk)
        rho_2 = compute_subspace_interference(2.5 * mixed_delta, Uk)
        self.assertAlmostEqual(rho_1, rho_2, places=12, msg="rho_k must be scale-invariant")

    def test_relative_update_magnitude_eta_edge_cases(self):
        """Test eta when delta_w is zero, W0 is zero, or under scaling."""
        d_out, d_in = 40, 20
        rng = np.random.default_rng(6304)
        W0 = rng.standard_normal((d_out, d_in))

        # 1. Zero update
        zero_delta = np.zeros((d_out, d_in))
        self.assertEqual(compute_relative_update_magnitude(zero_delta, W0), 0.0)

        # 2. Zero base weight
        zero_w0 = np.zeros((d_out, d_in))
        delta = rng.standard_normal((d_out, d_in))
        self.assertEqual(compute_relative_update_magnitude(delta, zero_w0), 0.0)

        # 3. Linearity with respect to update scale: eta(2 * delta, W0) == 2 * eta(delta, W0)
        eta_1 = compute_relative_update_magnitude(delta, W0)
        eta_2 = compute_relative_update_magnitude(2.0 * delta, W0)
        self.assertAlmostEqual(eta_2, 2.0 * eta_1, places=12)

    def test_activation_ratio_r_z_edge_cases(self):
        """Test R(z) when delta_w is zero, activation z is zero (1D and 2D), or extreme values."""
        d_out, d_in = 32, 16
        rng = np.random.default_rng(6304)
        W0 = rng.standard_normal((d_out, d_in))
        delta_w = rng.standard_normal((d_out, d_in))

        # 1. Delta_W is zero
        z_random = rng.standard_normal(d_in)
        r_zero_delta = compute_activation_ratio(np.zeros((d_out, d_in)), W0, z_random)
        self.assertEqual(r_zero_delta, 0.0, "R(z) must be 0 when Delta W is 0")

        # 2. Activation z is all zeros (1D)
        z_zero_1d = np.zeros(d_in)
        r_zero_z_1d = compute_activation_ratio(delta_w, W0, z_zero_1d)
        self.assertEqual(r_zero_z_1d, 0.0, "R(z) must be 0 when z is a zero vector")

        # 3. Activation z is all zeros (2D: (d_in, 1))
        z_zero_2d = np.zeros((d_in, 1))
        r_zero_z_2d = compute_activation_ratio(delta_w, W0, z_zero_2d)
        self.assertEqual(r_zero_z_2d, 0.0, "R(z) must be 0 when z is a 2D zero vector")

        # 4. Denominator epsilon prevents division by zero when W0 z == 0 and Delta_W z != 0
        W_rank_deficient = np.zeros((d_out, d_in))
        W_rank_deficient[:10, :] = rng.standard_normal((10, d_in))
        _, _, vh_rd = np.linalg.svd(W_rank_deficient)
        null_z = vh_rd[-1, :]
        self.assertLess(np.linalg.norm(W_rank_deficient @ null_z), 1e-12)

        r_null = compute_activation_ratio(delta_w, W_rank_deficient, null_z)
        self.assertFalse(math.isnan(r_null))
        self.assertFalse(math.isinf(r_null))
        self.assertGreater(r_null, 0.0)

        # 5. Approximate scale invariance for large activations: R(alpha * z) ~ R(z) when ||W0 z|| >> 1e-8
        r_base = compute_activation_ratio(delta_w, W0, z_random)
        r_scaled = compute_activation_ratio(delta_w, W0, 100.0 * z_random)
        self.assertAlmostEqual(r_base, r_scaled, places=6, msg="R(z) should be approximately scale invariant for large z")


if __name__ == "__main__":
    unittest.main()
